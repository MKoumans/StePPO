"""Compare the diffrax PIDController baseline with RL controllers on the cached test set.

Usage:
    # PID baseline only:
    python compare.py --config configs/envs/ode/van_der_pol/van_der_pol_default.yaml
    # PID vs a trained policy (config read from the checkpoint directory):
    python compare.py --checkpoint <run>/checkpoints/checkpoint_<N>
    # Scaling study over batch sizes:
    python compare.py --checkpoint ... -n 1 2 4 8 16 32 64 128 -r 64

Prints a table of success rate, mean steps and other metrics per method.
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import csv
import dataclasses
import os
import time
from pathlib import Path

import diffrax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from research.ode.post_run_analysis.rollout_cache import load_or_compute
from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv, ODEParams, _make_solver
from steppo.envs.ode.learned_controller import LearnedController
from steppo.envs.ode.systems import get_rhs, get_system
from steppo.training.error_dist import (
    PID_COMPARE_NUM_EPISODES,
    PID_COMPARE_SEED,
    PID_EVAL_REF_TOL_FACTOR,
    _aggregate,
    load_cached_pid_batch,
)
from steppo.utils.checkpoint import build_models, load_checkpoint, resolve_checkpoint


def _run_timed_lams(run_fn, lams, keys, batch_size, repeats):
    """Like error_dist._run_timed, but batches over a (lams, keys) pair together
    instead of just keys — for run_fn signatures that need the fixed λ array
    alongside the per-episode key (see eval_rl/eval_learned_controller, which
    replay a cached dataset's exact task_params rather than sampling their own)."""
    timings = []
    repeat_results = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        batch_results = []
        for start in range(0, len(keys), batch_size):
            batch_res = run_fn(lams[start : start + batch_size], keys[start : start + batch_size])
            batch_results.append(jax.tree.map(np.asarray, batch_res))
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        repeat_results.append(jax.tree.map(lambda *arrs: np.concatenate(arrs), *batch_results))
    return repeat_results, timings


# ─── diffrax PIDController baseline ─────────────────────────────────────────


def eval_diffrax_pid(
    config, bins, num_episodes, max_steps, seed=42, repeats=3, progress_label=None
):
    """Wall-clock timing of the diffrax PIDController baseline, read from the cached PID evaluation dataset."""
    is_unlimited = max_steps != config.rollout_steps
    data = load_cached_pid_batch(
        config.env,
        num_envs=1024,
        seed=0,
        max_steps=config.rollout_steps,
        bins=bins,
        timing_n=128,
        timing_repeats=8,
        timing_unlimited_repeats=None,
        timing_unlimited_max_steps=8192,
        split="test",
    )
    row = dict(data["timing"]["unlimited" if is_unlimited else "budget"])
    row["method"] = f"diffrax PID (budget={max_steps})"
    return row


# ─── RL evaluation (from checkpoint) ────────────────────────────────────────


def policy_method_label(config):
    """Return the table label for the primary learned-policy checkpoint."""
    algorithm = str(getattr(config, "algo", "")).lower()
    if algorithm == "ppo":
        return "StePPO (PPO)"
    backbone = getattr(config, "backbone", "policy")
    return f"{algorithm.upper() or 'RL'} ({backbone})"


def eval_rl(
    env,
    config,
    checkpoint_path,
    lams,
    episode_keys,
    num_episodes,
    seed=42,
    repeats=3,
    method_name="RL (VariBAD)",
):
    """Roll out the trained policy on the (λ, episode_key) pairs from the cached
    PID eval dataset — no live task sampling — so the RL row draws λ from the
    same fixed test set the PID/oracle rows were solved against."""
    from steppo.training.eval import eval_episode_early_exit

    vae, policy = build_models(config, env, config.seed)
    vae, policy = load_checkpoint(
        vae, policy, checkpoint_path, backbone=config.backbone, algo=config.algo
    )

    max_steps = getattr(config, "eval_max_steps", config.rollout_steps)

    @nnx.jit(static_argnames=("env", "max_steps_"))
    def run_batch(vae_, policy_, lams_, keys_, env, max_steps_: int):
        def run_one(lam, key_i):
            task_params = ODEParams(
                lam=lam,
                pulse_phase=jax.random.uniform(
                    jax.random.fold_in(key_i, 12345), shape=(), dtype=jnp.float32
                ),
                max_steps=env.max_steps,
            )
            return eval_episode_early_exit(
                vae_, policy_, env, task_params, key_i, max_steps=max_steps_
            )

        return jax.vmap(run_one)(lams_, keys_)

    lams = jnp.asarray(lams[:num_episodes], dtype=jnp.float32)
    keys = jnp.asarray(episode_keys[:num_episodes])
    batch_size = min(num_episodes, 32)

    # Warmup: trigger JIT compilation with a small batch
    t0 = time.perf_counter()
    _ = run_batch(
        vae, policy, lams[: min(batch_size, 4)], keys[: min(batch_size, 4)], env, max_steps
    )
    t_compile = time.perf_counter() - t0

    def run_fn(batch_lams, batch_keys):
        return run_batch(vae, policy, batch_lams, batch_keys, env, max_steps)

    repeat_results, timings = _run_timed_lams(run_fn, lams, keys, batch_size, repeats)
    t_end = float(env.t_end)

    per_repeat = []
    for i, results in enumerate(repeat_results):
        t_reached = results["t_reached"]
        per_repeat.append(
            {
                "success_rate": float(np.mean(t_reached >= t_end)),
                "mean_steps": float(np.mean(results["episode_length"])),
                "mean_step_count": float(
                    np.mean(results["budget_exhausted"].astype(float) * env.max_steps)
                ),
                "mean_accept_ema": float(np.mean(results["accept_ema"].astype(float))),
                "mean_t_reached": float(np.mean(t_reached)),
                "mean_return": float(np.mean(results["total_return"])),
                "ms_per_ep": timings[i] / max(num_episodes, 1) * 1000,
            }
        )

    row = _aggregate(per_repeat)
    row["method"] = method_name
    row["t_compile"] = t_compile
    row["t_runtime"] = float(np.median(timings))
    return row


# ─── Learned controller inside diffeqsolve ─────────────────────────────────


def eval_learned_controller(
    config, checkpoint_path, env, lams, episode_keys, num_episodes, max_steps, seed=42, repeats=3
):
    """Run diffeqsolve with the trained RL policy as a diffrax controller, on
    the same (λ, episode_key) pairs from the cached PID eval dataset — no live
    task sampling. Builds y0/pulse_phase from episode_key with the exact same
    key-split order pid_solve.solve_pid_batch uses, so this lands on the exact
    same y0/pulse_phase as the cached PID/oracle rows for the same episode."""
    rhs = get_rhs(config.env)
    solver = _make_solver(config.env.rtol, config.env.atol)
    sc = LearnedController.from_checkpoint(checkpoint_path, config, env)
    spec = get_system(config.env.system)

    @jax.jit
    def solve_batch(lams_, keys_):
        def solve_one(lam, key):
            key_y0, key_pulse = jax.random.split(key)
            pulse_phase = jax.random.uniform(key_pulse, shape=(), dtype=jnp.float32)
            task = jnp.array([lam, pulse_phase], dtype=jnp.float32)
            y0 = spec.y0(task, key_y0, config.env)
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(rhs),
                solver,
                t0=0.0,
                t1=config.env.t_end,
                dt0=config.env.dt0,
                y0=y0,
                args=task,
                stepsize_controller=sc,
                max_steps=max_steps,
                throw=False,
            )
            return {
                "t_reached": sol.ts[-1],
                "num_steps": sol.stats["num_steps"],
                "accepted": sol.stats["num_accepted_steps"],
                "rejected": sol.stats["num_rejected_steps"],
            }

        return jax.vmap(solve_one)(lams_, keys_)

    lams = jnp.asarray(lams[:num_episodes], dtype=jnp.float32)
    keys = jnp.asarray(episode_keys[:num_episodes])
    batch_size = min(num_episodes, 32)

    t0 = time.perf_counter()
    jax.tree.map(
        lambda x: x.block_until_ready(),
        solve_batch(lams[: min(batch_size, 4)], keys[: min(batch_size, 4)]),
    )
    t_compile = time.perf_counter() - t0

    repeat_results, timings = _run_timed_lams(solve_batch, lams, keys, batch_size, repeats)

    per_repeat = []
    for i, results in enumerate(repeat_results):
        t_reached = results["t_reached"]
        accepted = results["accepted"].astype(float)
        rejected = results["rejected"].astype(float)
        total = accepted + rejected
        per_repeat.append(
            {
                "success_rate": float(np.mean(t_reached >= config.env.t_end * 0.999)),
                "mean_steps": float(np.mean(results["num_steps"])),
                "mean_step_count": float(np.mean(total)),
                "mean_accept_ema": float(np.mean(accepted / np.maximum(total, 1))),
                "mean_t_reached": float(np.mean(t_reached)),
                "ms_per_ep": timings[i] / max(num_episodes, 1) * 1000,
            }
        )

    row = _aggregate(per_repeat)
    row["method"] = "RL (diffeqsolve)"
    row["t_compile"] = t_compile
    row["t_runtime"] = float(np.median(timings))
    return row


# ─── Table formatting & export ─────────────────────────────────────────────

COLUMNS = [
    ("Method", "method", "{:<30s}"),
    ("Success%", "success_rate", "{:>9.1%}"),
    ("Steps", "mean_steps", "{:>8.1f}"),
    ("StepCount", "mean_step_count", "{:>10.1f}"),
    ("AcceptEMA", "mean_accept_ema", "{:>10.3f}"),
    ("t reached", "mean_t_reached", "{:>12.2f}"),
    ("Return", "mean_return", "{:>10.2f}"),
    ("Compile(s)", "t_compile", "{:>10.2f}"),
    ("Runtime(s)", "t_runtime", "{:>10.2f}"),
    ("ms/ep", "ms_per_ep", "{:>8.1f}"),
]


def format_val(val, fmt):
    """Format a table value while preserving missing-value markers."""
    if isinstance(val, str):
        return fmt.format(val)
    if isinstance(val, float) and np.isnan(val):
        w = len(fmt.format(0.0))
        return "—".center(w)
    return fmt.format(val)


def _build_table_lines(rows: list[dict], t_end: float) -> list[str]:
    widths = []
    for name, key, fmt in COLUMNS:
        try:
            w = max(len(name), len(fmt.format("x" * 30 if "s" in fmt else 99999.99)))
        except (ValueError, TypeError):
            w = max(len(name), 14)
        widths.append(w)

    header = " | ".join(
        name.ljust(w) if "s" in fmt else name.rjust(w) for (name, _, fmt), w in zip(COLUMNS, widths)
    )
    sep = "-+-".join("-" * w for w in widths)
    rule = "=" * len(header)

    lines = [
        rule,
        f"  ODE Step-Size Controller Comparison  (t_end = {t_end})",
        rule,
        header,
        sep,
    ]
    for row in rows:
        parts = []
        for (name, key, fmt), w in zip(COLUMNS, widths):
            val = row.get(key, np.nan)
            parts.append(format_val(val, fmt))
        lines.append(" | ".join(parts))
    lines.append(rule)
    return lines


def print_table(rows: list[dict], t_end: float):
    """Print per-bin controller comparison rows."""
    for line in _build_table_lines(rows, t_end):
        print(line)
    print()


def export_txt(rows: list[dict], t_end: float, path: str):
    """Write per-bin comparison rows as a text table."""
    lines = _build_table_lines(rows, t_end)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → {path}")


def export_latex(rows: list[dict], t_end: float, path: str):
    """Write per-bin comparison rows as a LaTeX table."""
    headers = [name for name, _, _ in COLUMNS]

    with open(path, "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write(
            f"\\caption{{ODE Step-Size Controller Comparison ($t_{{\\mathrm{{end}}}} = {t_end}$)}}\n"
        )
        f.write("\\label{tab:controller-comparison}\n")
        f.write("\\begin{tabular}{l" + " r" * (len(headers) - 1) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(h.replace("%", "\\%") for h in headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            cells = []
            for name, key, fmt in COLUMNS:
                val = row.get(key, np.nan)
                if key == "success_rate":
                    cells.append(f"{val * 100:.1f}")
                elif isinstance(val, str):
                    cells.append(val)
                elif isinstance(val, float) and np.isnan(val):
                    cells.append("---")
                else:
                    cells.append(fmt.format(val).strip())
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  → {path}")


def export_xlsx(rows: list[dict], t_end: float, path: str):
    """Write per-bin comparison rows to an Excel workbook."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    wb = Workbook()
    ws = wb.active
    ws.title = "Controller Comparison"

    headers = [name for name, _, _ in COLUMNS]

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    ws["A1"] = f"ODE Step-Size Controller Comparison (t_end = {t_end})"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A1"].alignment = Alignment(horizontal="center")

    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font_white = Font(bold=True, size=11, color="FFFFFF")
    thick = Side(style="medium")
    row_fills = [
        PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid"),
        PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid"),
    ]

    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=3, column=col, value=h)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.border = Border(top=thick, bottom=thick)
        cell.alignment = Alignment(horizontal="center")

    for r, row in enumerate(rows, 4):
        fill = row_fills[(r - 4) % 2]
        for c, (name, key, fmt) in enumerate(COLUMNS, 1):
            val = row.get(key, np.nan)
            if key == "success_rate":
                val = val * 100
            cell = ws.cell(row=r, column=c, value=val)
            cell.fill = fill
            if c == 1:
                cell.alignment = Alignment(horizontal="left")
            else:
                cell.alignment = Alignment(horizontal="center")

    last_row = 3 + len(rows)
    for c in range(1, len(headers) + 1):
        ws.cell(row=last_row, column=c).border = Border(bottom=thick)

    ws.column_dimensions["A"].width = 28
    for i in range(1, len(headers)):
        col_letter = chr(ord("B") + i - 1)
        ws.column_dimensions[col_letter].width = 14

    wb.save(path)
    print(f"  → {path}")


# ─── Compact single-decimal summary (steps / wallclock / success / return) ──

SUMMARY_COLUMNS = [
    ("Method", "method", "{:<28s}"),
    ("Steps", "mean_steps", "{:>8.1f}"),
    ("Wallclock (ms/ep)", "ms_per_ep", "{:>18.1f}"),
    ("Success %", "success_rate", "{:>10.1f}"),
    ("Return", "mean_return", "{:>10.1f}"),
]


def _summary_row(row: dict) -> dict:
    """Same row dict, with success_rate rescaled to a percentage for display."""
    out = dict(row)
    if "success_rate" in out:
        out["success_rate"] = out["success_rate"] * 100
    return out


def _build_summary_table_lines(rows: list[dict], t_end: float) -> list[str]:
    display_rows = [_summary_row(r) for r in rows]
    widths = []
    for name, key, fmt in SUMMARY_COLUMNS:
        try:
            w = max(len(name), len(fmt.format("x" * 28 if "s" in fmt else 99999.9)))
        except (ValueError, TypeError):
            w = max(len(name), 14)
        widths.append(w)

    header = " | ".join(
        name.ljust(w) if "s" in fmt else name.rjust(w)
        for (name, _, fmt), w in zip(SUMMARY_COLUMNS, widths)
    )
    sep = "-+-".join("-" * w for w in widths)
    rule = "=" * len(header)

    lines = [
        rule,
        f"  ODE Controller Comparison — Summary  (t_end = {t_end})",
        rule,
        header,
        sep,
    ]
    for row in display_rows:
        parts = []
        for (name, key, fmt), w in zip(SUMMARY_COLUMNS, widths):
            val = row.get(key, np.nan)
            parts.append(format_val(val, fmt))
        lines.append(" | ".join(parts))
    lines.append(rule)
    return lines


def print_summary(rows: list[dict], t_end: float):
    """Print the aggregate controller comparison summary."""
    for line in _build_summary_table_lines(rows, t_end):
        print(line)
    print()


def export_summary_txt(rows: list[dict], t_end: float, path: str):
    """Write the aggregate comparison summary as text."""
    lines = _build_summary_table_lines(rows, t_end)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → {path}")


def export_summary_csv(rows: list[dict], path: str):
    """Write the aggregate comparison summary as CSV."""
    import csv

    display_rows = [_summary_row(r) for r in rows]
    headers = [name for name, _, _ in SUMMARY_COLUMNS]
    keys = [key for _, key, _ in SUMMARY_COLUMNS]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in display_rows:
            cells = []
            for key in keys:
                val = row.get(key, np.nan)
                if isinstance(val, str):
                    cells.append(val)
                elif isinstance(val, float) and np.isnan(val):
                    cells.append("")
                else:
                    cells.append(f"{val:.1f}")
            writer.writerow(cells)
    print(f"  → {path}")


def export_summary_latex(rows: list[dict], t_end: float, path: str):
    """Write the aggregate comparison summary as LaTeX."""
    display_rows = [_summary_row(r) for r in rows]
    headers = [name for name, _, _ in SUMMARY_COLUMNS]

    with open(path, "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write(
            f"\\caption{{ODE Controller Comparison — Summary ($t_{{\\mathrm{{end}}}} = {t_end}$)}}\n"
        )
        f.write("\\label{tab:controller-comparison-summary}\n")
        f.write("\\begin{tabular}{l" + " r" * (len(headers) - 1) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(h.replace("%", "\\%") for h in headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in display_rows:
            cells = []
            for name, key, fmt in SUMMARY_COLUMNS:
                val = row.get(key, np.nan)
                if isinstance(val, str):
                    cells.append(val)
                elif isinstance(val, float) and np.isnan(val):
                    cells.append("---")
                else:
                    cells.append(f"{val:.1f}")
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  → {path}")


# ─── Run all methods for a given n ─────────────────────────────────────────


def _run_methods(
    config, env, num_episodes, max_steps, checkpoint, seed, repeats, extra_checkpoints=None
):
    """Evaluate every method on the same cached test-set tasks; return one row dict per method.

    `extra_checkpoints` holds (label, path, config, env) tuples for additional policies.
    """
    rows = []
    bins = list(config.env.test_bins)
    dataset = load_cached_pid_batch(
        config.env,
        max_steps=max_steps,
        bins=bins,
        split="test",
        num_envs=num_episodes,
        seed=seed,
    )
    lams, episode_keys = dataset["task_params"], dataset["episode_keys"]

    print(f"  PID (budget={max_steps}) ...", end=" ", flush=True)
    row = eval_diffrax_pid(
        config,
        bins,
        num_episodes,
        max_steps=max_steps,
        seed=seed,
        repeats=repeats,
        progress_label="PID budget",
    )
    print(f"\n  done ({row['t_compile']:.1f}s compile, {row['t_runtime']:.1f}s run)")
    rows.append(row)

    print("  PID (unlimited) ...", end=" ", flush=True)
    # Same `repeats` as the budget call above (not a reduced count): both calls
    # land on the same load_pid_batch fingerprint, so this one is a cache hit
    # off the first call's generation instead of a second live benchmark.
    row = eval_diffrax_pid(
        config,
        bins,
        num_episodes,
        max_steps=8192,
        seed=seed,
        repeats=repeats,
        progress_label="PID unlimited",
    )
    row["method"] = "diffrax PID (unlimited)"
    print(f"\n  done ({row['t_compile']:.1f}s compile, {row['t_runtime']:.1f}s run)")
    rows.append(row)

    if checkpoint:
        primary_label = policy_method_label(config)
        print(f"  {primary_label} ...", end=" ", flush=True)
        row = eval_rl(
            env,
            config,
            checkpoint,
            lams,
            episode_keys,
            num_episodes,
            seed=seed,
            repeats=repeats,
            method_name=primary_label,
        )
        print(f"done ({row['t_compile']:.1f}s compile, {row['t_runtime']:.1f}s run)")
        rows.append(row)

        print("  RL (diffeqsolve) ...", end=" ", flush=True)
        row = eval_learned_controller(
            config,
            checkpoint,
            env,
            lams,
            episode_keys,
            num_episodes,
            max_steps,
            seed=seed,
            repeats=repeats,
        )
        print(f"done ({row['t_compile']:.1f}s compile, {row['t_runtime']:.1f}s run)")
        rows.append(row)

    for label, ckpt_path, ckpt_config, ckpt_env in extra_checkpoints or []:
        print(f"  {label} ...", end=" ", flush=True)
        row = eval_rl(
            ckpt_env,
            ckpt_config,
            ckpt_path,
            lams,
            episode_keys,
            num_episodes,
            seed=seed,
            repeats=repeats,
            method_name=label,
        )
        print(f"done ({row['t_compile']:.1f}s compile, {row['t_runtime']:.1f}s run)")
        rows.append(row)

    return rows


# ─── Scaling figure ───────────────────────────────────────────────────────


def plot_scaling(all_results: list[tuple[int, list[dict]]], t_end: float, out_dir: str):
    """Generate scaling figure: metrics vs number of episodes with error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_values = [n for n, _ in all_results]
    methods = []
    for _, rows in all_results:
        for row in rows:
            if row["method"] not in methods:
                methods.append(row["method"])

    metrics = [
        ("t_runtime", "Runtime (s)"),
        ("ms_per_ep", "ms / episode"),
        ("success_rate", "Success rate"),
        ("mean_steps", "Mean steps"),
        ("mean_accept_ema", "Accept EMA"),
    ]

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
    fig.suptitle(f"ODE Controller Scaling  (t_end = {t_end})", fontsize=14)
    colors = plt.cm.tab10.colors

    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)

    for ax, (metric, label) in zip(axes.flat, metrics):
        for i, method in enumerate(methods):
            means, stds = [], []
            for n, rows in all_results:
                row = next((r for r in rows if r["method"] == method), None)
                if row:
                    means.append(row[metric])
                    stds.append(row.get(f"{metric}_std", 0))
                else:
                    means.append(np.nan)
                    stds.append(0)
            ax.errorbar(
                n_values,
                means,
                yerr=stds,
                label=method,
                color=colors[i % len(colors)],
                marker="o",
                capsize=3,
                linewidth=1.5,
            )
        ax.set_xlabel("Number of episodes (n)")
        ax.set_ylabel(label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_values)
        ax.set_xticklabels([str(v) for v in n_values])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    path = os.path.join(out_dir, "scaling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


# ─── Per-bin system metadata ────────────────────────────────────────────────

META_NUMERIC_COLUMNS = [
    "bin_lo",
    "bin_hi",
    "n",
    "pid_steps",
    "ours_steps",
    "oracle_steps",
    "ours_step_improvement",
    "oracle_step_improvement",
    "ours_step_gap_to_oracle",
    "pid_error",
    "ours_error",
    "oracle_error",
    "ours_error_diff_vs_pid",
    "oracle_error_diff_vs_pid",
]


def _mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _bin_label(lo, hi, label=None):
    return str(label) if label is not None else f"[{lo:g},{hi:g}]"


def compute_bin_rows(
    *,
    bins,
    split,
    pid_steps,
    policy_steps=None,
    oracle_steps=None,
    pid_errors=None,
    policy_errors=None,
    oracle_errors=None,
    labels=None,
    trained=None,
):
    """Reduce episode arrays to one row per bin.

    Step improvement is the average of per-episode ``(pid - ours) / pid``
    ratios (with a one-step floor), not a ratio of bin means. Error columns
    are log10-relative-error differences, so negative is better than PID.
    """
    split = np.asarray(split)
    pid_steps = np.asarray(pid_steps, dtype=float)
    n = len(pid_steps)
    for name, values in {
        "policy_steps": policy_steps,
        "oracle_steps": oracle_steps,
        "pid_errors": pid_errors,
        "policy_errors": policy_errors,
        "oracle_errors": oracle_errors,
    }.items():
        if values is not None and len(values) != n:
            raise ValueError(f"{name} has {len(values)} values, expected {n}")
    rows = []
    for i, (lo, hi) in enumerate(bins):
        mask = split == i
        pid = pid_steps[mask]
        ours = None if policy_steps is None else np.asarray(policy_steps, float)[mask]
        oracle = None if oracle_steps is None else np.asarray(oracle_steps, float)[mask]
        pid_err = None if pid_errors is None else np.asarray(pid_errors, float)[mask]
        ours_err = None if policy_errors is None else np.asarray(policy_errors, float)[mask]
        oracle_err = None if oracle_errors is None else np.asarray(oracle_errors, float)[mask]
        pid_m, ours_m, oracle_m = _mean(pid), _mean(ours), _mean(oracle)
        pid_e, ours_e, oracle_e = _mean(pid_err), _mean(ours_err), _mean(oracle_err)
        rows.append(
            {
                "bin_label": _bin_label(float(lo), float(hi), labels[i] if labels else None),
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(mask.sum()),
                "trained": None if trained is None else bool(trained[i]),
                "pid_steps": pid_m,
                "ours_steps": ours_m,
                "oracle_steps": oracle_m,
                "ours_step_improvement": (
                    _mean((pid - ours) / np.maximum(pid, 1.0)) if ours is not None else float("nan")
                ),
                "oracle_step_improvement": (
                    _mean((pid - oracle) / np.maximum(pid, 1.0))
                    if oracle is not None
                    else float("nan")
                ),
                "ours_step_gap_to_oracle": (
                    ours_m - oracle_m
                    if np.isfinite(ours_m) and np.isfinite(oracle_m)
                    else float("nan")
                ),
                "pid_error": pid_e,
                "ours_error": ours_e,
                "oracle_error": oracle_e,
                "ours_error_diff_vs_pid": (
                    ours_e - pid_e if np.isfinite(ours_e) and np.isfinite(pid_e) else float("nan")
                ),
                "oracle_error_diff_vs_pid": (
                    oracle_e - pid_e
                    if np.isfinite(oracle_e) and np.isfinite(pid_e)
                    else float("nan")
                ),
            }
        )
    return rows


def aggregate_bin_rows(per_run_rows):
    """Average matching bin rows across seed runs."""
    grouped = {}
    for rows in per_run_rows:
        for row in rows:
            grouped.setdefault(row["bin_label"], []).append(row)
    result = []
    for label, entries in grouped.items():
        row = {"bin_label": label, "n_seeds": len(entries), "trained": entries[0].get("trained")}
        for key in META_NUMERIC_COLUMNS:
            row[key] = _mean([entry.get(key, float("nan")) for entry in entries])
        result.append(row)
    return result


def _value(value):
    try:
        return f"{float(value):.6f}" if np.isfinite(float(value)) else "nan"
    except (TypeError, ValueError):
        return "nan"


def write_meta_txt(rows, path, *, system="", n_seeds=1):
    """Write comparison metadata and rows as a text artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["bin_label", "trained", *META_NUMERIC_COLUMNS]
    with path.open("w") as f:
        f.write(f"# ODE system metadata: {system}\n# seed_count: {n_seeds}\n")
        f.write("  ".join(headers) + "\n")
        for row in rows:
            trained = row.get("trained")
            trained_value = "nan" if trained is None else str(int(bool(trained)))
            f.write(
                "  ".join(
                    [str(row["bin_label"]), trained_value]
                    + [_value(row.get(k)) for k in META_NUMERIC_COLUMNS]
                )
                + "\n"
            )


def read_meta_txt(path):
    """Read comparison metadata from a text artifact."""
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not lines:
        return []
    reader = csv.DictReader(lines, delimiter=" ", skipinitialspace=True)
    rows = []
    for raw in reader:
        row = {"bin_label": raw["bin_label"]}
        trained = raw.get("trained", "nan")
        row["trained"] = None if trained in (None, "", "nan") else bool(int(float(trained)))
        for key in META_NUMERIC_COLUMNS:
            value = raw.get(key, "nan")
            row[key] = float(value) if value not in (None, "", "nan") else float("nan")
        rows.append(row)
    return rows


def _escape(value):
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _latex_num(value, digits=1):
    return "---" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def write_meta_latex(rows, path, *, system="", n_seeds=1):
    """Write comparison metadata and rows as a LaTeX artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write(f"\\caption{{ODE controller metadata by bin ({_escape(system)}, n={n_seeds})}}\n")
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
        f.write(
            "\\textbf{Bin} & \\textbf{Trained} & \\textbf{PID steps} & \\textbf{Ours steps} & \\textbf{Oracle steps} & "
            "\\textbf{Ours step improvement} & \\textbf{Ours error $-$ PID} & "
            "\\textbf{Oracle error $-$ PID} \\\\\n\\midrule\n"
        )
        for row in rows:
            improvement = row.get("ours_step_improvement", float("nan"))
            imp = (
                "---"
                if not np.isfinite(float(improvement))
                else f"{100 * float(improvement):.1f}\\%"
            )
            trained = row.get("trained")
            trained_label = "---" if trained is None else ("yes" if trained else "no")
            cells = [
                _escape(row["bin_label"]),
                trained_label,
                _latex_num(row.get("pid_steps")),
                _latex_num(row.get("ours_steps")),
                _latex_num(row.get("oracle_steps")),
                imp,
                _latex_num(row.get("ours_error_diff_vs_pid"), 3),
                _latex_num(row.get("oracle_error_diff_vs_pid"), 3),
            ]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def _runtime_rows(config, checkpoint, *, num_episodes, seed, ref_tol_factor):
    from steppo.envs.ode import ODEEnv
    from steppo.envs.ode.learned_controller import LearnedController
    from steppo.training.error_dist import (
        collect_controller_steps_and_errors,
        load_cached_pid_batch,
        log10_rel_error,
    )

    test_bins = list(config.env.test_bins)
    batch = load_cached_pid_batch(
        config.env,
        max_steps=config.rollout_steps,
        num_envs=num_episodes,
        seed=seed,
        bins=test_bins,
        split="test",
        ref_tol_factor=ref_tol_factor,
    )
    cache_split = np.asarray(batch["split"])
    refs, pid_final = np.asarray(batch["ref_final_y"]), np.asarray(batch["pid_final_y"])
    pid_steps = np.asarray(batch["pid_steps"], float)
    pid_errors = log10_rel_error(pid_final, refs, config.env.atol)
    oracle_steps = (
        None if batch.get("oracle_steps") is None else np.asarray(batch["oracle_steps"], float)
    )
    oracle_errors = (
        None
        if batch.get("oracle_final_y") is None
        else log10_rel_error(np.asarray(batch["oracle_final_y"]), refs, config.env.atol)
    )
    ours_steps = ours_errors = None
    if checkpoint:
        env = ODEEnv(config.env, config.rollout_steps)
        controller = LearnedController.from_checkpoint(checkpoint, config, env)
        n_bins = len(test_bins)
        probe, references = {}, {}
        for i in range(n_bins):
            mask = cache_split == i
            probe[str(i)] = (batch["task_params"][mask], batch["episode_keys"][mask])
            references[str(i)] = refs[mask]
        # One fused solve (steps + error together) per bin, instead of a
        # whole-batch collect_policy_steps call plus a separate per-bin
        # collect_controller_errors pass over the same episodes.
        by_split = collect_controller_steps_and_errors(
            config.env, controller, probe, references, config.rollout_steps
        )
        ours_steps = np.full(len(cache_split), np.nan)
        ours_errors = np.full(len(cache_split), np.nan)
        for i in range(n_bins):
            mask = cache_split == i
            ours_steps[mask] = by_split[str(i)]["steps"]
            ours_errors[mask] = by_split[str(i)]["err"]
    bins, labels = test_bins, [f"bin_{i}" for i in range(len(test_bins))]
    trained = [
        not config.env.train_bins or any(lo >= a and hi <= b for a, b in config.env.train_bins)
        for lo, hi in bins
    ]
    eval_split = cache_split
    return compute_bin_rows(
        bins=bins,
        split=eval_split,
        pid_steps=pid_steps,
        policy_steps=ours_steps,
        oracle_steps=oracle_steps,
        pid_errors=pid_errors,
        policy_errors=ours_errors,
        oracle_errors=oracle_errors,
        labels=labels,
        trained=trained,
    )


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    """Run the configured PID-versus-controller comparison and exports."""
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

    parser = argparse.ArgumentParser(
        description="Compare diffrax PID vs RL on ODE step-size control"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config (auto-detected from checkpoint if omitted)",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="RL checkpoint path")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Additional labeled baselines for the summary export, each as "
        "LABEL=PATH (e.g. --checkpoints 'Memoryless=checkpoints/run_a/checkpoint_9' "
        "'RL2-style=checkpoints/run_b/checkpoint_9'). Each checkpoint's own "
        "bundled config.yaml is auto-detected, so architectures may differ "
        "across labels (e.g. a no-embedding policy vs. a deterministic-latent "
        "one). Evaluated alongside --checkpoint (labeled from its config) and PID.",
    )
    parser.add_argument(
        "-n",
        "--num_episodes",
        type=int,
        nargs="+",
        default=[PID_COMPARE_NUM_EPISODES],
        help="Number of evaluation episodes (multiple values for scaling study)",
    )
    parser.add_argument("-r", "--repeats", type=int, default=8, help="Timing repeats per n-value")
    parser.add_argument("--seed", type=int, default=PID_COMPARE_SEED, help="Random seed")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Directory for exported results (overrides the default outputs/compare/<checkpoint>/... location)",
    )
    parser.add_argument(
        "--meta_num_episodes",
        type=int,
        default=1024,
        help="Episodes used for the per-bin system metadata export.",
    )
    args = parser.parse_args()

    def _autodetect_config(checkpoint_path):
        bundled = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
        return bundled if os.path.isfile(bundled) else None

    if args.checkpoint is not None:
        args.checkpoint = resolve_checkpoint(args.checkpoint)

    checkpoint_specs = []  # (label, path) for --checkpoints entries, parsed before config resolution
    if args.checkpoints:
        for entry in args.checkpoints:
            label, sep, path = entry.partition("=")
            if not sep:
                parser.error(f"--checkpoints entries must be LABEL=PATH, got: {entry!r}")
            checkpoint_specs.append((label.strip(), resolve_checkpoint(path.strip())))

    if args.config is None and args.checkpoint is not None:
        bundled = _autodetect_config(args.checkpoint)
        if bundled:
            args.config = bundled
            print(f"[*] Auto-detected config: {bundled}")

    if args.config is None and checkpoint_specs:
        bundled = _autodetect_config(checkpoint_specs[0][1])
        if bundled:
            args.config = bundled
            print(f"[*] Auto-detected config from {checkpoint_specs[0][0]}: {bundled}")

    if args.config is None:
        parser.error("--config is required (no config.yaml found in checkpoint dir)")

    config = load_config_from_yaml(TrainConfig, args.config, strict=False)
    env = ODEEnv(config.env, config.rollout_steps)
    system = config.env.system
    max_steps = config.rollout_steps
    repeats = args.repeats
    n_values = sorted(args.num_episodes)

    extra_checkpoints = []  # (label, path, ckpt_config, ckpt_env) — see --checkpoints
    for label, path in checkpoint_specs:
        ckpt_config_path = _autodetect_config(path) or args.config
        ckpt_config = load_config_from_yaml(TrainConfig, ckpt_config_path, strict=False)
        ckpt_env = ODEEnv(ckpt_config.env, ckpt_config.rollout_steps)
        extra_checkpoints.append((label, path, ckpt_config, ckpt_env))
        print(f"[*] {label}: {path}  (config: {ckpt_config_path})")

    print(f"System: {system}  |  t_end={config.env.t_end}")
    print(f"Solver: Kvaerno5  rtol={config.env.rtol}  atol={config.env.atol}")
    print(f"Step budget: {max_steps}  |  dt range: [{config.env.dt_min}, {config.env.dt_max}]")
    print(f"n = {n_values}  |  r = {repeats} repeats per n")
    print()

    all_results = []
    for n in n_values:
        print(f"── n = {n} ──")
        cache_payload = {
            "checkpoint": args.checkpoint,
            "extra_checkpoints": [(label, path) for label, path, _, _ in extra_checkpoints],
            "env_config": dataclasses.asdict(config.env),
            "n": n,
            "max_steps": max_steps,
            "seed": args.seed,
            "repeats": repeats,
        }
        rows = load_or_compute(
            "compare",
            cache_payload,
            lambda: _run_methods(
                config,
                env,
                n,
                max_steps,
                args.checkpoint,
                args.seed,
                repeats,
                extra_checkpoints=extra_checkpoints,
            ),
        )
        print_table(rows, config.env.t_end)
        print_summary(rows, config.env.t_end)
        all_results.append((n, rows))

    if args.out:
        out_dir = Path(args.out)
    elif args.checkpoint:
        run_output_dir = Path(args.checkpoint.replace("checkpoints", "outputs", 1)).parent
        if len(n_values) > 1:
            out_dir = run_output_dir / "compare" / f"scaling_r{repeats:03d}"
        else:
            out_dir = run_output_dir / "compare" / f"n{n_values[0]:03d}_r{repeats:03d}"
    elif checkpoint_specs:
        run_output_dir = Path(checkpoint_specs[0][1].replace("checkpoints", "outputs", 1)).parent
        if len(n_values) > 1:
            out_dir = run_output_dir / "compare" / f"scaling_r{repeats:03d}"
        else:
            out_dir = run_output_dir / "compare" / f"n{n_values[0]:03d}_r{repeats:03d}"
    else:
        out_dir = None

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        print("Exporting results:")
        for n, rows in all_results:
            tag = f"_n{n:03d}" if len(n_values) > 1 else ""
            export_txt(rows, config.env.t_end, f"{out_dir}/results{tag}.txt")
            export_latex(rows, config.env.t_end, f"{out_dir}/results{tag}.tex")
            export_xlsx(rows, config.env.t_end, f"{out_dir}/results{tag}.xlsx")
            export_summary_txt(rows, config.env.t_end, f"{out_dir}/summary{tag}.txt")
            export_summary_csv(rows, f"{out_dir}/summary{tag}.csv")
            export_summary_latex(rows, config.env.t_end, f"{out_dir}/summary{tag}.tex")
        print("Exporting per-bin system metadata:")
        try:
            meta_cache_payload = {
                "checkpoint": args.checkpoint,
                "env_config": dataclasses.asdict(config.env),
                "num_episodes": args.meta_num_episodes,
                "seed": 0,
                "ref_tol_factor": PID_EVAL_REF_TOL_FACTOR,
            }
            meta_rows = load_or_compute(
                "compare_meta",
                meta_cache_payload,
                lambda: _runtime_rows(
                    config,
                    args.checkpoint,
                    num_episodes=args.meta_num_episodes,
                    seed=0,
                    ref_tol_factor=PID_EVAL_REF_TOL_FACTOR,
                ),
            )
            write_meta_txt(
                meta_rows, out_dir / "system_meta.txt", system=config.env.system, n_seeds=1
            )
            write_meta_latex(
                meta_rows, out_dir / "system_meta.tex", system=config.env.system, n_seeds=1
            )
            print(f"  → {out_dir / 'system_meta.txt'}")
            print(f"  → {out_dir / 'system_meta.tex'}")
        except Exception as exc:
            print(f"  [!] Per-bin system metadata export failed: {exc}")
        if len(n_values) > 1:
            plot_scaling(all_results, config.env.t_end, str(out_dir))
    else:
        print("No --out, --checkpoint, or --checkpoints specified, skipping export.")


if __name__ == "__main__":
    main()
