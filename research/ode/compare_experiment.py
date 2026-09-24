"""Compare runs within one experiment batch (scripts/launch_experiments.sh output:
runs sharing a UID under outputs/experiments/<DATE>/<UID>/{checkpoints,outputs}/<run_uid>/).

Produces, in outputs/experiments/<DATE>/<UID>/outputs/comparison/ (override via --out):
  1. reward_trajectories.png       — <dataset>/mean_return vs. logged point, per dataset subplot.
  2. latent_<i>_<param><value>.png — belief mean/variance vs. rollout step at each sampled task point.
  3. step_diff[_rel]_<split>.png   — histogram of pid_steps - rl_steps per run/split (train/val/test),
     from PID vs. trained controller rollouts on that split's configured bins; `_rel` normalizes by
     pid_steps for cross-μ comparability. "test" is OOD only if its bins lie outside train_bins.
  4. error_dist_<split>.png        — log10 relative L2 error vs. tight-tolerance reference, per run,
     with pooled PID error distribution and solver rtol as references.
  5. step_diff_stats_<split>.png/.txt — per-experiment mean/median/Q1/Q3 of (3)'s relative savings.

Thin CLI over research/ode/execute_comparison.py, which does run discovery and all heavy
compute (checkpoint loads, GPU solves), caching to <out_dir>/data/ so replotting is a pure load.

Usage:
    python research/ode/compare_experiment.py --experiment_uid <UUID>
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import glob
import json
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from research.ode.execute_comparison import discover_runs, run_and_save
from steppo.training.error_dist import PID_BIN_EVAL_NUM_ENVS, PID_EVAL_REF_TOL_FACTOR
from steppo.utils.plotting import get_series


def parse_args():
    """Parse experiment identity, comparison settings, and output options."""
    p = argparse.ArgumentParser(description="Compare runs within one experiment batch")
    p.add_argument(
        "--experiment_uid",
        type=str,
        required=True,
        help="UID assigned by scripts/launch_experiments.sh",
    )
    p.add_argument(
        "--experiment_date",
        type=str,
        default=None,
        help="YYYYMMDD date the experiment batch was launched under "
        "(outputs/experiments/<DATE>/<UID>/). Auto-discovered by "
        "searching outputs/experiments/*/<UID> if omitted.",
    )
    p.add_argument("--workdir", type=str, default=".")
    # p.add_argument("--num_envs",
    #                type=int,
    #                default=1024,
    #                help="Environments to average over per test task")
    p.add_argument(
        "--num_task_points",
        type=int,
        default=6,
        help="Number of task values (log-spaced across the system's full "
        "valid range) to plot latent trajectories at",
    )
    p.add_argument(
        "--num_step_diff_episodes",
        type=int,
        default=1024,
        help="Episodes (distinct μ samples) per run for the step-count-vs-PID distribution plots",
    )
    p.add_argument(
        "--step_diff_bins",
        type=int,
        default=96,
        help="Histogram bin count for the step-count and error distribution plots",
    )
    p.add_argument(
        "--ref_tol_factor",
        type=float,
        default=PID_EVAL_REF_TOL_FACTOR,
        help="Reference-solve tolerances = (rtol, atol) * this",
    )
    p.add_argument(
        "--num_bin_episodes",
        type=int,
        default=PID_BIN_EVAL_NUM_ENVS,
        help="Episodes per test_bin for the trained-vs-held-out bin comparison "
        "(skipped entirely if config.env.test_bins is empty)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: outputs/experiments/<DATE>/<UID>/outputs/comparison/)",
    )
    return p.parse_args()


def _gradient_colors(runs: list[dict]):
    """Return evenly spaced Viridis colors for the supplied runs."""
    return cm.viridis(np.linspace(0.1, 0.9, max(len(runs), 2)))[: len(runs)]


_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _markers(runs: list[dict]):
    """Cycle distinct marker shapes so adjacent-colored gradient lines stay visually separable."""
    return [_MARKERS[i % len(_MARKERS)] for i in range(len(runs))]


# ---------------------------------------------------------------------------
# 1. Reward trajectory comparison
# ---------------------------------------------------------------------------


def _config_label(exp_name: str) -> str:
    """Strip the "_r<seed>" repeat suffix (see discover_runs) to get the config's display name."""
    return re.sub(r"_r\d+$", "", exp_name)


_DATASET_ORDER = ("train", "val", "test", "noise", "eval")


def _discover_return_datasets(metrics_by_run: dict[str, list[dict]]) -> list[str]:
    """Return dataset prefixes that have a ``<dataset>/mean_return`` metric."""
    datasets = {
        key.rsplit("/", 1)[0]
        for metrics in metrics_by_run.values()
        for row in metrics
        for key in row
        if key.endswith("/mean_return") and "/" in key
    }
    preferred = [dataset for dataset in _DATASET_ORDER if dataset in datasets]
    return preferred + sorted(datasets.difference(preferred))


def _align_repeat_series(
    series: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Align repeat curves on their common logged points and stack their values."""
    common_points = set(series[0][0].tolist())
    for points, _ in series[1:]:
        common_points.intersection_update(points.tolist())
    points = np.array(sorted(common_points), dtype=int)
    values = np.stack(
        [
            np.array([returns[np.flatnonzero(iterations == point)[0]] for point in points])
            for iterations, returns in series
        ]
    )
    return points, values


def plot_reward_trajectories(runs: list[dict], out_path: str):
    """One vertically stacked mean-return subplot per dataset inferred from
    ``<dataset>/mean_return`` keys (also matches legacy ``eval/mean_return``).
    Runs sharing an experiment id are averaged across repeats."""
    groups: dict[int, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["eid"], []).append(run)
    eids = sorted(groups.keys())

    metrics_by_run = {}
    for run in runs:
        with open(run["metrics_path"]) as f:
            metrics_by_run[run["run_uid"]] = json.load(f)

    datasets = _discover_return_datasets(metrics_by_run)
    if not datasets:
        print("[!] No <dataset>/mean_return metrics found; writing an empty reward plot")

    series_by_dataset_eid = {}
    repeat_counts = set()
    for dataset in datasets:
        key = f"{dataset}/mean_return"
        for eid in eids:
            repeat_series = []
            for run in groups[eid]:
                it, ret = get_series(metrics_by_run[run["run_uid"]], key)
                if len(it) > 0:
                    repeat_series.append((it, ret))
            if not repeat_series:
                continue
            it, stacked = _align_repeat_series(repeat_series)
            if len(it) == 0:
                print(f"[!] {dataset}: no common logged points for experiment e{eid}, skipping")
                continue
            series_by_dataset_eid[(dataset, eid)] = {
                "it": it,
                "mean": stacked.mean(axis=0),
                "std": stacked.std(axis=0),
                "label": _config_label(groups[eid][0]["exp_name"]),
                "n": len(repeat_series),
            }
            repeat_counts.add(len(repeat_series))

    plot_datasets = datasets or ["none"]
    fig_height = max(3.0 * len(plot_datasets), 4.5)
    fig, axes = plt.subplots(
        len(plot_datasets),
        1,
        figsize=(10, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    colors = cm.viridis(np.linspace(0.1, 0.9, max(len(eids), 2)))[: len(eids)]
    color_by_eid = dict(zip(eids, colors))
    legend_handles = {}
    legend_labels = {}

    for ax, dataset in zip(axes, plot_datasets):
        plotted = False
        if dataset != "none":
            for eid in eids:
                s = series_by_dataset_eid.get((dataset, eid))
                if s is None:
                    continue
                (line,) = ax.plot(
                    s["it"],
                    s["mean"],
                    color=color_by_eid[eid],
                    linewidth=1.8,
                )
                if s["n"] > 1:
                    ax.fill_between(
                        s["it"],
                        s["mean"] - s["std"],
                        s["mean"] + s["std"],
                        color=color_by_eid[eid],
                        alpha=0.12,
                    )
                if eid not in legend_handles:
                    legend_handles[eid] = line
                    legend_labels[eid] = s["label"]
                plotted = True

        ax.set_title(dataset.title() if dataset != "none" else "No return data")
        ax.set_ylabel("mean return")
        ax.grid(True, linewidth=0.4)
        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No <dataset>/mean_return metrics",
                transform=ax.transAxes,
                ha="center",
                va="center",
                alpha=0.7,
            )

    title = "Reward trajectories across experiments"
    if repeat_counts and repeat_counts - {1}:
        title += (
            f" (n={min(repeat_counts)} repeats)"
            if len(repeat_counts) == 1
            else f" (n={min(repeat_counts)}-{max(repeat_counts)} repeats)"
        )
    fig.suptitle(title, y=0.995)
    axes[-1].set_xlabel("Logged point")
    if legend_handles:
        ordered_eids = sorted(legend_handles)
        fig.legend(
            [legend_handles[eid] for eid in ordered_eids],
            [legend_labels[eid] for eid in ordered_eids],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            fontsize=8,
            ncol=min(len(ordered_eids), 4),
            title="experiment",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.93 if legend_handles else 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# 2. Latent (belief mu / variance) comparison at fixed test tasks
# ---------------------------------------------------------------------------


def plot_latent_for_task(
    task_value: float, traces: dict, runs: list[dict], out_path: str, param_label: str
):
    """Plot latent means and variances for every run at one task value."""
    n_dims = max(t["latent_dim"] for t in traces.values())
    fig, axes = plt.subplots(2, n_dims, figsize=(4 * n_dims, 6), squeeze=False)

    fig.suptitle(f"Belief trajectories at {param_label} = {task_value:.2f}", fontsize=13)

    colors = _gradient_colors(runs)
    markers = _markers(runs)
    n_marks = 12  # roughly this many markers spread across the rollout, regardless of length
    for i, (color, marker, run) in enumerate(zip(colors, markers, runs)):
        t = traces[run["run_uid"]]
        steps = np.arange(t["mu"].shape[0])
        every = max(len(steps) // n_marks, 1)
        offset = (i * every) // len(runs)
        for d in range(t["latent_dim"]):
            axes[0][d].plot(
                steps,
                t["mu"][:, d],
                color=color,
                linewidth=1.5,
                marker=marker,
                markevery=(offset, every),
                markersize=5,
                label=run["exp_name"],
            )
            axes[1][d].plot(
                steps,
                t["var"][:, d],
                color=color,
                linewidth=1.5,
                marker=marker,
                markevery=(offset, every),
                markersize=5,
            )

    for d in range(n_dims):
        axes[0][d].set_title(f"z[{d}]", fontsize=10)
        axes[0][d].set_ylabel("latent mean")
        axes[0][d].grid(True, linewidth=0.4)
        axes[1][d].set_ylabel("latent variance")
        axes[1][d].set_xlabel("env steps")
        axes[1][d].grid(True, linewidth=0.4)

    axes[0][0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# 3. Step-count-vs-PID distribution comparison (per split: train / val / test)
# ---------------------------------------------------------------------------


def plot_step_diff_distribution(
    traces: dict,
    runs: list[dict],
    split: str,
    out_path: str,
    param_label: str,
    num_bins: int = 80,
    relative: bool = False,
):
    """Plot the distribution of step savings against the PID baseline."""
    field = "rel_diff" if relative else "diff"
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = _gradient_colors(runs)

    tag = f"{split} split"

    if relative:
        clip_lo, clip_hi = -1.0, 1.0
        bins = np.linspace(clip_lo, clip_hi, num_bins + 1)
    else:
        all_diffs = np.concatenate([traces[r["run_uid"]][split][field] for r in runs])
        bins = np.histogram_bin_edges(all_diffs, bins=num_bins)

    n_below = n_above = n_total = 0
    for color, run in zip(colors, runs):
        diffs = traces[run["run_uid"]][split][field]
        label_val = f"{np.median(diffs) * 100:+.0f}%" if relative else f"{np.median(diffs):+.1f}"
        plot_diffs = np.clip(diffs, bins[0], bins[-1]) if relative else diffs
        ax.hist(
            plot_diffs,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=color,
            density=True,
            label=f"{run['exp_name']} (median={label_val})",
        )
        if relative:
            n_below += int(np.sum(diffs < clip_lo))
            n_above += int(np.sum(diffs > clip_hi))
            n_total += diffs.size

    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.2, alpha=0.7, label="PID parity")
    if relative:
        ax.set_xlim(clip_lo, clip_hi)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")
        ax.set_xlabel(
            "Solver steps saved vs. PID, relative  ((pid_steps - rl_steps) / pid_steps, positive = RL better)"
            "  [clipped to ±100%]"
        )
        ax.set_title(f"Relative step-count improvement over PID across {param_label} ({tag})")
        if n_below > 0:
            ax.annotate(
                f"◀ {n_below}/{n_total} ({n_below / n_total:.0%}) below -100%",
                xy=(0.0, 1.0),
                xycoords="axes fraction",
                xytext=(4, -4),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8,
                color="firebrick",
            )
        if n_above > 0:
            ax.annotate(
                f"{n_above}/{n_total} ({n_above / n_total:.0%}) above +100% ▶",
                xy=(1.0, 1.0),
                xycoords="axes fraction",
                xytext=(-4, -4),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=8,
                color="firebrick",
            )
    else:
        ax.set_xlabel("Solver steps saved vs. PID  (pid_steps - rl_steps, positive = RL better)")
        ax.set_title(f"Step-count improvement over PID across {param_label} ({tag})")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def _fit_gmm_1d(data: np.ndarray, k: int, n_init: int = 8, n_iter: int = 200, seed: int = 0):
    """Fit a k-component 1D Gaussian mixture via EM. Returns (weights, means, stds, bic).
    Minimal in-house EM (no sklearn dep); several random restarts since 1D EM is cheap
    and prone to poor local optima with a single init."""
    rng = np.random.default_rng(seed)
    x = data.astype(np.float64)
    n = x.size
    data_std = max(x.std(), 1e-6)

    best_ll = -np.inf
    best = None
    for _ in range(n_init):
        means = rng.choice(x, size=k, replace=n < k)
        stds = np.full(k, data_std / max(k, 1) + 1e-3)
        weights = np.full(k, 1.0 / k)

        prev_ll = -np.inf
        for _ in range(n_iter):
            # E-step
            resp = np.stack(
                [
                    weights[j]
                    * np.exp(-0.5 * ((x - means[j]) / stds[j]) ** 2)
                    / (stds[j] * np.sqrt(2 * np.pi))
                    for j in range(k)
                ],
                axis=1,
            )
            resp_sum = resp.sum(axis=1, keepdims=True)
            resp_sum = np.clip(resp_sum, 1e-300, None)
            ll = np.log(resp_sum).sum()
            resp = resp / resp_sum

            # M-step
            nk = resp.sum(axis=0)
            nk = np.clip(nk, 1e-6, None)
            means = (resp * x[:, None]).sum(axis=0) / nk
            stds = np.sqrt((resp * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / nk)
            stds = np.clip(stds, 1e-3, None)
            weights = nk / n

            if abs(ll - prev_ll) < 1e-6 * max(abs(ll), 1.0):
                break
            prev_ll = ll

        if ll > best_ll:
            best_ll = ll
            best = (weights.copy(), means.copy(), stds.copy())

    weights, means, stds = best
    n_params = 3 * k - 1  # weights sum to 1
    bic = n_params * np.log(n) - 2 * best_ll
    return weights, means, stds, bic


def _fit_best_gmm_1d(data: np.ndarray, max_components: int = 3, seed: int = 0):
    """Fit k=1..max_components 1D GMMs and return the one with the lowest BIC."""
    best = None
    for k in range(1, max_components + 1):
        if k > data.size:
            break
        weights, means, stds, bic = _fit_gmm_1d(data, k, seed=seed)
        if best is None or bic < best[-1]:
            best = (weights, means, stds, bic)
    return best[:3]


def _save_curve_columns_txt(xs: np.ndarray, curves: dict, out_path: str, x_label: str = "x"):
    """Columnar .txt of one or more curves sharing an x-axis: `x_label  col1  col2  ...`,
    whitespace-delimited so it drops straight into external plotting software."""
    labels = list(curves.keys())
    with open(out_path, "w") as f:
        f.write("  ".join(f"{h:>16s}" for h in [x_label] + labels) + "\n")
        for i, x in enumerate(xs):
            row = [x] + [curves[label][i] for label in labels]
            f.write("  ".join(f"{v:>16.6g}" for v in row) + "\n")
    print(f"  saved -> {out_path}")


def plot_step_diff_gaussian_fit(
    traces: dict,
    runs: list[dict],
    split: str,
    out_path: str,
    param_label: str,
    max_components: int = 3,
    relative: bool = False,
):
    """Per-config (pooled across repeats) Gaussian-mixture fit of the step-diff distribution.
    BIC-selected 1-3 components so a real bimodal main-mode/failure-tail split (e.g.
    van_der_pol_e1) isn't smeared into one bell curve. Also writes a same-named .txt with
    the plotted curves as columns for import into external plotting tools."""
    field = "rel_diff" if relative else "diff"
    groups: dict[int, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["eid"], []).append(run)
    eids = sorted(groups.keys())

    pooled = {}
    for eid in eids:
        pooled[eid] = np.concatenate([traces[r["run_uid"]][split][field] for r in groups[eid]])

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = cm.viridis(np.linspace(0.1, 0.9, max(len(eids), 2)))[: len(eids)]

    tag = f"{split} split"
    all_pooled = np.concatenate(list(pooled.values()))
    lo, hi = (np.percentile(all_pooled, 0.5), np.percentile(all_pooled, 99.5))
    xs = np.linspace(lo, hi, 512)

    curves = {}
    for color, eid in zip(colors, eids):
        data = pooled[eid]
        weights, means, stds = _fit_best_gmm_1d(data, max_components=max_components)
        pdf = np.zeros_like(xs)
        for w, m, s in zip(weights, means, stds):
            pdf += w * np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        label = _config_label(groups[eid][0]["exp_name"])
        curves[label] = pdf
        k = len(weights)
        n_repeats = len(groups[eid])
        comp_str = f"{k} comp" if k > 1 else "1 comp"
        ax.plot(
            xs,
            pdf,
            color=color,
            linewidth=2.0,
            label=f"{label} (n={n_repeats} repeats, {comp_str}, median={np.median(data):+.1f})",
        )
        ax.fill_between(xs, pdf, color=color, alpha=0.08)

    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.2, alpha=0.7, label="PID parity")
    if relative:
        ax.xaxis.set_major_formatter(lambda x, _: f"{x * 100:.0f}%")
        ax.set_xlabel(
            "Solver steps saved vs. PID, relative  ((pid_steps - rl_steps) / pid_steps, positive = RL better)"
        )
        ax.set_title(f"Relative step-count improvement over PID — per-config GMM fit ({tag})")
    else:
        ax.set_xlabel("Solver steps saved vs. PID  (pid_steps - rl_steps, positive = RL better)")
        ax.set_title(
            f"Step-count improvement over PID — per-config GMM fit across {param_label} ({tag})"
        )
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=1)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")

    x_label = "rel_diff" if relative else "diff"
    _save_curve_columns_txt(xs, curves, os.path.splitext(out_path)[0] + ".txt", x_label=x_label)


def plot_error_dist_distribution(
    traces: dict,
    runs: list[dict],
    split: str,
    out_path: str,
    param_label: str,
    rtol: float,
    num_bins: int = 80,
):
    """Overlaid per-run histograms of log10 relative solution error (RL, solid),
    with the pooled PID error distribution (dashed black) and rtol as references."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = _gradient_colors(runs)

    tag = f"{split} split"
    pid_all = np.concatenate([traces[r["run_uid"]][split]["pid_err"] for r in runs])
    all_errs = np.concatenate([traces[r["run_uid"]][split]["err"] for r in runs] + [pid_all])
    bins = np.histogram_bin_edges(all_errs, bins=num_bins)

    ax.hist(
        pid_all,
        bins=bins,
        histtype="step",
        linewidth=1.4,
        color="k",
        linestyle="--",
        alpha=0.6,
        density=True,
        label=f"PID (median={np.median(pid_all):.2f})",
    )
    for color, run in zip(colors, runs):
        errs = traces[run["run_uid"]][split]["err"]
        ax.hist(
            errs,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=color,
            density=True,
            label=f"{run['exp_name']} (median={np.median(errs):.2f})",
        )

    ax.axvline(
        np.log10(rtol), color="k", linestyle=":", linewidth=1.2, alpha=0.7, label=f"rtol = {rtol:g}"
    )
    ax.set_xlabel("log10 relative L2 error at t_end  (vs tight-tolerance reference)")
    ax.set_ylabel("Density")
    ax.set_title(f"Solution-error distribution across {param_label} ({tag})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# 4b. Per-experiment summary stats table (mean/median/quartiles), pooled
#     across seed repeats -- the "which config is actually best" glance view.
# ---------------------------------------------------------------------------


def summarize_step_diff_stats(
    traces: dict, runs: list[dict], split: str, relative: bool = True
) -> list[dict]:
    """One row per experiment: mean/median/Q1/Q3 of step-savings-vs-PID, pooling
    every seed repeat's episodes into one sample (not averaging repeat-level
    summaries) so quartiles reflect actual per-episode spread."""
    field = "rel_diff" if relative else "diff"
    groups: dict[int, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["eid"], []).append(run)

    rows = []
    for eid in sorted(groups):
        group_runs = groups[eid]
        pooled = np.concatenate([traces[r["run_uid"]][split][field] for r in group_runs])
        q1, median, q3 = np.percentile(pooled, [25, 50, 75])
        rows.append(
            {
                "exp_name": _config_label(group_runs[0]["exp_name"]),
                "n_repeats": len(group_runs),
                "n_samples": int(pooled.size),
                "mean": float(np.mean(pooled)),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
            }
        )
    return rows


def save_step_diff_stats_txt(rows: list[dict], save_path: str, relative: bool):
    """Write pooled step-savings summary rows as a text table."""
    unit = (
        "  (fraction of PID's steps saved; positive = RL better)"
        if relative
        else "  (steps saved vs. PID; positive = RL better)"
    )
    header = ["exp_name", "n_repeats", "n_samples", "mean", "median", "q1", "q3"]
    with open(save_path, "w") as f:
        f.write(f"# Per-experiment step-savings-vs-PID stats, pooled across seed repeats{unit}.\n")
        f.write("  ".join(f"{h:>12s}" for h in header) + "\n")
        for row in rows:
            f.write(
                f"{row['exp_name']:>12s}  {row['n_repeats']:>12d}  {row['n_samples']:>12d}  "
                f"{row['mean']:>12.4f}  {row['median']:>12.4f}  {row['q1']:>12.4f}  {row['q3']:>12.4f}\n"
            )
    print(f"  saved -> {save_path}")


def plot_step_diff_stats_table(rows: list[dict], out_path: str, split: str, relative: bool):
    """Rendered table: one row per experiment, mean/median/Q1/Q3 of step-savings-vs-PID.
    Highest-median row is highlighted as a starting point, not a verdict -- check the
    matching distribution plots for shape (bimodal/heavy-tailed configs can have a
    middling median but a bad worst case)."""
    fmt = (lambda v: f"{v * 100:+.1f}%") if relative else (lambda v: f"{v:+.1f}")
    col_labels = ["experiment", "n (repeats × episodes)", "mean", "median", "Q1", "Q3"]
    cell_text = [
        [
            row["exp_name"],
            f"{row['n_repeats']} × {row['n_samples'] // row['n_repeats']}",
            fmt(row["mean"]),
            fmt(row["median"]),
            fmt(row["q1"]),
            fmt(row["q3"]),
        ]
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.2))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    best_idx = max(range(len(rows)), key=lambda i: rows[i]["median"])
    for c in range(len(col_labels)):
        table[(best_idx + 1, c)].set_facecolor("#d9f2d9")
    metric = "Relative step savings vs. PID" if relative else "Step savings vs. PID"
    ax.set_title(
        f"{metric} summary ({split} split) — highlighted row = highest median", fontsize=11, pad=14
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# 5. Trained-vs-held-out bin comparison (task_sample_scheme sweeps only)
# ---------------------------------------------------------------------------


def summarize_bin_comparison(bin_results: dict) -> list[dict]:
    """One row per run: mean PID-relative improvement split by trained/held-out label.
    A mean over zero bins is reported as NaN with n=0, not silently dropped."""
    rows = []
    for run_uid, r in bin_results.items():
        improvement = np.asarray(r["improvement"])
        trained_mask = np.asarray(r["trained"], dtype=bool)
        trained_vals = improvement[trained_mask]
        held_out_vals = improvement[~trained_mask]
        rows.append(
            {
                "run_uid": run_uid,
                "exp_name": r["exp_name"],
                "scheme": r["scheme"],
                "trained_mean": float(np.mean(trained_vals)) if trained_vals.size else float("nan"),
                "held_out_mean": float(np.mean(held_out_vals))
                if held_out_vals.size
                else float("nan"),
                "trained_n": int(trained_vals.size),
                "held_out_n": int(held_out_vals.size),
            }
        )
    return rows


def save_bin_comparison_txt(summary_rows: list[dict], save_path: str):
    """Columnar .txt: one row per run, trained-vs-held-out mean PID-relative improvement."""
    header = ["exp_name", "scheme", "trained_mean", "trained_n", "held_out_mean", "held_out_n"]
    with open(save_path, "w") as f:
        f.write("# Per-run PID-relative improvement ((pid-rl)/pid), split by whether each\n")
        f.write("# test_bin was inside that run's own train_bins (trained) or not (held_out).\n")
        f.write("  ".join(f"{h:>16s}" for h in header) + "\n")
        for row in summary_rows:
            f.write(
                f"{row['exp_name']:>16s}  {row['scheme']:>16s}  "
                f"{row['trained_mean']:>16.4f}  {row['trained_n']:>16d}  "
                f"{row['held_out_mean']:>16.4f}  {row['held_out_n']:>16d}\n"
            )
    print(f"  saved -> {save_path}")


def plot_bin_comparison(summary_rows: list[dict], out_path: str):
    """Grouped bar chart: trained vs. held-out mean PID-relative improvement per run —
    the cross-arm comparison a task_sample_scheme sweep is for."""
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(summary_rows)), 5.5))
    x = np.arange(len(summary_rows))
    width = 0.35
    trained_vals = [r["trained_mean"] for r in summary_rows]
    held_out_vals = [r["held_out_mean"] for r in summary_rows]
    ax.bar(x - width / 2, trained_vals, width, label="Trained bins", color="#2a78d6")
    ax.bar(x + width / 2, held_out_vals, width, label="Held-out bins", color="#e34948")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_config_label(r["exp_name"]) for r in summary_rows], rotation=20, ha="right"
    )
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_ylabel("Mean PID-relative improvement  ((pid-rl)/pid)")
    ax.set_title("Trained vs. held-out bin performance across schemes")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the multi-seed experiment comparison and save its outputs."""
    args = parse_args()

    experiment_uid = args.experiment_uid.removeprefix("experiment_")

    if args.experiment_date:
        experiment_root = os.path.join(
            args.workdir, "outputs/experiments", args.experiment_date, experiment_uid
        )
    else:
        matches = sorted(
            glob.glob(os.path.join(args.workdir, "outputs/experiments/*", experiment_uid))
        )
        if not matches:
            raise SystemExit(
                f"No experiment batch found for UID {experiment_uid} under "
                f"{os.path.join(args.workdir, 'outputs/experiments/*', experiment_uid)}"
            )
        if len(matches) > 1:
            print(f"[!] Multiple date dirs match UID {experiment_uid}, using latest: {matches[-1]}")
        experiment_root = matches[-1]

    checkpoints_root = os.path.join(experiment_root, "checkpoints")
    outputs_root = os.path.join(experiment_root, "outputs")
    out_dir = args.out or os.path.join(outputs_root, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    runs = discover_runs(checkpoints_root, outputs_root)
    if not runs:
        raise SystemExit(f"No runs found for experiment {experiment_uid} under {outputs_root}")
    print(f"[*] Found {len(runs)} run(s): {[r['exp_name'] for r in runs]}")

    print("[*] Plotting reward trajectories ...")
    plot_reward_trajectories(runs, os.path.join(out_dir, "reward_trajectories.png"))

    # Heavy compute happens here, once, cached to out_dir/data/.
    data = run_and_save(runs, args, out_dir)
    ref_config = data["ref_config"]
    param_label = data["param_label"]

    for i, task_value in enumerate(data["test_points"], start=1):
        out_path = os.path.join(out_dir, f"latent_{i}_{param_label}{task_value:.1f}.png")
        plot_latent_for_task(
            task_value, data["latent_traces_by_task"][i], runs, out_path, param_label
        )

    available_splits = [s for s in ("train", "val", "test") if getattr(ref_config.env, f"{s}_bins")]
    step_diff_traces = data["step_diff_traces"]
    for split in available_splits:
        plot_step_diff_distribution(
            step_diff_traces,
            runs,
            split,
            os.path.join(out_dir, f"step_diff_{split}.png"),
            param_label,
            num_bins=args.step_diff_bins,
        )
        plot_step_diff_distribution(
            step_diff_traces,
            runs,
            split,
            os.path.join(out_dir, f"step_diff_rel_{split}.png"),
            param_label,
            num_bins=args.step_diff_bins,
            relative=True,
        )
        plot_step_diff_gaussian_fit(
            step_diff_traces,
            runs,
            split,
            os.path.join(out_dir, f"step_diff_gmm_{split}.png"),
            param_label,
        )
        plot_step_diff_gaussian_fit(
            step_diff_traces,
            runs,
            split,
            os.path.join(out_dir, f"step_diff_gmm_rel_{split}.png"),
            param_label,
            relative=True,
        )
        plot_error_dist_distribution(
            step_diff_traces,
            runs,
            split,
            os.path.join(out_dir, f"error_dist_{split}.png"),
            param_label,
            rtol=ref_config.env.rtol,
            num_bins=args.step_diff_bins,
        )

        stats_rows = summarize_step_diff_stats(step_diff_traces, runs, split, relative=True)
        save_step_diff_stats_txt(
            stats_rows, os.path.join(out_dir, f"step_diff_stats_{split}.txt"), relative=True
        )
        plot_step_diff_stats_table(
            stats_rows, os.path.join(out_dir, f"step_diff_stats_{split}.png"), split, relative=True
        )

    if ref_config.env.test_bins:
        summary_rows = summarize_bin_comparison(data["bin_results"])
        save_bin_comparison_txt(summary_rows, os.path.join(out_dir, "bin_comparison.txt"))
        plot_bin_comparison(summary_rows, os.path.join(out_dir, "bin_comparison.png"))

    print(f"\n[+] All comparison plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
