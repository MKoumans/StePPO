"""Plot user-specified trajectories from a trained checkpoint.

Standalone — not wired into analyse_rollout.py. Rolls out the policy on an
explicit list of (task, y0) conditions (no sampling) and writes a two-panel
PNG (state-vs-time per dim, scaled error vs step) plus a matching Excel
workbook. Rejected steps can be highlighted (show_rejected) regardless of
step_interval, which only thins the marked/exported points.

Usage: --config points to a PlotCustomTrajectoriesConfig YAML
(configs/inference/plot_custom_trajectories/*.yaml); any field can be
overridden ad hoc, e.g. --step_interval 5.
    uv run python research/ode/inference/plot_custom_trajectories.py \\
        --config configs/inference/plot_custom_trajectories/scalar_decay_example.yaml
"""

from steppo.utils.device import setup_devices

setup_devices()

import dataclasses
import os

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from steppo.configs.base_config import TrainConfig, apply_dotted_overrides, load_config_from_yaml
from steppo.envs.ode import ODEEnv, ODEParams
from steppo.models.huggingface import download_model, load_model_artifact
from steppo.training.eval import run_diagnostic_episode
from steppo.utils.checkpoint import build_models, resolve_checkpoint
from steppo.utils.task_params import task_label

try:
    from ..post_run_analysis.analysis_common import _savefig, try_load_checkpoint
except ImportError:
    from research.ode.post_run_analysis.analysis_common import _savefig, try_load_checkpoint

from configs import PlotCustomTrajectoriesConfig

try:
    from ..cli_config import parse_config_args
    from ..output_dirs import resolve_output_dir
except ImportError:
    from research.ode.cli_config import parse_config_args
    from research.ode.output_dirs import resolve_output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse the plot config path and any CLI overrides."""
    return parse_config_args(
        description=__doc__,
        config_help="PlotCustomTrajectoriesConfig YAML, e.g. "
        "configs/inference/plot_custom_trajectories/<name>.yaml",
    )


def _parse_conditions(raw: list) -> list[tuple[float, tuple[float, ...] | None]]:
    """Normalize the config's `conditions` list into (task, y0_or_None) tuples."""
    conditions = []
    for entry in raw:
        if len(entry) == 1:
            conditions.append((float(entry[0]), None))
        elif len(entry) == 2:
            task, y0 = entry
            conditions.append((float(task), tuple(float(v) for v in y0)))
        else:
            raise ValueError(f"Each condition must be [task] or [task, y0]; got {entry!r}")
    return conditions


def _env_for_condition(base_env: ODEEnv, base_config, y0):
    """Return the env to roll out on for this condition's y0.

    y0 is baked in at env-construction time via the system spec, not
    passable to reset(), so forcing a specific y0 means rebuilding the env
    with sample_y0/y0_x/y0_y pinned. scalar_decay's y0 is hard-fixed at 1.0
    with no config hook, so an explicit y0 there is a user error.
    """
    if y0 is None:
        return base_env

    y_dim = base_env._spec.y_dim
    if len(y0) != y_dim:
        raise ValueError(
            f"y0 has {len(y0)} dims but system '{base_config.env.system}' expects {y_dim}."
        )
    if not hasattr(base_config.env, "sample_y0"):
        raise ValueError(
            f"System '{base_config.env.system}' has no sample_y0/y0_x/y0_y config hook — "
            "an explicit y0 isn't supported for it (its y0 is hard-fixed in the system spec)."
        )
    if y_dim != 2:
        raise ValueError(
            f"Explicit y0 override is only wired up for 2-D systems (sample_y0/y0_x/y0_y); "
            f"'{base_config.env.system}' has y_dim={y_dim}."
        )

    env_config = dataclasses.replace(base_config.env, sample_y0=False, y0_x=y0[0], y0_y=y0[1])
    return ODEEnv(env_config, base_env.max_steps)


def _rollout_condition(vae, policy, base_env, base_config, task, y0, key, max_steps):
    env = _env_for_condition(base_env, base_config, y0)
    env_params = ODEParams(lam=task, pulse_phase=0.0, max_steps=env.max_steps)

    key_reset, key_run = jax.random.split(key)
    obs0, env_state0 = env.reset(key_reset, env_params)
    diag = run_diagnostic_episode(
        vae, policy, env, key_run, max_steps=max_steps, env_params=env_params
    )

    y = np.concatenate([np.asarray(env_state0.y)[None, :], diag["y"]], axis=0)
    t = np.concatenate([[float(env_state0.t)], diag["t"]])
    scaled_error = np.concatenate([[np.nan], diag["scaled_error"]])
    accepted = np.concatenate([[True], diag["keep_step"].astype(bool)])

    return {"y": y, "t": t, "scaled_error": scaled_error, "accepted": accepted}


def _condition_label(system, task, y0):
    label = f"{task_label(system)}={task:g}"
    if y0 is not None:
        label += f", y0={tuple(round(v, 2) for v in y0)}"
    return label


def _condition_cmap(base_color):
    """White (start of trajectory) -> base_color (end of trajectory)."""
    return mcolors.LinearSegmentedColormap.from_list("cond", ["white", base_color])


def _draw_gradient_trajectory(ax, x, y, cmap, mark_idx, rejected_idx, show_rejected, linewidth=1.5):
    """One condition's line + step markers, colored white->cmap(1) along the
    trajectory. Rejected steps (if requested) are always shown in red,
    regardless of --step_interval."""
    n = len(x)
    if n > 1:
        points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        frac = np.linspace(0.0, 1.0, n - 1)
        lc = LineCollection(segments, colors=cmap(frac), linewidth=linewidth, alpha=0.85)
        ax.add_collection(lc)

    mark_idx = mark_idx[mark_idx != 0]
    if mark_idx.size:
        ax.scatter(
            x[mark_idx],
            y[mark_idx],
            c=cmap(mark_idx / max(n - 1, 1)),
            s=25,
            zorder=4,
            edgecolors="none",
        )

    # Step 0: white face, black ring — marks the initial condition.
    ax.scatter([x[0]], [y[0]], color=cmap(0.0), edgecolors="black", linewidth=1.3, s=45, zorder=6)

    if show_rejected and rejected_idx.size:
        ax.scatter(x[rejected_idx], y[rejected_idx], color="red", s=25, zorder=5)


_MAX_STATE_PLOTS = 4  # cap, mirroring trajectory_gif's y-t panel convention


def _plot(results, conditions, system, step_interval, show_rejected, save_path):
    y_dim = results[0]["y"].shape[1]
    n_state_plots = min(y_dim, _MAX_STATE_PLOTS)
    base_colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 2)))

    fig, axes = plt.subplots(n_state_plots + 1, 1, figsize=(8, 3.2 * (n_state_plots + 1)))
    state_axes = axes[:n_state_plots]
    err_ax = axes[-1]
    legend_handles = []

    for i, (res, (task, y0)) in enumerate(zip(results, conditions)):
        base_color = base_colors[i]
        cmap = _condition_cmap(base_color)
        label = _condition_label(system, task, y0)
        n = res["y"].shape[0]
        mark_idx = np.arange(0, n, step_interval)
        rejected_idx = np.where(~res["accepted"])[0]
        t = res["t"]

        for d, ax_d in enumerate(state_axes):
            _draw_gradient_trajectory(
                ax_d, t, res["y"][:, d], cmap, mark_idx, rejected_idx, show_rejected
            )

        steps = np.arange(n)
        _draw_gradient_trajectory(
            err_ax, steps, res["scaled_error"], cmap, mark_idx, rejected_idx, show_rejected
        )

        legend_handles.append(Line2D([0], [0], color=base_color, lw=2, label=label))

    for d, ax_d in enumerate(state_axes):
        ax_d.autoscale_view()
        ax_d.set_xlabel("ODE time t")
        ax_d.set_ylabel(f"y{d}")
        ax_d.set_title(f"y{d} vs time — {system}")
        ax_d.grid(alpha=0.3)
    state_axes[0].legend(handles=legend_handles, fontsize=8)

    err_ax.autoscale_view()
    err_ax.set_yscale("log")
    err_ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="accept threshold")
    err_ax.set_ylim(1e-2, 1.1)
    err_ax.set_xlabel("Step")
    err_ax.set_ylabel("Scaled Error")
    err_ax.set_title("Scaled error vs step")
    err_ax.legend(
        handles=legend_handles
        + [Line2D([0], [0], color="gray", linestyle="--", label="accept threshold")],
        fontsize=8,
    )
    err_ax.grid(alpha=0.3)

    fig.tight_layout()
    _savefig(fig, save_path)


def _export_excel(results, conditions, system, step_interval, save_path):
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        for i, (res, (task, y0)) in enumerate(zip(results, conditions)):
            idx = np.arange(0, res["y"].shape[0], step_interval)
            data = {"t": res["t"][idx]}
            for d in range(res["y"].shape[1]):
                data[f"y{d}"] = res["y"][idx, d]
            data["scaled_error"] = res["scaled_error"][idx]
            df = pd.DataFrame(data)
            sheet_name = f"cond{i}_{_condition_label(system, task, y0)}"[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  saved → {save_path}")


def main():
    """Plot trajectories from explicit task and initial-state conditions."""
    args = parse_args()
    cfg = (
        load_config_from_yaml(PlotCustomTrajectoriesConfig, args.config)
        if args.config
        else PlotCustomTrajectoriesConfig()
    )
    cfg = apply_dotted_overrides(cfg, args.overrides)

    sources = [s for s in (cfg.checkpoint, cfg.repo_id, cfg.model_dir) if s]
    if len(sources) != 1:
        raise SystemExit("config needs exactly one of checkpoint, repo_id, model_dir")
    if not cfg.conditions:
        raise SystemExit("config needs conditions: a list of [task] or [task, [y0...]] entries")

    conditions = _parse_conditions(cfg.conditions)

    resolved_checkpoint = None
    if cfg.checkpoint:
        resolved_checkpoint = resolve_checkpoint(cfg.checkpoint)
        run_dir = os.path.dirname(resolved_checkpoint)
        config_path = cfg.env_config
        if config_path is None:
            bundled = os.path.join(run_dir, "config.yaml")
            if not os.path.isfile(bundled):
                raise SystemExit(
                    f"env_config is required (no config.yaml found next to {resolved_checkpoint})."
                )
            config_path = bundled
            print(f"[*] Auto-detected config: {config_path}")
        config = load_config_from_yaml(TrainConfig, config_path, strict=False)
        env = None
        vae = policy = None
        source_label = f"checkpoint {resolved_checkpoint}"
    elif cfg.repo_id:
        loaded = download_model(cfg.repo_id, revision=cfg.revision)
        run_dir = os.getcwd()
        config = loaded.config
        env, vae, policy = loaded.env, loaded.vae, loaded.policy
        source_label = f"Hub repository {cfg.repo_id}@{cfg.revision}"
    else:
        if cfg.env_config is not None:
            raise SystemExit("env_config can only be used with checkpoint.")
        loaded = load_model_artifact(cfg.model_dir)
        run_dir = cfg.model_dir
        config = loaded.config
        env, vae, policy = loaded.env, loaded.vae, loaded.policy
        source_label = f"local model artifact {cfg.model_dir}"

    system = config.env.system
    if cfg.rollout_steps is not None:
        config.rollout_steps = cfg.rollout_steps
    out_dir = cfg.out_dir or resolve_output_dir(
        "inference", cfg, name=cfg.name, exclude=("name", "out_dir")
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] System     : {system}")
    print(f"[*] Model      : {source_label}")
    print(f"[*] Conditions : {conditions}")
    print(f"[*] rollout_steps (env step budget): {config.rollout_steps}  t_end: {config.env.t_end}")

    if cfg.checkpoint:
        env = ODEEnv(config.env, config.rollout_steps)
        vae, policy = build_models(config, env, cfg.seed)
        vae, policy = try_load_checkpoint(
            vae, policy, resolved_checkpoint, backbone=config.backbone, algo=config.algo
        )
    elif cfg.rollout_steps is not None:
        env = ODEEnv(config.env, config.rollout_steps)

    base_key = jax.random.PRNGKey(cfg.seed)
    results = []
    for i, (task, y0) in enumerate(conditions):
        key = jax.random.fold_in(base_key, i)
        res = _rollout_condition(vae, policy, env, config, task, y0, key, cfg.max_steps)
        results.append(res)
        print(
            f"  [{i}] {_condition_label(system, task, y0)}: "
            f"{res['y'].shape[0]} steps, {int((~res['accepted']).sum())} rejected"
        )

    png_path = os.path.join(out_dir, f"{cfg.tag}.png")
    xlsx_path = os.path.join(out_dir, f"{cfg.tag}.xlsx")
    _plot(results, conditions, system, cfg.step_interval, cfg.show_rejected, png_path)
    _export_excel(results, conditions, system, cfg.step_interval, xlsx_path)


if __name__ == "__main__":
    main()
