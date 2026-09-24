"""Analyse and compare a factorial experiment across scalar_decay configs.

Reads the metrics JSON files produced by each training run, and optionally
runs analysis rollouts from checkpoints (reusing analyze_rollout.py logic).

Produces a multi-panel comparison dashboard:

  1. Training curves (eval return, success rate) overlaid per experiment
  2. VAE loss breakdown per experiment
  3. Belief quality (sigma reduction, task MAE) over training
  4. Final performance bar charts grouped by factorial condition
  5. Interaction effects heatmaps (direction x task_loss x ema)
  6. Per-experiment rollout analysis (if checkpoints are available)

Usage:
    # Point at the outputs directory containing run folders:
    python research/ode/analyse_experiment.py \\
        --runs outputs/run_scalar_decay_e1 outputs/run_scalar_decay_e2 ... \\
        --out outputs/experiment_comparison

    # Or auto-discover runs by experiment name prefix:
    python research/ode/analyse_experiment.py \\
        --discover outputs/ --prefix scalar_decay_e \\
        --out outputs/experiment_comparison

    # Also run analysis rollouts from checkpoint:
    python research/ode/analyse_experiment.py \\
        --discover outputs/ --prefix scalar_decay_e \\
        --checkpoint checkpoint/ \\
        --out outputs/experiment_comparison
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
import flax.nnx as nnx
import jax
import matplotlib.pyplot as plt
from post_run_analysis.analysis_common import _savefig

from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.utils.plotting import get_series

# ---------------------------------------------------------------------------
# Experiment design: the 2^3 factorial
# ---------------------------------------------------------------------------

FACTORS = ["direction", "task_loss", "ema"]

EXPERIMENT_DESIGN = {
    "e1": {"direction": False, "task_loss": False, "ema": False},
    "e2": {"direction": False, "task_loss": False, "ema": True},
    "e3": {"direction": False, "task_loss": True, "ema": False},
    "e4": {"direction": False, "task_loss": True, "ema": True},
    "e5": {"direction": True, "task_loss": False, "ema": False},
    "e6": {"direction": True, "task_loss": False, "ema": True},
    "e7": {"direction": True, "task_loss": True, "ema": False},
    "e8": {"direction": True, "task_loss": True, "ema": True},
}


def exp_label(eid: str) -> str:
    """Return the readable description of a factorial experiment variant."""
    d = EXPERIMENT_DESIGN[eid]
    parts = []
    parts.append("dir" if d["direction"] else "no_dir")
    parts.append("task" if d["task_loss"] else "no_task")
    parts.append("ema" if d["ema"] else "no_ema")
    return f"{eid}: {'+'.join(parts)}"


def short_label(eid: str) -> str:
    """Return the compact legend label for an experiment variant."""
    d = EXPERIMENT_DESIGN[eid]
    symbols = []
    if d["direction"]:
        symbols.append("D")
    if d["task_loss"]:
        symbols.append("T")
    if d["ema"]:
        symbols.append("E")
    return f"{eid}" + (f" ({','.join(symbols)})" if symbols else " (baseline)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse run-discovery, plotting, and output options."""
    p = argparse.ArgumentParser(description="Factorial experiment comparison for ODE systems")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--runs", nargs="+", help="Explicit list of output directories (one per experiment)"
    )
    g.add_argument("--discover", type=str, help="Parent directory to auto-discover runs in")
    p.add_argument(
        "--prefix",
        type=str,
        default="scalar_decay_e",
        help="Experiment name prefix for auto-discovery (default: scalar_decay_e)",
    )
    p.add_argument(
        "--configs",
        type=str,
        default="configs/envs/ode/scalar_decay",
        help="Directory containing the experiment YAML configs",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint base directory (enables rollout analysis)",
    )
    p.add_argument("--num_envs", type=int, default=64, help="Environments for rollout analysis")
    p.add_argument(
        "--out",
        type=str,
        default="outputs/experiment_comparison",
        help="Output directory for comparison plots",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def discover_runs(parent_dir: str, prefix: str) -> dict[str, str]:
    """Find run directories matching the experiment prefix.

    Searches for metrics JSON files named <prefix>*_metrics.json inside
    run_* subdirectories of parent_dir.  Returns {experiment_id: run_dir}.
    """
    found = {}
    for run_dir in sorted(glob.glob(os.path.join(parent_dir, "run_*"))):
        if not os.path.isdir(run_dir):
            continue
        for json_file in glob.glob(os.path.join(run_dir, f"{prefix}*_metrics.json")):
            basename = os.path.basename(json_file)
            match = re.search(r"(e\d+)", basename)
            if match:
                eid = match.group(1)
                if eid in EXPERIMENT_DESIGN:
                    found[eid] = run_dir
    return found


def load_metrics(run_dir: str, exp_name: str) -> list[dict] | None:
    """Load one experiment's metrics JSON, if present."""
    json_path = os.path.join(run_dir, f"{exp_name}_metrics.json")
    if not os.path.isfile(json_path):
        for f in glob.glob(os.path.join(run_dir, "*_metrics.json")):
            json_path = f
            break
    if not os.path.isfile(json_path):
        print(f"  [!] No metrics JSON found in {run_dir}")
        return None
    with open(json_path) as f:
        return json.load(f)


def find_checkpoint(checkpoints_dir: str, exp_name: str) -> str | None:
    """Find the latest checkpoint for an experiment."""
    for run_dir in sorted(glob.glob(os.path.join(checkpoints_dir, "run_*")), reverse=True):
        checkpoints = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*")))
        if checkpoints:
            parent_json = os.path.join(
                run_dir.replace("checkpoints", "outputs", 1), f"{exp_name}_metrics.json"
            )
            if os.path.isfile(parent_json) or exp_name in run_dir:
                return checkpoints[-1]
    pattern = os.path.join(checkpoints_dir, f"*{exp_name}*", "checkpoint_*")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    return None


# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------

PALETTE = {
    "e1": "#1f77b4",
    "e2": "#ff7f0e",
    "e3": "#2ca02c",
    "e4": "#d62728",
    "e5": "#9467bd",
    "e6": "#8c564b",
    "e7": "#e377c2",
    "e8": "#7f7f7f",
}


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------


def plot_training_curves(all_data: dict, out_dir: str):
    """Overlaid eval/mean_return and eval/success_rate for all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Training Performance Across Experiments", fontsize=14, fontweight="bold")

    for eid, metrics in sorted(all_data.items()):
        color = PALETTE[eid]
        label = short_label(eid)

        it, ret = get_series(metrics, "eval/mean_return")
        if len(it) > 0:
            ax1.plot(it, ret, color=color, linewidth=1.8, label=label, alpha=0.85)

        it, sr = get_series(metrics, "eval/success_rate")
        if len(it) > 0:
            ax2.plot(it, sr, color=color, linewidth=1.8, label=label, alpha=0.85)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean Return")
    ax1.set_title("Evaluation Return")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, linewidth=0.4)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Success Rate (t_reached >= t_end)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, linewidth=0.4)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "training_curves.png"))


def plot_vae_losses(all_data: dict, out_dir: str):
    """VAE loss components for each experiment in a grid."""
    eids = sorted(all_data.keys())
    n = len(eids)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle("VAE Loss Breakdown per Experiment", fontsize=14, fontweight="bold")

    loss_keys = [
        ("vae/total_loss", "Total", "black"),
        ("vae/kl_loss", "KL", "tab:orange"),
        ("vae/rew_loss", "Reward", "tab:red"),
        ("vae/state_loss", "State", "tab:purple"),
        ("vae/task_loss", "Task", "tab:green"),
    ]

    for idx, eid in enumerate(eids):
        ax = axes[idx // cols][idx % cols]
        metrics = all_data[eid]

        for key, name, color in loss_keys:
            it, vals = get_series(metrics, key)
            if len(it) > 0:
                ax.plot(it, vals, color=color, linewidth=1.2, label=name, alpha=0.8)

        ax.set_title(short_label(eid), fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        try:
            ax.set_yscale("log")
        except ValueError:
            pass
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, which="both", linewidth=0.3)

    for idx in range(len(eids), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "vae_losses.png"))


def plot_belief_quality(all_data: dict, out_dir: str):
    """Sigma reduction ratio and task embedding MAE over training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Belief Quality Over Training", fontsize=14, fontweight="bold")

    for eid, metrics in sorted(all_data.items()):
        color = PALETTE[eid]
        label = short_label(eid)

        it, sr = get_series(metrics, "eval/sigma_reduction_ratio")
        if len(it) > 0:
            ax1.plot(it, sr, color=color, linewidth=1.8, label=label, alpha=0.85)

        it, mae = get_series(metrics, "eval/task_embedding_mae")
        if len(it) > 0:
            ax2.plot(it, mae, color=color, linewidth=1.8, label=label, alpha=0.85)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Sigma Reduction Ratio")
    ax1.set_title("Belief Uncertainty Reduction")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, linewidth=0.4)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Task Embedding MAE")
    ax2.set_title("Task Identification Accuracy")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, linewidth=0.4)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "belief_quality.png"))


def plot_final_bars(all_data: dict, out_dir: str):
    """Bar chart of final metrics for each experiment, grouped by condition."""
    eids = sorted(all_data.keys())

    final_metrics = {}
    for eid in eids:
        metrics = all_data[eid]
        last_eval = {}
        for m in reversed(metrics):
            if "eval/mean_return" in m and m["eval/mean_return"] is not None:
                last_eval = m
                break
        final_metrics[eid] = {
            "return": last_eval.get("eval/mean_return", 0.0),
            "success": last_eval.get("eval/success_rate", 0.0),
            "sigma_red": last_eval.get("eval/sigma_reduction_ratio", 0.0),
            "task_mae": last_eval.get("eval/task_embedding_mae", None),
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Final Performance Comparison", fontsize=14, fontweight="bold")

    x = np.arange(len(eids))
    labels = [short_label(eid) for eid in eids]
    colors = [PALETTE[eid] for eid in eids]

    # Return
    ax = axes[0, 0]
    vals = [final_metrics[eid]["return"] for eid in eids]
    ax.bar(x, vals, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean Return")
    ax.set_title("Final Eval Return")
    ax.grid(True, axis="y", linewidth=0.4)

    # Success rate
    ax = axes[0, 1]
    vals = [final_metrics[eid]["success"] for eid in eids]
    ax.bar(x, vals, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Success Rate")
    ax.set_title("Final Success Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linewidth=0.4)

    # Sigma reduction
    ax = axes[1, 0]
    vals = [final_metrics[eid]["sigma_red"] for eid in eids]
    ax.bar(x, vals, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Sigma Reduction Ratio")
    ax.set_title("Final Belief Uncertainty Reduction")
    ax.grid(True, axis="y", linewidth=0.4)

    # Task MAE
    ax = axes[1, 1]
    vals = [final_metrics[eid]["task_mae"] or 0.0 for eid in eids]
    has_task = [final_metrics[eid]["task_mae"] is not None for eid in eids]
    bar_colors = [colors[i] if has_task[i] else "lightgrey" for i in range(len(eids))]
    ax.bar(x, vals, color=bar_colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Task Embedding MAE")
    ax.set_title("Final Task Identification (lower = better)")
    ax.grid(True, axis="y", linewidth=0.4)
    if not all(has_task):
        ax.annotate(
            "grey = no task loss",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            color="grey",
            va="top",
        )

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "final_bars.png"))

    return final_metrics


def plot_interaction_effects(final_metrics: dict, out_dir: str):
    """2x2 heatmaps showing interaction effects between the three factors.

    Three panels: direction x task_loss (averaged over ema),
                  direction x ema (averaged over task_loss),
                  task_loss x ema (averaged over direction).
    """

    def avg_metric(factor_vals: dict, metric_key: str) -> float:
        vals = []
        for eid, design in EXPERIMENT_DESIGN.items():
            if eid not in final_metrics:
                continue
            if all(design[f] == v for f, v in factor_vals.items()):
                v = final_metrics[eid].get(metric_key, 0.0)
                if v is None:
                    v = 0.0
                vals.append(v)
        return np.mean(vals) if vals else 0.0

    metric_key = "return"
    metric_label = "Mean Return"

    pairs = [
        ("direction", "task_loss", "ema"),
        ("direction", "ema", "task_loss"),
        ("task_loss", "ema", "direction"),
    ]

    # Compute all grids first to find a shared color range
    grids = []
    for f1, f2, _ in pairs:
        grid = np.zeros((2, 2))
        for i, v1 in enumerate([False, True]):
            for j, v2 in enumerate([False, True]):
                grid[i, j] = avg_metric({f1: v1, f2: v2}, metric_key)
        grids.append(grid)

    all_vals = np.concatenate([g.ravel() for g in grids])
    vmin, vmax = all_vals.min(), all_vals.max()
    # Ensure a meaningful range: pad by at least 5% of the mean
    margin = max((vmax - vmin) * 0.1, abs(all_vals.mean()) * 0.05, 0.01)
    vmin -= margin
    vmax += margin

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Interaction Effects on {metric_label}", fontsize=14, fontweight="bold")

    for ax, (f1, f2, _), grid in zip(axes, pairs, grids):
        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"no {f2}", f2])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"no {f1}", f1])
        ax.set_title(f"{f1} x {f2}")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "interaction_effects.png"))


def plot_factor_main_effects(final_metrics: dict, out_dir: str):
    """Main effect of each factor averaged across the other two.

    Shows how much each factor improves/degrades each metric on average.
    """
    metric_keys = [
        ("return", "Mean Return"),
        ("success", "Success Rate"),
        ("sigma_red", "Sigma Reduction"),
    ]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(5 * len(metric_keys), 5))
    fig.suptitle("Main Effects of Each Factor", fontsize=14, fontweight="bold")

    for ax, (mk, mlabel) in zip(axes, metric_keys):
        effects = []
        for factor in FACTORS:
            on_vals, off_vals = [], []
            for eid, design in EXPERIMENT_DESIGN.items():
                if eid not in final_metrics:
                    continue
                v = final_metrics[eid].get(mk, 0.0)
                if v is None:
                    v = 0.0
                if design[factor]:
                    on_vals.append(v)
                else:
                    off_vals.append(v)
            on_mean = np.mean(on_vals) if on_vals else 0.0
            off_mean = np.mean(off_vals) if off_vals else 0.0
            effects.append(on_mean - off_mean)

        colors_bar = ["#2ca02c" if e >= 0 else "#d62728" for e in effects]
        x = np.arange(len(FACTORS))
        ax.bar(x, effects, color=colors_bar, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(FACTORS)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"Δ {mlabel}")
        ax.set_title(mlabel)
        ax.grid(True, axis="y", linewidth=0.4)

        for i, e in enumerate(effects):
            ax.text(
                i,
                e + 0.01 * (1 if e >= 0 else -1) * max(abs(e) for e in effects),
                f"{e:+.3f}",
                ha="center",
                va="bottom" if e >= 0 else "top",
                fontsize=9,
            )

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "main_effects.png"))


# ---------------------------------------------------------------------------
# Rollout analysis (reuses analyze_rollout.py)
# ---------------------------------------------------------------------------


def run_rollout_comparison(all_data: dict, args, out_dir: str):
    """Run analysis rollouts for experiments with available checkpoints."""
    from research.ode.post_run_analysis.analyse_rollout import (
        collect_analysis_rollout,
        get_task_param,
        plot_task_vs_belief,
    )
    from steppo.envs.ode import ODEEnv
    from steppo.utils.checkpoint import build_models, load_checkpoint

    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

    rollout_dir = os.path.join(out_dir, "rollouts")
    os.makedirs(rollout_dir, exist_ok=True)

    # Collect final belief correlation per experiment
    correlations = {}

    # Derive system name from the config directory (e.g. "scalar_decay" or "van_der_pol")
    system_prefix = os.path.basename(args.configs.rstrip("/"))

    for eid in sorted(EXPERIMENT_DESIGN.keys()):
        exp_name = f"{system_prefix}_{eid}"
        config_path = os.path.join(args.configs, f"{exp_name}.yaml")
        if not os.path.isfile(config_path):
            continue

        ckpt = find_checkpoint(args.checkpoints, exp_name)
        if ckpt is None:
            print(f"  [{eid}] No checkpoint found, skipping rollout")
            continue

        print(f"  [{eid}] Loading checkpoint: {ckpt}")
        config = load_config_from_yaml(TrainConfig, config_path)
        env = ODEEnv(config.env, config.rollout_steps)
        vae, policy = build_models(config, env, config.seed)

        try:
            vae, policy = load_checkpoint(
                vae, policy, ckpt, backbone=config.backbone, algo=config.algo
            )
        except Exception as e:
            print(f"  [{eid}] Checkpoint load failed: {e}")
            continue

        rng = jax.random.PRNGKey(args.seed)
        rng, key_task, key_rollout = jax.random.split(rng, 3)
        task_keys = jax.random.split(key_task, args.num_envs)
        env_params_batch = jax.vmap(env.sample_task)(task_keys)
        true_params, param_label = get_task_param(env_params_batch, config.env.system)

        vae_graphdef, vae_params = nnx.split(vae)
        policy_graphdef, policy_params = nnx.split(policy)

        @jax.jit
        def run_rollout(vae_p, policy_p, env_p, key):
            v = nnx.merge(vae_graphdef, vae_p)
            p = nnx.merge(policy_graphdef, policy_p)
            return collect_analysis_rollout(
                v, p, env, env_p, args.num_envs, config.rollout_steps, key
            )

        traces = run_rollout(vae_params, policy_params, env_params_batch, key_rollout)
        traces_np = jax.tree.map(np.array, traces)

        # Save per-experiment task_vs_belief plot
        exp_dir = os.path.join(rollout_dir, eid)
        os.makedirs(exp_dir, exist_ok=True)
        plot_task_vs_belief(traces_np, exp_dir, true_params, config.env.system)

        # Compute correlation for summary
        mu = traces_np["belief_mu"]
        active = traces_np["active"]
        T, E, latent_dim = mu.shape
        last_step = np.maximum(active.sum(axis=0).astype(int) - 1, 0)
        final_mu = mu[last_step, np.arange(E), :]

        best_r = 0.0
        for d in range(latent_dim):
            if true_params.std() > 1e-8 and final_mu[:, d].std() > 1e-8:
                r = abs(float(np.corrcoef(true_params, final_mu[:, d])[0, 1]))
                best_r = max(best_r, r)
        correlations[eid] = best_r
        print(f"  [{eid}] Best |r| = {best_r:.3f}")

    if correlations:
        plot_correlation_comparison(correlations, out_dir)

    return correlations


def plot_correlation_comparison(correlations: dict, out_dir: str):
    """Bar chart of belief-task correlation per experiment."""
    eids = sorted(correlations.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Belief-Task Parameter Correlation (|r|)", fontsize=14, fontweight="bold")

    x = np.arange(len(eids))
    vals = [correlations[eid] for eid in eids]
    colors = [PALETTE[eid] for eid in eids]
    labels = [short_label(eid) for eid in eids]

    ax.bar(x, vals, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|Pearson r|")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="strong (0.9)")
    ax.axhline(
        0.7, color="orange", linewidth=0.8, linestyle="--", alpha=0.5, label="moderate (0.7)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linewidth=0.4)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "correlation_comparison.png"))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(final_metrics: dict, correlations: dict | None = None):
    """Print a text summary table to stdout."""
    print("\n" + "=" * 90)
    print("  FACTORIAL EXPERIMENT SUMMARY")
    print("=" * 90)

    header = (
        f"{'Exp':>4s}  {'Dir':>3s} {'Task':>4s} {'EMA':>3s}  "
        f"{'Return':>8s} {'Success':>8s} {'SigmaRed':>9s} {'TaskMAE':>8s}"
    )
    if correlations:
        header += f"  {'|r|':>6s}"
    print(header)
    print("-" * len(header))

    for eid in sorted(EXPERIMENT_DESIGN.keys()):
        if eid not in final_metrics:
            continue
        d = EXPERIMENT_DESIGN[eid]
        fm = final_metrics[eid]

        row = (
            f"{eid:>4s}  {'Y' if d['direction'] else 'N':>3s} "
            f"{'Y' if d['task_loss'] else 'N':>4s} "
            f"{'Y' if d['ema'] else 'N':>3s}  "
            f"{fm['return']:>8.2f} {fm['success']:>8.2%} "
            f"{fm['sigma_red']:>9.2f} "
        )

        if fm["task_mae"] is not None:
            row += f"{fm['task_mae']:>8.4f}"
        else:
            row += f"{'N/A':>8s}"

        if correlations and eid in correlations:
            row += f"  {correlations[eid]:>6.3f}"
        print(row)

    print("=" * 90)

    # Main effects summary
    print("\nMain Effects (ON - OFF average):")
    for factor in FACTORS:
        on_returns, off_returns = [], []
        for eid, design in EXPERIMENT_DESIGN.items():
            if eid not in final_metrics:
                continue
            v = final_metrics[eid].get("return", 0.0) or 0.0
            if design[factor]:
                on_returns.append(v)
            else:
                off_returns.append(v)
        if on_returns and off_returns:
            effect = np.mean(on_returns) - np.mean(off_returns)
            print(f"  {factor:>12s}: {effect:>+.4f} return")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Discover factorial runs, compare their metrics, and save plots."""
    args = parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # ── Discover or use explicit runs ────────────────────────────
    if args.discover:
        run_map = discover_runs(args.discover, args.prefix)
        if not run_map:
            print(f"[!] No runs found in {args.discover} matching prefix '{args.prefix}'")
            print("    Looking for *_metrics.json files directly...")
            for json_file in sorted(
                glob.glob(os.path.join(args.discover, "**", "*_metrics.json"), recursive=True)
            ):
                match = re.search(r"(e\d+)", os.path.basename(json_file))
                if match and match.group(1) in EXPERIMENT_DESIGN:
                    run_map[match.group(1)] = os.path.dirname(json_file)
    else:
        run_map = {}
        for run_dir in args.runs:
            for json_file in glob.glob(os.path.join(run_dir, "*_metrics.json")):
                match = re.search(r"(e\d+)", os.path.basename(json_file))
                if match and match.group(1) in EXPERIMENT_DESIGN:
                    run_map[match.group(1)] = run_dir

    print(f"[*] Found {len(run_map)} experiment runs: {sorted(run_map.keys())}")

    # ── Load metrics ─────────────────────────────────────────────
    all_data = {}
    # Derive system name from prefix (e.g. "scalar_decay_e" -> "scalar_decay")
    system_prefix = args.prefix.rstrip("_e").rstrip("_")

    for eid, run_dir in sorted(run_map.items()):
        exp_name = f"{system_prefix}_{eid}"
        metrics = load_metrics(run_dir, exp_name)
        if metrics is not None:
            all_data[eid] = metrics
            print(f"  [{eid}] Loaded {len(metrics)} iterations from {run_dir}")

    if not all_data:
        print("[!] No metrics data loaded. Nothing to compare.")
        return

    # ── Generate comparison plots ────────────────────────────────
    print(f"\n[*] Generating comparison plots in {out_dir} ...")
    plot_training_curves(all_data, out_dir)
    plot_vae_losses(all_data, out_dir)
    plot_belief_quality(all_data, out_dir)
    final_metrics = plot_final_bars(all_data, out_dir)
    plot_interaction_effects(final_metrics, out_dir)
    plot_factor_main_effects(final_metrics, out_dir)

    # ── Rollout analysis (optional) ──────────────────────────────
    correlations = None
    if args.checkpoints:
        print(f"\n[*] Running rollout analysis from checkpoints in {args.checkpoints} ...")
        correlations = run_rollout_comparison(all_data, args, out_dir)

    # ── Summary ──────────────────────────────────────────────────
    print_summary_table(final_metrics, correlations)
    print(f"\n[+] All comparison plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
