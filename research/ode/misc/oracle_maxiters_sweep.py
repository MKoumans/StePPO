"""Measure oracle search-budget sensitivity on one fixed ODE evaluation set.

Edit the constants below, then run this file directly. The same task parameters
and episode keys are reused for every search budget, so differences in the
plotted metric are attributable to the oracle search budget.
"""

import logging
import time
from pathlib import Path

import diffrax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv  # noqa: F401 — initialize before error_dist
from steppo.training.error_dist import (
    sample_eval_tasks,
    solve_oracle_batch,
    solve_pid_baseline_batch,
)
from steppo.training.pid_solve import solve_pid_batch
from steppo.utils.device import setup_devices

CONFIG_PATH = "configs/envs/ode/van_der_pol/van_der_pol_default.yaml"
SEED = 0
NUM_ENVS = 1024
MAX_ITERS = 8

LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = "outputs/analysis/oracle_maxiters_sweep"


def build_threshold_grid(
    threshold_min: float, threshold_max: float, num_thresholds: int
) -> np.ndarray:
    """Return a positive geometric threshold grid including both endpoints."""
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2")
    if threshold_min <= 0 or threshold_max <= threshold_min:
        raise ValueError("threshold bounds must satisfy 0 < min < max")
    return np.geomspace(threshold_min, threshold_max, num_thresholds)


def summarize_threshold(pid_steps: np.ndarray, oracle_steps: np.ndarray) -> dict[str, float]:
    """Aggregate one threshold's per-environment relative improvements."""
    pid = np.asarray(pid_steps, dtype=np.float64)
    oracle = np.asarray(oracle_steps, dtype=np.float64)
    if pid.shape != oracle.shape or pid.ndim != 1:
        raise ValueError("pid_steps and oracle_steps must be same-shaped 1D arrays")
    if pid.size == 0:
        raise ValueError("at least one environment is required")

    per_env = (pid - oracle) / np.maximum(pid, 1.0)
    return {
        "mean_relative_improvement": float(np.mean(per_env)),
        "std_relative_improvement": float(np.std(per_env)),
        "mean_pid_steps": float(np.mean(pid)),
        "mean_oracle_steps": float(np.mean(oracle)),
    }


def run_threshold_sweep(
    config,
    task_params,
    episode_keys,
    max_steps: int,
    thresholds,
    max_iters: int,
) -> dict[str, np.ndarray]:
    """Run PID once and the real-env oracle once per threshold."""
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("thresholds must be a non-empty 1D array")

    sweep_start = time.perf_counter()
    LOGGER.info(
        "Starting oracle threshold sweep: %d environments, %d thresholds, max_iters=%d",
        len(task_params),
        len(thresholds),
        max_iters,
    )

    pid_controller = diffrax.PIDController(
        rtol=config.rtol,
        atol=config.atol,
        dtmin=config.dt_min,
        dtmax=config.dt_max,
        force_dtmin=False,
    )
    LOGGER.info("Computing PID baseline...")
    pid_start = time.perf_counter()
    pid_out = solve_pid_batch(config, pid_controller, task_params, episode_keys, max_steps)
    pid_steps = np.asarray(pid_out["steps"], dtype=np.float64)
    LOGGER.info(
        "PID baseline complete: mean_steps=%.2f (%.1fs)",
        float(np.mean(pid_steps)),
        time.perf_counter() - pid_start,
    )

    rows = []
    for index, threshold in enumerate(thresholds, start=1):
        threshold_start = time.perf_counter()
        LOGGER.info(
            "[%d/%d] Solving oracle for threshold=%.6g...",
            index,
            len(thresholds),
            threshold,
        )
        oracle_out = solve_oracle_batch(
            config,
            task_params,
            episode_keys,
            max_steps,
            threshold=float(threshold),
            max_iters=int(max_iters),
        )

        summary = summarize_threshold(pid_steps, oracle_out["steps"])
        rows.append(summary)
        LOGGER.info(
            "[%d/%d] threshold=%.6g complete: improvement=%.4f, "
            "mean_oracle_steps=%.2f (%.1fs; total %.1fs)",
            index,
            len(thresholds),
            threshold,
            summary["mean_relative_improvement"],
            summary["mean_oracle_steps"],
            time.perf_counter() - threshold_start,
            time.perf_counter() - sweep_start,
        )
    LOGGER.info(
        "Oracle threshold sweep complete in %.1fs",
        time.perf_counter() - sweep_start,
    )

    return {
        "threshold": thresholds,
        "mean_relative_improvement": np.array([row["mean_relative_improvement"] for row in rows]),
        "std_relative_improvement": np.array([row["std_relative_improvement"] for row in rows]),
        "mean_pid_steps": np.array([row["mean_pid_steps"] for row in rows]),
        "mean_oracle_steps": np.array([row["mean_oracle_steps"] for row in rows]),
    }


def run_max_iters_sweep(
    config,
    task_params,
    episode_keys,
    max_steps: int,
    max_iters_values,
) -> dict[str, np.ndarray]:
    """Run the oracle for several per-step search budgets.

    Threshold-based early stopping has been removed from the oracle. The
    meaningful sensitivity axis is therefore the number of candidate-search
    iterations allowed for each trajectory step.
    """
    max_iters_values = np.asarray(max_iters_values, dtype=np.int32)
    if max_iters_values.ndim != 1 or max_iters_values.size == 0:
        raise ValueError("max_iters_values must be a non-empty 1D array")
    if np.any(max_iters_values < 0):
        raise ValueError("all max_iters values must be non-negative")

    sweep_start = time.perf_counter()
    LOGGER.info(
        "Starting oracle max_iters sweep: %d environments, %d budgets, max_iters=%d..%d",
        len(task_params),
        len(max_iters_values),
        int(max_iters_values.min()),
        int(max_iters_values.max()),
    )

    LOGGER.info("Computing PID baseline with the oracle's exact fallback...")
    pid_start = time.perf_counter()
    pid_out = solve_pid_baseline_batch(config, task_params, episode_keys, max_steps)
    pid_steps = np.asarray(pid_out["steps"], dtype=np.float64)
    LOGGER.info(
        "PID baseline complete: mean_steps=%.2f (%.1fs)",
        float(np.mean(pid_steps)),
        time.perf_counter() - pid_start,
    )

    rows = []
    for index, search_iters in enumerate(max_iters_values, start=1):
        iteration_start = time.perf_counter()
        LOGGER.info(
            "[%d/%d] Solving oracle for max_iters=%d...",
            index,
            len(max_iters_values),
            int(search_iters),
        )
        oracle_out = (
            pid_out
            if int(search_iters) == 0
            else solve_oracle_batch(
                config,
                task_params,
                episode_keys,
                max_steps,
                max_iters=int(search_iters),
            )
        )
        summary = summarize_threshold(pid_steps, oracle_out["steps"])
        rows.append(summary)
        LOGGER.info(
            "[%d/%d] max_iters=%d complete: improvement=%.4f, "
            "mean_oracle_steps=%.2f (%.1fs; total %.1fs)",
            index,
            len(max_iters_values),
            int(search_iters),
            summary["mean_relative_improvement"],
            summary["mean_oracle_steps"],
            time.perf_counter() - iteration_start,
            time.perf_counter() - sweep_start,
        )

    LOGGER.info(
        "Oracle max_iters sweep complete in %.1fs",
        time.perf_counter() - sweep_start,
    )
    return {
        "max_iters": max_iters_values,
        "mean_relative_improvement": np.array([row["mean_relative_improvement"] for row in rows]),
        "std_relative_improvement": np.array([row["std_relative_improvement"] for row in rows]),
        "mean_pid_steps": np.array([row["mean_pid_steps"] for row in rows]),
        "mean_oracle_steps": np.array([row["mean_oracle_steps"] for row in rows]),
    }


def save_max_iters_sweep_csv(result: dict[str, np.ndarray], output_path) -> None:
    """Write the max-iteration sweep values and step-count summaries."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "max_iters",
        "mean_relative_improvement",
        "std_relative_improvement",
        "mean_pid_steps",
        "mean_oracle_steps",
    )
    data = np.column_stack([result[column] for column in columns])
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header=",".join(columns),
        comments="",
    )


def plot_max_iters_sweep(
    result: dict[str, np.ndarray],
    system: str,
    num_envs: int,
    max_iters: int,
    output_path,
) -> None:
    """Save mean relative step improvement versus search budget."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        result["max_iters"],
        result["mean_relative_improvement"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        color="#4472C4",
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Oracle search iterations per trajectory step")
    ax.set_ylabel("Mean relative step improvement vs PID")
    ax.set_title(
        f"Oracle search-budget sensitivity — {system} (N={num_envs}, max_iters≤{max_iters})"
    )
    ax.set_xticks(result["max_iters"])
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_threshold_sweep_csv(result: dict[str, np.ndarray], output_path) -> None:
    """Write the plotted values and supporting step-count summaries."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "threshold",
        "mean_relative_improvement",
        "std_relative_improvement",
        "mean_pid_steps",
        "mean_oracle_steps",
    )
    data = np.column_stack([result[column] for column in columns])
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header=",".join(columns),
        comments="",
    )


def plot_threshold_sweep(
    result: dict[str, np.ndarray],
    system: str,
    num_envs: int,
    max_iters: int,
    output_path,
) -> None:
    """Save mean relative step improvement versus oracle threshold."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        result["threshold"],
        result["mean_relative_improvement"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        color="#4472C4",
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Oracle convergence threshold")
    ax.set_ylabel("Mean relative step improvement vs PID")
    ax.set_title(f"Oracle threshold sensitivity — {system} (N={num_envs}, max_iters={max_iters})")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run and plot oracle sensitivity to the per-step search budget."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    setup_devices()
    LOGGER.info("Loading config from %s", CONFIG_PATH)
    config = load_config_from_yaml(TrainConfig, CONFIG_PATH)
    if not config.env.immediate_dt_action:
        raise ValueError("oracle threshold sweep requires env.immediate_dt_action=True")

    tasks = sample_eval_tasks(config.env, NUM_ENVS, SEED)
    max_iters_values = np.arange(0, MAX_ITERS + 1, dtype=np.int32)
    result = run_max_iters_sweep(
        config.env,
        tasks["task_params"],
        tasks["episode_keys"],
        config.rollout_steps,
        max_iters_values,
    )

    output_dir = Path("outputs/analysis/oracle_max_iters_sweep")
    save_max_iters_sweep_csv(result, output_dir / "oracle_max_iters_sweep.csv")
    plot_max_iters_sweep(
        result,
        config.env.system,
        NUM_ENVS,
        MAX_ITERS,
        output_dir / "oracle_max_iters_sweep.png",
    )
    print(f"saved max_iters sweep outputs to {output_dir}")


if __name__ == "__main__":
    main()
