"""Diffrax-only Bayesian optimisation of PID step-size gains.

The frozen split supplies task keys and tight reference states. Every candidate
is solved afresh in Diffrax; no environment rollouts or stored trajectories are
used by this module.

``Kd`` defaults to 0 (pure PI control, matching the paper baseline). Pass
``kd_bounds`` to also search the derivative gain — diffrax's PIDController
already accepts ``dcoeff``, so this only widens the search space, it doesn't
require any solver changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np

from steppo.configs.base_config import TrainConfig, apply_precision, load_config_from_yaml


def load_cached_pid_batch(*args, **kwargs):
    from steppo.training.error_dist import load_cached_pid_batch as loader

    return loader(*args, **kwargs)


def solve_pid_batch(*args, **kwargs):
    from steppo.training.pid_solve import solve_pid_batch as solver

    return solver(*args, **kwargs)


def log10_rel_error(y, y_ref, atol):
    diff = np.linalg.norm(np.asarray(y) - np.asarray(y_ref), axis=-1)
    denom = np.linalg.norm(np.asarray(y_ref), axis=-1) + atol
    relative = diff / denom
    relative = np.where(np.isfinite(relative), relative, 10.0**3)
    return np.clip(np.log10(relative + 10.0**-16), -16.0, 3.0)


LogFn = Callable[[str], None]


def _resolve_logger(log: bool, log_fn: LogFn | None) -> LogFn | None:
    if log_fn is not None:
        return log_fn
    if log:
        return lambda message: print(message, flush=True)
    return None


def _emit(log_fn: LogFn | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


@dataclass(frozen=True)
class PIGains:
    kp: float
    ki: float
    kd: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.kp) or not np.isfinite(self.ki) or not np.isfinite(self.kd):
            raise ValueError("PID gains must be finite")
        if self.kp < 0.0 or self.ki < 0.0 or self.kd < 0.0:
            raise ValueError("PID gains must be non-negative")


@dataclass(frozen=True)
class PIData:
    task_params: np.ndarray
    episode_keys: np.ndarray
    ref_final_y: np.ndarray
    fingerprint: str
    split: str

    def __post_init__(self):
        if len({len(self.task_params), len(self.episode_keys), len(self.ref_final_y)}) != 1:
            raise ValueError(
                "task_params, episode_keys, and ref_final_y must have the same number of episodes"
            )
        if self.task_params.ndim != 1:
            raise ValueError("task_params must be a one-dimensional episode array")
        if self.episode_keys.ndim != 2:
            raise ValueError("episode_keys must be a two-dimensional key array")


@dataclass(frozen=True)
class PIConstraints:
    max_p95_error: float
    min_completion_rate: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.max_p95_error):
            raise ValueError("max_p95_error must be finite")
        if not 0.0 <= self.min_completion_rate <= 1.0:
            raise ValueError("min_completion_rate must be in [0, 1]")


@dataclass(frozen=True)
class PIEvaluation:
    gains: PIGains
    steps: np.ndarray
    errors: np.ndarray
    t_reached: np.ndarray
    completed: np.ndarray
    mean_steps: float
    median_steps: float
    p95_error: float
    completion_rate: float
    feasible: bool


@dataclass(frozen=True)
class PISearch:
    evaluations: tuple[PIEvaluation, ...]
    baseline: PIEvaluation
    constraints: PIConstraints
    best: PIEvaluation


def load_split(
    env_config,
    split: str,
    *,
    num_envs: int = 1024,
    seed: int = 0,
    max_steps: int | None = None,
    cache_dir: str | Path = "data",
) -> PIData:
    """Load one existing frozen PID evaluation split without sampling."""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be one of: train, val, test")
    bins = list(getattr(env_config, f"{split}_bins"))
    if not bins:
        raise ValueError(f"config.env.{split}_bins is empty")
    max_steps = max_steps or getattr(env_config, "rollout_steps", None)
    if max_steps is None:
        raise ValueError("max_steps must be provided when env_config has no rollout_steps")
    batch = load_cached_pid_batch(
        env_config=env_config,
        max_steps=max_steps,
        bins=bins,
        split=split,
        num_envs=num_envs,
        seed=seed,
        cache_dir=str(cache_dir),
    )
    return PIData(
        task_params=np.asarray(batch["task_params"]),
        episode_keys=np.asarray(batch["episode_keys"]),
        ref_final_y=np.asarray(batch["ref_final_y"]),
        fingerprint=str(batch["fingerprint"]),
        split=split,
    )


def _config_payload(config) -> dict:
    return dataclasses.asdict(config) if dataclasses.is_dataclass(config) else vars(config)


def _cache_path(cache_dir, env_config, data: PIData, gains: PIGains, max_steps: int) -> Path:
    payload = {
        "kind": "pi_candidate",
        "data": data.fingerprint,
        "env": _config_payload(env_config),
        "kp": gains.kp,
        "ki": gains.ki,
        "kd": gains.kd,
        "max_steps": int(max_steps),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]
    return Path(cache_dir) / getattr(env_config, "system", "ode") / f"pid_{digest}.npz"


def _make_evaluation(gains, data, env_config, raw, constraints=None) -> PIEvaluation:
    final_y = np.asarray(raw["final_y"])
    steps = np.asarray(raw["steps"], dtype=float)
    t_reached = np.asarray(raw["t_reached"], dtype=float)
    errors = np.asarray(log10_rel_error(final_y, data.ref_final_y, env_config.atol), dtype=float)
    tolerance = 1e-6 * max(1.0, float(env_config.t_end))
    completed = t_reached >= float(env_config.t_end) - tolerance
    completion_rate = float(np.mean(completed))
    p95_error = float(np.percentile(errors, 95))
    feasible = constraints is None or (
        completion_rate >= constraints.min_completion_rate
        and p95_error <= constraints.max_p95_error
    )
    return PIEvaluation(
        gains=gains,
        steps=steps,
        errors=errors,
        t_reached=t_reached,
        completed=completed,
        mean_steps=float(np.mean(steps)),
        median_steps=float(np.median(steps)),
        p95_error=p95_error,
        completion_rate=completion_rate,
        feasible=feasible,
    )


def evaluate_pi(
    env_config,
    dataset: PIData,
    gains: PIGains,
    *,
    max_steps: int,
    constraints: PIConstraints | None = None,
    cache_dir: str | Path | None = None,
    solve_fn: Callable | None = None,
) -> PIEvaluation:
    """Evaluate one PID controller on the dataset's fixed episodes."""
    path = _cache_path(cache_dir, env_config, dataset, gains, max_steps) if cache_dir else None
    if path is not None and path.is_file():
        with np.load(path) as raw:
            return _make_evaluation(gains, dataset, env_config, raw, constraints)
    from steppo.training.pid_solve import env_pid_controller

    controller = env_pid_controller(env_config, pcoeff=gains.kp, icoeff=gains.ki, dcoeff=gains.kd)
    raw = (solve_fn or solve_pid_batch)(
        env_config,
        controller,
        dataset.task_params,
        dataset.episode_keys,
        max_steps,
        save_steps=False,
    )
    missing = {"final_y", "steps"}.difference(raw)
    if missing:
        raise ValueError(f"candidate solver result is missing: {sorted(missing)}")
    if "t_reached" not in raw:
        raw = dict(
            raw,
            t_reached=np.where(
                np.asarray(raw["steps"]) < max_steps,
                float(env_config.t_end),
                0.0,
            ),
        )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, final_y=raw["final_y"], steps=raw["steps"], t_reached=raw["t_reached"]
        )
    return _make_evaluation(gains, dataset, env_config, raw, constraints)


def _evaluate_with_logging(
    label, env_config, dataset, gains, *, max_steps, constraints, cache_dir, solve_fn, log_fn
):
    _emit(
        log_fn, f"[PID-BO] {label} start: Kp={gains.kp:.6g}, Ki={gains.ki:.6g}, Kd={gains.kd:.6g}"
    )
    started = time.perf_counter()
    try:
        result = evaluate_pi(
            env_config,
            dataset,
            gains,
            max_steps=max_steps,
            constraints=constraints,
            cache_dir=cache_dir,
            solve_fn=solve_fn,
        )
    except Exception as error:
        _emit(
            log_fn, f"[PID-BO] {label} failed after {time.perf_counter() - started:.2f}s: {error}"
        )
        raise
    _emit(
        log_fn,
        f"[PID-BO] {label} done: mean_steps={result.mean_steps:.3f}, "
        f"p95_error={result.p95_error:.5g}, completion={result.completion_rate:.3f}, "
        f"feasible={result.feasible}, elapsed={time.perf_counter() - started:.2f}s",
    )
    return result


def _normalise(points, bounds):
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)
    return (np.asarray(points, dtype=float) - lo) / (hi - lo), lo, hi


def _gains_to_point(gains: PIGains, dim: int) -> tuple[float, ...]:
    return (gains.kp, gains.ki, gains.kd)[:dim]


def _next_gain(observed: list[PIGains], scores: np.ndarray, bounds, rng, pool_size=4096) -> PIGains:
    dim = len(bounds)
    points = np.asarray([_gains_to_point(g, dim) for g in observed], dtype=float)
    x, lo, hi = _normalise(points, bounds)
    y = np.asarray(scores, dtype=float)
    y = (y - float(np.mean(y))) / max(float(np.std(y)), 1e-6)
    distances = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    kernel = np.exp(-0.5 * distances / 0.15**2) + 1e-6 * np.eye(len(x))
    chol = np.linalg.cholesky(kernel)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    pool = rng.uniform(0.0, 1.0, size=(pool_size, dim))
    cross = np.exp(-0.5 * np.sum((pool[:, None, :] - x[None, :, :]) ** 2, axis=-1) / 0.15**2)
    mean = cross @ alpha
    variance = np.maximum(1.0 - np.sum(np.linalg.solve(chol, cross.T) ** 2, axis=0), 0.0)
    for i in np.argsort(mean - 1.5 * np.sqrt(variance)):
        if np.min(np.sum((x - pool[i]) ** 2, axis=1)) > 1e-8:
            candidate = lo + pool[i] * (hi - lo)
            values = list(candidate) + [0.0] * (3 - dim)
            return PIGains(*values)
    raise RuntimeError("could not propose a non-duplicate gain")


def _score(evaluation: PIEvaluation, constraints: PIConstraints, baseline_steps: float) -> float:
    if evaluation.feasible:
        return evaluation.mean_steps
    completion_violation = max(0.0, constraints.min_completion_rate - evaluation.completion_rate)
    error_violation = max(0.0, evaluation.p95_error - constraints.max_p95_error)
    return baseline_steps + 100.0 * max(1.0, baseline_steps) * (
        1.0 + completion_violation + error_violation
    )


def optimise_pi(
    env_config,
    dataset: PIData,
    *,
    max_steps: int,
    kp_bounds: tuple[float, float] = (0.0, 4.0),
    ki_bounds: tuple[float, float] = (0.01, 2.0),
    kd_bounds: tuple[float, float] | None = None,
    initial_gains: Iterable[PIGains] = (),
    iterations: int = 20,
    seed: int = 0,
    cache_dir: str | Path | None = None,
    solve_fn: Callable | None = None,
    log: bool = False,
    log_fn: LogFn | None = None,
    p95_error_slack: float = 0.10,
) -> PISearch:
    """Run deterministic, low-dimensional GP Bayesian optimisation.

    Searches ``(Kp, Ki)`` by default; pass ``kd_bounds`` to also search ``Kd``.

    ``p95_error_slack`` is a fractional allowance in raw relative-error
    space. The stored error is log10 relative error, so the resulting limit is
    ``baseline.p95_error + log10(1 + p95_error_slack)``.

    ``log=True`` prints flushed progress lines after every candidate. Supply
    ``log_fn`` to route the same lines to a notebook, logger, or test sink.
    """
    bounds = (kp_bounds, ki_bounds) if kd_bounds is None else (kp_bounds, ki_bounds, kd_bounds)
    if any(hi <= lo for lo, hi in bounds):
        raise ValueError("each gain bound must be increasing")
    if any(lo < 0.0 for lo, _ in bounds):
        raise ValueError("PID gain bounds must be non-negative")
    if not np.isfinite(p95_error_slack) or p95_error_slack < 0.0:
        raise ValueError("p95_error_slack must be finite and non-negative")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    logger = _resolve_logger(log, log_fn)
    rng = np.random.default_rng(seed)
    baseline = _evaluate_with_logging(
        "baseline",
        env_config,
        dataset,
        PIGains(0.0, 1.0),
        max_steps=max_steps,
        constraints=None,
        cache_dir=cache_dir,
        solve_fn=solve_fn,
        log_fn=logger,
    )
    max_p95_error = baseline.p95_error + np.log10(1.0 + p95_error_slack)
    constraints = PIConstraints(max_p95_error, baseline.completion_rate)
    _emit(
        logger,
        f"[PID-BO] constraints: max_p95_error={constraints.max_p95_error:.5g} "
        f"(baseline={baseline.p95_error:.5g}, raw_slack={p95_error_slack:.1%}), "
        f"min_completion={constraints.min_completion_rate:.3f}",
    )
    bounds_msg = f"[PID-BO] bounds: Kp=[{kp_bounds[0]:.6g}, {kp_bounds[1]:.6g}], Ki=[{ki_bounds[0]:.6g}, {ki_bounds[1]:.6g}]"
    if kd_bounds is not None:
        bounds_msg += f", Kd=[{kd_bounds[0]:.6g}, {kd_bounds[1]:.6g}]"
    _emit(logger, bounds_msg)
    candidates = [baseline.gains, *tuple(initial_gains)]
    unique_candidates = []
    seen = set()
    for gains in candidates:
        if gains not in seen:
            seen.add(gains)
            unique_candidates.append(gains)
    target_count = len(unique_candidates) + iterations
    evaluations = []
    for gains in unique_candidates:
        if gains == baseline.gains:
            result = baseline
        else:
            result = _evaluate_with_logging(
                f"candidate {len(evaluations) + 1}/{target_count}",
                env_config,
                dataset,
                gains,
                max_steps=max_steps,
                constraints=constraints,
                cache_dir=cache_dir,
                solve_fn=solve_fn,
                log_fn=logger,
            )
        evaluations.append(result)
    while len(evaluations) < target_count:
        scores = np.asarray([_score(e, constraints, baseline.mean_steps) for e in evaluations])
        gains = _next_gain([e.gains for e in evaluations], scores, bounds, rng)
        if gains in seen:
            continue
        seen.add(gains)
        evaluations.append(
            _evaluate_with_logging(
                f"candidate {len(evaluations) + 1}/{target_count}",
                env_config,
                dataset,
                gains,
                max_steps=max_steps,
                constraints=constraints,
                cache_dir=cache_dir,
                solve_fn=solve_fn,
                log_fn=logger,
            )
        )
    feasible = [e for e in evaluations if e.feasible]
    best = min(feasible or evaluations, key=lambda e: (e.mean_steps, e.p95_error))
    _emit(
        logger,
        f"[PID-BO] train optimum: Kp={best.gains.kp:.6g}, Ki={best.gains.ki:.6g}, Kd={best.gains.kd:.6g}, mean_steps={best.mean_steps:.3f}",
    )
    return PISearch(tuple(evaluations), baseline, constraints, best)


def plot_search(
    search: PISearch, output_path: str | Path, selected: PIEvaluation | None = None
) -> Path:
    """Write a triangulated 3D (Kp, Ki, steps) response surface and highlight the optimum.

    Always projects onto the (Kp, Ki) plane, even when Kd was also searched —
    a 4D surface has no direct plot, and Kp/Ki dominate step count in practice.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = selected or search.best
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    feasible = [e for e in search.evaluations if e.feasible]
    if len(feasible) >= 3:
        try:
            surface = ax.plot_trisurf(
                [e.gains.kp for e in feasible],
                [e.gains.ki for e in feasible],
                [e.mean_steps for e in feasible],
                cmap="viridis",
                alpha=0.72,
                linewidth=0.25,
                edgecolor="none",
            )
            fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.10, label="mean solver steps")
        except (RuntimeError, ValueError):
            # A small or nearly collinear BO sample cannot form a triangulation.
            pass
    if feasible:
        ax.scatter(
            [e.gains.kp for e in feasible],
            [e.gains.ki for e in feasible],
            [e.mean_steps for e in feasible],
            color="tab:blue",
            s=24,
            label="feasible samples",
        )
    infeasible = [e for e in search.evaluations if not e.feasible]
    if infeasible:
        ax.scatter(
            [e.gains.kp for e in infeasible],
            [e.gains.ki for e in infeasible],
            [e.mean_steps for e in infeasible],
            color="tab:red",
            s=28,
            label="infeasible samples",
        )
    ax.scatter(
        [search.baseline.gains.kp],
        [search.baseline.gains.ki],
        [search.baseline.mean_steps],
        color="black",
        marker="D",
        s=55,
        label="baseline (0, 1)",
    )
    ax.scatter(
        [selected.gains.kp],
        [selected.gains.ki],
        [selected.mean_steps],
        color="gold",
        edgecolor="black",
        marker="*",
        s=180,
        label="selected optimum",
    )
    ax.text(
        selected.gains.kp,
        selected.gains.ki,
        selected.mean_steps,
        f"  Kp={selected.gains.kp:.4g}, Ki={selected.gains.ki:.4g}, Kd={selected.gains.kd:.4g}\n  steps={selected.mean_steps:.2f}",
    )
    ax.set_xlabel("Kp")
    ax.set_ylabel("Ki")
    ax.set_zlabel("mean solver steps")
    ax.set_title("Bayesian optimisation of Diffrax PID gains")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def export_xyz(search: PISearch, output_path: str | Path) -> Path:
    """Export evaluated ``Kp, Ki, Kd, mean_steps`` rows for external plotting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(
        [[e.gains.kp, e.gains.ki, e.gains.kd, e.mean_steps] for e in search.evaluations],
        dtype=float,
    )
    np.savetxt(output_path, rows, delimiter=",", header="Kp,Ki,Kd,mean_steps", comments="")
    return output_path


def _best_on_split(search, env_config, data, max_steps, cache_dir, log_fn=None):
    evaluations = [
        _evaluate_with_logging(
            f"validation candidate {index}/{len(search.evaluations)}",
            env_config,
            data,
            candidate.gains,
            max_steps=max_steps,
            constraints=search.constraints,
            cache_dir=cache_dir,
            solve_fn=None,
            log_fn=log_fn,
        )
        for index, candidate in enumerate(search.evaluations, start=1)
    ]
    feasible = [e for e in evaluations if e.feasible]
    selected = min(feasible or evaluations, key=lambda e: (e.mean_steps, e.p95_error))
    _emit(
        log_fn,
        f"[PID-BO] validation optimum: Kp={selected.gains.kp:.6g}, Ki={selected.gains.ki:.6g}, Kd={selected.gains.kd:.6g}, mean_steps={selected.mean_steps:.3f}",
    )
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-max-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--kp-bounds", nargs=2, type=float, default=(0.0, 4.0), metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--ki-bounds", nargs=2, type=float, default=(0.01, 2.0), metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--kd-bounds",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="also search the derivative gain over this range (default: Kd fixed at 0)",
    )
    parser.add_argument(
        "--p95-error-slack",
        type=float,
        default=0.10,
        help="fractional raw p95-error allowance over baseline",
    )
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output", default="outputs/pid_bo/pid_response_3d.png")
    parser.add_argument(
        "--xyz-output", default=None, help="CSV path for evaluated Kp, Ki, Kd, mean_steps rows"
    )
    parser.add_argument("--log", action="store_true", help="print flushed progress for every solve")
    args = parser.parse_args(argv)
    logger = _resolve_logger(args.log, None)
    config = load_config_from_yaml(TrainConfig, args.config)
    apply_precision(config)
    dataset_max_steps = args.dataset_max_steps or config.rollout_steps
    solve_max_steps = args.max_steps or dataset_max_steps * 5
    train = load_split(
        config.env,
        "train",
        num_envs=args.num_envs,
        seed=args.seed,
        max_steps=dataset_max_steps,
        cache_dir=args.cache_dir,
    )
    _emit(logger, f"[PID-BO] loaded train split: {len(train.task_params)} episodes")
    search = optimise_pi(
        config.env,
        train,
        max_steps=solve_max_steps,
        iterations=args.iterations,
        seed=args.seed,
        kp_bounds=tuple(args.kp_bounds),
        ki_bounds=tuple(args.ki_bounds),
        kd_bounds=tuple(args.kd_bounds) if args.kd_bounds else None,
        cache_dir=Path(args.cache_dir) / "pid_candidates",
        p95_error_slack=args.p95_error_slack,
        log_fn=logger,
    )
    val = load_split(
        config.env,
        "val",
        num_envs=args.num_envs,
        seed=args.seed,
        max_steps=dataset_max_steps,
        cache_dir=args.cache_dir,
    )
    _emit(logger, f"[PID-BO] loaded validation split: {len(val.task_params)} episodes")
    selected = _best_on_split(
        search,
        config.env,
        val,
        solve_max_steps,
        Path(args.cache_dir) / "pid_candidates",
        log_fn=logger,
    )
    test = load_split(
        config.env,
        "test",
        num_envs=args.num_envs,
        seed=args.seed,
        max_steps=dataset_max_steps,
        cache_dir=args.cache_dir,
    )
    _emit(logger, f"[PID-BO] loaded test split: {len(test.task_params)} episodes")
    test_result = _evaluate_with_logging(
        "test selected",
        config.env,
        test,
        selected.gains,
        max_steps=solve_max_steps,
        constraints=search.constraints,
        cache_dir=Path(args.cache_dir) / "pid_candidates",
        solve_fn=None,
        log_fn=logger,
    )
    plot_search(search, args.output, selected=selected)
    xyz_output = args.xyz_output or str(Path(args.output).with_suffix(".xyz.csv"))
    export_xyz(search, xyz_output)
    _emit(logger, f"[PID-BO] plot written: {args.output}")
    _emit(logger, f"[PID-BO] XYZ written: {xyz_output}")
    print(
        json.dumps(
            {
                "kp": selected.gains.kp,
                "ki": selected.gains.ki,
                "kd": selected.gains.kd,
                "val_mean_steps": selected.mean_steps,
                "test_mean_steps": test_result.mean_steps,
                "test_p95_error": test_result.p95_error,
                "test_completion_rate": test_result.completion_rate,
                "plot": str(args.output),
                "xyz": str(xyz_output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
