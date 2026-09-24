"""Cache-backed steps-vs-μ data production for PID, policy, and Oracle."""

import numpy as np

from steppo.training.error_dist import (
    DEFAULT_PID_CACHE_DIR,
    PID_EVAL_NUM_ENVS,
    PID_EVAL_REF_TOL_FACTOR,
    PID_EVAL_SEED,
    PID_UNLIMITED_BUDGET_MULTIPLIER,
    load_cached_pid_batch,
    log10_rel_error,
)
from steppo.training.pid_solve import solve_pid_batch
from steppo.training.trajectory_compare import (
    final_state_from_trajectory,
    log10_time_integrated_error,
    time_average,
    to_log10_clipped,
)


def _dense_trajectory_errors(
    env_config,
    controller,
    task_params,
    episode_keys,
    max_steps,
    ref_final_y,
    ref_ts,
    ref_ys,
    atol,
) -> dict:
    """Step count (accepted + rejected) and last-step / time-integrated error of one
    dense controller solve against the cached reference."""
    dense = solve_pid_batch(
        env_config,
        controller,
        task_params,
        episode_keys,
        max_steps,
        save_steps=True,
        save_ys=True,
    )
    final_y = final_state_from_trajectory(dense["ts"], dense["ys"])
    return {
        "mu": np.asarray(task_params),
        "err": log10_rel_error(final_y, ref_final_y, atol),
        "err_integrated": log10_time_integrated_error(
            ref_ts, ref_ys, dense["ts"], dense["ys"], atol
        ),
        "steps": np.asarray(dense["accepted"]) + np.asarray(dense["rejected"]),
        "completed": np.asarray(dense["completed"]),
    }


def _graded_completion(cached: dict, own_completed) -> np.ndarray:
    """An error is trustworthy only where the graded solve *and* the
    reference it is compared against both reached t_end — a reference that
    stopped early makes every error measured against it meaningless, and
    `throw=False` means neither failure is otherwise visible."""
    return np.asarray(own_completed) & np.asarray(cached["ref_completed"])


def collect_pid_errors(cached: dict, atol: float) -> dict:
    """PID last-step and time-integrated error, read from the cached dataset (no solve)."""
    return {
        "mu": np.asarray(cached["task_params"]),
        "err": log10_rel_error(cached["pid_final_y"], cached["ref_final_y"], atol),
        "err_integrated": time_average(cached["pid_ts"], to_log10_clipped(cached["pid_local_err"])),
        "completed": _graded_completion(cached, cached["pid_completed"]),
    }


def collect_policy_errors(
    env_config,
    policy_controller,
    task_params,
    episode_keys,
    max_steps,
    ref_final_y,
    ref_ts,
    ref_ys,
    atol,
    ref_completed=None,
) -> dict:
    """Learned-controller last-step and time-integrated error against the
    cached tight-tolerance reference trajectory. `ref_completed` narrows the
    returned `completed` mask to episodes whose reference is real — pass it
    whenever it is available (see `_graded_completion`)."""
    errors = _dense_trajectory_errors(
        env_config,
        policy_controller,
        task_params,
        episode_keys,
        max_steps,
        ref_final_y,
        ref_ts,
        ref_ys,
        atol,
    )
    if ref_completed is not None:
        errors["completed"] = errors["completed"] & np.asarray(ref_completed)
    return errors


def collect_policy_steps(
    env_config,
    policy_controller,
    task_params,
    episode_keys,
    max_steps: int,
) -> dict:
    """Solve the cached episodes with a learned controller (via pid_solve.solve_pid_batch)."""
    out = solve_pid_batch(env_config, policy_controller, task_params, episode_keys, max_steps)
    return {
        "mu": np.asarray(task_params),
        "steps": out["steps"],
        "completed": np.asarray(out["completed"]),
    }


def build_steps_vs_mu_data(
    env_config,
    *,
    max_steps: int,
    bins,
    split: str = "test",
    policy_controller=None,
    num_envs: int = PID_EVAL_NUM_ENVS,
    seed: int = PID_EVAL_SEED,
    ref_tol_factor: float = PID_EVAL_REF_TOL_FACTOR,
    timing_n: int = 128,
    timing_repeats: int = 8,
    timing_unlimited_repeats: int | None = None,
    timing_unlimited_max_steps: int = 8192,
    oracle_max_iters: int = 6,
    budget_multiplier: int = PID_UNLIMITED_BUDGET_MULTIPLIER,
    cache_dir: str = DEFAULT_PID_CACHE_DIR,
    with_error: bool = False,
) -> dict:
    """Load the cached evaluation dataset and optionally solve its episodes with the policy.

    `with_error=True` adds `pid_error` (from the cache), `policy_error` (one
    dense solve) and `oracle_error` (from the cache; time-integrated only for
    datasets that store the Oracle trajectory).
    """
    cached = load_cached_pid_batch(
        env_config,
        max_steps=max_steps,
        num_envs=num_envs,
        seed=seed,
        bins=bins,
        split=split,
        ref_tol_factor=ref_tol_factor,
        timing_n=timing_n,
        timing_repeats=timing_repeats,
        timing_unlimited_repeats=timing_unlimited_repeats,
        timing_unlimited_max_steps=timing_unlimited_max_steps,
        oracle_max_iters=oracle_max_iters,
        budget_multiplier=budget_multiplier,
        cache_dir=cache_dir,
    )
    task_params = np.asarray(cached["task_params"])
    pid_unlimited = None
    if cached.get("pid_steps_unlimited") is not None:
        pid_unlimited = {
            "mu": task_params,
            "steps": np.asarray(cached["pid_steps_unlimited"]),
            "completed": np.asarray(cached["pid_completed_unlimited"]),
        }
    result = {
        "cache": cached,
        "pid": {
            "mu": task_params,
            "steps": np.asarray(cached["pid_steps"]),
            "completed": np.asarray(cached["pid_completed"]),
        },
        "pid_unlimited": pid_unlimited,
        "policy": None,
        "oracle": None,
        "pid_error": None,
        "policy_error": None,
        "oracle_error": None,
    }
    if with_error:
        result["pid_error"] = collect_pid_errors(cached, env_config.atol)
        if cached.get("oracle_final_y") is not None:
            result["oracle_error"] = {
                "mu": task_params,
                "err": log10_rel_error(
                    cached["oracle_final_y"], cached["ref_final_y"], env_config.atol
                ),
                "completed": _graded_completion(cached, cached["oracle_completed"]),
            }
            if cached.get("oracle_local_err") is not None:
                result["oracle_error"]["err_integrated"] = time_average(
                    cached["oracle_ts"], to_log10_clipped(cached["oracle_local_err"])
                )
    if policy_controller is not None:
        result["policy"] = collect_policy_steps(
            env_config,
            policy_controller,
            task_params,
            cached["episode_keys"],
            max_steps,
        )
        if with_error:
            result["policy_error"] = collect_policy_errors(
                env_config,
                policy_controller,
                task_params,
                cached["episode_keys"],
                max_steps,
                cached["ref_final_y"],
                cached["ref_ts"],
                cached["ref_ys"],
                env_config.atol,
                ref_completed=cached["ref_completed"],
            )
    if cached.get("oracle_steps") is not None:
        result["oracle"] = {
            "mu": task_params,
            "steps": np.asarray(cached["oracle_steps"]),
            "completed": np.asarray(cached["oracle_completed"]),
        }
    return result


def build_steps_vs_mu_data_by_split(
    env_config,
    *,
    max_steps: int,
    split_bins: dict,
    policy_controller=None,
    num_envs: int = PID_EVAL_NUM_ENVS,
    seed: int = PID_EVAL_SEED,
    oracle_max_iters: int = 6,
    budget_multiplier: int = PID_UNLIMITED_BUDGET_MULTIPLIER,
    with_error: bool = False,
) -> dict:
    """Build cache-backed PID/RL/oracle series for every configured split."""
    result = {}
    for split in ("train", "val", "test"):
        bins = split_bins.get(split)
        if not bins:
            continue
        result[split] = build_steps_vs_mu_data(
            env_config,
            max_steps=max_steps,
            bins=list(bins),
            split=split,
            policy_controller=policy_controller,
            num_envs=num_envs,
            seed=seed,
            oracle_max_iters=oracle_max_iters,
            budget_multiplier=budget_multiplier,
            with_error=with_error,
        )
    return result
