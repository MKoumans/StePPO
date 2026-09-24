"""Training-time evaluation: step efficiency and L2 error per split, from one dense
policy solve per split and the cached PID results."""

import numpy as np

from steppo.training.steps_vs_mu import collect_pid_errors, collect_policy_errors


def eval_metrics(
    env_config,
    policy_controller,
    datasets: dict,
    num_envs: int,
    max_steps: int,
) -> dict:
    """Efficiency relative to PID and L2 error for each split in `datasets`.

    Uses the first `num_envs` cached episodes per split; missing splits are
    skipped. Returns {"efficiency": {split: ...}, "l2_error": {split: ...}}.
    """
    efficiency: dict = {}
    l2_error: dict = {}

    for split, cached in datasets.items():
        if cached is None:
            continue

        task_params = np.asarray(cached["task_params"])[:num_envs]
        episode_keys = np.asarray(cached["episode_keys"])[:num_envs]
        # Must match task_params/episode_keys's [:num_envs] slice, or the
        # dense solve's per-row comparison against the reference shape-mismatches.
        ref_final_y = np.asarray(cached["ref_final_y"])[:num_envs]
        ref_ts = np.asarray(cached["ref_ts"])[:num_envs]
        ref_ys = np.asarray(cached["ref_ys"])[:num_envs]

        # One dense diffrax solve yields both step count and error aggregates.
        policy = collect_policy_errors(
            env_config,
            policy_controller,
            task_params,
            episode_keys,
            max_steps,
            ref_final_y,
            ref_ts,
            ref_ys,
            env_config.atol,
        )
        policy_steps = np.asarray(policy["steps"], dtype=np.float64)
        policy_err = np.asarray(policy["err"])
        policy_err_integrated = np.asarray(policy["err_integrated"])

        # PID side: read from cache (no live solve), sliced to match.
        pid = collect_pid_errors(cached, env_config.atol)
        pid_steps = np.asarray(cached["pid_steps"], dtype=np.float64)[:num_envs]
        pid_err = np.asarray(pid["err"])[:num_envs]
        pid_err_integrated = np.asarray(pid["err_integrated"])[:num_envs]

        improvement = (pid_steps - policy_steps) / np.maximum(pid_steps, 1.0)
        efficiency[split] = {
            "mean": float(np.mean(improvement)),
            "std": float(np.std(improvement)),
            "median": float(np.median(improvement)),
            "mean_pid_steps": float(np.mean(pid_steps)),
            "std_pid_steps": float(np.std(pid_steps)),
            "mean_policy_steps": float(np.mean(policy_steps)),
            "std_policy_steps": float(np.std(policy_steps)),
        }

        l2_error[split] = {
            "policy_err_mean": float(np.mean(policy_err)),
            "policy_err_std": float(np.std(policy_err)),
            "policy_err_integrated_mean": float(np.mean(policy_err_integrated)),
            "policy_err_integrated_std": float(np.std(policy_err_integrated)),
            "pid_err_mean": float(np.mean(pid_err)),
            "pid_err_std": float(np.std(pid_err)),
            "pid_err_integrated_mean": float(np.mean(pid_err_integrated)),
            "pid_err_integrated_std": float(np.std(pid_err_integrated)),
        }

    return {"efficiency": efficiency, "l2_error": l2_error}
