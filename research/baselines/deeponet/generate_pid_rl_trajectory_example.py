"""Solve one representative task per system with the reference, PID, PID with an
unlimited budget, and the RL policy from Hugging Face, for Fig. 2 (plotted by
plot_pid_rl_deeponet_trajectory.py). The task defaults to the median test-set
task parameter.

Needs the JAX environment:
    uv run python research/baselines/deeponet/generate_pid_rl_trajectory_example.py [--system SYSTEM ...]
"""

import argparse
import os

import jax.numpy as jnp
import numpy as np
from overleaf_tables import load_oracle_testset

from steppo.configs.base_config import TrainConfig, apply_precision, load_config_from_yaml
from steppo.envs.ode import ODEEnv  # noqa: F401 — import order avoids a circular import
from steppo.training.error_dist import PID_UNLIMITED_BUDGET_MULTIPLIER
from steppo.training.pid_solve import env_pid_controller, solve_pid_batch

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..", "..")

SYSTEMS = {
    "scalar_decay": dict(
        config="configs/envs/ode/scalar_decay/scalar_decay_default.yaml",
        data_dir="data/scalar_decay/evaluation",
        hf_repo="MKoumans/sd",
        hf_revision=None,
    ),
    "van_der_pol": dict(
        config="configs/envs/ode/van_der_pol/van_der_pol_default.yaml",
        data_dir="data/van_der_pol/evaluation",
        hf_repo="MKoumans/vdp",
        hf_revision=None,
    ),
    "brusselator": dict(
        config="configs/envs/ode/brusselator/brusselator_default.yaml",
        data_dir="data/brusselator/evaluation",
        hf_repo="MKoumans/bru",
        hf_revision=None,
    ),
}


def ensure_warp_cache(repo_id: str, revision: str | None):
    """Build the progress-warp cache for an artifact trained with progress_warp,
    since LearnedController.from_models only reads that cache.
    """
    from pathlib import Path

    import yaml
    from huggingface_hub import HfApi

    from steppo.configs.base_config import TrainConfig, load_config_from_dict
    from steppo.training.reward_norm import build_pid_warp_grid

    api = HfApi()
    download_kwargs = dict(repo_id=repo_id, repo_type="model")
    if revision is not None:
        download_kwargs["revision"] = revision
    artifact_dir = Path(api.snapshot_download(**download_kwargs))
    config_data = yaml.safe_load((artifact_dir / "config.yaml").read_bytes()) or {}
    config = load_config_from_dict(TrainConfig, config_data, strict=True)
    if not getattr(config.env, "progress_warp", False):
        return  # no warp grid needed for this artifact
    env = ODEEnv(config.env, config.rollout_steps)
    lo, hi = env._task_bounds()
    rn = config.training.reward_normalization
    build_pid_warp_grid(
        config.env,
        float(lo),
        float(hi),
        grid_points=rn.grid_points,
        num_repeats=rn.num_repeats,
        max_steps=rn.max_steps or config.rollout_steps,
        require_cache=False,
        verbose=1,
    )


def generate_one(system: str, spec: dict, task_param: float | None = None):
    print(f"\n=== {system} ===")
    config = load_config_from_yaml(TrainConfig, os.path.join(REPO_ROOT, spec["config"]))
    apply_precision(config)
    env_config = config.env
    max_steps = config.rollout_steps

    d = load_oracle_testset(os.path.join(REPO_ROOT, spec["data_dir"]))
    task_params = d["task_params"].astype(np.float64)
    episode_keys = d["episode_keys"]
    ref_ts = d["ref_ts"]
    ref_ys = d["ref_ys"]

    if task_param is None:
        # Default task: the median task parameter of the test set.
        order = np.argsort(task_params)
        idx = order[len(order) // 2]
    else:
        # Use the nearest test-set task (reference paths exist only for those).
        idx = int(np.argmin(np.abs(task_params - task_param)))
    mu = float(task_params[idx])
    key = jnp.asarray(episode_keys[idx])
    print(f"[*] Selected task index {idx}: task_param={mu:.4g}")

    mu_batch = jnp.asarray([mu], dtype=jnp.float32)
    key_batch = key[None, :]

    # --- PID baseline ---
    pid_sc = env_pid_controller(env_config)
    pid_out = solve_pid_batch(
        env_config,
        pid_sc,
        mu_batch,
        key_batch,
        max_steps,
        save_steps=True,
        save_ys=True,
    )
    pid_ts, pid_ys = pid_out["ts"][0], pid_out["ys"][0]
    print(
        f"[*] PID baseline solved: accepted={pid_out['accepted'][0]} rejected={pid_out['rejected'][0]}"
    )

    # --- PID with budget_multiplier x the step budget ---
    pid_unlimited_out = solve_pid_batch(
        env_config,
        pid_sc,
        mu_batch,
        key_batch,
        max_steps * PID_UNLIMITED_BUDGET_MULTIPLIER,
        save_steps=True,
        save_ys=True,
    )
    pid_unlimited_ts, pid_unlimited_ys = pid_unlimited_out["ts"][0], pid_unlimited_out["ys"][0]
    print(
        f"[*] PID unlimited solved: accepted={pid_unlimited_out['accepted'][0]} "
        f"rejected={pid_unlimited_out['rejected'][0]}"
    )

    # --- RL: LearnedController from the Hugging Face artifact ---
    from steppo.models.huggingface.hub import download_model

    ensure_warp_cache(spec["hf_repo"], spec["hf_revision"])
    print(
        f"[*] Downloading RL policy: {spec['hf_repo']}"
        f"{'@' + spec['hf_revision'] if spec['hf_revision'] else ''} ..."
    )
    download_kwargs = {"revision": spec["hf_revision"]} if spec["hf_revision"] else {}
    artifact = download_model(spec["hf_repo"], **download_kwargs)
    assert artifact.config.env.system == system, (
        f"HF repo {spec['hf_repo']} is for system '{artifact.config.env.system}', not '{system}'"
    )
    rl_out = solve_pid_batch(
        env_config,
        artifact.controller,
        mu_batch,
        key_batch,
        max_steps,
        save_steps=True,
        save_ys=True,
    )
    rl_ts, rl_ys = rl_out["ts"][0], rl_out["ys"][0]
    print(f"[*] RL solved: accepted={rl_out['accepted'][0]} rejected={rl_out['rejected'][0]}")

    out_path = os.path.join(
        HERE, "output-deeponet", system, "results", "pid_rl_vs_reference_trajectory.npz"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        task_param=mu,
        ref_ts=ref_ts[idx],
        ref_ys=ref_ys[idx],
        pid_ts=pid_ts,
        pid_ys=pid_ys,
        pid_unlimited_ts=pid_unlimited_ts,
        pid_unlimited_ys=pid_unlimited_ys,
        rl_ts=rl_ts,
        rl_ys=rl_ys,
        atol=float(env_config.atol),
        rtol=float(env_config.rtol),
    )
    print(f"[+] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system", action="append", choices=list(SYSTEMS), help="repeatable; default = all systems"
    )
    parser.add_argument(
        "--task-param",
        type=float,
        default=None,
        help="task parameter (mu/lambda/B) to use instead of the test set's "
        "median; snapped to the nearest task actually in the held-out "
        "test set. Only valid with exactly one --system.",
    )
    args = parser.parse_args()
    systems = args.system or list(SYSTEMS)
    if args.task_param is not None and len(systems) != 1:
        parser.error("--task-param requires exactly one --system")
    for system in systems:
        generate_one(system, SYSTEMS[system], task_param=args.task_param)


if __name__ == "__main__":
    main()
