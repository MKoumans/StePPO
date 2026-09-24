"""Solve a few representative tasks per system with the reference, the oracle and
(optionally) the RL policy from Hugging Face, for plot_trajectory_examples.py.

Needs the JAX environment:
    uv run python research/baselines/deeponet/generate_trajectory_examples.py --system van_der_pol --hf_repo MKoumans/vdp
    uv run python research/baselines/deeponet/generate_trajectory_examples.py --system brusselator   # no --hf_repo: RL is skipped
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from overleaf_tables import load_oracle_testset

from steppo.configs.base_config import TrainConfig, apply_precision, load_config_from_yaml
from steppo.envs.ode import ODEEnv  # noqa: F401 — import order avoids a circular import
from steppo.envs.ode_env import ODEParams
from steppo.training.error_dist import _oracle_hill_climb
from steppo.training.pid_solve import solve_pid_batch

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..", "..")

_CONFIG_PATHS = {
    "scalar_decay": "configs/envs/ode/scalar_decay/scalar_decay_default.yaml",
    "van_der_pol": "configs/envs/ode/van_der_pol/van_der_pol_default.yaml",
    "brusselator": "configs/envs/ode/brusselator/brusselator_default.yaml",
}
_DATA_DIRS = {
    "scalar_decay": "data/scalar_decay/evaluation",
    "van_der_pol": "data/van_der_pol/evaluation",
    "brusselator": "data/brusselator/evaluation",
}


def pick_example_indices(task_params: np.ndarray, n: int) -> np.ndarray:
    """Pick n tasks at log-spaced percentiles of the sorted task parameters."""
    order = np.argsort(task_params)
    percentiles = np.linspace(0.05, 0.95, n)
    positions = (percentiles * (len(order) - 1)).astype(int)
    return order[positions]


def solve_oracle_single_with_trajectory(env_config, mu: float, key, max_steps: int, max_iters: int):
    """Oracle hill-climb solve that records (t, y) at every step, +inf-padded past the end."""
    env = ODEEnv(env_config, max_steps)
    dt_log_gain = jnp.asarray(env_config.dt_log_gain, dtype=jnp.float32)
    dummy_key = jax.random.PRNGKey(0)
    key_y0, key_pulse = jax.random.split(key)
    pulse_phase = jax.random.uniform(key_pulse, shape=(), dtype=jnp.float32)
    params = ODEParams(
        lam=jnp.asarray(mu, dtype=jnp.float32), pulse_phase=pulse_phase, max_steps=max_steps
    )
    _, state0 = env.reset(key_y0, params)

    def scan_step(carry, i):
        state, done = carry
        hill_state, step_done = _oracle_hill_climb(
            env, dt_log_gain, dummy_key, state, params, max_iters
        )
        new_state = jax.tree.map(lambda n, o: jnp.where(done, o, n), hill_state, state)
        new_done = jnp.logical_or(done, step_done)
        out_t = jnp.where(done, jnp.inf, new_state.t)  # `done` refers to the state before this step
        return (new_state, new_done), (out_t, new_state.y)

    @jax.jit
    def run():
        (_, _), (ts, ys) = jax.lax.scan(
            scan_step, (state0, jnp.bool_(False)), jnp.arange(max_steps)
        )
        return ts, ys

    ts, ys = run()
    return np.asarray(ts), np.asarray(ys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=list(_CONFIG_PATHS))
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="Hugging Face repo ID of the trained RL policy; omit to skip RL",
    )
    parser.add_argument("--n_examples", type=int, default=4)
    parser.add_argument("--oracle_max_iters", type=int, default=6)
    args = parser.parse_args()

    config = load_config_from_yaml(TrainConfig, os.path.join(REPO_ROOT, _CONFIG_PATHS[args.system]))
    apply_precision(config)
    env_config = config.env
    max_steps = config.rollout_steps

    d = load_oracle_testset(os.path.join(REPO_ROOT, _DATA_DIRS[args.system]))
    task_params = d["task_params"].astype(np.float64)
    episode_keys = d["episode_keys"]
    ref_ts = d["ref_ts"]
    ref_ys = d["ref_ys"]

    idx = pick_example_indices(task_params, args.n_examples)
    print(f"[*] Selected examples ({args.system}): task_params={task_params[idx]}")

    ex_task_params = task_params[idx]
    ex_keys = episode_keys[idx]
    ex_ref_ts = ref_ts[idx]
    ex_ref_ys = ref_ys[idx]

    # --- Oracle: one example at a time (episode lengths differ) ---
    oracle_ts_list, oracle_ys_list = [], []
    for i, (mu, key) in enumerate(zip(ex_task_params, ex_keys)):
        print(f"[*] Oracle-trajectory {i + 1}/{len(idx)} (mu={mu:.4g}) ...")
        ts_i, ys_i = solve_oracle_single_with_trajectory(
            env_config,
            mu,
            jnp.asarray(key),
            max_steps,
            args.oracle_max_iters,
        )
        oracle_ts_list.append(ts_i)
        oracle_ys_list.append(ys_i)
    oracle_ts = np.stack(oracle_ts_list)
    oracle_ys = np.stack(oracle_ys_list)

    # --- RL (optional), from a Hugging Face artifact ---
    rl_ts = rl_ys = None
    if args.hf_repo:
        from steppo.models.huggingface.hub import download_model

        print(f"[*] Downloading RL policy: {args.hf_repo} ...")
        artifact = download_model(args.hf_repo)
        assert artifact.config.env.system == args.system, (
            f"HF-repo {args.hf_repo} is for system '{artifact.config.env.system}', "
            f"not '{args.system}'"
        )
        rl_out = solve_pid_batch(
            env_config,
            artifact.controller,
            jnp.asarray(ex_task_params, dtype=jnp.float32),
            jnp.asarray(ex_keys),
            max_steps,
            save_steps=True,
            save_ys=True,
        )
        rl_ts, rl_ys = np.asarray(rl_out["ts"]), np.asarray(rl_out["ys"])

    out_dir = os.path.join(HERE, "output-deeponet", args.system, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "trajectory_examples.npz")
    save_kwargs = dict(
        task_params=ex_task_params,
        ref_ts=ex_ref_ts,
        ref_ys=ex_ref_ys,
        oracle_ts=oracle_ts,
        oracle_ys=oracle_ys,
        atol=float(env_config.atol),
    )
    if rl_ts is not None:
        save_kwargs["rl_ts"] = rl_ts
        save_kwargs["rl_ys"] = rl_ys
    np.savez(out_path, **save_kwargs)
    print(f"[+] Saved: {out_path}")


if __name__ == "__main__":
    main()
