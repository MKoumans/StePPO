"""Oracle warmstart dataset: offline generation and cache.

The oracle search is too expensive to run during training, so the dataset is
generated explicitly by generate_pid_training_dataset.py and training only
reads it. The dataset size is a generation option and not part of the cache key.
"""

import dataclasses
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from steppo.envs.ode_env import ODEEnv
from steppo.training.rollout import _broadcast_params, _env_step_and_reset
from steppo.utils.task_params import task_bounds

DEFAULT_ORACLE_CACHE_DIR = "data"
DEFAULT_ORACLE_DATASET_EPISODES = (
    2048  # generate_pid_training_dataset.py's default; not part of the cache identity
)


@partial(
    jax.jit,
    static_argnames=("env", "num_envs", "rollout_steps", "episodes_per_trial", "oracle_max_iters"),
)
def _generate_oracle_batch_jit(
    env,
    env_params,
    num_envs: int,
    rollout_steps: int,
    episodes_per_trial: int,
    oracle_max_iters: int,
    rng_key,
) -> dict:
    """Roll out the oracle expert; returns (T, num_envs, ...) arrays without beliefs."""
    from steppo.training.error_dist import oracle_action

    params_batch = _broadcast_params(env_params, num_envs)
    dt_log_gain = env.dt_log_gain

    key_reset, key_run = jax.random.split(rng_key)
    reset_keys = jax.random.split(key_reset, num_envs)
    obs_batch, env_state_batch = jax.vmap(env.reset)(reset_keys, params_batch)

    def scan_step(carry, rng_key):
        env_state, obs, episode_count, trial_done = carry
        key_search, key_env, key_reset = jax.random.split(rng_key, 3)

        actions = jax.vmap(
            lambda s, p: oracle_action(env, dt_log_gain, key_search, s, p, oracle_max_iters)
        )(env_state, params_batch)

        (
            final_obs,
            final_env_state,
            encoder_obs,
            rewards,
            dones,
            keep_step,
            new_episode_count,
            new_trial_done,
            dones_ep,
        ) = _env_step_and_reset(
            env,
            env_state,
            obs,
            actions,
            params_batch,
            key_env,
            key_reset,
            num_envs,
            episode_count,
            trial_done,
            episodes_per_trial,
        )

        transition = (obs, actions, rewards, encoder_obs, dones, keep_step, dones_ep, trial_done)
        new_carry = (final_env_state, final_obs, new_episode_count, new_trial_done)
        return new_carry, transition

    init_carry = (
        env_state_batch,
        obs_batch,
        jnp.zeros(num_envs, dtype=jnp.int32),
        jnp.zeros(num_envs, dtype=jnp.bool_),
    )
    keys = jax.random.split(key_run, rollout_steps)
    _, transitions = jax.lax.scan(scan_step, init_carry, keys)

    (states, actions, rewards, next_states, dones, keep_steps, dones_ep, trial_done_before) = (
        transitions
    )

    if hasattr(env, "get_task_params"):
        task_params = jax.vmap(env.get_task_params)(params_batch)
    else:
        task_params = jnp.zeros((num_envs, 0), dtype=jnp.float32)

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "keep_steps": keep_steps,
        "dones_ep": dones_ep,
        "trial_done_before": trial_done_before,
        "task_params": task_params,
    }


def generate_oracle_batch(
    env,
    num_envs: int,
    rollout_steps: int,
    episodes_per_trial: int,
    oracle_max_iters: int,
    seed: int,
) -> dict:
    """Sample `num_envs` fresh tasks from env.sample_task (the same
    distribution live training draws from) and run the oracle rollout over
    them. Returns numpy arrays. Pure compute, no caching — see
    generate_and_cache_oracle_training_batch for the disk-cached entry point."""
    key = jax.random.PRNGKey(seed)
    key_task, key_run = jax.random.split(key)
    task_keys = jax.random.split(key_task, num_envs)
    env_params = jax.vmap(env.sample_task)(task_keys)
    batch = _generate_oracle_batch_jit(
        env,
        env_params,
        num_envs,
        rollout_steps,
        episodes_per_trial,
        oracle_max_iters,
        key_run,
    )
    return jax.tree.map(np.asarray, batch)


def oracle_training_fingerprint(
    env_config,
    rollout_steps: int,
    episodes_per_trial: int,
    seed: int,
    oracle_max_iters: int,
) -> str:
    """Hash of the full env config and oracle options (the dataset size is excluded)."""
    from steppo.training.pid_solve import fingerprint as _fingerprint

    oracle_task_bounds = list(task_bounds(env_config))

    payload = {
        "kind": "oracle_training_rollout",
        "env_config": dataclasses.asdict(env_config),
        "oracle_task_bounds": oracle_task_bounds,
        "rollout_steps": int(rollout_steps),
        "episodes_per_trial": int(episodes_per_trial),
        "seed": int(seed),
        "oracle_max_iters": int(oracle_max_iters),
    }
    return _fingerprint(payload)


def _cache_path(cache_dir: str, system: str, fp: str) -> str:
    from steppo.training.pid_solve import cache_dir as _cache_dir

    return os.path.join(_cache_dir("training", system, root=cache_dir), f"oracle_rollout_{fp}.npz")


def generate_and_cache_oracle_training_batch(
    env_config,
    num_envs: int,
    rollout_steps: int,
    episodes_per_trial: int = 0,
    seed: int = 0,
    oracle_max_iters: int = 20,
    cache_dir: str = DEFAULT_ORACLE_CACHE_DIR,
) -> dict:
    """Generate the oracle dataset and write `<cache_dir>/<system>/training/oracle_rollout_<fingerprint>.npz`."""
    from steppo.training.pid_solve import atomic_savez

    assert env_config.immediate_dt_action, (
        "oracle training-set generation requires env.immediate_dt_action=True "
        "— the oracle's search candidate must affect the step it's testing."
    )
    lo, hi = task_bounds(env_config)
    oracle_config = dataclasses.replace(
        env_config,
        train_bins=((lo, hi),),
    )
    env = ODEEnv(oracle_config, rollout_steps)
    batch = generate_oracle_batch(
        env,
        num_envs,
        rollout_steps,
        episodes_per_trial,
        oracle_max_iters,
        seed,
    )
    fp = oracle_training_fingerprint(
        env_config,
        rollout_steps,
        episodes_per_trial,
        seed,
        oracle_max_iters,
    )
    path = _cache_path(cache_dir, env_config.system, fp)
    atomic_savez(path, **batch)
    return batch


def load_oracle_training_batch(
    env_config,
    rollout_steps: int,
    episodes_per_trial: int = 0,
    seed: int = 0,
    oracle_max_iters: int = 20,
    cache_dir: str = DEFAULT_ORACLE_CACHE_DIR,
) -> dict:
    """Load the cached oracle dataset; raises FileNotFoundError with the generation command if absent."""
    fp = oracle_training_fingerprint(
        env_config,
        rollout_steps,
        episodes_per_trial,
        seed,
        oracle_max_iters,
    )
    path = _cache_path(cache_dir, env_config.system, fp)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No cached oracle training set for system={env_config.system!r} at {path}. "
            f"Generate it first: PYTHONPATH=. python src/steppo/envs/ode/generate_pid_training_dataset.py "
            f"--config <the config you're about to train with>.yaml"
        )
    print(f"[+] Loaded oracle training set: {path}")
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}
