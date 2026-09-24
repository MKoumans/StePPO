"""JAX rollout collection for policy, belief, and expert training data."""

from functools import partial
from typing import Any

import flax.nnx as nnx
import flax.struct
import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.configs.base_config import TrainConfig, WarmstartConfig
from steppo.utils.debug import maybe_jit


@flax.struct.dataclass
class RolloutState:
    """Mutable state during a rollout."""

    env_state: Any
    obs: Array  # (num_envs, obs_dim)
    gru_hidden: Any  # recurrent carry: Array for GRU, richer pytree for other encoders
    belief_mu: Array  # (num_envs, latent_dim)
    belief_logvar: Array  # (num_envs, latent_dim)
    episode_count: Array  # (num_envs,) int32 — episodes completed so far
    trial_done: Array  # (num_envs,) bool — whether trial is frozen
    step: int = flax.struct.field(pytree_node=False)


@flax.struct.dataclass
class RolloutBatch:
    """Output of one rollout: data for VAE + PPO buffers."""

    states: Array  # (T, num_envs, obs_dim)
    actions: Array  # (T, num_envs) discrete or (T, num_envs, action_dim) continuous
    rewards: Array  # (T, num_envs, 1)
    next_states: Array  # (T, num_envs, obs_dim)
    dones: Array  # (T, num_envs)
    beliefs_mu: Array  # (T, num_envs, latent_dim)
    beliefs_logvar: Array  # (T, num_envs, latent_dim)
    log_probs: Array  # (T, num_envs)
    values: Array  # (T, num_envs)
    bootstrap_value: Array  # (num_envs,)
    task_params: Array  # (num_envs, task_dim) — constant per env; (num_envs, 0) when unused
    keep_steps: Array  # (T, num_envs) — 1.0 if solver step accepted, 0.0 if rejected


def _broadcast_params(env_params, num_envs: int):
    """Return `env_params` batched over `num_envs`, broadcasting a single template if needed."""
    leaves = jax.tree.leaves(env_params)
    if leaves and all(jnp.ndim(leaf) >= 1 and jnp.shape(leaf)[0] == num_envs for leaf in leaves):
        return env_params
    return jax.tree.map(lambda x: jnp.stack([x] * num_envs), env_params)


def _build_z(belief_mu, belief_logvar, use_latent_sample, vae_encoder, key):
    """Build z input for policy: sample or concatenate [mu, logvar]."""
    if use_latent_sample:
        num_envs = belief_mu.shape[0]
        keys = jax.random.split(key, num_envs)
        return jax.vmap(vae_encoder.sample)(belief_mu, belief_logvar, keys)
    else:
        return jnp.concatenate([belief_mu, belief_logvar], axis=-1)


def _encode_action_for_encoder(actions, action_space: str, action_dim: int):
    """Convert actions to float array suitable for encoder input."""
    if action_space == "discrete":
        return jax.nn.one_hot(actions, action_dim).astype(jnp.float32)
    return actions.astype(jnp.float32)


def _policy_step(policy, obs, z, rng_key, num_envs, task):
    """Vmap policy.act over envs, clamp log-probs to avoid NaN from tanh saturation."""
    keys = jax.random.split(rng_key, num_envs)
    actions, log_probs, values = jax.vmap(policy.act)(obs, z, keys, task=task)
    log_probs = jnp.where(jnp.isfinite(log_probs), log_probs, jnp.full_like(log_probs, -100.0))
    return actions, log_probs, values


def _task_batch_for(env, params_batch, num_envs: int):
    """Per-env task vector, constant over a rollout; (num_envs, 0) when the env has none."""
    if hasattr(env, "get_task_params"):
        return jax.vmap(env.get_task_params)(params_batch)
    return jnp.zeros((num_envs, 0), dtype=jnp.float32)


def _env_step_and_reset(
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
):
    """Step all envs with multi-episode trials.

    Finished episodes reset to the same task while episodes remain; finished
    trials are frozen (unchanged obs/state, zero reward, done).
    """

    # Step all envs (frozen ones will be overridden below)
    key_envs = jax.random.split(key_env, num_envs)
    next_obs, next_env_state, rewards, dones_ep, info = jax.vmap(env.step)(
        key_envs, env_state, actions, params_batch
    )
    rewards = rewards.reshape(num_envs, 1)
    if "keep_step" in info:
        keep_step = info["keep_step"].astype(jnp.float32)
    else:
        keep_step = jnp.ones(num_envs, dtype=jnp.float32)

    # Track episode completions
    new_episode_count = episode_count + dones_ep.astype(jnp.int32)

    has_limit = episodes_per_trial > 0
    new_trial_done = jnp.where(
        has_limit,
        new_episode_count >= episodes_per_trial,
        jnp.zeros_like(trial_done),
    )
    new_trial_done = new_trial_done | trial_done

    # Auto-reset envs that finished an episode but still have episodes left
    should_reset = dones_ep & ~new_trial_done
    key_resets = jax.random.split(key_reset, num_envs)
    reset_obs, reset_env_state = jax.vmap(env.reset)(key_resets, params_batch)

    live_obs = jnp.where(should_reset[:, None], reset_obs, next_obs)
    live_env_state = jax.tree.map(
        lambda r, n: jnp.where(
            should_reset.reshape(should_reset.shape + (1,) * (r.ndim - 1))
            if r.ndim > 1
            else should_reset,
            r,
            n,
        ),
        reset_env_state,
        next_env_state,
    )

    # Freeze already-done trials: keep carry obs/state, zero reward
    active = ~trial_done
    final_obs = jnp.where(active[:, None], live_obs, obs)
    final_env_state = jax.tree.map(
        lambda live, prev: jnp.where(
            active.reshape(active.shape + (1,) * (live.ndim - 1)) if live.ndim > 1 else active,
            live,
            prev,
        ),
        live_env_state,
        env_state,
    )
    rewards = jnp.where(active[:, None], rewards, jnp.zeros_like(rewards))
    keep_step = jnp.where(active, keep_step, jnp.zeros_like(keep_step))

    # next_obs for encoder: the raw observation before any freeze override
    encoder_obs = jnp.where(active[:, None], next_obs, obs)

    dones_out = dones_ep | trial_done

    return (
        final_obs,
        final_env_state,
        encoder_obs,
        rewards,
        dones_out,
        keep_step,
        new_episode_count,
        new_trial_done,
        dones_ep,
    )


def _encode_step(vae_encoder, actions_enc, next_obs, rewards, task, gru_hidden):
    """Update GRU belief with the new transition."""
    return jax.vmap(vae_encoder.encode_step)(actions_enc, next_obs, rewards, gru_hidden, task)


def init_rollout_state(
    vae,
    env,
    env_params,
    num_envs: int,
    rng_key: PRNGKey,
) -> RolloutState:
    """Reset all environments and initialize their prior beliefs."""
    keys = jax.random.split(rng_key, num_envs)

    params_batch = _broadcast_params(env_params, num_envs)
    obs_batch, env_state_batch = jax.vmap(env.reset)(keys, params_batch)

    latent_dim = vae.config.total_latent_dim
    gru_hidden = vae.encoder.init_hidden((num_envs,))

    prior_mu, prior_logvar = vae.get_prior()
    belief_mu = jnp.broadcast_to(prior_mu, (num_envs, latent_dim))
    belief_logvar = jnp.broadcast_to(prior_logvar, (num_envs, latent_dim))

    return RolloutState(
        env_state=env_state_batch,
        obs=obs_batch,
        gru_hidden=gru_hidden,
        belief_mu=belief_mu,
        belief_logvar=belief_logvar,
        episode_count=jnp.zeros(num_envs, dtype=jnp.int32),
        trial_done=jnp.zeros(num_envs, dtype=jnp.bool_),
        step=0,
    )


@partial(maybe_jit, static_argnames=("vae_graphdef", "policy_graphdef", "env", "config"))
def _collect_rollout_jit(
    rollout_state: RolloutState,
    vae_params: Any,
    vae_graphdef: Any,
    policy_params: Any,
    policy_graphdef: Any,
    env: Any,
    env_params: Any,
    config: TrainConfig,
    rng_key: PRNGKey,
) -> tuple[RolloutState, RolloutBatch]:
    """JIT-compiled environment rollout collection loop."""

    # Reconstruct VAE and Policy once at the outer compile level
    vae = nnx.merge(vae_graphdef, vae_params)
    policy = nnx.merge(policy_graphdef, policy_params)

    num_envs = rollout_state.obs.shape[0]
    use_latent_sample = policy.use_latent_sample
    action_space = policy.action_space
    action_dim = policy.action_dim

    params_batch = _broadcast_params(env_params, num_envs)
    task_batch = _task_batch_for(env, params_batch, num_envs)
    episodes_per_trial = config.episodes_per_trial
    reset_belief = config.reset_belief_between_episodes

    prior_mu, prior_logvar = vae.get_prior()
    prior_mu_batch = jnp.broadcast_to(prior_mu, (num_envs, prior_mu.shape[-1]))
    prior_logvar_batch = jnp.broadcast_to(prior_logvar, (num_envs, prior_logvar.shape[-1]))

    def scan_step(carry, rng_key):
        env_state, obs, gru_hidden, belief_mu, belief_logvar, episode_count, trial_done = carry
        key_act, key_env, key_reset, key_z = jax.random.split(rng_key, 4)

        z = _build_z(belief_mu, belief_logvar, use_latent_sample, vae.encoder, key_z)
        actions, log_probs, values = _policy_step(policy, obs, z, key_act, num_envs, task_batch)

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

        actions_enc = _encode_action_for_encoder(actions, action_space, action_dim)
        new_mu, new_logvar, new_hidden = _encode_step(
            vae.encoder, actions_enc, encoder_obs, rewards, task_batch, gru_hidden
        )

        if reset_belief:
            should_reset_belief = dones_ep & ~trial_done
            zero_hidden = vae.encoder.init_hidden((num_envs,))
            new_hidden = jax.tree.map(
                lambda z, n: jnp.where(
                    should_reset_belief.reshape(should_reset_belief.shape + (1,) * (n.ndim - 1)),
                    z,
                    n,
                ),
                zero_hidden,
                new_hidden,
            )
            new_mu = jnp.where(should_reset_belief[:, None], prior_mu_batch, new_mu)
            new_logvar = jnp.where(should_reset_belief[:, None], prior_logvar_batch, new_logvar)

        transition = (
            obs,
            actions,
            rewards,
            encoder_obs,
            dones,
            belief_mu,
            belief_logvar,
            log_probs,
            values,
            keep_step,
        )
        new_carry = (
            final_env_state,
            final_obs,
            new_hidden,
            new_mu,
            new_logvar,
            new_episode_count,
            new_trial_done,
        )
        return new_carry, transition

    init_carry = (
        rollout_state.env_state,
        rollout_state.obs,
        rollout_state.gru_hidden,
        rollout_state.belief_mu,
        rollout_state.belief_logvar,
        rollout_state.episode_count,
        rollout_state.trial_done,
    )

    T = config.rollout_steps
    keys = jax.random.split(rng_key, T)
    final_carry, transitions = jax.lax.scan(scan_step, init_carry, keys)

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        beliefs_mu,
        beliefs_logvar,
        log_probs,
        values,
        keep_steps,
    ) = transitions

    (
        final_env_state,
        final_obs,
        final_hidden,
        final_mu,
        final_logvar,
        final_episode_count,
        final_trial_done,
    ) = final_carry

    final_z = _build_z(final_mu, final_logvar, use_latent_sample, vae.encoder, rng_key)
    bootstrap_value = jax.vmap(lambda obs_s, z_s, task_s: policy(obs_s, z_s, task_s)[1])(
        final_obs, final_z, task_batch
    )

    new_rollout_state = RolloutState(
        env_state=final_env_state,
        obs=final_obs,
        gru_hidden=final_hidden,
        belief_mu=final_mu,
        belief_logvar=final_logvar,
        episode_count=final_episode_count,
        trial_done=final_trial_done,
        step=rollout_state.step + T,
    )

    # Same task_batch computed once above (constant across this rollout window).
    task_params = task_batch

    batch = RolloutBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        beliefs_mu=beliefs_mu,
        beliefs_logvar=beliefs_logvar,
        log_probs=log_probs,
        values=values,
        bootstrap_value=bootstrap_value,
        task_params=task_params,
        keep_steps=keep_steps,
    )

    return new_rollout_state, batch


def collect_rollout(
    rollout_state: RolloutState,
    vae,
    policy,
    env,
    env_params,
    config: TrainConfig,
    rng_key: PRNGKey,
) -> tuple[RolloutState, RolloutBatch]:
    """Collect one configured rollout and return its updated carry and batch."""

    vae_graphdef, vae_params = nnx.split(vae)
    policy_graphdef, policy_params = nnx.split(policy)

    new_rollout_state, batch = _collect_rollout_jit(
        rollout_state,
        vae_params,
        vae_graphdef,
        policy_params,
        policy_graphdef,
        env,
        env_params,
        config,
        rng_key,
    )

    return new_rollout_state, batch


# ---------------------------------------------------------------------------
# PID expert rollout (for warmstart)
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=(
        "vae_graphdef",
        "env",
        "config",
        "step_context_offset",
        "pid_q",
        "pid_safety",
        "pid_min_factor",
        "pid_max_factor",
        "pid_kp",
        "pid_ki",
        "pid_kd",
    ),
)
def _collect_pid_rollout_jit(
    vae_params: Any,
    vae_graphdef: Any,
    env: Any,
    env_params: Any,
    config: TrainConfig,
    step_context_offset: int,
    pid_q: int,
    pid_safety: float,
    pid_min_factor: float,
    pid_max_factor: float,
    rng_key: PRNGKey,
    pid_kp: float = 0.0,
    pid_ki: float = 1.0,
    pid_kd: float = 0.0,
) -> RolloutBatch:
    """Collect rollouts using the PID step-size controller instead of a learned policy."""
    from steppo.training.pid_controller import pid_action_from_obs

    vae = nnx.merge(vae_graphdef, vae_params)

    num_envs = config.num_envs
    params_batch = _broadcast_params(env_params, num_envs)
    task_batch = _task_batch_for(env, params_batch, num_envs)

    key_reset, key_run = jax.random.split(rng_key)
    reset_keys = jax.random.split(key_reset, num_envs)
    obs_batch, env_state_batch = jax.vmap(env.reset)(reset_keys, params_batch)

    latent_dim = vae.config.total_latent_dim
    gru_hidden = vae.encoder.init_hidden((num_envs,))
    prior_mu, prior_logvar = vae.get_prior()
    belief_mu = jnp.broadcast_to(prior_mu, (num_envs, latent_dim))
    belief_logvar = jnp.broadcast_to(prior_logvar, (num_envs, latent_dim))
    prior_mu_batch = belief_mu
    prior_logvar_batch = belief_logvar
    episodes_per_trial = config.episodes_per_trial
    reset_belief = config.reset_belief_between_episodes

    def scan_step(carry, rng_key):
        env_state, obs, gru_hidden, belief_mu, belief_logvar, episode_count, trial_done = carry

        key_env, key_reset = jax.random.split(rng_key)

        # Clip to the policy action range: these act as policy actions downstream.
        actions = jax.vmap(
            lambda o: jnp.clip(
                pid_action_from_obs(
                    o,
                    step_context_offset,
                    pid_q,
                    pid_safety,
                    pid_min_factor,
                    pid_max_factor,
                    dt_log_gain=env.dt_log_gain,
                    kp=pid_kp,
                    ki=pid_ki,
                    kd=pid_kd,
                ),
                -1.0,
                1.0,
            )
        )(obs)
        log_probs = jnp.zeros(num_envs)
        values = jnp.zeros(num_envs)

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

        actions_enc = actions.astype(jnp.float32)
        new_mu, new_logvar, new_hidden = jax.vmap(
            lambda a, o, r, h, t: vae.encoder.encode_step(a, o, r, h, t)
        )(actions_enc, encoder_obs, rewards, gru_hidden, task_batch)

        if reset_belief:
            should_reset_belief = dones_ep & ~trial_done
            zero_hidden = vae.encoder.init_hidden((num_envs,))
            new_hidden = jax.tree.map(
                lambda z, n: jnp.where(
                    should_reset_belief.reshape(should_reset_belief.shape + (1,) * (n.ndim - 1)),
                    z,
                    n,
                ),
                zero_hidden,
                new_hidden,
            )
            new_mu = jnp.where(should_reset_belief[:, None], prior_mu_batch, new_mu)
            new_logvar = jnp.where(should_reset_belief[:, None], prior_logvar_batch, new_logvar)

        transition = (
            obs,
            actions,
            rewards,
            encoder_obs,
            dones,
            belief_mu,
            belief_logvar,
            log_probs,
            values,
            keep_step,
        )
        new_carry = (
            final_env_state,
            final_obs,
            new_hidden,
            new_mu,
            new_logvar,
            new_episode_count,
            new_trial_done,
        )
        return new_carry, transition

    init_carry = (
        env_state_batch,
        obs_batch,
        gru_hidden,
        belief_mu,
        belief_logvar,
        jnp.zeros(num_envs, dtype=jnp.int32),
        jnp.zeros(num_envs, dtype=jnp.bool_),
    )
    keys = jax.random.split(key_run, config.rollout_steps)
    _, transitions = jax.lax.scan(scan_step, init_carry, keys)

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        beliefs_mu,
        beliefs_logvar,
        log_probs,
        values,
        keep_steps,
    ) = transitions

    task_params = task_batch

    return RolloutBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        beliefs_mu=beliefs_mu,
        beliefs_logvar=beliefs_logvar,
        log_probs=log_probs,
        values=values,
        bootstrap_value=jnp.zeros(num_envs),
        task_params=task_params,
        keep_steps=keep_steps,
    )


def collect_pid_rollout(
    vae,
    env,
    env_params,
    config: TrainConfig,
    warmstart_config: WarmstartConfig,
    step_context_offset: int,
    rng_key: PRNGKey,
) -> RolloutBatch:
    """Public wrapper for PID expert rollout collection."""
    vae_graphdef, vae_params = nnx.split(vae)
    return _collect_pid_rollout_jit(
        vae_params,
        vae_graphdef,
        env,
        env_params,
        config,
        step_context_offset,
        warmstart_config.pid_order,
        warmstart_config.pid_safety,
        warmstart_config.pid_min_factor,
        warmstart_config.pid_max_factor,
        rng_key,
        pid_kp=warmstart_config.pid_kp,
        pid_ki=warmstart_config.pid_ki,
        pid_kd=warmstart_config.pid_kd,
    )


# ---------------------------------------------------------------------------
# Oracle expert rollout (for warmstart)
# ---------------------------------------------------------------------------
# Oracle trajectories are generated once offline (oracle_dataset.py) because the
# search is expensive. They do not depend on the belief, so each warmstart
# iteration only re-encodes the beliefs with the current VAE.


@partial(jax.jit, static_argnames=("vae_graphdef", "config"))
def _encode_oracle_batch_jit(
    vae_params: Any,
    vae_graphdef: Any,
    static_batch: dict,
    config: TrainConfig,
) -> RolloutBatch:
    """Replay a precomputed oracle trajectory (oracle_dataset.py) through the
    current vae's GRU encoder to attach beliefs_mu/beliefs_logvar. No env.step
    calls — states/actions/rewards/dones are already fixed."""
    vae = nnx.merge(vae_graphdef, vae_params)

    states = static_batch["states"]
    actions = static_batch["actions"]
    rewards = static_batch["rewards"]
    next_states = static_batch["next_states"]
    dones_ep = static_batch["dones_ep"]
    trial_done_before = static_batch["trial_done_before"]
    task_batch = static_batch["task_params"]  # (num_envs, task_dim) — constant per trajectory

    T, num_envs = states.shape[0], states.shape[1]
    latent_dim = vae.config.total_latent_dim
    gru_hidden = vae.encoder.init_hidden((num_envs,))
    prior_mu, prior_logvar = vae.get_prior()
    belief_mu = jnp.broadcast_to(prior_mu, (num_envs, latent_dim))
    belief_logvar = jnp.broadcast_to(prior_logvar, (num_envs, latent_dim))
    prior_mu_batch = belief_mu
    prior_logvar_batch = belief_logvar
    reset_belief = config.reset_belief_between_episodes

    def scan_step(carry, xs):
        gru_hidden, belief_mu, belief_logvar = carry
        actions_t, next_obs_t, rewards_t, dones_ep_t, trial_done_t = xs

        actions_enc = actions_t.astype(jnp.float32)
        new_mu, new_logvar, new_hidden = jax.vmap(
            lambda a, o, r, h, t: vae.encoder.encode_step(a, o, r, h, t)
        )(actions_enc, next_obs_t, rewards_t, gru_hidden, task_batch)

        if reset_belief:
            should_reset_belief = dones_ep_t & ~trial_done_t
            zero_hidden = vae.encoder.init_hidden((num_envs,))
            new_hidden = jax.tree.map(
                lambda z, n: jnp.where(
                    should_reset_belief.reshape(should_reset_belief.shape + (1,) * (n.ndim - 1)),
                    z,
                    n,
                ),
                zero_hidden,
                new_hidden,
            )
            new_mu = jnp.where(should_reset_belief[:, None], prior_mu_batch, new_mu)
            new_logvar = jnp.where(should_reset_belief[:, None], prior_logvar_batch, new_logvar)

        new_carry = (new_hidden, new_mu, new_logvar)
        return new_carry, (belief_mu, belief_logvar)

    init_carry = (gru_hidden, belief_mu, belief_logvar)
    xs = (actions, next_states, rewards, dones_ep, trial_done_before)
    _, (beliefs_mu, beliefs_logvar) = jax.lax.scan(scan_step, init_carry, xs)

    return RolloutBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=static_batch["dones"],
        beliefs_mu=beliefs_mu,
        beliefs_logvar=beliefs_logvar,
        log_probs=jnp.zeros((T, num_envs)),
        values=jnp.zeros((T, num_envs)),
        bootstrap_value=jnp.zeros(num_envs),
        task_params=static_batch["task_params"],
        keep_steps=static_batch["keep_steps"],
    )


def encode_oracle_batch(vae, static_batch: dict, config: TrainConfig) -> RolloutBatch:
    """Public wrapper: attach the current vae's belief encoding to a
    precomputed oracle trajectory minibatch (see oracle_dataset.py)."""
    vae_graphdef, vae_params = nnx.split(vae)
    return _encode_oracle_batch_jit(vae_params, vae_graphdef, static_batch, config)
