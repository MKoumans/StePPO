"""Evaluation rollouts and diagnostics for trained policies."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

from steppo.envs.ode import ODEParams


def _format_t_per_episode(mean_t_list: list[float], precision: int = 1) -> str:
    """Format per-episode t values as 'N-N-N', dropping trailing zeros."""
    parts = []
    for t in mean_t_list:
        parts.append(f"{t:.{precision}f}")
    while len(parts) > 1 and parts[-1] == f"{0.0:.{precision}f}":
        parts.pop()
    return "-".join(parts)


def eval_episode(
    vae,
    policy,
    env,
    env_params,
    rng_key: PRNGKey,
    deterministic: bool = True,
    max_steps: int = 5000,
    return_trajectories: bool = False,
    episodes_per_trial: int = 0,
    reset_belief_between_episodes: bool = False,
    z_noise: bool = False,
    z_noise_alpha_mu: float = 0.0,
    z_noise_alpha_logvar: float = 0.0,
) -> dict:
    """Roll out one trial (one or more episodes of the same task) with lax.scan.

    `episodes_per_trial` = 0 means a single episode. `z_noise` feeds the policy
    standard-normal noise instead of the belief; `z_noise_alpha_mu/_logvar`
    blend the belief mean/log-variance towards that noise (0 = belief,
    1 = noise). Only the policy input is perturbed.
    """
    use_latent_sample = policy.use_latent_sample
    action_space = policy.action_space
    action_dim = policy.action_dim

    key_reset, key_run = jax.random.split(rng_key)
    obs0, env_state0 = env.reset(key_reset, env_params)

    gru_hidden0 = vae.encoder.init_hidden()
    belief_mu0, belief_logvar0 = vae.get_prior()

    task = env.get_task_params(env_params) if hasattr(env, "get_task_params") else jnp.zeros(0)

    has_limit = episodes_per_trial > 0
    _MAX_EP_BUFFER = 20
    max_ep = episodes_per_trial if has_limit else _MAX_EP_BUFFER
    has_t = hasattr(env_state0, "t")

    def step_fn(carry, i):
        (
            obs,
            env_state,
            gru_hidden,
            belief_mu,
            belief_logvar,
            total_return,
            trial_frozen,
            episode_count,
            ep_t_reached,
            accepted_count,
        ) = carry

        key_i = jax.random.fold_in(key_run, i)
        key_act, key_step, key_enc, key_reset_ep, key_noise = jax.random.split(key_i, 5)

        key_noise_mu, key_noise_logvar = jax.random.split(key_noise)
        eps_mu = jax.random.normal(key_noise_mu, belief_mu.shape, dtype=belief_mu.dtype)
        eps_logvar = jax.random.normal(
            key_noise_logvar, belief_logvar.shape, dtype=belief_logvar.dtype
        )
        belief_mu_z = (1.0 - z_noise_alpha_mu) * belief_mu + z_noise_alpha_mu * eps_mu
        belief_logvar_z = (
            1.0 - z_noise_alpha_logvar
        ) * belief_logvar + z_noise_alpha_logvar * eps_logvar

        if use_latent_sample:
            z = vae.encoder.sample(belief_mu_z, belief_logvar_z, key_enc)
        else:
            z = jnp.concatenate([belief_mu_z, belief_logvar_z], axis=-1)

        if z_noise:
            z = jax.random.normal(key_noise, z.shape, dtype=z.dtype)

        action, _, _ = policy.act(obs, z, key_act, deterministic=deterministic, task=task)
        obs_next, env_state_next, reward, done_ep, info = env.step(
            key_step, env_state, action, env_params
        )

        if action_space == "discrete":
            action_enc = jax.nn.one_hot(action, action_dim).astype(jnp.float32)
        else:
            action_enc = action.astype(jnp.float32)

        reward_enc = jnp.reshape(reward, (1,)).astype(jnp.float32)
        new_mu, new_logvar, new_hidden = vae.encoder.encode_step(
            action_enc, obs_next, reward_enc, gru_hidden, task
        )

        new_episode_count = episode_count + done_ep.astype(jnp.int32)
        new_trial_frozen = jnp.where(
            has_limit,
            new_episode_count >= episodes_per_trial,
            jnp.asarray(False),
        )
        new_trial_frozen = new_trial_frozen | trial_frozen

        should_reset_env = done_ep & ~new_trial_frozen
        reset_obs, reset_env_state = env.reset(key_reset_ep, env_params)

        if reset_belief_between_episodes:
            should_reset_belief = done_ep & ~trial_frozen
            zero_hidden = vae.encoder.init_hidden()
            new_hidden = jax.tree_util.tree_map(
                lambda z, n: jnp.where(should_reset_belief, z, n), zero_hidden, new_hidden
            )
            new_mu = jnp.where(should_reset_belief, belief_mu0, new_mu)
            new_logvar = jnp.where(should_reset_belief, belief_logvar0, new_logvar)

        active = ~trial_frozen

        def select(new, old):
            return jnp.where(active, new, old)

        if has_t:
            new_accepted_count = accepted_count + (info["keep_step"] & active).astype(jnp.int32)
        else:
            new_accepted_count = accepted_count

        live_obs = jnp.where(should_reset_env, reset_obs, obs_next)
        live_env_state = jax.tree_util.tree_map(
            lambda r, n: jnp.where(should_reset_env, r, n),
            reset_env_state,
            env_state_next,
        )

        if has_t:
            ep_idx = jnp.minimum(episode_count, max_ep - 1)
            ep_t_reached = jnp.where(
                active,
                ep_t_reached.at[ep_idx].set(
                    jnp.maximum(ep_t_reached[ep_idx], env_state_next.t.astype(jnp.float32))
                ),
                ep_t_reached,
            )

        carry_next = (
            jax.tree_util.tree_map(select, live_obs, obs),
            jax.tree_util.tree_map(select, live_env_state, env_state),
            jax.tree_util.tree_map(select, new_hidden, gru_hidden),
            select(new_mu, belief_mu),
            select(new_logvar, belief_logvar),
            total_return + jnp.where(active, reward, 0.0),
            new_trial_frozen,
            new_episode_count,
            ep_t_reached,
            new_accepted_count,
        )
        if return_trajectories:
            ys = (carry_next[3], carry_next[4], active)
        else:
            ys = active
        return carry_next, ys

    init_carry = (
        obs0,
        env_state0,
        gru_hidden0,
        belief_mu0,
        belief_logvar0,
        jnp.asarray(0.0),
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros(max_ep, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    final_carry, scan_ys = jax.lax.scan(step_fn, init_carry, jnp.arange(max_steps))
    if return_trajectories:
        mus, logvars, active_mask = scan_ys
    else:
        active_mask = scan_ys

    total_return = final_carry[5]
    episode_length = jnp.sum(active_mask.astype(jnp.int32))
    final_episode_count = final_carry[7]
    ep_t_reached = final_carry[8]
    num_episodes = jnp.minimum(final_episode_count + 1, max_ep)

    final_mu = final_carry[3]
    final_logvar = final_carry[4]

    sigma_initial = jnp.mean(jnp.exp(0.5 * belief_logvar0))
    sigma_final = jnp.mean(jnp.exp(0.5 * final_logvar))
    sigma_reduction_ratio = sigma_initial / (sigma_final + 1e-8)

    final_env_state = final_carry[1]
    out = {
        "total_return": total_return,
        "episode_length": episode_length,
        "final_mu": final_mu,
        "final_logvar": final_logvar,
        "sigma_reduction_ratio": sigma_reduction_ratio,
    }

    if has_t:
        out["ep_t_reached"] = ep_t_reached
        out["num_episodes"] = num_episodes
        accepted_count = final_carry[9]
        out["accepted_steps"] = accepted_count
        out["rejected_steps"] = episode_length - accepted_count
    if hasattr(final_env_state, "budget_exhausted"):
        out["budget_exhausted"] = final_env_state.budget_exhausted

    if return_trajectories:
        out["belief_mu_trajectory"] = jnp.concatenate([belief_mu0[None], mus], axis=0)
        out["belief_logvar_trajectory"] = jnp.concatenate([belief_logvar0[None], logvars], axis=0)

    has_task_decoder = vae.task_decoder is not None
    has_task_params = hasattr(env, "get_task_params")
    if has_task_params:
        out["task_true"] = env.get_task_params(env_params)
    if has_task_decoder and has_task_params:
        task_pred = vae.task_decoder(final_mu)
        if isinstance(task_pred, tuple):
            task_pred = task_pred[0]
        out["task_embedding_mae"] = jnp.mean(jnp.abs(task_pred - out["task_true"]))

    return out


def eval_episode_early_exit(
    vae,
    policy,
    env,
    env_params,
    rng_key: PRNGKey,
    deterministic: bool = True,
    max_steps: int = 5000,
) -> dict:
    """Like eval_episode but uses while_loop for early termination at inference."""
    use_latent_sample = policy.use_latent_sample
    action_space = policy.action_space
    action_dim = policy.action_dim

    key_reset, key_run = jax.random.split(rng_key)
    obs0, env_state0 = env.reset(key_reset, env_params)

    gru_hidden0 = vae.encoder.init_hidden()
    belief_mu0, belief_logvar0 = vae.get_prior()

    task = env.get_task_params(env_params) if hasattr(env, "get_task_params") else jnp.zeros(0)

    def cond_fn(carry):
        _, _, _, _, _, _, done, step_i = carry
        return ~done & (step_i < max_steps)

    def body_fn(carry):
        obs, env_state, gru_hidden, belief_mu, belief_logvar, total_return, _, step_i = carry

        key_i = jax.random.fold_in(key_run, step_i)
        key_act, key_step, key_enc = jax.random.split(key_i, 3)

        if use_latent_sample:
            z = vae.encoder.sample(belief_mu, belief_logvar, key_enc)
        else:
            z = jnp.concatenate([belief_mu, belief_logvar], axis=-1)

        action, _, _ = policy.act(obs, z, key_act, deterministic=deterministic, task=task)
        obs_next, env_state_next, reward, done, _ = env.step(
            key_step, env_state, action, env_params
        )

        if action_space == "discrete":
            action_enc = jax.nn.one_hot(action, action_dim).astype(jnp.float32)
        else:
            action_enc = action.astype(jnp.float32)

        reward_enc = jnp.reshape(reward, (1,)).astype(jnp.float32)
        new_mu, new_logvar, new_hidden = vae.encoder.encode_step(
            action_enc, obs_next, reward_enc, gru_hidden, task
        )

        return (
            obs_next,
            env_state_next,
            new_hidden,
            new_mu,
            new_logvar,
            total_return + reward,
            done,
            step_i + 1,
        )

    init_carry = (
        obs0,
        env_state0,
        gru_hidden0,
        belief_mu0,
        belief_logvar0,
        jnp.asarray(0.0),
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
    )
    final = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    total_return = final[5]
    episode_length = final[7]
    final_env_state = final[1]

    out = {
        "total_return": total_return,
        "episode_length": episode_length,
    }
    if hasattr(final_env_state, "t"):
        out["t_reached"] = final_env_state.t
    if hasattr(final_env_state, "budget_exhausted"):
        out["budget_exhausted"] = final_env_state.budget_exhausted
    if hasattr(final_env_state, "accept_ema"):
        out["accept_ema"] = final_env_state.accept_ema

    return out


def _diagnostic_episode_core(vae, policy, env, env_params, rng_key, max_steps: int):
    """Run one episode, recording per-step diagnostics.

    Not jitted, so callers can vmap it inside their own nnx.jit.
    """
    use_latent_sample = policy.use_latent_sample
    action_space = policy.action_space
    action_dim = policy.action_dim

    key_reset, key_run = jax.random.split(rng_key)
    obs0, env_state0 = env.reset(key_reset, env_params)

    gru_hidden0 = vae.encoder.init_hidden()
    belief_mu0, belief_logvar0 = vae.get_prior()

    task = env.get_task_params(env_params) if hasattr(env, "get_task_params") else jnp.zeros(0)

    def step_fn(carry, i):
        obs, env_state, gru_hidden, belief_mu, belief_logvar, total_return, done_so_far = carry

        key_i = jax.random.fold_in(key_run, i)
        key_act, key_step, key_enc = jax.random.split(key_i, 3)

        if use_latent_sample:
            z = vae.encoder.sample(belief_mu, belief_logvar, key_enc)
        else:
            z = jnp.concatenate([belief_mu, belief_logvar], axis=-1)

        action, _, _ = policy.act(obs, z, key_act, deterministic=True, task=task)
        dt_before = env_state.dt
        next_obs, next_env_state, reward, done, info = env.step(
            key_step, env_state, action, env_params
        )

        if action_space == "discrete":
            action_enc = jax.nn.one_hot(action, action_dim).astype(jnp.float32)
        else:
            action_enc = action.astype(jnp.float32)

        reward_enc = jnp.reshape(reward, (1,)).astype(jnp.float32)
        new_mu, new_logvar, new_hidden = vae.encoder.encode_step(
            action_enc, next_obs, reward_enc, gru_hidden, task
        )

        active = jnp.logical_not(done_so_far)

        def select(new, old):
            return jnp.where(active, new, old)

        carry_next = (
            jax.tree_util.tree_map(select, next_obs, obs),
            jax.tree_util.tree_map(select, next_env_state, env_state),
            jax.tree_util.tree_map(select, new_hidden, gru_hidden),
            select(new_mu, belief_mu),
            select(new_logvar, belief_logvar),
            total_return + jnp.where(active, reward, 0.0),
            jnp.logical_or(done_so_far, done),
        )

        diag = {
            "y": next_env_state.y,
            "t": next_env_state.t,
            "dt": dt_before,
            "reward": reward,
            "scaled_error": info["scaled_error"],
            "keep_step": info["keep_step"],
            "active": active,
            "reject_streak": next_env_state.reject_streak,
            "accept_ema": next_env_state.accept_ema,
            "log_error_ema": next_env_state.log_error_ema,
        }
        return carry_next, diag

    init_carry = (
        obs0,
        env_state0,
        gru_hidden0,
        belief_mu0,
        belief_logvar0,
        jnp.asarray(0.0),
        jnp.asarray(False),
    )
    _, diagnostics = jax.lax.scan(step_fn, init_carry, jnp.arange(max_steps))
    return diagnostics


_run_diagnostic_episode = nnx.jit(static_argnames=("env", "max_steps"))(_diagnostic_episode_core)


def run_diagnostic_episode(vae, policy, env, rng_key, max_steps=500, env_params=None):
    """Public wrapper: run one diagnostic episode and return numpy arrays."""
    if hasattr(env, "t_end"):
        if env_params is None:
            env_params = env.sample_task(rng_key)
        dataset = {
            "task_params": jnp.asarray([env_params.lam]),
            "episode_keys": jax.random.split(rng_key, 1),
        }
        return run_cached_diagnostic_episode(vae, policy, env, dataset, rng_key, max_steps)
    if env_params is None:
        env_params = env.sample_task(rng_key)
    diag = _run_diagnostic_episode(vae, policy, env, env_params, rng_key, max_steps)
    active = np.asarray(diag["active"])
    n_active = int(np.sum(active))
    return {k: np.asarray(v)[:n_active] for k, v in diag.items()}


# name -> (per-episode reduction, direction) for the PID fallback control limits.
# "upper": higher is worse (limit = envelope * multiplier); "lower": the opposite.
CONTROL_STATS = {
    "reject_streak": (jnp.max, "upper"),
    "accept_ema": (jnp.min, "lower"),
    "log_error_ema": (jnp.max, "upper"),
}


@nnx.jit(static_argnames=("env", "max_steps"))
def _collect_diagnostic_batch(vae, policy, env, keys, max_steps: int):
    """Batches _diagnostic_episode_core via vmap inside this single nnx.jit
    boundary — see _diagnostic_episode_core's docstring for why the core must
    stay undecorated for this to work."""
    return jax.vmap(
        lambda k: _diagnostic_episode_core(vae, policy, env, env.sample_task(k), k, max_steps)
    )(keys)


def _dataset_ode_params(env, lam, key):
    """Rebuild ODE task params for a cached task, using solve_pid_batch's key split."""
    key_pulse = jax.random.split(key)[1]
    pulse_phase = jax.random.uniform(key_pulse, shape=(), dtype=jnp.float32)
    return ODEParams(lam=lam, pulse_phase=pulse_phase, max_steps=env.max_steps)


@nnx.jit(static_argnames=("env", "max_steps"))
def _collect_diagnostic_dataset_batch(vae, policy, env, lams, keys, max_steps: int):
    """Run diagnostics on explicit cached ODE tasks, without task sampling."""

    def run_one(lam, key):
        params = _dataset_ode_params(env, lam, key)
        return _diagnostic_episode_core(vae, policy, env, params, key, max_steps)

    return jax.vmap(run_one)(lams, keys)


def run_cached_diagnostic_episode(
    vae, policy, env, dataset: dict, rng_key: PRNGKey, max_steps: int, index: int = 0
):
    """Run one diagnostic episode from a cached ODE task."""
    import numpy as np

    lams = jnp.asarray(dataset["task_params"], dtype=jnp.float32)[index : index + 1]
    keys = jnp.asarray(dataset["episode_keys"])[index : index + 1]
    diag = _collect_diagnostic_dataset_batch(vae, policy, env, lams, keys, max_steps)
    active = np.asarray(diag["active"])[0]
    n_active = int(np.sum(active))
    return {k: np.asarray(v)[0, :n_active] for k, v in diag.items()}


def collect_control_stat_envelope(
    vae,
    policy,
    env,
    rng_key,
    num_envs: int,
    max_steps: int,
    stat_name: str,
    percentile: float,
    dataset: dict | None = None,
) -> float:
    """One control-chart calibration 'subgroup': percentile (across num_envs
    episodes) of each episode's worst value of `stat_name`, direction per
    CONTROL_STATS. When ``dataset`` is supplied, its cached ``task_params`` and
    ``episode_keys`` are replayed instead of sampling fresh tasks."""
    if dataset is None:
        keys = jax.random.split(rng_key, num_envs)
        diag = _collect_diagnostic_batch(vae, policy, env, keys, max_steps)
    else:
        lams = jnp.asarray(dataset["task_params"][:num_envs], dtype=jnp.float32)
        keys = jnp.asarray(dataset["episode_keys"][:num_envs])
        diag = _collect_diagnostic_dataset_batch(vae, policy, env, lams, keys, max_steps)

    reduce_fn, _ = CONTROL_STATS[stat_name]
    fill = -jnp.inf if reduce_fn is jnp.max else jnp.inf  # inactive steps never win the reduction
    masked = jnp.where(diag["active"], diag[stat_name], fill)
    per_env = reduce_fn(masked, axis=1)  # (num_envs,)
    return float(jnp.percentile(per_env, percentile))


@nnx.jit(
    static_argnames=("env", "max_steps", "episodes_per_trial", "reset_belief_between_episodes")
)
def _run_eval_batch(
    vae,
    policy,
    env,
    keys,
    max_steps: int,
    episodes_per_trial: int = 0,
    reset_belief_between_episodes: bool = False,
):
    """Module-level nnx.jit so the compiled function is cached across eval calls."""

    def run_episode(key_i):
        key_task, key_ep = jax.random.split(key_i)
        task_params = env.sample_task(key_task)
        return eval_episode(
            vae,
            policy,
            env,
            task_params,
            key_ep,
            max_steps=max_steps,
            episodes_per_trial=episodes_per_trial,
            reset_belief_between_episodes=reset_belief_between_episodes,
        )

    return jax.vmap(run_episode)(keys)


@nnx.jit(
    static_argnames=("env", "max_steps", "episodes_per_trial", "reset_belief_between_episodes")
)
def _run_eval_dataset_batch(
    vae,
    policy,
    env,
    lams,
    keys,
    max_steps: int,
    episodes_per_trial: int = 0,
    reset_belief_between_episodes: bool = False,
):
    """Evaluate explicit cached ODE tasks; no call to ``env.sample_task``."""

    def run_one(lam, key):
        params = _dataset_ode_params(env, lam, key)
        return eval_episode(
            vae,
            policy,
            env,
            params,
            key,
            max_steps=max_steps,
            episodes_per_trial=episodes_per_trial,
            reset_belief_between_episodes=reset_belief_between_episodes,
        )

    return jax.vmap(run_one)(lams, keys)


def eval_policy(
    vae,
    policy,
    env,
    env_params,
    rng_key: PRNGKey,
    num_episodes: int,
    num_envs: int = 1,
    max_steps: int = 5000,
    episodes_per_trial: int = 0,
    reset_belief_between_episodes: bool = False,
    dataset: dict | None = None,
) -> dict:
    """Average metrics over episodes, on the cached tasks of ``dataset`` or on freshly sampled ones."""

    if dataset is None:
        keys = jax.vmap(lambda i: jax.random.fold_in(rng_key, i))(jnp.arange(num_episodes))
    else:
        task_values = jnp.asarray(dataset["task_params"], dtype=jnp.float32)
        task_keys = jnp.asarray(dataset["episode_keys"])
        num_episodes = min(num_episodes, int(task_values.shape[0]))
        if num_episodes <= 0:
            raise ValueError("Cached evaluation dataset is empty")
        task_values = task_values[:num_episodes]
        task_keys = task_keys[:num_episodes]

    returns, lengths, final_logvars = [], [], []
    sigma_ratios, task_maes = [], []
    ep_t_all, step_counts_all = [], []
    final_mus_all, task_trues_all = [], []
    for start in range(0, num_episodes, num_envs):
        if dataset is None:
            batch_metrics = _run_eval_batch(
                vae,
                policy,
                env,
                keys[start : start + num_envs],
                max_steps,
                episodes_per_trial,
                reset_belief_between_episodes,
            )
        else:
            batch_metrics = _run_eval_dataset_batch(
                vae,
                policy,
                env,
                task_values[start : start + num_envs],
                task_keys[start : start + num_envs],
                max_steps,
                episodes_per_trial,
                reset_belief_between_episodes,
            )
        returns.append(batch_metrics["total_return"])
        lengths.append(batch_metrics["episode_length"])
        final_logvars.append(jnp.mean(jnp.exp(batch_metrics["final_logvar"]), axis=-1))
        sigma_ratios.append(batch_metrics["sigma_reduction_ratio"])
        if "ep_t_reached" in batch_metrics:
            ep_t_all.append(batch_metrics["ep_t_reached"])
        if "task_embedding_mae" in batch_metrics:
            task_maes.append(batch_metrics["task_embedding_mae"])
        if "budget_exhausted" in batch_metrics:
            step_counts_all.append(batch_metrics["budget_exhausted"] * max_steps)
        if "task_true" in batch_metrics:
            final_mus_all.append(batch_metrics["final_mu"])
            task_trues_all.append(batch_metrics["task_true"])

    returns = jnp.concatenate(returns)
    lengths = jnp.concatenate(lengths)
    final_logvars = jnp.concatenate(final_logvars)
    sigma_ratios = jnp.concatenate(sigma_ratios)

    result = {
        "mean_return": float(jnp.mean(returns)),
        "std_return": float(jnp.std(returns)),
        "mean_length": float(jnp.mean(lengths)),
        "std_length": float(jnp.std(lengths)),
        "belief_final_variance": float(jnp.mean(final_logvars)),
        "sigma_reduction_ratio": float(jnp.mean(sigma_ratios)),
        "std_sigma_reduction_ratio": float(jnp.std(sigma_ratios)),
    }
    if ep_t_all and hasattr(env, "t_end"):
        ep_t = jnp.concatenate(ep_t_all, axis=0)
        t_end = float(env.t_end)
        max_ep = ep_t.shape[1]
        mean_t_per_ep = [float(jnp.mean(ep_t[:, i])) for i in range(max_ep)]
        result["t_per_episode"] = _format_t_per_episode(mean_t_per_ep)
        per_ep_success = [float(jnp.mean(ep_t[:, i] >= t_end)) for i in range(max_ep)]
        result["success_per_episode"] = per_ep_success
        result["success_rate"] = per_ep_success[0] if per_ep_success else 0.0
    else:
        result["success_rate"] = 0.0
    if task_maes:
        all_maes = jnp.concatenate(task_maes)
        result["task_embedding_mae"] = float(jnp.mean(all_maes))
        result["std_task_embedding_mae"] = float(jnp.std(all_maes))
    if step_counts_all:
        all_steps = jnp.concatenate(step_counts_all)
        result["mean_step_count"] = float(jnp.mean(all_steps))
        result["std_step_count"] = float(jnp.std(all_steps))
    if final_mus_all:
        all_mu = np.asarray(jnp.concatenate(final_mus_all))  # (N, latent_dim)
        all_task = np.asarray(jnp.concatenate(task_trues_all))  # (N, task_dim)
        # Task param is scalar in the ODE envs (lam); correlate each latent dim
        # against it directly, matching analyse_rollout.plot_task_vs_belief.
        task_scalar = all_task[:, 0] if all_task.ndim > 1 else all_task
        per_dim_r = []
        if task_scalar.std() > 1e-8:
            for d in range(all_mu.shape[-1]):
                z = all_mu[:, d]
                r = float(np.corrcoef(task_scalar, z)[0, 1]) if z.std() > 1e-8 else 0.0
                per_dim_r.append(0.0 if np.isnan(r) else r)
        else:
            per_dim_r = [0.0] * all_mu.shape[-1]
        result["task_belief_corr_per_dim"] = per_dim_r
        result["task_belief_corr_sum"] = float(np.sum(np.abs(per_dim_r)))
    return result


@nnx.jit(
    static_argnames=(
        "env",
        "max_steps",
        "num_repeats",
        "num_mus",
        "episodes_per_trial",
        "reset_belief_between_episodes",
    )
)
def _run_mu_table_batch(
    vae,
    policy,
    env,
    mu_arr,
    rng_key,
    max_steps: int,
    num_repeats: int,
    num_mus: int,
    episodes_per_trial: int = 0,
    reset_belief_between_episodes: bool = False,
):
    """Run all mu values × repeats in a single vmap call."""
    keys = jax.random.split(rng_key, num_mus * num_repeats).reshape(num_mus, num_repeats, 2)

    def run_single_mu(mu_val, mu_keys):
        def run_one(k):
            key_pulse = jax.random.split(k)[1]
            pulse_phase = jax.random.uniform(key_pulse, shape=(), dtype=jnp.float32)
            ep = ODEParams(lam=mu_val, pulse_phase=pulse_phase, max_steps=max_steps)
            return eval_episode(
                vae,
                policy,
                env,
                ep,
                k,
                max_steps=max_steps,
                episodes_per_trial=episodes_per_trial,
                reset_belief_between_episodes=reset_belief_between_episodes,
            )

        return jax.vmap(run_one)(mu_keys)

    return jax.vmap(run_single_mu)(mu_arr, keys)


def eval_mu_table(
    vae,
    policy,
    env,
    mus: list[float],
    rng_key: PRNGKey,
    max_steps: int = 1000,
    num_repeats: int = 16,
    episodes_per_trial: int = 0,
    reset_belief_between_episodes: bool = False,
) -> list[dict]:
    """Evaluate policy behavior repeatedly at each supplied task value."""
    mu_arr = jnp.array(mus, dtype=jnp.float32)
    batch = _run_mu_table_batch(
        vae,
        policy,
        env,
        mu_arr,
        rng_key,
        max_steps,
        num_repeats,
        len(mus),
        episodes_per_trial,
        reset_belief_between_episodes,
    )
    t_end = float(env.t_end)
    results = []
    for i, mu in enumerate(mus):
        ep_t = batch["ep_t_reached"][i]
        max_ep = ep_t.shape[1]
        mean_t_per_ep = [float(jnp.mean(ep_t[:, j])) for j in range(max_ep)]
        per_ep_success = [float(jnp.mean(ep_t[:, j] >= t_end)) for j in range(max_ep)]
        results.append(
            {
                "mu": mu,  #
                "steps": float(jnp.mean(batch["episode_length"][i])),  #
                "t_str": _format_t_per_episode(mean_t_per_ep),  #
                "success_str": _format_t_per_episode(per_ep_success, precision=0),  #
                "returns": [float(v) for v in batch["total_return"][i]],  #
                "accepted": float(jnp.mean(batch["accepted_steps"][i])),
                "rejected": float(jnp.mean(batch["rejected_steps"][i])),
            }
        )
    return results


@nnx.jit(static_argnames=("env", "max_steps", "num_samples"))
def _run_latent_scatter_batch(vae, policy, env, rng_key, max_steps: int, num_samples: int):
    """Roll out `num_samples` new tasks; return the final belief means and the true task values."""
    keys = jax.random.split(rng_key, num_samples)

    def run_one(key):
        key_task, key_ep = jax.random.split(key)
        params = env.sample_task(key_task)
        out = eval_episode(vae, policy, env, params, key_ep, max_steps=max_steps)
        return out["final_mu"], params.lam

    return jax.vmap(run_one)(keys)


def collect_latent_scatter(vae, policy, env, rng_key: PRNGKey, num_samples: int, max_steps: int):
    """Numpy-returning wrapper around `_run_latent_scatter_batch` (ODE envs only)."""
    final_mu, lam = _run_latent_scatter_batch(vae, policy, env, rng_key, max_steps, num_samples)
    return np.asarray(final_mu), np.asarray(lam)


def compute_pid_baseline_steps(
    env_config,
    mus: list[float],
    num_repeats: int = 16,
    max_steps: int = 1000,
    use_cache: bool = True,
    require_cache: bool = False,
    force_regenerate: bool = False,
    verbose: int = 0,
) -> dict[float, dict]:
    """Mean PID step counts per mu, cached under data/<system>/training/.

    Each result has "steps" (accepted + rejected), "accepted", "rejected" and
    "t_reached" (< t_end when the budget ran out).
    """
    import os

    from steppo.training.pid_solve import (
        PID_BASELINE_SEED,
        atomic_savez,
        cache_dir,
        env_pid_controller,
        fingerprint,
        pid_env_payload,
        solve_pid_batch,
    )

    path = None
    if require_cache and not use_cache:
        raise ValueError("require_cache=True requires use_cache=True")
    if use_cache:
        payload = {
            "kind": "pid_baseline_steps",
            "env_config": pid_env_payload(env_config),
            "pid_seed": PID_BASELINE_SEED,
            "mus": [float(m) for m in mus],
            "num_repeats": int(num_repeats),
            "max_steps": int(max_steps),
        }
        fp = fingerprint(payload)
        path = os.path.join(cache_dir("training", env_config.system), f"baseline_{fp}.npz")
        if not force_regenerate and os.path.isfile(path):
            if verbose > 0:
                print(f"[+] Loaded PID baseline cache: {path}")
            data = np.load(path)
            return {
                float(mu): {
                    "steps": float(data["steps"][i]),
                    "accepted": float(data["accepted"][i]),
                    "rejected": float(data["rejected"][i]),
                    "t_reached": float(data["t_reached"][i]),
                }
                for i, mu in enumerate(mus)
            }
        if require_cache:
            raise FileNotFoundError(
                f"Missing pre-generated PID training cache: {path}. "
                "Run bash scripts/generate-pid-datasets-all-ode.sh "
                f"{env_config.system} --force before training."
            )

    sc = env_pid_controller(env_config)

    rng = jax.random.PRNGKey(PID_BASELINE_SEED)
    result = {}
    for i, mu in enumerate(mus):
        keys = jax.random.split(rng, num_repeats)
        rng = jax.random.fold_in(rng, i)
        out = solve_pid_batch(
            env_config,
            sc,
            np.full(num_repeats, mu, dtype=np.float32),
            keys,
            max_steps,
            save_steps=True,
        )
        mean_acc = float(np.mean(out["accepted"]))
        mean_rej = float(np.mean(out["rejected"]))
        result[mu] = {
            "steps": mean_acc + mean_rej,
            "accepted": mean_acc,
            "rejected": mean_rej,
            "t_reached": float(np.mean(out["t_reached"])),
        }

    if path is not None:
        mus_list = list(mus)
        atomic_savez(
            path,
            steps=np.array([result[m]["steps"] for m in mus_list]),
            accepted=np.array([result[m]["accepted"] for m in mus_list]),
            rejected=np.array([result[m]["rejected"] for m in mus_list]),
            t_reached=np.array([result[m]["t_reached"] for m in mus_list]),
        )

    return result


def print_mu_table(
    results: list[dict], t_end: float, max_steps: int, pid_baseline: dict[float, dict] | None = None
):
    """Print a task-value comparison table from evaluation results."""
    has_pid = pid_baseline is not None and len(pid_baseline) > 0
    has_ar = bool(results) and "accepted" in results[0] and "rejected" in results[0]
    t_width = max(len(r["t_str"]) for r in results) if results else 6
    t_width = max(t_width, 1)
    s_width = max(len(r["success_str"]) for r in results) if results else 7
    s_width = max(s_width, 7)
    steps_col = "steps" if not has_pid else "steps (PID)"
    steps_width = max(len(steps_col), 12) if has_pid else 8
    ar_col = "accepted/rejected"
    pid_ar_col = "PID acc/rej"
    if has_ar:
        ar_strs = [f"{r['accepted']:.0f}/{r['rejected']:.0f}" for r in results]
        ar_width = max(len(ar_col), max(len(s) for s in ar_strs))
    has_pid_ar = has_pid and all("accepted" in v and "rejected" in v for v in pid_baseline.values())
    if has_pid_ar:
        pid_ar_strs = [
            f"{pid_baseline[r['mu']]['accepted']:.0f}/{pid_baseline[r['mu']]['rejected']:.0f}"
            for r in results
            if r["mu"] in pid_baseline
        ]
        pid_ar_width = max([len(pid_ar_col)] + [len(s) for s in pid_ar_strs])
    row_width = (
        steps_width
        + 1
        + t_width
        + 1
        + s_width
        + (ar_width + 1 if has_ar else 0)
        + (pid_ar_width + 1 if has_pid_ar else 0)
    )
    sep = "─" * 6 + "──┼" + "─" * (row_width + 2) + "┼"
    hdr = (
        f"{'mu':>6s}  │"
        f"{steps_col:>{steps_width}s} "
        + (f"{pid_ar_col:>{pid_ar_width}s} " if has_pid_ar else "")
        + (f"{ar_col:>{ar_width}s} " if has_ar else "")
        + f"{'t':>{t_width}s} {'success':>{s_width}s} │"
    )
    print(f"  Per-μ breakdown (budget={max_steps}, t_end={t_end}):")
    print(f"  {sep}")
    print(f"  {hdr}")
    print(f"  {sep}")
    for r in results:
        if has_pid and r["mu"] in pid_baseline:
            pid_steps = pid_baseline[r["mu"]]["steps"]
            steps_str = f"{r['steps']:.0f} ({pid_steps:.0f})"
        else:
            steps_str = f"{r['steps']:.0f}"
        mu_str = f"{r['mu']:.0f}" if r["mu"] >= 1 else f"{r['mu']:g}"
        ar_str = f"{r['accepted']:.0f}/{r['rejected']:.0f}" if has_ar else ""
        if has_pid_ar and r["mu"] in pid_baseline:
            pv = pid_baseline[r["mu"]]
            pid_ar_str = f"{pv['accepted']:.0f}/{pv['rejected']:.0f}"
        else:
            pid_ar_str = ""
        print(
            f"  {mu_str:>6s}  │"
            f"{steps_str:>{steps_width}s} "
            + (f"{pid_ar_str:>{pid_ar_width}s} " if has_pid_ar else "")
            + (f"{ar_str:>{ar_width}s} " if has_ar else "")
            + f"{r['t_str']:>{t_width}s} "
            f"{r['success_str']:>{s_width}s} │"
        )
    print(f"  {sep}")
