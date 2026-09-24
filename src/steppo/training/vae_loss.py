"""ELBO and reconstruction losses for the variational belief model."""

from typing import Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.configs.base_config import TrainingConfig, VAEConfig
from steppo.utils.replay_buffer import TrajectoryBatch

_DEFAULT_TRAINING_CONFIG = TrainingConfig()

_LOG_VAR_MIN = -10.0
_LOG_VAR_MAX = 10.0


def kl_divergence_standard(mu: Array, logvar: Array, free_bits: float = 0.0) -> Array:
    """Return the per-sample KL divergence to a standard normal prior."""
    logvar = jnp.clip(logvar, _LOG_VAR_MIN, _LOG_VAR_MAX)
    kl_per_dim = -0.5 * (1.0 + logvar - jnp.square(mu) - jnp.exp(logvar))
    return jnp.sum(jnp.maximum(kl_per_dim, free_bits), axis=-1)


def kl_divergence_sequential(
    mu_t: Array,
    logvar_t: Array,
    mu_prev: Array,
    logvar_prev: Array,
    free_bits: float = 0.0,
    anchor_weight: float = 0.0,
) -> Array:
    """Return the KL divergence between consecutive latent posteriors."""
    logvar_t = jnp.clip(logvar_t, _LOG_VAR_MIN, _LOG_VAR_MAX)
    logvar_prev = jnp.clip(logvar_prev, _LOG_VAR_MIN, _LOG_VAR_MAX)
    var_t = jnp.exp(logvar_t)
    var_prev = jnp.exp(logvar_prev)

    kl_per_dim = 0.5 * (
        logvar_prev - logvar_t - 1.0 + var_t / var_prev + jnp.square(mu_t - mu_prev) / var_prev
    )
    seq_kl = jnp.sum(jnp.maximum(kl_per_dim, free_bits), axis=-1)
    if anchor_weight <= 0.0:
        return seq_kl
    anchor_kl = kl_divergence_standard(mu_t, logvar_t, free_bits=free_bits)
    return (1.0 - anchor_weight) * seq_kl + anchor_weight * anchor_kl


def reconstruction_loss_mse(pred: Array, target: Array) -> Array:
    """Return the mean squared reconstruction error."""
    return jnp.mean(jnp.square(pred - target))


def bernoulli_predictive_entropy(logit: Array) -> Array:
    """Return the mean Bernoulli entropy implied by acceptance logits."""
    p = jax.nn.sigmoid(logit)
    p = jnp.clip(p, 1e-7, 1.0 - 1e-7)
    return jnp.mean(-p * jnp.log(p) - (1.0 - p) * jnp.log1p(-p))


def gaussian_nll(pred_mean: Array, pred_logvar: Array, target: Array) -> Array:
    """Heteroscedastic Gaussian NLL."""
    logvar = jnp.clip(pred_logvar, _LOG_VAR_MIN, _LOG_VAR_MAX)
    resid_sq = jnp.square(pred_mean - target)
    return jnp.mean(0.5 * (resid_sq * jnp.exp(-logvar) + logvar))


def binary_cross_entropy_with_logits(logit: Array, target: Array) -> Array:
    """Return numerically stable binary cross-entropy from logits."""
    return jnp.maximum(logit, 0.0) - logit * target + jnp.log1p(jnp.exp(-jnp.abs(logit)))


def _elbo_single(
    vae,
    states,
    actions,
    rewards,
    next_states,
    masks,
    rng_key,
    config: VAEConfig,
    task_params,
    keep_steps,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    """Compute masked loss components and diagnostics for one trajectory."""
    mus, logvars = vae.encode(actions, states, rewards, task_params)

    logvars_full = jnp.clip(logvars[1:], _LOG_VAR_MIN, _LOG_VAR_MAX)
    var_sum_per_step = jnp.sum(jnp.exp(logvars_full), axis=-1)
    n_valid_full = jnp.maximum(jnp.sum(masks), 1.0)
    total_var_sum = jnp.sum(var_sum_per_step * masks) / n_valid_full

    T = states.shape[0]
    key_sub, key_z = jax.random.split(rng_key)
    keys = jax.random.split(key_z, T)

    rewards_flat = rewards.squeeze(-1)
    return_to_go = jnp.flip(jnp.cumsum(jnp.flip(rewards_flat, axis=0)), axis=0)

    all_xs = (
        mus[1:],
        logvars[1:],
        mus[:-1],
        logvars[:-1],
        states,
        actions,
        rewards,
        next_states,
        masks,
        keys,
        keep_steps,
        return_to_go,
    )

    S = config.subsample_len
    if S > 0 and S < T:
        start = jax.random.randint(key_sub, (), 0, T - S)
        all_xs = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, S, axis=0), all_xs)

    def scan_fn(carry, xs):
        (
            mu_t,
            logvar_t,
            mu_prev,
            logvar_prev,
            state,
            action,
            reward,
            next_state,
            mask,
            key,
            keep,
            rtg,
        ) = xs
        z = vae.sample_z(mu_t, logvar_t, key)
        decoded = vae.decode(z, state, action)

        if config.heteroscedastic_recon:
            rew_loss = gaussian_nll(decoded["reward"], decoded["reward_logvar"], reward.squeeze(-1))
            rew_pred_var = jnp.mean(
                jnp.exp(jnp.clip(decoded["reward_logvar"], _LOG_VAR_MIN, _LOG_VAR_MAX))
            )
        else:
            rew_loss = reconstruction_loss_mse(decoded["reward"], reward.squeeze(-1))
            rew_pred_var = jnp.zeros(())
        if decoded["state"] is not None:
            if config.heteroscedastic_recon:
                state_loss = gaussian_nll(decoded["state"], decoded["state_logvar"], next_state)
                state_pred_var = jnp.mean(
                    jnp.exp(jnp.clip(decoded["state_logvar"], _LOG_VAR_MIN, _LOG_VAR_MAX))
                )
            else:
                state_loss = reconstruction_loss_mse(decoded["state"], next_state)
                state_pred_var = jnp.zeros(())
        else:
            state_loss = jnp.zeros(())
            state_pred_var = jnp.zeros(())
        if decoded["task"] is not None:
            if config.heteroscedastic_recon:
                task_loss = gaussian_nll(decoded["task"], decoded["task_logvar"], task_params)
                task_pred_var = jnp.mean(
                    jnp.exp(jnp.clip(decoded["task_logvar"], _LOG_VAR_MIN, _LOG_VAR_MAX))
                )
            else:
                task_loss = reconstruction_loss_mse(decoded["task"], task_params)
                task_pred_var = jnp.zeros(())
        else:
            task_loss = jnp.zeros(())
            task_pred_var = jnp.zeros(())
        if decoded["accept_logit"] is not None:
            accept_loss = binary_cross_entropy_with_logits(decoded["accept_logit"], keep)
            accept_pred_entropy = bernoulli_predictive_entropy(decoded["accept_logit"])
        else:
            accept_loss = jnp.zeros(())
            accept_pred_entropy = jnp.zeros(())
        end_reward_loss = (
            reconstruction_loss_mse(decoded["end_reward"].squeeze(-1), rtg)
            if decoded["end_reward"] is not None
            else jnp.zeros(())
        )

        if config.sequential_kl_propagate_gradient:
            mu_prev_kl, logvar_prev_kl = mu_prev, logvar_prev
        else:
            mu_prev_kl = jax.lax.stop_gradient(mu_prev)
            logvar_prev_kl = jax.lax.stop_gradient(logvar_prev)
        if config.deterministic_latent:
            kl = jnp.zeros(())
        else:
            kl = jax.lax.cond(
                config.sequential_kl,
                lambda: kl_divergence_sequential(
                    mu_t,
                    logvar_t,
                    mu_prev_kl,
                    logvar_prev_kl,
                    free_bits=config.kl_free_bits,
                    anchor_weight=config.sequential_kl_anchor_weight,
                ),
                lambda: kl_divergence_standard(mu_t, logvar_t, free_bits=config.kl_free_bits),
            )

        return None, (
            rew_loss * mask,
            state_loss * mask,
            kl * mask,
            task_loss * mask,
            accept_loss * mask,
            end_reward_loss * mask,
            rew_pred_var * mask,
            state_pred_var * mask,
            task_pred_var * mask,
            accept_pred_entropy * mask,
        )

    (
        _,
        (
            rew_losses,
            state_losses,
            kl_losses,
            task_losses,
            accept_losses,
            end_reward_losses,
            rew_pred_vars,
            state_pred_vars,
            task_pred_vars,
            accept_pred_entropies,
        ),
    ) = jax.lax.scan(scan_fn, None, all_xs)

    window_masks = all_xs[8]
    n_valid = jnp.maximum(jnp.sum(window_masks), 1.0)
    total_rew = jnp.sum(rew_losses) / n_valid
    total_state = jnp.sum(state_losses) / n_valid
    total_kl = jnp.sum(kl_losses) / n_valid
    total_task = jnp.sum(task_losses) / n_valid
    total_accept = jnp.sum(accept_losses) / n_valid
    total_end_reward = jnp.sum(end_reward_losses) / n_valid
    total_rew_pred_var = jnp.sum(rew_pred_vars) / n_valid
    total_state_pred_var = jnp.sum(state_pred_vars) / n_valid
    total_task_pred_var = jnp.sum(task_pred_vars) / n_valid
    total_accept_pred_entropy = jnp.sum(accept_pred_entropies) / n_valid

    return (
        total_rew,
        total_state,
        total_kl,
        total_task,
        total_accept,
        total_end_reward,
        total_var_sum,
        total_rew_pred_var,
        total_state_pred_var,
        total_task_pred_var,
        total_accept_pred_entropy,
    )


def elbo_loss(
    vae,
    trajectories: TrajectoryBatch,
    rng_key: PRNGKey,
    config: VAEConfig,
    training_config: Optional[TrainingConfig] = None,
    ema: Optional[Tuple[Array, Array, Array, Array, Array, Array]] = None,
    kl_anneal_factor: float = 1.0,
):
    """Compute the batch ELBO and return it with unweighted diagnostics."""
    if training_config is None:
        training_config = _DEFAULT_TRAINING_CONFIG
    B = trajectories.states.shape[0]
    keys = jax.random.split(rng_key, B)

    graphdef, params = nnx.split(vae)

    def single(states, actions, rewards, next_states, masks, key, task_params, keep_steps):
        vae_ = nnx.merge(graphdef, params)
        return _elbo_single(
            vae_,
            states,
            actions,
            rewards,
            next_states,
            masks,
            key,
            config,
            task_params,
            keep_steps,
        )

    (
        rew_arr,
        state_arr,
        kl_arr,
        task_arr,
        accept_arr,
        end_reward_arr,
        var_sum_arr,
        rew_pred_var_arr,
        state_pred_var_arr,
        task_pred_var_arr,
        accept_pred_entropy_arr,
    ) = jax.vmap(single)(
        trajectories.states,
        trajectories.actions,
        trajectories.rewards,
        trajectories.next_states,
        trajectories.masks,
        keys,
        trajectories.task_params,
        trajectories.keep_steps,
    )

    mean_rew = jnp.mean(rew_arr)
    mean_state = jnp.mean(state_arr)
    mean_kl = jnp.mean(kl_arr)
    mean_task = jnp.mean(task_arr)
    mean_accept = jnp.mean(accept_arr)
    mean_end_reward = jnp.mean(end_reward_arr)
    mean_var_sum = jnp.mean(var_sum_arr)
    mean_rew_pred_var = jnp.mean(rew_pred_var_arr)
    mean_state_pred_var = jnp.mean(state_pred_var_arr)
    mean_task_pred_var = jnp.mean(task_pred_var_arr)
    mean_accept_pred_entropy = jnp.mean(accept_pred_entropy_arr)

    if training_config.ema_norm and ema is not None:
        ema_rew, ema_state, ema_kl, ema_task, ema_accept, ema_end_reward = ema
        norm_rew = mean_rew / jnp.maximum(jnp.abs(ema_rew), jnp.maximum(jnp.abs(mean_rew), 1e-8))
        norm_state = mean_state / jnp.maximum(
            jnp.abs(ema_state), jnp.maximum(jnp.abs(mean_state), 1e-8)
        )
        norm_task = mean_task / jnp.maximum(
            jnp.abs(ema_task), jnp.maximum(jnp.abs(mean_task), 1e-8)
        )
        norm_accept = mean_accept / jnp.maximum(
            jnp.abs(ema_accept), jnp.maximum(jnp.abs(mean_accept), 1e-8)
        )
        norm_end_reward = mean_end_reward / jnp.maximum(
            jnp.abs(ema_end_reward), jnp.maximum(jnp.abs(mean_end_reward), 1e-8)
        )
    else:
        norm_rew, norm_state, norm_task, norm_accept, norm_end_reward = (
            mean_rew,
            mean_state,
            mean_task,
            mean_accept,
            mean_end_reward,
        )
    norm_kl = mean_kl

    effective_kl_weight = config.kl_weight * kl_anneal_factor

    mean_loss = (
        config.rew_loss_coeff * norm_rew
        + config.state_loss_coeff * norm_state
        + effective_kl_weight * norm_kl
        + config.task_loss_coeff * norm_task
        + config.accept_loss_coeff * norm_accept
        + config.end_reward_loss_coeff * norm_end_reward
    )

    weighted_rew = config.rew_loss_coeff * norm_rew
    weighted_state = config.state_loss_coeff * norm_state
    weighted_kl = effective_kl_weight * norm_kl
    weighted_task = config.task_loss_coeff * norm_task
    weighted_accept = config.accept_loss_coeff * norm_accept
    weighted_end_reward = config.end_reward_loss_coeff * norm_end_reward

    metrics = {
        "rew_loss": mean_rew,
        "state_loss": mean_state,
        "kl_loss": mean_kl,
        "task_loss": mean_task,
        "accept_loss": mean_accept,
        "end_reward_loss": mean_end_reward,
        "w_rew_loss": weighted_rew,
        "w_state_loss": weighted_state,
        "w_kl_loss": weighted_kl,
        "w_task_loss": weighted_task,
        "w_accept_loss": weighted_accept,
        "w_end_reward_loss": weighted_end_reward,
        "total_loss": mean_loss,
        "kl_weight_effective": effective_kl_weight,
        "latent_var_sum": mean_var_sum,
        "rew_pred_var": mean_rew_pred_var,
        "state_pred_var": mean_state_pred_var,
        "task_pred_var": mean_task_pred_var,
        "accept_pred_entropy": mean_accept_pred_entropy,
    }
    return mean_loss, metrics
