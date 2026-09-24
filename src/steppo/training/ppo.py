"""Pure PPO data structures and loss calculations."""

from typing import Optional, Tuple

import distrax
import flax.struct
import jax
import jax.numpy as jnp
from jax import Array

from steppo.configs.base_config import PPOConfig, TrainingConfig


@flax.struct.dataclass
class PPOBatch:
    """Mini-batch for one PPO update step."""

    states: Array  # (B, state_dim)
    z: Array  # (B, z_dim)
    task: Array  # (B, task_dim) — true task parameter; (B, 0) when unused
    actions: Array  # (B,) discrete or (B, action_dim) continuous
    log_probs: Array  # (B,)
    advantages: Array  # (B,)
    returns: Array  # (B,)


def compute_gae(
    rewards: Array,  # (T, num_envs)
    values: Array,  # (T+1, num_envs)  — last entry = bootstrap value
    dones: Array,  # (T, num_envs)
    gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    """Compute GAE advantages and value targets for a rollout."""

    def scan_fn(carry, xs):
        reward, value, next_value, done = xs
        delta = reward + gamma * next_value * (1.0 - done) - value
        adv = delta + gamma * gae_lambda * (1.0 - done) * carry
        return adv, adv

    # Scan in reverse: xs = (r_T-1, ..., r_0)
    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.zeros(rewards.shape[1:]),
        (rewards[::-1], values[:-1][::-1], values[1:][::-1], dones[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values[:-1]
    return advantages, returns


def ppo_loss(
    policy,
    batch: PPOBatch,
    config: PPOConfig,
    rng_key,
    training_config: Optional[TrainingConfig] = None,
    ema: Optional[Tuple[Array, Array]] = None,
) -> tuple[Array, dict]:
    """Compute the clipped actor, value, entropy, and KL metrics."""
    if training_config is None:
        training_config = TrainingConfig()

    dist, values = policy(batch.states, batch.z, batch.task)

    actions = batch.actions  #
    if isinstance(dist, distrax.Transformed):  #
        actions = jnp.clip(actions, -1.0 + 1e-6, 1.0 - 1e-6)  #
    log_probs_new = dist.log_prob(actions)  #

    adv = batch.advantages  #
    adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)  #

    log_ratio = jnp.clip(log_probs_new - batch.log_probs, -10.0, 10.0)
    ratio = jnp.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
    actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    # Approximate KL(old || new) between the pre-update (rollout) policy and the
    # policy at the params this loss was evaluated on — the min() clip above only
    # bounds the objective, not the actual divergence (Engstrom & Ilyas et al.,
    # ICLR 2020). Checked by the caller against config.target_kl to early-stop a
    # PPO epoch once updates have moved too far from the data they were computed on.
    approx_kl = jnp.mean(batch.log_probs - log_probs_new)

    value_loss = jnp.mean(jnp.square(values - batch.returns))

    # Analytical entropy ensures a clean gradient on log_std for Tanh-squashed Gaussians.
    if isinstance(dist, distrax.Transformed):
        entropy = dist.distribution.entropy()
    else:
        entropy = dist.entropy()
    entropy_loss = -jnp.mean(entropy)

    if training_config.ema_norm and ema is not None:
        _, ema_value = ema
        C = training_config.ema_clamp
        ema_v = jnp.maximum(ema_value, 1e-8)
        norm_actor = actor_loss
        norm_value = jnp.clip(value_loss, -C * ema_v, C * ema_v) / ema_v
    else:
        norm_actor, norm_value = actor_loss, value_loss

    entropy_coeff = jnp.float32(config.entropy_coeff)

    # https://arxiv.org/pdf/1707.06347
    total = norm_actor + config.value_coeff * norm_value + entropy_coeff * entropy_loss

    metrics = {
        "actor_loss": actor_loss,  #
        "value_loss": value_loss,  #
        "entropy": -entropy_loss,  #
        "entropy_coeff": entropy_coeff,  #
        "total_loss": total,  #
        "norm_actor_loss": norm_actor,  #
        "norm_value_loss": config.value_coeff * norm_value,  #
        "norm_entropy_loss": entropy_coeff * entropy_loss,  #
        "approx_kl": approx_kl,  #
    }

    return total, metrics
