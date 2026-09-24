"""Tests for steppo.training.ppo (GAE, ppo_loss) and steppo.training.ppo_trainer."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from steppo.configs.base_config import PPOConfig
from steppo.models.policy import ActorCritic
from steppo.training.ppo import PPOBatch, compute_gae, ppo_loss
from steppo.training.ppo_trainer import create_ppo_train_state, ppo_train_step

STATE_DIM = 2
ACTION_DIM = 4
LATENT_DIM = 5
Z_DIM = LATENT_DIM * 2
T = 10
NUM_ENVS = 4


def _make_discrete_policy():
    return ActorCritic(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        action_space="discrete",
        use_latent_sample=False,
        rngs=nnx.Rngs(0),
    )


def _make_ppo_batch(B=32):
    key = jax.random.PRNGKey(42)
    ks = jax.random.split(key, 6)
    return PPOBatch(
        states=jax.random.normal(ks[0], (B, STATE_DIM)),
        z=jax.random.normal(ks[1], (B, Z_DIM)),
        task=jnp.zeros((B, 0)),
        actions=jax.random.randint(ks[2], (B,), 0, ACTION_DIM),
        log_probs=jax.random.normal(ks[3], (B,)),
        advantages=jax.random.normal(ks[4], (B,)),
        returns=jax.random.normal(ks[5], (B,)),
    )


# ── GAE ──────────────────────────────────────────────────────────────────────


def test_compute_gae_shapes():
    rewards = jnp.ones((T, NUM_ENVS))
    values = jnp.ones((T + 1, NUM_ENVS))
    dones = jnp.zeros((T, NUM_ENVS))
    adv, ret = compute_gae(rewards, values, dones, gamma=0.95, gae_lambda=0.95)
    assert adv.shape == (T, NUM_ENVS)
    assert ret.shape == (T, NUM_ENVS)


def test_compute_gae_zero_reward_constant_value():
    rewards = jnp.zeros((T, NUM_ENVS))
    values = jnp.ones((T + 1, NUM_ENVS))
    dones = jnp.zeros((T, NUM_ENVS))
    adv, _ = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=1.0)
    assert jnp.allclose(adv, jnp.zeros_like(adv), atol=1e-5)


# ── PPO loss ─────────────────────────────────────────────────────────────────


def test_ppo_loss_finite():
    policy = _make_discrete_policy()
    config = PPOConfig()
    batch = _make_ppo_batch(B=32)
    key = jax.random.PRNGKey(1)
    loss, metrics = ppo_loss(policy, batch, config, key)
    assert jnp.isfinite(loss)


# ── PPO train step ───────────────────────────────────────────────────────────


def test_ppo_loss_decreases():
    policy = _make_discrete_policy()
    config = PPOConfig(lr=1e-3)
    train_state = create_ppo_train_state(policy, config)
    batch = _make_ppo_batch(B=64)
    key = jax.random.PRNGKey(0)

    losses = []
    for i in range(50):
        k = jax.random.fold_in(key, i)
        train_state, metrics = ppo_train_step(train_state, batch, config, k)
        losses.append(float(metrics["total_loss"]))

    assert sum(losses[:10]) > sum(losses[-10:]), "PPO loss should decrease over 50 steps"


def test_gradient_clipping_ppo():
    """PPO gradients should be finite after clipping."""
    policy = _make_discrete_policy()
    config = PPOConfig()
    train_state = create_ppo_train_state(policy, config)
    batch = _make_ppo_batch()
    key = jax.random.PRNGKey(2)
    new_state, metrics = ppo_train_step(train_state, batch, config, key)
    assert jnp.isfinite(metrics["total_loss"])
