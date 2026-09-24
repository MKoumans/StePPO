"""Tests for steppo.training.vae_loss and steppo.training.vae_trainer."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from steppo.configs.base_config import EncoderArchConfig, VAEConfig
from steppo.models.vae import VariBADVAE
from steppo.training.vae_loss import (
    bernoulli_predictive_entropy,
    binary_cross_entropy_with_logits,
    elbo_loss,
    gaussian_nll,
    kl_divergence_sequential,
    kl_divergence_standard,
    reconstruction_loss_mse,
)
from steppo.training.vae_trainer import create_vae_train_state, vae_train_step
from steppo.utils.replay_buffer import TrajectoryBatch

STATE_DIM = 2
ACTION_DIM = 4
LATENT_DIM = 5
T = 10
B = 6


def _make_vae():
    config = VAEConfig(
        latent_dim=LATENT_DIM,
        encoder=EncoderArchConfig(hidden_size=64),
        rew_loss_coeff=1.0,
        state_loss_coeff=0.0,
        kl_weight=1e-2,
    )
    vae = VariBADVAE(STATE_DIM, ACTION_DIM, config, nnx.Rngs(0))
    return vae, config


def _make_batch(B=B, T=T):
    key = jax.random.PRNGKey(99)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return TrajectoryBatch(
        states=jax.random.normal(k1, (B, T, STATE_DIM)),
        actions=jax.random.normal(k2, (B, T, ACTION_DIM)),
        rewards=jax.random.normal(k3, (B, T, 1)),
        next_states=jax.random.normal(k4, (B, T, STATE_DIM)),
        masks=jnp.ones((B, T)),
        task_params=jnp.zeros((B, 0)),
        keep_steps=jnp.ones((B, T)),
    )


# ── KL divergence ────────────────────────────────────────────────────────────


def test_reconstruction_loss_helpers_match_closed_form_and_remain_stable():
    assert float(
        reconstruction_loss_mse(jnp.array([1.0, 3.0]), jnp.array([0.0, 1.0]))
    ) == pytest.approx(2.5)

    means = jnp.array([1.0, -1.0])
    logvars = jnp.array([0.0, jnp.log(4.0)])
    targets = jnp.array([0.0, 1.0])
    expected = np.mean(
        0.5
        * (
            (np.array([1.0, 4.0]) * np.exp(-np.array([0.0, np.log(4.0)])))
            + np.array([0.0, np.log(4.0)])
        )
    )
    assert float(gaussian_nll(means, logvars, targets)) == pytest.approx(expected)

    extreme_nll = gaussian_nll(jnp.zeros(2), jnp.array([-100.0, 100.0]), jnp.zeros(2))
    assert jnp.isfinite(extreme_nll)


def test_binary_decoder_losses_are_stable_for_extreme_logits():
    logits = jnp.array([-1000.0, 1000.0])
    matching_targets = jnp.array([0.0, 1.0])
    opposing_targets = 1.0 - matching_targets

    matching_loss = binary_cross_entropy_with_logits(logits, matching_targets)
    opposing_loss = binary_cross_entropy_with_logits(logits, opposing_targets)
    entropy = bernoulli_predictive_entropy(jnp.array([0.0, 100.0, -100.0]))

    assert jnp.all(jnp.isfinite(matching_loss))
    assert jnp.allclose(matching_loss, 0.0, atol=1e-6)
    assert jnp.all(jnp.isfinite(opposing_loss))
    assert jnp.allclose(opposing_loss, jnp.array([1000.0, 1000.0]))
    assert jnp.isfinite(entropy)
    assert 0.0 < float(entropy) <= np.log(2.0)


def test_elbo_masks_inactive_transitions_and_weights_heteroscedastic_losses():
    class _ToyVAE(nnx.Module):
        def __init__(self):
            self.shift = nnx.Param(jnp.array(0.0))

        def encode(self, actions, states, rewards, task_params):
            del actions, rewards, task_params
            steps = states.shape[0]
            mus = jnp.zeros((steps + 1, 1)) + self.shift[...]
            return mus, jnp.zeros_like(mus)

        @staticmethod
        def sample_z(mu, logvar, key):
            del logvar, key
            return mu

        def decode(self, z, state, action):
            del z, state, action
            return {
                "reward": self.shift[...],
                "reward_logvar": jnp.array(0.0),
                "state": None,
                "state_logvar": None,
                "task": None,
                "task_logvar": None,
                "accept_logit": jnp.array(0.0),
                "end_reward": None,
            }

    trajectory = TrajectoryBatch(
        states=jnp.zeros((1, 2, 1)),
        actions=jnp.zeros((1, 2, 1)),
        rewards=jnp.array([[[2.0], [100.0]]]),
        next_states=jnp.zeros((1, 2, 1)),
        masks=jnp.array([[1.0, 0.0]]),
        task_params=jnp.zeros((1, 0)),
        keep_steps=jnp.array([[1.0, 0.0]]),
    )
    config = VAEConfig(
        latent_dim=1,
        rew_loss_coeff=1.0,
        accept_loss_coeff=1.0,
        heteroscedastic_recon=True,
    )

    loss, metrics = elbo_loss(
        _ToyVAE(),
        trajectory,
        jax.random.PRNGKey(8),
        config,
    )

    assert metrics["rew_loss"] == pytest.approx(2.0)
    assert metrics["accept_loss"] == pytest.approx(np.log(2.0))
    assert metrics["rew_pred_var"] == pytest.approx(1.0)
    assert metrics["total_loss"] == pytest.approx(2.0 + np.log(2.0))
    assert loss == pytest.approx(metrics["total_loss"])


@pytest.mark.parametrize("free_bits", [0.0, 0.15])
def test_kl_standard_matches_per_dimension_closed_form(free_bits):
    mu = np.array([[0.0, 0.0], [1.0, -2.0]], dtype=np.float32)
    logvar = np.array([[0.0, 0.0], [np.log(2.0), np.log(0.5)]], dtype=np.float32)
    per_dim = -0.5 * (1.0 + logvar - mu**2 - np.exp(logvar))
    expected = np.maximum(per_dim, free_bits).sum(axis=-1)

    actual = kl_divergence_standard(jnp.asarray(mu), jnp.asarray(logvar), free_bits)

    assert np.allclose(actual, expected, atol=1e-6)


def test_kl_sequential_matches_closed_form_and_anchor_mixture():
    mu = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32)
    logvar = np.array([[0.0, np.log(2.0)], [np.log(0.5), 0.0]], dtype=np.float32)
    mu_prev = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    logvar_prev = np.array([[0.0, 0.0], [0.0, np.log(2.0)]], dtype=np.float32)
    free_bits = 0.1
    anchor_weight = 0.25

    seq_per_dim = 0.5 * (
        logvar_prev
        - logvar
        - 1.0
        + np.exp(logvar) / np.exp(logvar_prev)
        + (mu - mu_prev) ** 2 / np.exp(logvar_prev)
    )
    seq_kl = np.maximum(seq_per_dim, free_bits).sum(axis=-1)
    anchor_per_dim = -0.5 * (1.0 + logvar - mu**2 - np.exp(logvar))
    anchor_kl = np.maximum(anchor_per_dim, free_bits).sum(axis=-1)
    expected = (1.0 - anchor_weight) * seq_kl + anchor_weight * anchor_kl

    actual = kl_divergence_sequential(
        jnp.asarray(mu),
        jnp.asarray(logvar),
        jnp.asarray(mu_prev),
        jnp.asarray(logvar_prev),
        free_bits,
        anchor_weight,
    )

    assert np.allclose(actual, expected, atol=1e-6)


def test_kl_divergences_clip_extreme_log_variances_before_exponentiation():
    mu = jnp.array([[0.0, 1.0]])
    logvar = jnp.array([[-100.0, 100.0]])
    previous_mu = jnp.zeros_like(mu)
    previous_logvar = jnp.zeros_like(logvar)

    standard = kl_divergence_standard(mu, logvar)
    sequential = kl_divergence_sequential(mu, logvar, previous_mu, previous_logvar)

    assert jnp.all(jnp.isfinite(standard))
    assert jnp.all(jnp.isfinite(sequential))


# ── ELBO loss ────────────────────────────────────────────────────────────────


def test_elbo_loss_jit_compilable():
    vae, config = _make_vae()
    batch = _make_batch()
    key = jax.random.PRNGKey(0)

    graphdef, params = nnx.split(vae)

    def run(params):
        vae_ = nnx.merge(graphdef, params)
        return elbo_loss(vae_, batch, key, config)

    jit_fn = jax.jit(run)
    loss, metrics = jit_fn(params)
    assert jnp.isfinite(loss)


def test_elbo_loss_finite_on_random_data():
    vae, config = _make_vae()
    batch = _make_batch()
    key = jax.random.PRNGKey(1)
    loss, metrics = elbo_loss(vae, batch, key, config)
    assert jnp.isfinite(loss)
    for v in metrics.values():
        assert jnp.isfinite(v)


def test_elbo_metrics_keys():
    vae, config = _make_vae()
    batch = _make_batch()
    key = jax.random.PRNGKey(2)
    _, metrics = elbo_loss(vae, batch, key, config)
    assert "kl_loss" in metrics
    assert "rew_loss" in metrics
    assert "total_loss" in metrics


def test_masks_zero_out_loss():
    """Setting all masks to 0 should produce zero reconstruction loss."""
    vae, config = _make_vae()
    key = jax.random.PRNGKey(7)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    batch_zero_mask = TrajectoryBatch(
        states=jax.random.normal(k1, (B, T, STATE_DIM)),
        actions=jax.random.normal(k2, (B, T, ACTION_DIM)),
        rewards=jax.random.normal(k3, (B, T, 1)),
        next_states=jax.random.normal(k4, (B, T, STATE_DIM)),
        masks=jnp.zeros((B, T)),
        task_params=jnp.zeros((B, 0)),
        keep_steps=jnp.ones((B, T)),
    )
    loss, metrics = elbo_loss(vae, batch_zero_mask, key, config)
    assert float(metrics["rew_loss"]) == pytest.approx(0.0, abs=1e-5)


# ── VAE train step ───────────────────────────────────────────────────────────


def test_train_step_loss_decreases():
    vae, config = _make_vae()
    train_state = create_vae_train_state(vae, config)
    batch = _make_batch()
    key = jax.random.PRNGKey(3)

    losses = []
    for i in range(100):
        k = jax.random.fold_in(key, i)
        train_state, metrics = vae_train_step(train_state, batch, k, config)
        losses.append(float(metrics["total_loss"]))

    assert sum(losses[:10]) > sum(losses[-10:]), "Loss should decrease over 100 steps"


def test_train_step_finite():
    vae, config = _make_vae()
    train_state = create_vae_train_state(vae, config)
    batch = _make_batch()
    key = jax.random.PRNGKey(4)
    new_state, metrics = vae_train_step(train_state, batch, key, config)
    assert jnp.isfinite(metrics["total_loss"])


def test_gradient_finite():
    """Gradients should be finite (precondition for clipping to work)."""
    vae, config = _make_vae()
    batch = _make_batch()
    key = jax.random.PRNGKey(5)

    graphdef, params = nnx.split(vae)

    def loss_fn(params):
        vae_ = nnx.merge(graphdef, params)
        loss, _ = elbo_loss(vae_, batch, key, config)
        return loss

    grads = jax.grad(loss_fn)(params)
    flat_grads = jax.tree.leaves(grads)
    total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in flat_grads))
    assert jnp.isfinite(total_norm)


def test_ode_vae_loss_finite_after_rollout():
    """VAE ELBO on ODE trajectories must stay finite (no NaN poisoning)."""
    from steppo.configs.base_config import ODEEnvConfig, PPOConfig, TrainConfig
    from steppo.envs.ode import ODEEnv
    from steppo.models.policy import ActorCritic
    from steppo.training.rollout import collect_rollout, init_rollout_state
    from steppo.utils.replay_buffer import get_trajectory_batch

    # Keep this as a small ODE -> rollout -> ELBO integration check. The
    # separate stiff-rollout tests cover Van der Pol and float64 solver behavior.
    env_cfg = ODEEnvConfig(
        system="scalar_decay",
        precision="float32",
        t_end=1.0,
        dt0=1e-2,
        lam_min=2.0,
        lam_max=2.0,
        sample_lam=False,
    )
    num_envs, rollout_steps = 4, 16
    env = ODEEnv(env_cfg, max_steps=rollout_steps)
    obs_dim = env.obs_shape()[0]
    action_dim = env.num_actions

    config = TrainConfig(
        vae=VAEConfig(
            latent_dim=5,
            encoder=EncoderArchConfig(hidden_size=64),
            state_loss_coeff=0.5,
            task_loss_coeff=0.05,
        ),
        ppo=PPOConfig(),
        env=env_cfg,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
    )
    rngs = nnx.Rngs(42)
    vae = VariBADVAE(obs_dim, action_dim, config.vae, rngs, task_dim=1)
    policy = ActorCritic(
        obs_dim,
        action_dim,
        5,
        action_space="continuous",
        use_latent_sample=False,
        rngs=nnx.Rngs(43),
    )

    key = jax.random.PRNGKey(77)
    task_keys = jax.random.split(key, num_envs)
    env_params_batch = jax.vmap(env.sample_task)(task_keys)

    rs = init_rollout_state(vae, env, env_params_batch, num_envs, key)
    _, batch = collect_rollout(
        rs, vae, policy, env, env_params_batch, config, jax.random.PRNGKey(78)
    )

    traj = get_trajectory_batch(batch, policy.action_dim)
    loss, metrics = elbo_loss(vae, traj, jax.random.PRNGKey(79), config.vae)

    assert jnp.isfinite(loss), f"ELBO loss is {loss}"
    for k, v in metrics.items():
        assert jnp.isfinite(v), f"metric {k} is {v}"
