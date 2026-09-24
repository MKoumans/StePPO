"""Tests for steppo.training.rollout."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from steppo.configs.base_config import (
    EncoderArchConfig,
    ODEEnvConfig,
    PPOConfig,
    TrainConfig,
    VAEConfig,
)
from steppo.envs.ode import ODEEnv, ODEParams
from steppo.models.policy import ActorCritic
from steppo.models.vae import VariBADVAE
from steppo.training.rollout import collect_rollout, init_rollout_state

NUM_ENVS = 4
T_STEPS = 20
STATE_DIM = 10  # default obs_features=("state","step_context","solver_trend") -> 1+5+4
ACTION_DIM = 1
LATENT_DIM = 5


def _make_components():
    config = TrainConfig(
        vae=VAEConfig(latent_dim=LATENT_DIM, encoder=EncoderArchConfig(hidden_size=64)),
        ppo=PPOConfig(),
        env=ODEEnvConfig(system="scalar_decay", train_bins=((1.0, 100.0),)),
        num_envs=NUM_ENVS,
        rollout_steps=T_STEPS,
    )
    rngs = nnx.Rngs(0)
    vae = VariBADVAE(STATE_DIM, ACTION_DIM, config.vae, rngs)
    policy = ActorCritic(
        STATE_DIM,
        ACTION_DIM,
        LATENT_DIM,
        action_space="continuous",
        use_latent_sample=False,
        rngs=nnx.Rngs(1),
    )
    env = ODEEnv(config.env, max_steps=T_STEPS)
    params = ODEParams(lam=10.0)
    return vae, policy, env, params, config


def test_init_rollout_state():
    vae, policy, env, params, config = _make_components()
    key = jax.random.PRNGKey(0)
    rs = init_rollout_state(vae, env, params, NUM_ENVS, key)
    assert rs.obs.shape == (NUM_ENVS, STATE_DIM)
    assert rs.gru_hidden.shape == (NUM_ENVS, 64)
    assert rs.belief_mu.shape == (NUM_ENVS, LATENT_DIM)
    assert rs.belief_logvar.shape == (NUM_ENVS, LATENT_DIM)


def test_init_rollout_prior_belief():
    """Initial belief should equal encoder.prior()."""
    vae, policy, env, params, config = _make_components()
    key = jax.random.PRNGKey(0)
    rs = init_rollout_state(vae, env, params, NUM_ENVS, key)
    prior_mu, prior_logvar = vae.get_prior()
    assert jnp.allclose(rs.belief_mu[0], prior_mu, atol=1e-6)
    assert jnp.allclose(rs.belief_logvar[0], prior_logvar, atol=1e-6)


def test_init_rollout_gru_zeros():
    """GRU hidden should be zeros at trial start."""
    vae, policy, env, params, config = _make_components()
    key = jax.random.PRNGKey(0)
    rs = init_rollout_state(vae, env, params, NUM_ENVS, key)
    assert jnp.allclose(rs.gru_hidden, jnp.zeros_like(rs.gru_hidden))


def test_collect_rollout_batch_shapes():
    vae, policy, env, params, config = _make_components()
    key = jax.random.PRNGKey(0)
    rs = init_rollout_state(vae, env, params, NUM_ENVS, key)
    new_rs, batch = collect_rollout(rs, vae, policy, env, params, config, jax.random.PRNGKey(1))

    assert batch.states.shape == (T_STEPS, NUM_ENVS, STATE_DIM)
    assert batch.actions.shape == (T_STEPS, NUM_ENVS, ACTION_DIM)  # continuous action space
    assert batch.rewards.shape == (T_STEPS, NUM_ENVS, 1)
    assert batch.next_states.shape == (T_STEPS, NUM_ENVS, STATE_DIM)
    assert batch.dones.shape == (T_STEPS, NUM_ENVS)
    assert batch.beliefs_mu.shape == (T_STEPS, NUM_ENVS, LATENT_DIM)
    assert batch.beliefs_logvar.shape == (T_STEPS, NUM_ENVS, LATENT_DIM)
    assert batch.log_probs.shape == (T_STEPS, NUM_ENVS)
    assert batch.values.shape == (T_STEPS, NUM_ENVS)
    assert batch.bootstrap_value.shape == (NUM_ENVS,)


def test_gru_hidden_not_zero_after_rollout():
    """GRU hidden should accumulate and be non-zero after steps."""
    vae, policy, env, params, config = _make_components()
    key = jax.random.PRNGKey(0)
    rs = init_rollout_state(vae, env, params, NUM_ENVS, key)
    new_rs, _ = collect_rollout(rs, vae, policy, env, params, config, jax.random.PRNGKey(2))
    assert not jnp.allclose(new_rs.gru_hidden, jnp.zeros_like(new_rs.gru_hidden))


def test_logvar_stays_clipped():
    """beliefs_logvar should remain within [-10, 10]."""
    vae, policy, env, params, config = _make_components()
    key = jax.random.PRNGKey(0)
    rs = init_rollout_state(vae, env, params, NUM_ENVS, key)
    _, batch = collect_rollout(rs, vae, policy, env, params, config, jax.random.PRNGKey(3))
    assert jnp.all(batch.beliefs_logvar >= -10.0)
    assert jnp.all(batch.beliefs_logvar <= 10.0)


def test_log_probs_finite():
    vae, policy, env, params, config = _make_components()
    key = jax.random.PRNGKey(0)
    rs = init_rollout_state(vae, env, params, NUM_ENVS, key)
    _, batch = collect_rollout(rs, vae, policy, env, params, config, jax.random.PRNGKey(4))
    assert jnp.all(jnp.isfinite(batch.log_probs))


def test_collect_rollout_linear_encoder_under_x64():
    """Regression test for a scan carry dtype mismatch: `_BeliefEncoderMixin.prior()`/
    `init_hidden()` built their zero seed with `jnp.zeros(...)` and no explicit dtype,
    which silently picks up float64 once `jax_enable_x64` is on (as it is for any
    float64-precision ODE env, e.g. van_der_pol's stiff solver). GRU encoders
    happened to self-consistently promote every downstream op to float64 too, masking
    the bug — but LinearEncoder.encode_step never reads `hidden`, so its per-step
    mu/logvar came out float32 while the float64 prior() seeded the initial carry,
    and jax.lax.scan rejected the mismatch. See src/steppo/models/encoder.py."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        config = TrainConfig(
            vae=VAEConfig(
                latent_dim=LATENT_DIM,
                encoder=EncoderArchConfig(encoder_type="linear", linear_hidden_dims=(32, 32)),
            ),
            ppo=PPOConfig(),
            env=ODEEnvConfig(system="scalar_decay", train_bins=((1.0, 100.0),)),
            num_envs=NUM_ENVS,
            rollout_steps=T_STEPS,
        )
        rngs = nnx.Rngs(0)
        vae = VariBADVAE(STATE_DIM, ACTION_DIM, config.vae, rngs)
        policy = ActorCritic(
            STATE_DIM,
            ACTION_DIM,
            LATENT_DIM,
            action_space="continuous",
            use_latent_sample=False,
            rngs=nnx.Rngs(1),
        )
        env = ODEEnv(config.env, max_steps=T_STEPS)
        params = ODEParams(lam=10.0)

        key = jax.random.PRNGKey(0)
        rs = init_rollout_state(vae, env, params, NUM_ENVS, key)
        _, batch = collect_rollout(rs, vae, policy, env, params, config, jax.random.PRNGKey(1))
        assert batch.beliefs_mu.shape == (T_STEPS, NUM_ENVS, LATENT_DIM)
        assert jnp.all(jnp.isfinite(batch.beliefs_mu))
    finally:
        jax.config.update("jax_enable_x64", prev)
