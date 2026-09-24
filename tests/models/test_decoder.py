"""Tests for steppo.models.decoder (RewardDecoder, StateDecoder)."""

import flax.nnx as nnx
import jax.numpy as jnp

from steppo.configs.base_config import DecoderArchConfig
from steppo.models.decoder import RewardDecoder, StateDecoder, StepAcceptDecoder, TaskDecoder

STATE_DIM = 2
ACTION_DIM = 4
LATENT_DIM = 5


def test_reward_decoder_scalar():
    arch = DecoderArchConfig(hidden_dims=(32, 32))
    dec = RewardDecoder(STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, rngs=nnx.Rngs(0))
    z = jnp.zeros(LATENT_DIM)
    s = jnp.zeros(STATE_DIM)
    a = jnp.zeros(ACTION_DIM)
    r_hat = dec(z, s, a)
    assert r_hat.shape == ()


def test_state_decoder_shape():
    arch = DecoderArchConfig(hidden_dims=(32, 32))
    dec = StateDecoder(STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, rngs=nnx.Rngs(0))
    z = jnp.zeros(LATENT_DIM)
    s = jnp.zeros(STATE_DIM)
    a = jnp.zeros(ACTION_DIM)
    s_hat = dec(z, s, a)
    assert s_hat.shape == (STATE_DIM,)


def test_heteroscedastic_decoder_heads_preserve_batch_and_target_widths():
    batch_size = 3
    arch = DecoderArchConfig(hidden_dims=(8,))
    z = jnp.zeros((batch_size, LATENT_DIM))
    states = jnp.zeros((batch_size, STATE_DIM))
    actions = jnp.zeros((batch_size, ACTION_DIM))

    reward_mean, reward_logvar = RewardDecoder(
        STATE_DIM,
        ACTION_DIM,
        LATENT_DIM,
        arch,
        nnx.Rngs(1),
        heteroscedastic=True,
    )(z, states, actions)
    state_mean, state_logvar = StateDecoder(
        STATE_DIM,
        ACTION_DIM,
        LATENT_DIM,
        arch,
        nnx.Rngs(2),
        heteroscedastic=True,
    )(z, states, actions)
    task_mean, task_logvar = TaskDecoder(
        LATENT_DIM,
        task_dim=3,
        arch=arch,
        rngs=nnx.Rngs(3),
        heteroscedastic=True,
    )(z)
    accept_logit = StepAcceptDecoder(
        STATE_DIM,
        ACTION_DIM,
        LATENT_DIM,
        arch,
        nnx.Rngs(4),
    )(z, states, actions)

    assert reward_mean.shape == reward_logvar.shape == (batch_size,)
    assert state_mean.shape == state_logvar.shape == (batch_size, STATE_DIM)
    assert task_mean.shape == task_logvar.shape == (batch_size, 3)
    assert accept_logit.shape == (batch_size,)
    assert all(
        jnp.all(jnp.isfinite(output))
        for output in (
            reward_mean,
            reward_logvar,
            state_mean,
            state_logvar,
            task_mean,
            task_logvar,
            accept_logit,
        )
    )
