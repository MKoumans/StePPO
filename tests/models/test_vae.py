"""Tests for steppo.models.vae.VariBADVAE."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from steppo.configs.base_config import DecoderArchConfig, EncoderArchConfig, VAEConfig
from steppo.models.vae import VariBADVAE

STATE_DIM = 2
ACTION_DIM = 4
LATENT_DIM = 5
T = 10


def _make_vae():
    config = VAEConfig(
        latent_dim=LATENT_DIM,
        encoder=EncoderArchConfig(hidden_size=64),
        decoder=DecoderArchConfig(hidden_dims=(32, 32)),
    )
    return VariBADVAE(STATE_DIM, ACTION_DIM, config, nnx.Rngs(0)), config


def test_encode_decode_runs():
    vae, _ = _make_vae()
    actions = jnp.zeros((T, ACTION_DIM))
    states = jnp.zeros((T, STATE_DIM))
    rewards = jnp.zeros((T, 1))
    mus, logvars = vae.encode(actions, states, rewards)
    assert mus.shape == (T + 1, LATENT_DIM)
    z = mus[1]
    result = vae.decode(z, states[0], actions[0])
    assert "reward" in result


def test_jit_compatible():
    vae, _ = _make_vae()
    graphdef, params = nnx.split(vae)

    def encode(params, actions, states, rewards):
        v = nnx.merge(graphdef, params)
        return v.encode(actions, states, rewards)

    jit_encode = jax.jit(encode)
    actions = jnp.zeros((T, ACTION_DIM))
    states = jnp.zeros((T, STATE_DIM))
    rewards = jnp.zeros((T, 1))
    mus, _ = jit_encode(params, actions, states, rewards)
    assert mus.shape == (T + 1, LATENT_DIM)


def test_task_decoder_is_disabled_for_zero_width_task():
    config = VAEConfig(
        latent_dim=LATENT_DIM,
        task_loss_coeff=1.0,
        encoder=EncoderArchConfig(hidden_size=8),
        decoder=DecoderArchConfig(hidden_dims=(8,)),
    )
    vae = VariBADVAE(STATE_DIM, ACTION_DIM, config, nnx.Rngs(0), task_dim=0)

    decoded = vae.decode(jnp.zeros(LATENT_DIM), jnp.zeros(STATE_DIM), jnp.zeros(ACTION_DIM))

    assert vae.task_decoder is None
    assert decoded["task"] is None
