"""Tests for steppo.models.encoder.LSTMEncoder."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from steppo.configs.base_config import EncoderArchConfig
from steppo.models.encoder import LSTMEncoder

STATE_DIM = 2
ACTION_DIM = 4
LATENT_DIM = 5
T = 10


def _make_encoder():
    arch = EncoderArchConfig(hidden_size=64)
    return LSTMEncoder(STATE_DIM, ACTION_DIM, LATENT_DIM, arch, nnx.Rngs(0))


def test_prior_shape():
    enc = _make_encoder()
    mu, logvar = enc.prior()
    assert mu.shape == (LATENT_DIM,)
    assert logvar.shape == (LATENT_DIM,)


def test_encode_step_shapes():
    enc = _make_encoder()
    a = jnp.zeros(ACTION_DIM)
    s = jnp.zeros(STATE_DIM)
    r = jnp.zeros(1)
    h = jnp.zeros(64)
    mu, logvar, new_h = enc.encode_step(a, s, r, h)
    assert mu.shape == (LATENT_DIM,)
    assert logvar.shape == (LATENT_DIM,)
    assert new_h.shape == (64,)


def test_encode_trajectory_shape():
    enc = _make_encoder()
    actions = jnp.zeros((T, ACTION_DIM))
    states = jnp.zeros((T, STATE_DIM))
    rewards = jnp.zeros((T, 1))
    mus, logvars = enc.encode_trajectory(actions, states, rewards)
    assert mus.shape == (T + 1, LATENT_DIM)
    assert logvars.shape == (T + 1, LATENT_DIM)


def test_encode_trajectory_jit():
    enc = _make_encoder()
    actions = jnp.zeros((T, ACTION_DIM))
    states = jnp.zeros((T, STATE_DIM))
    rewards = jnp.zeros((T, 1))

    graphdef, params = nnx.split(enc)

    def run(params, actions, states, rewards):
        enc_ = nnx.merge(graphdef, params)
        return enc_.encode_trajectory(actions, states, rewards)

    jit_fn = jax.jit(run)
    mus, logvars = jit_fn(params, actions, states, rewards)
    assert mus.shape == (T + 1, LATENT_DIM)


def test_vmap_encoder():
    enc = _make_encoder()
    B = 3
    actions = jnp.zeros((B, T, ACTION_DIM))
    states = jnp.zeros((B, T, STATE_DIM))
    rewards = jnp.zeros((B, T, 1))

    graphdef, params = nnx.split(enc)

    def encode_one(actions, states, rewards):
        enc_ = nnx.merge(graphdef, params)
        return enc_.encode_trajectory(actions, states, rewards)

    mus, logvars = jax.vmap(encode_one)(actions, states, rewards)
    assert mus.shape == (B, T + 1, LATENT_DIM)


def test_logvar_clipped():
    enc = _make_encoder()
    a = jnp.ones(ACTION_DIM) * 1000.0
    s = jnp.ones(STATE_DIM) * 1000.0
    r = jnp.ones(1) * 1000.0
    h = jnp.ones(64) * 1000.0
    _, logvar, _ = enc.encode_step(a, s, r, h)
    assert jnp.all(logvar >= -10.0)
    assert jnp.all(logvar <= 10.0)


def test_sample_different_keys_different_results():
    enc = _make_encoder()
    mu = jnp.ones(LATENT_DIM)
    logvar = jnp.zeros(LATENT_DIM)
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)
    z1 = enc.sample(mu, logvar, key1)
    z2 = enc.sample(mu, logvar, key2)
    assert not jnp.allclose(z1, z2)
