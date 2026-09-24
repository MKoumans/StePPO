"""Tests for steppo.models.policy.ActorCritic."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from steppo.models.policy import ActorCritic

STATE_DIM = 2
ACTION_DIM = 4
LATENT_DIM = 5
Z_DIM = LATENT_DIM * 2


def _make_discrete_policy():
    return ActorCritic(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        action_space="discrete",
        use_latent_sample=False,
        rngs=nnx.Rngs(0),
    )


def _make_continuous_policy():
    return ActorCritic(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        action_space="continuous",
        use_latent_sample=False,
        rngs=nnx.Rngs(0),
    )


def test_act_shapes_discrete():
    policy = _make_discrete_policy()
    key = jax.random.PRNGKey(0)
    action, log_prob, value = policy.act(jnp.zeros(STATE_DIM), jnp.zeros(Z_DIM), key)
    assert action.shape == ()
    assert log_prob.shape == ()
    assert value.shape == ()


def test_act_shapes_continuous():
    policy = _make_continuous_policy()
    key = jax.random.PRNGKey(0)
    action, log_prob, value = policy.act(jnp.zeros(STATE_DIM), jnp.zeros(Z_DIM), key)
    assert action.shape == (ACTION_DIM,)
    assert log_prob.shape == ()
    assert value.shape == ()


def test_discrete_action_in_range():
    policy = _make_discrete_policy()
    key = jax.random.PRNGKey(7)
    for i in range(20):
        k = jax.random.fold_in(key, i)
        action, _, _ = policy.act(jnp.zeros(STATE_DIM), jnp.zeros(Z_DIM), k)
        assert 0 <= int(action) < ACTION_DIM


def test_call_jit_compilable():
    policy = _make_discrete_policy()
    graphdef, params = nnx.split(policy)

    def run(params):
        p = nnx.merge(graphdef, params)
        return p(jnp.zeros(STATE_DIM), jnp.zeros(Z_DIM))

    dist, value = jax.jit(run)(params)
    assert value.shape == ()
