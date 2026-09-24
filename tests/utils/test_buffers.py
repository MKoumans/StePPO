"""Tests for steppo.utils.replay_buffer and steppo.utils.policy_buffer."""

import jax
import jax.numpy as jnp

from steppo.utils.policy_buffer import PolicyBuffer
from steppo.utils.replay_buffer import TrajectoryBatch, VAEReplayBuffer

STATE_DIM = 2
ACTION_DIM = 4
LATENT_DIM = 5
Z_DIM = LATENT_DIM * 2
T = 10
B = 6


def _make_trajectory_batch(B=B, T=T):
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


# ── VAEReplayBuffer ──────────────────────────────────────────────────────────


def test_replay_buffer_sample_shapes():
    buf = VAEReplayBuffer(capacity=50, trajectory_len=T, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    buf.insert(_make_trajectory_batch(B=10))
    key = jax.random.PRNGKey(6)
    sampled = buf.sample(5, key)
    assert sampled.states.shape == (5, T, STATE_DIM)
    assert sampled.actions.shape == (5, T, ACTION_DIM)
    assert sampled.rewards.shape == (5, T, 1)
    assert sampled.next_states.shape == (5, T, STATE_DIM)
    assert sampled.masks.shape == (5, T)


def test_replay_buffer_size_grows_and_wraps():
    buf = VAEReplayBuffer(capacity=20, trajectory_len=T, state_dim=STATE_DIM, action_dim=ACTION_DIM)
    assert buf.size == 0
    buf.insert(_make_trajectory_batch(B=10))
    assert buf.size == 10
    buf.insert(_make_trajectory_batch(B=10))
    assert buf.size == 20
    buf.insert(_make_trajectory_batch(B=10))
    assert buf.size == 20


# ── PolicyBuffer ─────────────────────────────────────────────────────────────


def test_policy_buffer_minibatches_cover_each_transition_once_and_preserve_alignment():
    buf = PolicyBuffer()
    T_buf, N = 10, 4
    transition_ids = jnp.arange(T_buf * N).reshape(T_buf, N)
    states = jnp.broadcast_to(transition_ids[..., None], (T_buf, N, STATE_DIM))
    z = jnp.broadcast_to((transition_ids + 100)[..., None], (T_buf, N, Z_DIM))
    task = jnp.zeros((T_buf, N, 0))
    actions = transition_ids % ACTION_DIM
    log_probs = transition_ids + 200
    advantages = transition_ids + 300
    returns = transition_ids + 400

    buf.store(states, z, task, actions, log_probs, advantages, returns)
    assert buf.size == T_buf * N

    key = jax.random.PRNGKey(6)
    batches = buf.get_minibatches(num_minibatches=4, key=key)
    assert len(batches) == 4
    batch_ids = jnp.concatenate([batch.states[:, 0] for batch in batches])
    assert jnp.array_equal(jnp.sort(batch_ids), jnp.arange(T_buf * N))

    for batch in batches:
        ids = batch.states[:, 0]
        assert jnp.all(batch.states[:, 1] == ids)
        assert jnp.all(batch.z[:, 0] == ids + 100)
        assert jnp.all(batch.actions == ids % ACTION_DIM)
        assert jnp.all(batch.log_probs == ids + 200)
        assert jnp.all(batch.advantages == ids + 300)
        assert jnp.all(batch.returns == ids + 400)


def test_policy_buffer_active_mask_keeps_matching_transition_fields():
    buf = PolicyBuffer()
    T_buf, N = 3, 4
    transition_ids = jnp.arange(T_buf * N).reshape(T_buf, N)
    states = jnp.broadcast_to(transition_ids[..., None], (T_buf, N, STATE_DIM))
    z = jnp.broadcast_to((transition_ids + 100)[..., None], (T_buf, N, Z_DIM))
    task = jnp.zeros((T_buf, N, 0))
    actions = transition_ids % ACTION_DIM
    log_probs = transition_ids + 200
    advantages = transition_ids + 300
    returns = transition_ids + 400
    active_mask = jnp.array(
        [[True, False, True, False], [False, True, True, False], [True, False, False, True]]
    )

    buf.store(states, z, task, actions, log_probs, advantages, returns, active_mask)

    assert buf.size == 6
    batch = buf.get_minibatches(num_minibatches=1, key=jax.random.PRNGKey(9))[0]
    ids = batch.states[:, 0]
    assert jnp.array_equal(jnp.sort(ids), jnp.array([0, 2, 5, 6, 8, 11]))
    assert jnp.all(batch.z[:, 0] == ids + 100)
    assert jnp.all(batch.actions == ids % ACTION_DIM)
    assert jnp.all(batch.log_probs == ids + 200)
    assert jnp.all(batch.advantages == ids + 300)
    assert jnp.all(batch.returns == ids + 400)


def test_policy_buffer_all_inactive_mask_keeps_nonempty_fallback():
    buf = PolicyBuffer()
    T_buf, N = 2, 2
    transition_ids = jnp.arange(T_buf * N).reshape(T_buf, N)
    states = transition_ids[..., None]
    z = jnp.broadcast_to((transition_ids + 100)[..., None], (T_buf, N, Z_DIM))
    task = jnp.zeros((T_buf, N, 0))
    actions = transition_ids % ACTION_DIM
    log_probs = transition_ids + 200
    advantages = transition_ids + 300
    returns = transition_ids + 400
    active_mask = jnp.zeros((T_buf, N), dtype=bool)

    buf.store(states, z, task, actions, log_probs, advantages, returns, active_mask)

    # A completely frozen rollout still yields nonempty minibatches for PPO.
    assert buf.size == T_buf * N
    batch = buf.get_minibatches(num_minibatches=1, key=jax.random.PRNGKey(10))[0]
    ids = batch.states[:, 0]
    assert jnp.array_equal(jnp.sort(ids), jnp.arange(T_buf * N))
    assert jnp.all(batch.z[:, 0] == ids + 100)
