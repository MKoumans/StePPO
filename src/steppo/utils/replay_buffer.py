"""Trajectory replay storage for VAE and other sequence learners."""

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.random import PRNGKey


# -- Helper functions --
def transpose_action(batch, _transpose_tuple: tuple = (1, 0, 2)) -> Array:
    """Transpose discrete or continuous actions to batch-major trajectory order."""
    #   actions     = jnp.transpose(
    #     batch.actions if batch.actions.ndim == 3
    #     else batch.actions[:, :, None], (1, 0, 2)
    # ).squeeze(-1) if batch.actions.ndim == 2 else
    #     jnp.transpose(batch.actions, (1, 0, 2)),

    n_dims = batch.actions.ndim

    if n_dims == 3:
        return jnp.transpose(batch.actions, _transpose_tuple)
    elif n_dims == 2:
        return jnp.transpose(batch.actions[:, :, None], _transpose_tuple).squeeze(-1)
    else:
        raise ValueError(
            f"Unexpected n_dim. n_dim (2) = discrete action space, n_dim (3) = continuous action space. Got n_dim = {n_dims}"
        )


@flax.struct.dataclass
class TrajectoryBatch:
    """Pytree-compatible batch of batch-major trajectories."""

    states: Array  # (B, T, state_dim)
    actions: Array  # (B, T, action_dim)
    rewards: Array  # (B, T, 1)
    next_states: Array  # (B, T, state_dim)
    masks: Array  # (B, T)  — 0.0 after episode end
    task_params: Array  # (B, task_dim) — hidden task params; shape (B, 0) when unused
    keep_steps: Array  # (B, T)  — 1.0 if solver step accepted, 0.0 if rejected


class VAEReplayBuffer:
    """Ring buffer storing complete trajectories for VAE training."""

    def __init__(
        self, capacity: int, trajectory_len: int, state_dim: int, action_dim: int, task_dim: int = 0
    ):
        """Allocate a ring buffer for fixed-length trajectories."""
        self.capacity = capacity
        self.trajectory_len = trajectory_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.task_dim = task_dim
        self._size = 0
        self._ptr = 0
        self._rng = np.random.default_rng(0)

        self._states = np.zeros((capacity, trajectory_len, state_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, trajectory_len, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, trajectory_len, 1), dtype=np.float32)
        self._next_states = np.zeros((capacity, trajectory_len, state_dim), dtype=np.float32)
        self._masks = np.ones((capacity, trajectory_len), dtype=np.float32)
        self._task_params = np.zeros((capacity, task_dim), dtype=np.float32)
        self._keep_steps = np.zeros((capacity, trajectory_len), dtype=np.float32)

    def insert(self, trajectory: TrajectoryBatch):
        """Insert a batch of trajectories (B trajectories at once)."""
        batch_size = trajectory.states.shape[0]
        # Single bulk device→host transfer; numpy fancy indexing handles ring-buffer wrap-around
        states = np.asarray(trajectory.states)
        actions = np.asarray(trajectory.actions)
        rewards = np.asarray(trajectory.rewards)
        next_states = np.asarray(trajectory.next_states)
        masks = np.asarray(trajectory.masks)
        task_params = np.asarray(trajectory.task_params)
        keep_steps = np.asarray(trajectory.keep_steps)
        indices = (self._ptr + np.arange(batch_size)) % self.capacity
        self._states[indices] = states
        self._actions[indices] = actions
        self._rewards[indices] = rewards
        self._next_states[indices] = next_states
        self._masks[indices] = masks
        self._task_params[indices] = task_params
        self._keep_steps[indices] = keep_steps
        self._ptr += batch_size
        self._size = min(self._size + batch_size, self.capacity)

    def sample(self, batch_size: int, key: PRNGKey = None) -> TrajectoryBatch:
        """Sample a random batch of trajectories."""
        indices = self._rng.integers(0, self._size, size=batch_size)
        return TrajectoryBatch(
            states=jnp.array(self._states[indices]),
            actions=jnp.array(self._actions[indices]),
            rewards=jnp.array(self._rewards[indices]),
            next_states=jnp.array(self._next_states[indices]),
            masks=jnp.array(self._masks[indices]),
            task_params=jnp.array(self._task_params[indices]),
            keep_steps=jnp.array(self._keep_steps[indices]),
        )

    @property
    def size(self) -> int:
        """Return the number of stored trajectories."""
        return self._size


def get_trajectory_batch(batch, action_dim, _transpose_tuple: tuple = (1, 0, 2)) -> TrajectoryBatch:
    """Convert time-major rollout transitions to batch-major VAE trajectories."""
    # Reshape (T, num_envs, ...) → per-env trajectories
    T, N = batch.states.shape[:2]
    traj = TrajectoryBatch(
        states=jnp.transpose(batch.states, _transpose_tuple),  # (N, T, dim)
        actions=transpose_action(
            batch, _transpose_tuple
        ),  # (N, T, dim) or (N, T) depending on action space
        rewards=jnp.transpose(batch.rewards, _transpose_tuple),  # (N, T, 1)
        next_states=jnp.transpose(batch.next_states, _transpose_tuple),  # (N, T, dim)
        masks=jnp.transpose(1.0 - batch.dones, _transpose_tuple[:2]),  # (N, T)
        task_params=batch.task_params,  # (N, task_dim) — already per-env
        keep_steps=jnp.transpose(batch.keep_steps, _transpose_tuple[:2]),  # (N, T)
    )

    # actions for buffer: one-hot or float
    if batch.actions.ndim == 2:
        traj = TrajectoryBatch(
            states=traj.states,
            actions=jax.nn.one_hot(
                jnp.transpose(batch.actions, (1, 0)),
                num_classes=action_dim,
            ).astype(jnp.float32),
            rewards=traj.rewards,
            next_states=traj.next_states,
            masks=traj.masks,
            task_params=traj.task_params,
            keep_steps=traj.keep_steps,
        )

    return traj
