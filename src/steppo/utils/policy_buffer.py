"""Short-lived on-policy storage used to form PPO minibatches."""

from typing import Optional

import jax
import numpy as np
from jax import Array
from jax.random import PRNGKey

from steppo.training.ppo import PPOBatch


class PolicyBuffer:
    """Buffer for PPO: stores one rollout, produces mini-batches.

    Discarded after each update cycle.
    """

    def __init__(self):
        """Create an empty buffer for one PPO rollout."""
        self._states: Optional[Array] = None
        self._z: Optional[Array] = None
        self._task: Optional[Array] = None
        self._actions: Optional[Array] = None
        self._log_probs: Optional[Array] = None
        self._advantages: Optional[Array] = None
        self._returns: Optional[Array] = None
        self._size = 0

    def store(
        self,
        states: Array,
        z: Array,
        task: Array,
        actions: Array,
        log_probs: Array,
        advantages: Array,
        returns: Array,
        active_mask: Optional[Array] = None,
    ):
        """Flatten (T, num_envs, ...) arrays into (T*num_envs, ...) and store.

        If active_mask is provided, only timesteps where mask is True are kept.
        This filters out frozen trial steps from multi-episode rollouts.
        """

        def flat(x):
            # Explicit leading dim (not -1): with a zero-width trailing dim (e.g. an
            # unused (T, num_envs, 0) task array), letting reshape infer -1 divides by
            # zero internally (JAX/numpy can't solve for -1 against a zero-size axis).
            return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        self._states = flat(states)
        self._z = flat(z)
        self._task = flat(task)
        self._actions = flat(actions)
        self._log_probs = flat(log_probs)
        self._advantages = flat(advantages)
        self._returns = flat(returns)

        if active_mask is not None:
            mask = np.asarray(flat(active_mask)).astype(bool)
            if mask.any():
                self._states = self._states[mask]
                self._z = self._z[mask]
                self._task = self._task[mask]
                self._actions = self._actions[mask]
                self._log_probs = self._log_probs[mask]
                self._advantages = self._advantages[mask]
                self._returns = self._returns[mask]

        self._size = self._states.shape[0]

    def get_minibatches(self, num_minibatches: int, key: PRNGKey) -> list:
        """Shuffle data and split into mini-batches."""
        indices = jax.random.permutation(key, self._size)
        mb_size = self._size // num_minibatches
        batches = []
        for i in range(num_minibatches):
            idx = indices[i * mb_size : (i + 1) * mb_size]
            batches.append(
                PPOBatch(
                    states=self._states[idx],
                    z=self._z[idx],
                    task=self._task[idx],
                    actions=self._actions[idx],
                    log_probs=self._log_probs[idx],
                    advantages=self._advantages[idx],
                    returns=self._returns[idx],
                )
            )
        return batches

    @property
    def size(self) -> int:
        """Return the number of active transitions in the buffer."""
        return self._size
