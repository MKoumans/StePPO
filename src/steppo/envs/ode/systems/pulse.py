"""Periodic pulse forcing: narrow Gaussian bumps spaced `period` apart.

`random_ratio` in [0, 1] jitters each bump by up to ±random_ratio·period/2,
deterministically from the per-episode seed in task[1], so the forcing is a
pure function of (t, task).
"""

import jax.numpy as jnp
from jax import Array


def _cycle_hash(k: Array, seed: Array) -> Array:
    """Deterministic pseudo-random value in [0, 1) for a given (cycle, seed)."""
    x = jnp.sin(k * 12.9898 + seed * 78.233) * 43758.5453
    return x - jnp.floor(x)


def gaussian_pulse_train_jittered(
    period: float, width: float, amplitude: float, random_ratio: float
):
    """Build a deterministic, optionally jittered Gaussian pulse train."""

    def forcing(t: Array, seed: Array) -> Array:
        """Evaluate the nearest pulse at time ``t`` for an episode seed."""
        k = jnp.round(t / period)
        jitter = (_cycle_hash(k, seed) - 0.5) * random_ratio * period
        center = k * period + jitter
        phase = t - center
        return amplitude * jnp.exp(-0.5 * (phase / width) ** 2)

    return forcing


def make_pulsed_rhs(base_rhs, dim: int, forcing):
    """Wrap a system's `rhs(t, y, task)` to add `forcing(t, task[1])` into
    state dimension `dim`. `task[1]` is the per-episode pulse seed (see
    module docstring); systems that don't use pulses never read task[1]."""

    def forced_rhs(t: Array, y: Array, task: Array) -> Array:
        """Evaluate the base RHS and add forcing to one state dimension."""
        seed = task[1]
        return base_rhs(t, y, task).at[dim].add(forcing(t, seed))

    return forced_rhs
