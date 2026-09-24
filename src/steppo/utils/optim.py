"""Optimiser helpers shared by the PPO and VAE trainers."""

import jax.numpy as jnp
from jax import Array


def current_lr(schedule, step) -> Array:
    """Learning rate applied at `step` for a schedule (callable) or constant."""
    return schedule(step) if callable(schedule) else jnp.asarray(schedule, dtype=jnp.float32)
