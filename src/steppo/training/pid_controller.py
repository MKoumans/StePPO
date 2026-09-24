"""PID step-size controller for ODE environments.

Provides a pure-JAX PID policy and behavioral cloning loss
for the imitation warmstart pipeline.
"""

from typing import Optional, Tuple

import flax.nnx as nnx
import jax.numpy as jnp
from jax import Array

_LN2: float = 0.693147181


def compute_step_context_offset(obs_features: tuple, feature_dims: dict) -> int:
    """Return the starting index of step_context in the observation vector."""
    offset = 0
    for feat in obs_features:
        if feat == "step_context":
            return offset
        offset += feature_dims[feat]
    raise ValueError("step_context not in obs_features — PID controller requires it")


def pid_action_from_obs(
    obs: Array,
    step_context_offset: int,
    q: int = 4,
    safety: float = 0.9,
    min_factor: float = 0.2,
    max_factor: float = 10.0,
    dt_log_gain: float = _LN2,
    kp: float = 0.0,
    ki: float = 1.0,
    kd: float = 0.0,
) -> Array:
    """Return a JIT-safe PID action from normalized step-context features.

    The action is ``log(step_factor) / dt_log_gain``. The factor is bounded
    by ``min_factor`` and ``max_factor``; ``kd`` is reserved for future use.
    """
    log_error_obs = obs[step_context_offset + 0]  #
    log_error_delta_obs = obs[step_context_offset + 1]  #
    keep_step_obs = obs[step_context_offset + 2]  #

    log_error = log_error_obs * jnp.float32(5.0)
    log_error_delta = log_error_delta_obs * jnp.float32(5.0)
    scaled_error = jnp.exp(log_error)
    prev_scaled_error = jnp.exp(log_error - log_error_delta)

    inv_scaled_error = jnp.float32(1.0) / jnp.maximum(scaled_error, jnp.float32(1e-10))
    prev_inv_scaled_error = jnp.float32(1.0) / jnp.maximum(prev_scaled_error, jnp.float32(1e-10))

    error_order = jnp.float32(q + 1)
    coeff1 = jnp.float32(ki + kp) / error_order
    coeff2 = jnp.float32(-kp) / error_order
    factor = (
        jnp.float32(safety)
        * jnp.power(inv_scaled_error, coeff1)
        * jnp.power(prev_inv_scaled_error, coeff2)
    )
    factor = jnp.clip(factor, jnp.float32(min_factor), jnp.float32(max_factor))

    keep_step = keep_step_obs > jnp.float32(0.5)
    factor = jnp.where(
        keep_step,
        jnp.maximum(factor, jnp.float32(1.0)),
        jnp.minimum(factor, jnp.float32(safety)),
    )

    action = jnp.log(factor) / jnp.float32(dt_log_gain)
    return jnp.array([action], dtype=jnp.float32)


def bc_loss(
    graphdef,
    params,
    states: Array,
    z: Array,
    expert_actions: Array,
    task: Optional[Array] = None,
) -> Tuple[Array, dict]:
    """Behavioral cloning loss: MSE in pre-tanh (Gaussian mean) space."""
    policy = nnx.merge(graphdef, params)
    dist, _ = policy(states, z, task)

    expert_clipped = jnp.clip(expert_actions, -1.0 + 1e-6, 1.0 - 1e-6)
    target_pretanh = jnp.arctanh(expert_clipped)
    pred_pretanh = dist.distribution.loc

    loss = jnp.mean(jnp.square(pred_pretanh - target_pretanh))
    return loss, {"bc_loss": loss}
