"""Brusselator — chemical oscillator with tunable stiffness.

  dy1/dt = A + y1^2·y2 - (B+1)·y1
  dy2/dt = B·y1 - y1^2·y2

Fixed point: (A, B/A). Hopf bifurcation at B = 1 + A^2.
For A=1: limit cycle when B > 2, increasingly stiff as B grows.
Hidden task parameter: B ~ LogUniform[B_min, B_max].

System-specific observation feature (shared ones: see obs_factory.py):
  state:        [y1/5, sign(y2)·log(|y2|+1)]                                          (2D)
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.envs.ode.systems import ODESystemSpec
from steppo.envs.ode.systems.obs_factory import base_obs_factory

_A = 1.0


def _rhs(t: Array, y: Array, task: Array) -> Array:
    """Brusselator RHS. task = [B]."""
    B = task[0]
    return jnp.array(
        [
            _A + y[0] ** 2 * y[1] - (B + 1.0) * y[0],
            B * y[0] - y[0] ** 2 * y[1],
        ]
    )


def _sample_task(key: PRNGKey, config) -> Array:
    """Sample or deterministically select the Brusselator stiffness parameter."""
    if config.sample_B:
        log_B = jax.random.uniform(
            key,
            shape=(1,),
            minval=jnp.log(jnp.float32(config.B_min)),
            maxval=jnp.log(jnp.float32(config.B_max)),
        )
        return jnp.exp(log_B)
    B_mid = jnp.exp((jnp.log(jnp.float32(config.B_min)) + jnp.log(jnp.float32(config.B_max))) / 2.0)
    return jnp.array([B_mid], dtype=jnp.float32)


def _y0(task: Array, key: PRNGKey, config) -> Array:
    """Return the initial Brusselator state for one task."""
    if config.sample_y0:
        key_x, key_y = jax.random.split(key)
        y1 = jax.random.uniform(key_x, shape=(), minval=0.5, maxval=3.0)
        y2 = jax.random.uniform(key_y, shape=(), minval=0.5, maxval=5.0)
        return jnp.array([y1, y2], dtype=jnp.float32)
    return jnp.array([config.y0_x, config.y0_y], dtype=jnp.float32)


def _state(state, config) -> Array:
    """Transform the physical state into bounded observation features."""
    return jnp.array(
        [
            state.y[0] / jnp.float32(5.0),
            jnp.sign(state.y[1]) * jnp.log(jnp.abs(state.y[1]) + 1.0),
        ],
        dtype=jnp.float32,
    )


_obs, _FEATURE_DIMS = base_obs_factory(_state, state_dim=2, y_dim=2)


SPEC = ODESystemSpec(
    name="brusselator",
    y_dim=2,
    obs_dim=13,
    rhs=_rhs,
    sample_task=_sample_task,
    y0=_y0,
    obs=_obs,
    feature_dims=_FEATURE_DIMS,
)
