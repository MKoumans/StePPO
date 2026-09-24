"""Van der Pol oscillator ODE system.

  ẏ₀ = y₁
  ẏ₁ = μ(1 - y₀²)y₁ - y₀

Stiffness ≈ μ². Hidden task parameter: μ ~ LogUniform[mu_min, mu_max].
μ is NOT in the observation — VariBAD infers it from error/accept patterns.

System-specific observation feature (shared ones: see obs_factory.py):
  state:        [y0/2, sign(y1)·log(|y1|+1)]                                        (2D)
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.envs.ode.systems import ODESystemSpec
from steppo.envs.ode.systems.obs_factory import base_obs_factory


def _rhs(t: Array, y: Array, task: Array) -> Array:
    """Van der Pol RHS. task = [μ]."""
    mu = task[0]
    return jnp.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]])


def _sample_task(key: PRNGKey, config) -> Array:
    """Sample or deterministically select the Van der Pol stiffness parameter."""
    if config.sample_mu:
        log_mu = jax.random.uniform(
            key,
            shape=(1,),
            minval=jnp.log(jnp.float32(config.mu_min)),
            maxval=jnp.log(jnp.float32(config.mu_max)),
        )
        return jnp.exp(log_mu)
    mu_mid = (
        config.mu_min + config.mu_max
    ) / 2.0  # If not sampling, return midpoint of range for deterministic behavior
    return jnp.array([mu_mid], dtype=jnp.float32)


def _y0(task: Array, key: PRNGKey, config) -> Array:
    """Return the configured or randomized Van der Pol initial state."""
    if not config.sample_y0:
        return jnp.array([config.y0_x, config.y0_y], dtype=jnp.float32)
    key_x, key_y = jax.random.split(key)
    y0_x = jax.random.uniform(key_x, shape=(), minval=-2.0, maxval=2.0)
    y0_y = jax.random.uniform(key_y, shape=(), minval=-3.0, maxval=3.0)

    return jnp.array([y0_x, y0_y], dtype=jnp.float32)


def _state(state, config) -> Array:
    """Transform the oscillator state into bounded observation features."""
    return jnp.array(
        [
            state.y[0] / jnp.float32(2.0),
            jnp.sign(state.y[1]) * jnp.log(jnp.abs(state.y[1]) + 1.0),
        ],
        dtype=jnp.float32,
    )


_obs, _FEATURE_DIMS = base_obs_factory(_state, state_dim=2, y_dim=2)


SPEC = ODESystemSpec(
    name="van_der_pol",
    y_dim=2,
    obs_dim=13,
    rhs=_rhs,
    sample_task=_sample_task,
    y0=_y0,
    obs=_obs,
    feature_dims=_FEATURE_DIMS,
)
