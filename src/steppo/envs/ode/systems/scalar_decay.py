"""Scalar exponential decay ODE system.

  dy/dt = -λ·y,  y(0) = 1,  t ∈ [0, t_end]

Exact solution: y(t) = exp(-λt).  λ is the hidden task parameter.

Task: λ ∈ [lam_min, lam_max].
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.envs.ode.systems import ODESystemSpec
from steppo.envs.ode.systems.obs_factory import base_obs_factory
from steppo.envs.ode.systems.pulse import gaussian_pulse_train_jittered, make_pulsed_rhs


def _rhs(t: Array, y: Array, task: Array) -> Array:
    """dy/dt = -λ·y. task = [λ]."""
    return -task[0] * y


def make_rhs(config):
    """Return the RHS, with optional pulse forcing on `config.sd_pulse_dim` (see pulse.py)."""
    if not getattr(config, "sd_pulse_enabled", False):
        return _rhs
    forcing = gaussian_pulse_train_jittered(
        config.sd_pulse_period,
        config.sd_pulse_width,
        config.sd_pulse_amplitude,
        getattr(config, "sd_pulse_random", 0.0),
    )
    return make_pulsed_rhs(_rhs, config.sd_pulse_dim, forcing)


def _sample_task(key: PRNGKey, config) -> Array:
    """Sample or deterministically select the decay rate."""
    if config.sample_lam:
        return jax.random.uniform(
            key,
            shape=(1,),
            minval=jnp.float32(config.lam_min),
            maxval=jnp.float32(config.lam_max),
        )
    return jnp.array([config.lam_min], dtype=jnp.float32)


def _y0(task: Array, key: PRNGKey, config) -> Array:
    """Return the scalar-decay initial state."""
    return jnp.ones((1,), dtype=jnp.float32)


def _state(state, config) -> Array:
    """Clip the scalar state for use as an observation feature."""
    return jnp.clip(state.y[0], -10.0, 10.0)


_obs, _FEATURE_DIMS = base_obs_factory(_state, state_dim=1, y_dim=1)


SPEC = ODESystemSpec(
    name="scalar_decay",
    y_dim=1,
    obs_dim=11,
    rhs=_rhs,
    sample_task=_sample_task,
    y0=_y0,
    obs=_obs,
    feature_dims=_FEATURE_DIMS,
)
