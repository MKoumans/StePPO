"""FitzHugh-Nagumo — simplified neuron model with tunable time-scale separation.

  dv/dt = v - v^3/3 - w + I_ext
  dw/dt = eps · (v + a - b·w)

Fixed: a=0.7, b=0.8, I_ext=0.5.
eps controls time-scale separation: small eps → stiff relaxation oscillations.
Hidden task parameter: eps ~ LogUniform[eps_min, eps_max].

System-specific observation feature (shared ones: see obs_factory.py):
  state:        [v/2, w/2]                                                            (2D)
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.envs.ode.systems import ODESystemSpec
from steppo.envs.ode.systems.obs_factory import base_obs_factory
from steppo.envs.ode.systems.pulse import gaussian_pulse_train_jittered, make_pulsed_rhs

_A = 0.7
_B = 0.8
_I_EXT = 0.5


def _rhs(t: Array, y: Array, task: Array) -> Array:
    """FitzHugh-Nagumo RHS. task = [eps]."""
    eps = task[0]
    v, w = y[0], y[1]
    return jnp.array(
        [
            v - v**3 / 3.0 - w + _I_EXT,
            eps * (v + _A - _B * w),
        ]
    )


def make_rhs(config):
    """Return the RHS, with optional pulse forcing on `config.fhn_pulse_dim` (see pulse.py)."""
    if not getattr(config, "fhn_pulse_enabled", False):
        return _rhs
    forcing = gaussian_pulse_train_jittered(
        config.fhn_pulse_period,
        config.fhn_pulse_width,
        config.fhn_pulse_amplitude,
        getattr(config, "fhn_pulse_random", 0.0),
    )
    return make_pulsed_rhs(_rhs, config.fhn_pulse_dim, forcing)


def _sample_task(key: PRNGKey, config) -> Array:
    """Sample or deterministically select the time-scale parameter."""
    if config.sample_eps:
        log_eps = jax.random.uniform(
            key,
            shape=(1,),
            minval=jnp.log(jnp.float32(config.eps_min)),
            maxval=jnp.log(jnp.float32(config.eps_max)),
        )
        return jnp.exp(log_eps)
    eps_mid = jnp.exp(
        (jnp.log(jnp.float32(config.eps_min)) + jnp.log(jnp.float32(config.eps_max))) / 2.0
    )
    return jnp.array([eps_mid], dtype=jnp.float32)


def _y0(task: Array, key: PRNGKey, config) -> Array:
    """Return the configured or randomized FitzHugh-Nagumo initial state."""
    if config.sample_y0:
        key_v, key_w = jax.random.split(key)
        v0 = jax.random.uniform(key_v, shape=(), minval=-2.0, maxval=2.0)
        w0 = jax.random.uniform(key_w, shape=(), minval=-1.0, maxval=2.0)
        return jnp.array([v0, w0], dtype=jnp.float32)
    return jnp.array([config.y0_x, config.y0_y], dtype=jnp.float32)


def _state(state, config) -> Array:
    """Scale the neuron state into observation features."""
    return jnp.array(
        [
            state.y[0] / jnp.float32(2.0),
            state.y[1] / jnp.float32(2.0),
        ],
        dtype=jnp.float32,
    )


_obs, _FEATURE_DIMS = base_obs_factory(_state, state_dim=2, y_dim=2)


SPEC = ODESystemSpec(
    name="fitzhugh_nagumo",
    y_dim=2,
    obs_dim=13,
    rhs=_rhs,
    sample_task=_sample_task,
    y0=_y0,
    obs=_obs,
    feature_dims=_FEATURE_DIMS,
)
