"""Robertson chemical kinetics — canonical stiff test problem.

  dy1/dt = -k1*y1 + k3*y2*y3
  dy2/dt =  k1*y1 - k2*y2^2 - k3*y2*y3
  dy3/dt =  k2*y2^2

Conservation law: y1 + y2 + y3 = 1.
Standard values: k1=0.04, k3=1e4. k2 controls stiffness (~1e4 to ~1e9).
Hidden task parameter: k2 ~ LogUniform[k2_min, k2_max].

System-specific observation feature (shared ones: see obs_factory.py):
  state:        [y1, sign(y2)·log(|y2|+1e-10)/10, y3]                                 (3D)
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.envs.ode.systems import ODESystemSpec
from steppo.envs.ode.systems.obs_factory import base_obs_factory

_K1 = 0.04
_K3 = 1e4


def _rhs(t: Array, y: Array, task: Array) -> Array:
    """Robertson RHS. task = [k2]."""
    k2 = task[0]
    return jnp.array(
        [
            -_K1 * y[0] + _K3 * y[1] * y[2],
            _K1 * y[0] - k2 * y[1] ** 2 - _K3 * y[1] * y[2],
            k2 * y[1] ** 2,
        ]
    )


def _sample_task(key: PRNGKey, config) -> Array:
    """Sample or deterministically select the Robertson stiffness parameter."""
    if config.sample_k2:
        log_k2 = jax.random.uniform(
            key,
            shape=(1,),
            minval=jnp.log(jnp.float32(config.k2_min)),
            maxval=jnp.log(jnp.float32(config.k2_max)),
        )
        return jnp.exp(log_k2)
    k2_mid = jnp.exp(
        (jnp.log(jnp.float32(config.k2_min)) + jnp.log(jnp.float32(config.k2_max))) / 2.0
    )
    return jnp.array([k2_mid], dtype=jnp.float32)


def _y0(task: Array, key: PRNGKey, config) -> Array:
    """Return Robertson's conserved initial concentration state."""
    return jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)


def _state(state, config) -> Array:
    """Transform Robertson concentrations into stable observation features."""
    return jnp.array(
        [
            state.y[0],
            jnp.sign(state.y[1])
            * jnp.log(jnp.abs(state.y[1]) + jnp.float32(1e-10))
            / jnp.float32(10.0),
            state.y[2],
        ],
        dtype=jnp.float32,
    )


_obs, _FEATURE_DIMS = base_obs_factory(_state, state_dim=3, y_dim=3)


SPEC = ODESystemSpec(
    name="robertson",
    y_dim=3,
    obs_dim=15,
    rhs=_rhs,
    sample_task=_sample_task,
    y0=_y0,
    obs=_obs,
    feature_dims=_FEATURE_DIMS,
)
