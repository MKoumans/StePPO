"""Chua-inspired electrical circuit, Case 2 of Riley et al. (2025), arXiv:2501.08934.

State y = [V1, V2, i]:

  V1_dot = -V1/(C1*R) + V2/(C1*R) + F_r(V1)
  V2_dot =  V1/(C2*R) - V2/(C2*R) + i/C2
  i_dot  = -V2/L

F_r ∈ -(m/C1)*sign(V1) is an ideal Zener-diode-like element. C1, C2, R and L
are fixed; the diode gain m is the task parameter (fixed by default).

`chua` uses sign(V1); `chua_smooth` replaces it with tanh(V1 / eps)
(env.chua_eps). The initial condition is the paper's training one
(env.chua_v1_0, chua_v2_0, chua_i_0).
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.envs.ode.systems import ODESystemSpec
from steppo.envs.ode.systems.obs_factory import base_obs_factory

_C1 = 1.0 / 9.0  # farads (Riley et al. Case 2, §5.2.1)
_C2 = 1.0  # farads
_R = 1.0 / 0.7  # ohms
_L = 1.0 / 7.0  # henries


def _rhs(t: Array, y: Array, task: Array) -> Array:
    """Chua-circuit RHS. task = [m]; C1, C2, R, L fixed at Case 2 values."""
    m = task[0]
    v1, v2, i = y[0], y[1], y[2]
    f_r = -(m / _C1) * jnp.sign(v1)
    v1_dot = -v1 / (_C1 * _R) + v2 / (_C1 * _R) + f_r
    v2_dot = v1 / (_C2 * _R) - v2 / (_C2 * _R) + i / _C2
    i_dot = -v2 / _L
    return jnp.array([v1_dot, v2_dot, i_dot])


def _rhs_smooth_with_eps(eps: Array):
    """Build the tanh(V1/eps)-regularized RHS closure for a given eps."""

    def _rhs_smooth(t: Array, y: Array, task: Array) -> Array:
        m = task[0]
        v1, v2, i = y[0], y[1], y[2]
        f_r = -(m / _C1) * jnp.tanh(v1 / eps)
        v1_dot = -v1 / (_C1 * _R) + v2 / (_C1 * _R) + f_r
        v2_dot = v1 / (_C2 * _R) - v2 / (_C2 * _R) + i / _C2
        i_dot = -v2 / _L
        return jnp.array([v1_dot, v2_dot, i_dot])

    return _rhs_smooth


_EPS = 1e-3  # boundary-layer half-width used by SPEC_SMOOTH.rhs (sanity_check.py default)
_rhs_smooth = _rhs_smooth_with_eps(jnp.float32(_EPS))


def make_rhs(config):
    """Return the chua_smooth RHS using `config.chua_eps` (falls back to
    `_EPS` if unset) — same pattern as fosm.make_rhs."""
    eps = jnp.float32(getattr(config, "chua_eps", _EPS))
    return _rhs_smooth_with_eps(eps)


def _sample_task(key: PRNGKey, config) -> Array:
    """Sample or deterministically select the diode-gain parameter m."""
    if config.sample_m:
        return jax.random.uniform(
            key,
            shape=(1,),
            minval=jnp.float32(config.m_min),
            maxval=jnp.float32(config.m_max),
        )
    return jnp.array([config.m_min], dtype=jnp.float32)


def _y0(task: Array, key: PRNGKey, config) -> Array:
    """Return the configured Chua-circuit initial state (V1, V2, i)."""
    return jnp.array(
        [config.chua_v1_0, config.chua_v2_0, config.chua_i_0],
        dtype=jnp.float32,
    )


def _state(state, config) -> Array:
    """Clip the circuit state for use as an observation feature."""
    return jnp.clip(state.y, -20.0, 20.0)


_obs, _FEATURE_DIMS = base_obs_factory(_state, state_dim=3, y_dim=3)


SPEC = ODESystemSpec(
    name="chua",
    y_dim=3,
    obs_dim=15,
    rhs=_rhs,
    sample_task=_sample_task,
    y0=_y0,
    obs=_obs,
    feature_dims=_FEATURE_DIMS,
)

SPEC_SMOOTH = ODESystemSpec(
    name="chua_smooth",
    y_dim=3,
    obs_dim=15,
    rhs=_rhs_smooth,
    sample_task=_sample_task,
    y0=_y0,
    obs=_obs,
    feature_dims=_FEATURE_DIMS,
)
