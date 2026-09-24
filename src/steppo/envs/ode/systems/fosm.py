"""First-order sliding-mode controller (FOSM), a nonsmooth ODE system.

  ẇ = φ(t) + F_r,   φ(t) = D·sin(ω·t),   F_r ∈ -k·sign(w)

Case 1 of Riley et al. (2025), arXiv:2501.08934. k > D is required for
finite-time convergence to the sliding surface w = 0; k and ω are fixed.
Task: D ~ Uniform[D_min, D_max].

`fosm` keeps the discontinuous sign(w); `fosm_smooth` replaces it with
tanh(w / eps) (env.fosm_eps), which avoids chattering under an implicit RK
solver. Smaller eps is closer to sign(w) but costs more steps.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.envs.ode.systems import ODESystemSpec
from steppo.envs.ode.systems.obs_factory import base_obs_factory

_K = 2.0  # sliding-mode gain (Riley et al. Case 1, §5.1.1)
_OMEGA = 50.0  # forcing frequency, rad/s (Riley et al. Case 1, §5.1.1)


def _rhs(t: Array, y: Array, task: Array) -> Array:
    """FOSM RHS. task = [D]; k, omega fixed at Riley et al. Case 1 values."""
    D = task[0]
    phi = D * jnp.sin(_OMEGA * t)
    F_r = -_K * jnp.sign(y[0])
    return jnp.array([phi + F_r])


_EPS = 1e-3  # boundary-layer half-width for the smoothed sign() below


def _rhs_smooth(t: Array, y: Array, task: Array) -> Array:
    """FOSM RHS with sign(w) replaced by tanh(w/eps) (chattering fix)."""
    D = task[0]
    phi = D * jnp.sin(_OMEGA * t)
    F_r = -_K * jnp.tanh(y[0] / _EPS)
    return jnp.array([phi + F_r])


def make_rhs(config):
    """Return the fosm_smooth RHS using `config.fosm_eps` (falls back to
    `_EPS` if unset). This is what actual training runs go through
    (steppo.envs.ode.systems.get_rhs -> ode_env.py); sanity_check.py bypasses
    it and uses `SPEC_SMOOTH.rhs` (fixed at `_EPS`) directly."""
    eps = jnp.float32(getattr(config, "fosm_eps", _EPS))

    def _rhs_configured(t: Array, y: Array, task: Array) -> Array:
        D = task[0]
        phi = D * jnp.sin(_OMEGA * t)
        F_r = -_K * jnp.tanh(y[0] / eps)
        return jnp.array([phi + F_r])

    return _rhs_configured


def _sample_task(key: PRNGKey, config) -> Array:
    """Sample or deterministically select the forcing amplitude D."""
    if config.sample_D:
        return jax.random.uniform(
            key,
            shape=(1,),
            minval=jnp.float32(config.D_min),
            maxval=jnp.float32(config.D_max),
        )
    return jnp.array([config.D_min], dtype=jnp.float32)


def _y0(task: Array, key: PRNGKey, config) -> Array:
    """Return the configured FOSM initial state w(0)."""
    return jnp.array([config.fosm_w0], dtype=jnp.float32)


def _state(state, config) -> Array:
    """Clip the sliding variable for use as an observation feature."""
    return jnp.clip(state.y[0], -10.0, 10.0)


_obs, _FEATURE_DIMS = base_obs_factory(_state, state_dim=1, y_dim=1)


SPEC = ODESystemSpec(
    name="fosm",
    y_dim=1,
    obs_dim=11,
    rhs=_rhs,
    sample_task=_sample_task,
    y0=_y0,
    obs=_obs,
    feature_dims=_FEATURE_DIMS,
)

SPEC_SMOOTH = ODESystemSpec(
    name="fosm_smooth",
    y_dim=1,
    obs_dim=11,
    rhs=_rhs_smooth,
    sample_task=_sample_task,
    y0=_y0,
    obs=_obs,
    feature_dims=_FEATURE_DIMS,
)
