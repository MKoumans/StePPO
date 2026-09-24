"""Chemical cascade — layered reaction-diffusion network, linear and stiff.

  10 layers x 5 chemicals -> y in R^50 (state = concat of per-layer 5-vectors).

  dy/dt = (A_base - lambda*I) @ y

A_base is block-tridiagonal with fixed constants: diagonal block R is a
within-layer reaction chain X1->X2->X3->X4->X5 with widely separated rate
constants (1e3, 1e2, 1e1, 1e0) — the source of baseline stiffness, same idea
as Robertson's k1/k3 split — plus a small terminal leak on X5 for per-layer
stability. Off-diagonal blocks D = d*I diffusively couple the same chemical
between adjacent layers (interior layers subtract 2D, boundary layers
subtract D); this diffusive chain produces the layer-1 -> layer-10 delay,
since a pulse in layer 1 takes O(L^2/d) time to reach layer 10.

Hidden task parameter: lambda ~ LogUniform[cc_lam_min, cc_lam_max], a uniform
decay rate added to every state — larger lambda drains the cascade faster
and increases the stiffness ratio further.

Initial condition: a concentration pulse in layer 1's chemicals, zero
elsewhere, to exercise the cross-layer delay.

System-specific observation feature (shared ones: see obs_factory.py):
  state:        [clip(y_i, -5, 5)/5 for i in 0..49]                                    (50D)
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.envs.ode.systems import ODESystemSpec
from steppo.envs.ode.systems.obs_factory import base_obs_factory
from steppo.envs.ode.systems.pulse import gaussian_pulse_train_jittered, make_pulsed_rhs

_NUM_LAYERS = 10
_NUM_CHEMICALS = 5
_N = _NUM_LAYERS * _NUM_CHEMICALS  # 50

_CHAIN_RATES = jnp.array([1e3, 1e2, 1e1, 1e0], dtype=jnp.float32)  # widely separated -> stiff
_TERMINAL_LEAK = 0.1  # keeps each layer block stable alone
_DIFFUSION = 0.5  # inter-layer coupling strength d


def _reaction_block() -> Array:
    """5x5 within-layer reaction chain X1->X2->X3->X4->X5 (+ terminal leak)."""
    R = jnp.zeros((_NUM_CHEMICALS, _NUM_CHEMICALS), dtype=jnp.float32)
    for i in range(_NUM_CHEMICALS - 1):
        R = R.at[i, i].add(-_CHAIN_RATES[i])
        R = R.at[i + 1, i].add(_CHAIN_RATES[i])
    R = R.at[_NUM_CHEMICALS - 1, _NUM_CHEMICALS - 1].add(-_TERMINAL_LEAK)
    return R


_R_BLOCK = _reaction_block()
_D_BLOCK = _DIFFUSION * jnp.eye(_NUM_CHEMICALS, dtype=jnp.float32)


def _base_matrix() -> Array:
    """Block-tridiagonal reaction-diffusion matrix (decay added per-task)."""
    A = jnp.zeros((_N, _N), dtype=jnp.float32)
    for i in range(_NUM_LAYERS):
        n_neighbors = (i > 0) + (i < _NUM_LAYERS - 1)
        diag_block = _R_BLOCK - n_neighbors * _D_BLOCK
        rows = slice(i * _NUM_CHEMICALS, (i + 1) * _NUM_CHEMICALS)
        A = A.at[rows, rows].set(diag_block)
        if i > 0:
            prev = slice((i - 1) * _NUM_CHEMICALS, i * _NUM_CHEMICALS)
            A = A.at[rows, prev].set(_D_BLOCK)
        if i < _NUM_LAYERS - 1:
            nxt = slice((i + 1) * _NUM_CHEMICALS, (i + 2) * _NUM_CHEMICALS)
            A = A.at[rows, nxt].set(_D_BLOCK)
    return A


_A_BASE = _base_matrix()  # (50, 50), constant


def _rhs(t: Array, y: Array, task: Array) -> Array:
    """Chemical cascade RHS. task = [lambda] (uniform decay rate)."""
    lam = task[0]
    A = _A_BASE - lam * jnp.eye(_N, dtype=jnp.float32)
    return A @ y


def make_rhs(config):
    """RHS with optional periodic pulse forcing added into `config.cc_pulse_dim`;
    disabled (`cc_pulse_enabled=False`) returns plain `_rhs` unchanged.
    `cc_pulse_random` in [0, 1] jitters the pulse period per-episode (0 = fixed,
    1 = fully random within neighboring half-periods)."""
    if not getattr(config, "cc_pulse_enabled", False):
        return _rhs
    forcing = gaussian_pulse_train_jittered(
        config.cc_pulse_period,
        config.cc_pulse_width,
        config.cc_pulse_amplitude,
        getattr(config, "cc_pulse_random", 0.0),
    )
    return make_pulsed_rhs(_rhs, config.cc_pulse_dim, forcing)


def _sample_task(key: PRNGKey, config) -> Array:
    """Sample or deterministically select the cascade decay parameter."""
    if config.sample_cc_lam:
        log_lam = jax.random.uniform(
            key,
            shape=(1,),
            minval=jnp.log(jnp.float32(config.cc_lam_min)),
            maxval=jnp.log(jnp.float32(config.cc_lam_max)),
        )
        return jnp.exp(log_lam)
    lam_mid = jnp.exp(
        (jnp.log(jnp.float32(config.cc_lam_min)) + jnp.log(jnp.float32(config.cc_lam_max))) / 2.0
    )
    return jnp.array([lam_mid], dtype=jnp.float32)


def _y0(task: Array, key: PRNGKey, config) -> Array:
    """Return the initial concentration vector for one task."""
    y0 = jnp.zeros((_N,), dtype=jnp.float32)
    if config.sample_y0:
        pulse = jax.random.uniform(key, shape=(), minval=0.5, maxval=2.0, dtype=jnp.float32)
    else:
        pulse = jnp.float32(1.0)
    return y0.at[:_NUM_CHEMICALS].set(pulse)


def _state(state, config) -> Array:
    """Clip and normalize cascade concentrations for the observation."""
    return jnp.clip(state.y, -5.0, 5.0) / jnp.float32(5.0)


_obs, _FEATURE_DIMS = base_obs_factory(_state, state_dim=_N, y_dim=_N)


SPEC = ODESystemSpec(
    name="chemical_cascade",
    y_dim=_N,
    obs_dim=2 * _N + 9,
    rhs=_rhs,
    sample_task=_sample_task,
    y0=_y0,
    obs=_obs,
    feature_dims=_FEATURE_DIMS,
)
