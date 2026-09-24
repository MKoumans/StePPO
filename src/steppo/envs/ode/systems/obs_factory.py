"""Observation builders shared by all ODE systems.

Features, selected and ordered by `config.obs_features`:
  state:        system-specific transform of y
  direction:    the solver's RHS evaluation f                                        (y_dim)
  step_context: [log_err/5, clip(log_err_delta, -10, 10)/5, last_keep, t/t_end]      (4)
  solver_trend: [log(dt/dt0)/10, accept_ema, log_error_ema/5, tanh(reject_streak/4),
                 budget_exhausted]                                                   (5)
"""

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import Array

# ── Shared feature builders, each with signature (state, config) ────────────


def _direction(state, config) -> Array:
    """Return the derivative-direction features for the current solver state."""
    f = state.solver_state_f
    return f


def _step_context(state, config) -> Array:
    """Return normalized time, error, and acceptance context features."""
    return jnp.array(
        [
            state.last_log_error / jnp.float32(5.0),
            jnp.clip(state.log_error_delta, -10.0, 10.0) / jnp.float32(5.0),
            state.last_keep_step.astype(jnp.float32),
            state.t / jnp.float32(config.t_end),
        ],
        dtype=jnp.float32,
    )


def _solver_trend(state, config) -> Array:
    """Return normalized step-size and solver-history features."""
    return jnp.array(
        [
            jnp.log(state.dt / jnp.float32(config.dt0) + jnp.float32(1e-10)) / jnp.float32(10.0),
            state.accept_ema,
            state.log_error_ema / jnp.float32(5.0),
            jnp.tanh(state.reject_streak.astype(jnp.float32) / jnp.float32(4.0)),
            state.budget_exhausted,
        ],
        dtype=jnp.float32,
    )


SHARED_FEATURE_DIMS = {"direction": None, "step_context": 4, "solver_trend": 5}
"""Dims for the shared features. `direction` is None here since it depends on
the system's y_dim — `base_obs_factory` fills in the concrete value."""


def base_obs_factory(
    state_fn: Callable[[object, object], Array],
    state_dim: int,
    y_dim: int,
) -> Tuple[Callable[[object, object], Array], dict]:
    """Build an observation function and feature widths for one ODE system."""
    builders = {
        "state": state_fn,
        "direction": _direction,
        "step_context": _step_context,
        "solver_trend": _solver_trend,
    }
    feature_dims = {"state": state_dim, "direction": y_dim, "step_context": 4, "solver_trend": 5}

    def _obs(state, config) -> Array:
        unknown = [f for f in config.obs_features if f not in builders]
        if unknown:
            raise ValueError(
                f"obs_features contains unrecognized feature(s) {unknown} — known "
                f"features: {sorted(builders)}. A checkpoint's saved obs_features "
                f"list must match the feature-builder set it was trained under; "
                f"silently dropping an unknown feature would feed the policy a "
                f"scrambled observation. If this checkpoint predates a feature-"
                f"layout change, evaluate it with the code version it was trained "
                f"under instead."
            )
        parts = [jnp.atleast_1d(builders[f](state, config)) for f in config.obs_features]
        return jnp.concatenate(parts).astype(jnp.float32)

    return _obs, feature_dims
