"""Equation-level invariants for registered ODE system definitions."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from steppo.envs.ode.systems import get_rhs, get_system
from steppo.envs.ode.systems.pulse import gaussian_pulse_train_jittered, make_pulsed_rhs


def test_robertson_kinetics_preserve_total_mass_in_the_rhs():
    state = jnp.array([0.3, 0.2, 0.5], dtype=jnp.float32)
    for k2 in (1.0e3, 1.0e6, 1.0e9):
        derivative = get_system("robertson").rhs(0.0, state, jnp.array([k2]))
        assert derivative.shape == state.shape
        # The k2=1e9 case is stiff enough that float32 cancellation leaves a
        # small residual even though each transfer cancels symbolically.
        assert float(jnp.sum(derivative)) == pytest.approx(0.0, abs=1e-4)


def test_brusselator_fixed_point_is_stationary_for_each_stiffness():
    rhs = get_system("brusselator").rhs
    for B in (2.0, 10.0, 50.0):
        derivative = rhs(0.0, jnp.array([1.0, B]), jnp.array([B]))
        assert np.allclose(derivative, [0.0, 0.0], atol=1e-6)


def test_chua_rhs_matches_the_circuit_equations():
    state = jnp.array([0.5, -0.25, 0.1])
    diode_gain = 0.2
    rhs = get_rhs(SimpleNamespace(system="chua"))

    derivative = rhs(0.0, state, jnp.array([diode_gain]))
    c1, c2, resistance, inductance = 1.0 / 9.0, 1.0, 1.0 / 0.7, 1.0 / 7.0
    expected = [
        -state[0] / (c1 * resistance) + state[1] / (c1 * resistance) - diode_gain / c1,
        state[0] / (c2 * resistance) - state[1] / (c2 * resistance) + state[2] / c2,
        -state[1] / inductance,
    ]

    assert np.allclose(derivative, expected, rtol=1e-6)


def test_van_der_pol_rhs_matches_its_task_conditioned_oscillator_equation():
    state = jnp.array([1.25, -0.75])
    mu = 3.0
    derivative = get_system("van_der_pol").rhs(0.0, state, jnp.array([mu]))
    expected = [state[1], mu * (1.0 - state[0] ** 2) * state[1] - state[0]]

    assert np.allclose(derivative, expected, rtol=1e-6)


def test_chemical_cascade_transfers_conserve_mass_except_decay_and_terminal_leak():
    # Transfer terms reach ~5e3 and cancel in the sum; float32 leaves a ~2e-3 residual.
    jax.config.update("jax_enable_x64", True)  # restored by conftest
    state = jnp.linspace(0.1, 5.0, 50)
    decay = 0.3
    derivative = get_system("chemical_cascade").rhs(0.0, state, jnp.array([decay]))

    expected_mass_change = -decay * jnp.sum(state) - 0.1 * jnp.sum(state[4::5])
    assert float(jnp.sum(derivative)) == pytest.approx(float(expected_mass_change), abs=1e-5)


def test_pulse_wrapper_changes_only_the_configured_derivative_component():
    def base_rhs(_t, _y, _task):
        return jnp.array([1.0, -2.0, 3.0])

    forcing = gaussian_pulse_train_jittered(
        period=2.0,
        width=0.25,
        amplitude=4.0,
        random_ratio=0.0,
    )
    forced_rhs = make_pulsed_rhs(base_rhs, dim=1, forcing=forcing)

    derivative = forced_rhs(2.0, jnp.zeros(3), jnp.array([0.0, 17.0]))

    assert np.allclose(derivative, [1.0, 2.0, 3.0])
    # The episode seed is part of the forcing input and repeated evaluation is
    # deterministic, as required for JAX tracing/replay.
    assert np.array_equal(
        forced_rhs(2.0, jnp.zeros(3), jnp.array([0.0, 17.0])),
        derivative,
    )


def test_fitzhugh_nagumo_pulse_is_applied_to_configured_state_only():
    base = get_rhs(SimpleNamespace(system="fitzhugh_nagumo", fhn_pulse_enabled=False))
    pulsed = get_rhs(
        SimpleNamespace(
            system="fitzhugh_nagumo",
            fhn_pulse_enabled=True,
            fhn_pulse_period=1.0,
            fhn_pulse_width=0.25,
            fhn_pulse_amplitude=2.0,
            fhn_pulse_random=0.0,
            fhn_pulse_dim=1,
        )
    )
    state = jnp.array([0.5, -0.25])
    task = jnp.array([0.1, 123.0])

    base_derivative = base(1.0, state, task)
    pulse_derivative = pulsed(1.0, state, task)

    assert np.allclose(pulse_derivative - base_derivative, [0.0, 2.0])


def test_fosm_smooth_rhs_uses_configured_boundary_layer_width():
    t = 0.02
    state = jnp.array([0.5])
    task = jnp.array([0.8])
    rhs = get_rhs(SimpleNamespace(system="fosm_smooth", fosm_eps=0.25))

    expected = 0.8 * np.sin(50.0 * t) - 2.0 * np.tanh(0.5 / 0.25)
    assert float(rhs(t, state, task)[0]) == pytest.approx(expected, rel=1e-6)
