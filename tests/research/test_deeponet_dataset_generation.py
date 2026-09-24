"""CPU-safe invariants for the DeepONet trajectory dataset generator."""

import numpy as np
import pytest

from research.baselines.deeponet.dataset_generation import (
    generate_trajectories,
    sample_binned_values,
    sample_tasks,
    scalar_decay_trajectories,
)


def test_binned_sampling_is_seeded_and_never_crosses_gaps():
    bins = ((1.0, 2.0), (10.0, 20.0))
    first = sample_binned_values(np.random.default_rng(4), bins, 500)
    second = sample_binned_values(np.random.default_rng(4), bins, 500)

    assert np.array_equal(first, second)
    assert np.all(((first >= 1.0) & (first <= 2.0)) | ((first >= 10.0) & (first <= 20.0)))


def test_log_binned_sampling_is_uniform_in_log_space():
    values = sample_binned_values(
        np.random.default_rng(8),
        ((1.0, 100.0),),
        10_000,
        scheme="log-binned",
    )

    assert np.median(values) == pytest.approx(10.0, rel=0.04)


def test_sample_tasks_applies_the_fixed_system_initial_conditions():
    rng = np.random.default_rng(3)
    bins = ((2.0, 3.0),)

    vdp = sample_tasks("van_der_pol", rng, {}, bins, 4)
    brusselator = sample_tasks(
        "brusselator",
        np.random.default_rng(3),
        {"y0_x": 1.25, "y0_y": 2.5},
        bins,
        4,
    )

    assert vdp.shape == brusselator.shape == (4, 3)
    assert np.allclose(vdp[:, 1:], [[0.0, -2.0]] * 4)
    assert np.allclose(brusselator[:, 1:], [[1.25, 2.5]] * 4)
    with pytest.raises(ValueError, match="not defined"):
        sample_tasks("robertson", rng, {}, bins, 1)


def test_scalar_decay_generator_matches_analytic_solution():
    parameters = np.array([0.5, 2.0])
    times = np.array([0.0, 0.25, 1.0])

    values = scalar_decay_trajectories(parameters, times)

    assert values.shape == (2, 3)
    assert np.allclose(values, np.exp(-np.outer(parameters, times)))
    assert np.array_equal(values[:, 0], np.ones(2))


@pytest.mark.parametrize(
    ("system", "tasks"),
    [
        ("van_der_pol", np.array([[1.0, 0.0, -2.0]])),
        ("brusselator", np.array([[2.0, 1.0, 1.0]])),
    ],
)
def test_cpu_trajectory_generation_preserves_initial_state_and_shape(system, tasks):
    times = np.linspace(0.0, 0.05, 3)

    trajectories = generate_trajectories(system, tasks, times, t_end=0.05)

    assert trajectories.shape == (1, 3, 2)
    assert np.all(np.isfinite(trajectories))
    assert np.allclose(trajectories[0, 0], tasks[0, 1:])


def test_cpu_trajectory_generation_rejects_unknown_system():
    with pytest.raises(ValueError, match="not defined"):
        generate_trajectories("robertson", np.array([[1.0, 0.0]]), np.array([0.0, 0.1]), 0.1)
