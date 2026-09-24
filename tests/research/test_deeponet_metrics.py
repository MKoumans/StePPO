"""The NumPy-only DeepONet metrics must match StePPO's shared definitions."""

import numpy as np

from research.baselines.deeponet import relerr_metrics as deeponet
from steppo.training import trajectory_compare as steppo_metrics


def test_relative_and_log_errors_match_shared_trajectory_metrics():
    reference = np.array([[1.0, -2.0], [0.0, 0.0]])
    prediction = np.array([[1.2, -1.7], [1e-8, 0.0]])

    expected = steppo_metrics.relative_l2_error(prediction, reference, atol=1e-5)
    actual = deeponet.relative_l2_error(prediction, reference, atol=1e-5)

    assert np.allclose(actual, expected)
    assert np.allclose(
        deeponet.log10_rel_error(prediction, reference, atol=1e-5),
        steppo_metrics.to_log10_clipped(expected),
    )


def test_nonfinite_errors_map_to_ceiling_and_extremes_stay_clipped():
    errors = np.array([0.0, 1e-30, 1.0, np.inf, np.nan])

    transformed = deeponet.to_log10_clipped(errors)

    assert np.all(np.isfinite(transformed))
    assert np.all(transformed >= deeponet.LOG10_ERR_FLOOR)
    assert np.all(transformed <= deeponet.LOG10_ERR_CEIL)
    assert transformed[-2:].tolist() == [deeponet.LOG10_ERR_CEIL] * 2


def test_pid_scaled_error_uses_statewise_max_reference_scale():
    reference = np.array([[[2.0, -4.0], [0.5, 0.25]]])
    prediction = reference + np.array([[[0.3, 0.4], [0.1, -0.1]]])
    expected = np.linalg.norm(prediction - reference, axis=-1) / (
        0.01 + 0.1 * np.max(np.abs(reference), axis=-1)
    )

    assert np.allclose(deeponet.pid_scaled_error(prediction, reference, 0.01, 0.1), expected)


def test_time_integrated_error_is_a_trapezoidal_time_average():
    times = np.array([0.0, 0.2, 1.0])
    reference = np.array([[1.0], [2.0], [4.0]])
    prediction = np.array([[1.1], [1.8], [4.8]])
    pointwise = deeponet.to_log10_clipped(
        deeponet.relative_l2_error(prediction, reference, atol=0.0)
    )
    expected = np.trapezoid(pointwise, times) / (times[-1] - times[0])

    assert (
        deeponet.time_integrated_log10_rel_error(
            times,
            prediction,
            reference,
            atol=0.0,
        )
        == expected
    )


def test_zero_duration_uses_the_last_log_error():
    times = np.array([2.0, 2.0])
    reference = np.array([[1.0], [2.0]])
    prediction = np.array([[1.1], [2.2]])
    expected = deeponet.log10_rel_error(prediction[-1], reference[-1], atol=0.0)

    assert (
        deeponet.time_integrated_log10_rel_error(
            times,
            prediction,
            reference,
            atol=0.0,
        )
        == expected
    )
