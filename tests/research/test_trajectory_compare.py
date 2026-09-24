import numpy as np
import pytest


def test_relative_l2_error_matches_hand_computed_ratio():
    from steppo.training.trajectory_compare import relative_l2_error

    y = np.array([[3.0, 4.0]])  # norm 5
    y_ref = np.array([[0.0, 0.0]])  # norm 0

    err = relative_l2_error(y, y_ref, atol=1.0)

    assert np.allclose(err, [5.0 / 1.0])


def test_final_state_from_trajectory_picks_last_finite_step_per_episode():
    from steppo.training.trajectory_compare import final_state_from_trajectory

    # Episode 0 reaches t_end at step 2 (index 1), episode 1 at step 3 (index 2).
    ts = np.array(
        [
            [0.0, 1.0, np.inf],
            [0.0, 0.5, 1.0],
        ]
    )
    ys = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [np.inf, np.inf]],
            [[1.0, 1.0], [1.5, 1.5], [3.0, 3.0]],
        ]
    )

    final_y = final_state_from_trajectory(ts, ys)

    assert np.allclose(final_y, [[2.0, 2.0], [3.0, 3.0]])


def test_interp_at_linearly_interpolates_within_range_and_clips_outside():
    from steppo.training.trajectory_compare import interp_at

    # One episode: y0(t) = 2t, y1(t) = t, valid up to t=2.
    ref_ts = np.array([[0.0, 1.0, 2.0, np.inf, np.inf]])
    ref_ys = np.array([[[0.0, 0.0], [2.0, 1.0], [4.0, 2.0], [np.inf, np.inf], [np.inf, np.inf]]])
    query_ts = np.array([[0.5, 1.5, 5.0]])  # last query is out-of-range -> clip to t=2

    out = interp_at(ref_ts, ref_ys, query_ts)

    assert out.shape == (1, 3, 2)
    assert np.allclose(out[0, 0], [1.0, 0.5])
    assert np.allclose(out[0, 1], [3.0, 1.5])
    assert np.allclose(out[0, 2], [4.0, 2.0])


def test_last_valid_value_picks_last_finite_step_per_episode():
    from steppo.training.trajectory_compare import last_valid_value

    ts = np.array(
        [
            [0.0, 1.0, np.inf],
            [0.0, 0.5, 1.0],
        ]
    )
    values = np.array(
        [
            [10.0, 20.0, np.inf],
            [10.0, 15.0, 30.0],
        ]
    )

    result = last_valid_value(ts, values)

    assert np.allclose(result, [20.0, 30.0])


def test_time_average_matches_hand_computed_trapezoid_over_duration():
    from steppo.training.trajectory_compare import time_average

    ts = np.array([[0.0, 1.0, 2.0]])
    values = np.array([[3.0, -0.30103, -0.60206]])

    result = time_average(ts, values)

    # trapz = 0.5*(3.0-0.30103) + 0.5*(-0.30103-0.60206) = 1.349485 - 0.451545 = 0.89794
    # average = 0.89794 / duration(2.0) = 0.44897
    assert np.allclose(result, [0.44897], atol=1e-4)


def test_time_average_ignores_padded_tail():
    from steppo.training.trajectory_compare import time_average

    ts = np.array([[0.0, 1.0, np.inf]])
    values = np.array([[0.0, 2.0, np.inf]])

    result = time_average(ts, values)

    # trapz(|0,2| over [0,1]) = 1.0; duration = 1.0; average = 1.0
    assert np.allclose(result, [1.0])


def test_local_error_matches_pointwise_relative_l2_ratio_in_linear_space():
    from steppo.training.trajectory_compare import local_error

    ref_ts = np.array([[0.0, 1.0, 2.0]])
    ref_ys = np.array([[[0.0], [1.0], [2.0]]])
    ts = np.array([[0.0, 1.0, 2.0]])
    ys = np.array([[[0.5], [1.5], [2.5]]])

    result = local_error(ref_ts, ref_ys, ts, ys, atol=0.0)

    # t=0: |0.5-0|/(0+0) = inf. t=1: 0.5/1=0.5. t=2: 0.5/2=0.25. Un-logged.
    assert np.isinf(result[0, 0])
    assert np.allclose(result[0, 1:], [0.5, 0.25])


def test_local_error_pads_the_tail_with_inf():
    from steppo.training.trajectory_compare import local_error

    ref_ts = np.array([[0.0, 1.0, 2.0]])
    ref_ys = np.array([[[0.0], [1.0], [2.0]]])
    ts = np.array([[0.0, 1.0, np.inf]])
    ys = np.array([[[0.0], [1.0], [np.inf]]])

    result = local_error(ref_ts, ref_ys, ts, ys, atol=0.0)

    assert np.isinf(result[0, 2])


def test_local_error_log10_kind_matches_pointwise_to_log10_clipped_ratio():
    from steppo.training.trajectory_compare import local_error, to_log10_clipped

    ref_ts = np.array([[0.0, 1.0, 2.0]])
    ref_ys = np.array([[[0.0], [1.0], [2.0]]])
    ts = np.array([[0.0, 1.0, 2.0]])
    ys = np.array([[[0.5], [1.5], [2.5]]])

    result = local_error(ref_ts, ref_ys, ts, ys, atol=0.0, kind="log10")

    # t=0: |0.5-0|/(0+0) = inf -> ceiling. t=1: 0.5/1=0.5. t=2: 0.5/2=0.25.
    expected = to_log10_clipped(np.array([np.inf, 0.5, 0.25]))
    assert np.allclose(result[0], expected)


def test_local_error_log10_kind_maps_padded_tail_to_the_ceiling():
    from steppo.training.trajectory_compare import LOG10_ERR_CEIL, local_error

    ref_ts = np.array([[0.0, 1.0, 2.0]])
    ref_ys = np.array([[[0.0], [1.0], [2.0]]])
    ts = np.array([[0.0, 1.0, np.inf]])
    ys = np.array([[[0.0], [1.0], [np.inf]]])

    result = local_error(ref_ts, ref_ys, ts, ys, atol=0.0, kind="log10")

    assert result[0, 2] == LOG10_ERR_CEIL


def test_local_error_rejects_an_unknown_kind():
    from steppo.training.trajectory_compare import local_error

    ref_ts = np.array([[0.0]])
    ref_ys = np.array([[[0.0]]])
    ts = np.array([[0.0]])
    ys = np.array([[[0.0]]])

    with pytest.raises(ValueError, match="kind"):
        local_error(ref_ts, ref_ys, ts, ys, atol=0.0, kind="bogus")


def test_to_log10_clipped_maps_nonfinite_error_to_ceiling():
    from steppo.training.trajectory_compare import LOG10_ERR_CEIL, to_log10_clipped

    result = to_log10_clipped(np.array([np.inf, np.nan]))

    assert np.allclose(result, [LOG10_ERR_CEIL, LOG10_ERR_CEIL])


def test_to_log10_clipped_floors_a_near_zero_error():
    from steppo.training.trajectory_compare import LOG10_ERR_FLOOR, to_log10_clipped

    result = to_log10_clipped(np.array([0.0]))

    assert np.allclose(result, [LOG10_ERR_FLOOR])


def test_log10_time_integrated_error_averages_local_error_in_log_space():
    from steppo.training.trajectory_compare import log10_time_integrated_error

    ref_ts = np.array([[0.0, 1.0, 2.0]])
    ref_ys = np.array([[[0.0], [1.0], [2.0]]])
    ts = np.array([[0.0, 1.0, 2.0]])
    ys = np.array([[[0.5], [1.5], [2.5]]])

    result = log10_time_integrated_error(ref_ts, ref_ys, ts, ys, atol=0.0)

    assert np.allclose(result, [0.44897], atol=1e-4)


def test_log10_time_integrated_error_matches_time_average_of_local_error_log10_kind():
    from steppo.training.trajectory_compare import (
        local_error,
        log10_time_integrated_error,
        time_average,
    )

    ref_ts = np.array([[0.0, 1.0, 2.0]])
    ref_ys = np.array([[[0.0], [1.0], [2.0]]])
    ts = np.array([[0.0, 1.0, 2.0]])
    ys = np.array([[[0.5], [1.5], [2.5]]])

    expected = time_average(ts, local_error(ref_ts, ref_ys, ts, ys, atol=0.0, kind="log10"))
    result = log10_time_integrated_error(ref_ts, ref_ys, ts, ys, atol=0.0)

    assert np.allclose(result, expected)


def test_log10_time_integrated_error_stays_bounded_when_reference_passes_near_zero():
    from steppo.training.trajectory_compare import (
        local_error,
        log10_time_integrated_error,
        time_average,
    )

    # Reference magnitude passes very close to zero at the middle sample
    # (t=1) — e.g. a decaying system's trajectory near its asymptote.
    ref_ts = np.array([[0.0, 1.0, 2.0]])
    ref_ys = np.array([[[1.0], [1e-8], [1.0]]])
    ts = np.array([[0.0, 1.0, 2.0]])
    ys = np.array([[[1.01], [0.01], [1.01]]])

    # The naive linear-space average of the raw pointwise ratio blows up:
    # the local ratio at t=1 is |0.01-1e-8|/(1e-8+0) ~= 1e6, so a plain
    # trapezoidal average of the unlogged local errors is dominated by it.
    naive_linear_average = time_average(
        ts,
        local_error(ref_ts, ref_ys, ts, ys, atol=0.0),
    )
    assert naive_linear_average[0] > 1e5  # the blowup the fix avoids

    result = log10_time_integrated_error(ref_ts, ref_ys, ts, ys, atol=0.0)

    # Clipping each point to the log10 ceiling *before* averaging keeps the
    # log-space result near the ordinary-error points' own scale (~0.5),
    # orders of magnitude away from log10(naive_linear_average) (~5.7).
    assert np.allclose(result, [0.5])
    assert result[0] < np.log10(naive_linear_average[0]) - 4
