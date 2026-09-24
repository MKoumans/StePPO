"""Unit tests for the trained-vs-held-out bin comparison in
research/ode/compare_experiment.py and research/ode/post_run_analysis/steps_vs_mu.py.

Pure-function tests only — no training, no checkpoints, no GPU solves needed to
verify this wiring (see session history: manual end-to-end runs were previously
used to verify similar wiring, which is slow and doesn't persist as regression
coverage; these atomic tests replace that approach for the new bin-comparison
seams)."""

import numpy as np
import pytest

from research.ode.compare_experiment import (
    _align_repeat_series,
    _config_label,
    _discover_return_datasets,
    _fit_best_gmm_1d,
    _fit_gmm_1d,
    summarize_bin_comparison,
    summarize_step_diff_stats,
)
from research.ode.execute_comparison import _bin_is_trained


def test_bin_is_trained_empty_train_bins_means_continuous_scheme_saw_everything():
    assert _bin_is_trained(10.0, 20.0, ()) is True
    assert _bin_is_trained(60.0, 100.0, ()) is True


def test_bin_is_trained_bin_inside_a_train_bin():
    train_bins = ((5.0, 10.0), (20.0, 35.0), (60.0, 100.0))
    assert _bin_is_trained(5.0, 10.0, train_bins) is True
    assert _bin_is_trained(20.0, 35.0, train_bins) is True


def test_bin_is_trained_bin_in_a_gap():
    train_bins = ((5.0, 10.0), (20.0, 35.0), (60.0, 100.0))
    assert _bin_is_trained(10.0, 20.0, train_bins) is False
    assert _bin_is_trained(35.0, 60.0, train_bins) is False


def test_summarize_bin_comparison_splits_trained_and_held_out_means():
    bin_results = {
        "run_a": {
            "exp_name": "van_der_pol_e1",
            "scheme": "uniform",
            "bins": [(5.0, 10.0), (10.0, 20.0), (20.0, 35.0)],
            "improvement": np.array([0.4, 0.5, 0.6]),
            "trained": [True, True, True],  # continuous scheme: everything trained
        },
        "run_b": {
            "exp_name": "van_der_pol_e3",
            "scheme": "binned",
            "bins": [(5.0, 10.0), (10.0, 20.0), (20.0, 35.0)],
            "improvement": np.array([0.7, 0.1, 0.8]),
            "trained": [True, False, True],  # [10,20] is a held-out gap
        },
    }
    rows = {r["exp_name"]: r for r in summarize_bin_comparison(bin_results)}

    assert rows["van_der_pol_e1"]["trained_mean"] == np.mean([0.4, 0.5, 0.6])
    assert rows["van_der_pol_e1"]["trained_n"] == 3
    assert rows["van_der_pol_e1"]["held_out_n"] == 0
    assert np.isnan(rows["van_der_pol_e1"]["held_out_mean"])

    assert rows["van_der_pol_e3"]["trained_mean"] == np.mean([0.7, 0.8])
    assert rows["van_der_pol_e3"]["trained_n"] == 2
    assert rows["van_der_pol_e3"]["held_out_mean"] == 0.1
    assert rows["van_der_pol_e3"]["held_out_n"] == 1


def test_repeat_curves_align_on_sorted_intersection_without_mixing_values():
    points, values = _align_repeat_series(
        [
            (np.array([3, 1, 2]), np.array([30.0, 10.0, 20.0])),
            (np.array([2, 3, 4]), np.array([200.0, 300.0, 400.0])),
        ]
    )

    assert np.array_equal(points, [2, 3])
    assert np.array_equal(values, [[20.0, 30.0], [200.0, 300.0]])


def test_return_dataset_order_is_stable_and_ignores_unrelated_metrics():
    datasets = _discover_return_datasets(
        {
            "run": [
                {
                    "eval/mean_return": 1.0,
                    "train/mean_return": 2.0,
                    "test/mean_return": 3.0,
                    "noise/mean_return": 4.0,
                    "custom/mean_return": 5.0,
                    "custom/success_rate": 0.8,
                    "iteration": 12,
                }
            ]
        }
    )

    assert datasets == ["train", "test", "noise", "eval", "custom"]


def test_repeat_summaries_pool_episode_values_before_computing_quantiles():
    rows = summarize_step_diff_stats(
        traces={
            "seed_a": {"test": {"rel_diff": np.array([0.0, 1.0])}},
            "seed_b": {"test": {"rel_diff": np.array([0.8, 1.2])}},
        },
        runs=[
            {"eid": 1, "exp_name": "van_der_pol_e1_r001", "run_uid": "seed_a"},
            {"eid": 1, "exp_name": "van_der_pol_e1_r002", "run_uid": "seed_b"},
        ],
        split="test",
    )

    [row] = rows
    assert row["exp_name"] == "van_der_pol_e1"
    assert row["n_repeats"] == 2
    assert row["n_samples"] == 4
    assert np.allclose(
        [row["mean"], row["median"], row["q1"], row["q3"]],
        [0.75, 0.9, 0.6, 1.05],
    )
    assert _config_label("van_der_pol_e1_r002") == "van_der_pol_e1"


def test_gmm_fit_returns_normalized_positive_components_deterministically():
    rng = np.random.default_rng(29)
    data = np.concatenate((rng.normal(-2.0, 0.15, 100), rng.normal(2.5, 0.2, 100)))

    fit_a = _fit_gmm_1d(data, k=2, seed=17)
    fit_b = _fit_gmm_1d(data, k=2, seed=17)
    weights, means, stds, bic = fit_a

    assert np.all(weights > 0)
    assert np.sum(weights) == pytest.approx(1.0)
    assert np.all(stds > 0)
    assert np.allclose(np.sort(means), [-2.0, 2.5], atol=0.1)
    assert np.isfinite(bic)
    assert all(
        np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(fit_a, fit_b)
    )


def test_best_gmm_uses_bic_to_recover_separated_modes():
    rng = np.random.default_rng(31)
    data = np.concatenate((rng.normal(-3.0, 0.1, 80), rng.normal(3.0, 0.1, 80)))

    weights, means, stds = _fit_best_gmm_1d(data, max_components=2, seed=11)

    assert len(weights) == len(means) == len(stds) == 2
    assert np.allclose(np.sort(means), [-3.0, 3.0], atol=0.15)
    assert np.sum(weights) == pytest.approx(1.0)
