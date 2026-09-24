"""Tests for pure metrics and report rows in research/ode rollout analysis."""

import numpy as np
import pytest

from research.ode.post_run_analysis.analyse_rollout import (
    _acceptance_rate_per_env,
    _build_rows,
    _format_speedup,
    _rejection_rate_arrays,
    _solver_stats,
    _steps_to_completion,
    save_mu_table_latex,
    save_mu_table_txt,
)


def test_rollout_counts_ignore_inactive_padding():
    traces = {
        "active": np.array([[1, 1], [0, 1], [0, 0]], dtype=bool),
        "keep_step": np.array([[1, 0], [1, 0], [0, 0]], dtype=bool),
    }

    assert np.array_equal(_steps_to_completion(traces), [1, 2])
    accepted, rejected = _solver_stats(traces)
    assert np.array_equal(accepted, [1, 0])
    assert np.array_equal(rejected, [0, 2])
    assert np.array_equal(accepted + rejected, _steps_to_completion(traces))


def test_rejection_rate_averages_only_active_rollout_rows():
    traces = {
        "keep_step": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
        "active": np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=bool),
    }

    mean, std = _rejection_rate_arrays(traces)

    assert np.allclose(mean, [2 / 3, 2 / 3, 0.0])
    assert np.allclose(std[:2], [np.std([0.0, 1.0, 1.0]), np.std([1.0, 0.0, 1.0])])
    assert std[2] == 0.0


def test_per_environment_acceptance_uses_active_step_denominators():
    traces = {
        "keep_step": np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=bool),
        "active": np.array([[1, 0, 1], [1, 0, 1], [0, 0, 1]], dtype=bool),
    }

    with np.errstate(divide="ignore", invalid="ignore"):
        rates = _acceptance_rate_per_env(traces)

    assert rates[0] == 0.5
    assert np.isnan(rates[1])
    assert rates[2] == pytest.approx(2 / 3)


def test_build_rows_computes_acceptance_rate_and_policy_pid_speedup():
    [row] = _build_rows(
        [
            {
                "mu": 10.0,
                "pol_steps": 4.0,
                "pol_acc": 3.0,
                "pol_rej": 1.0,
                "pol_t": 0.75,
                "pid_steps": 8.0,
                "pid_t": 1.5,
            }
        ]
    )

    assert row["pol_rate"] == 75.0
    assert row["speedup"] == " 2.00×"
    assert "ds_steps" not in row


def test_build_rows_includes_diffrax_metrics_when_present():
    [row] = _build_rows(
        [
            {
                "mu": 10.0,
                "pol_steps": 4.0,
                "pol_acc": 3.0,
                "pol_rej": 1.0,
                "pol_t": 0.75,
                "pid_steps": 8.0,
                "pid_t": 1.5,
                "ds_steps": 2.0,
                "ds_acc": 0.625,
                "ds_t": 2.25,
            }
        ]
    )

    assert row["ds_acc"] == 62.5
    assert row["ds_speedup"] == " 4.00×"


def test_speedup_marks_missing_step_counts_as_unavailable():
    assert _format_speedup(0.0, 10.0).strip() == "N/A"
    assert _format_speedup(10.0, 0.0).strip() == "N/A"


def test_mu_tables_export_matching_diffrax_columns(tmp_path):
    results = [
        {
            "mu": 10.0,
            "pol_steps": 4.0,
            "pol_acc": 3.0,
            "pol_rej": 1.0,
            "pol_t": 0.75,
            "pid_steps": 8.0,
            "pid_t": 1.5,
            "ds_steps": 2.0,
            "ds_acc": 0.625,
            "ds_t": 2.25,
        }
    ]
    txt_path = tmp_path / "mu_table.txt"
    latex_path = tmp_path / "mu_table.tex"

    save_mu_table_txt(results, T=100, t_end=1.0, path=str(txt_path))
    save_mu_table_latex(results, T=100, t_end=1.0, path=str(latex_path))

    txt = txt_path.read_text()
    latex = latex_path.read_text()
    assert "DS steps" in txt and "2" in txt and "62.5%" in txt
    assert "RL (diffeqsolve)" in latex and "62.5\\%" in latex
