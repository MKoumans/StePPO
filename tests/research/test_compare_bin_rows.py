"""Scientific regression tests for per-bin comparison aggregation."""

import numpy as np
import pytest

from research.ode.post_run_analysis.compare import (
    aggregate_bin_rows,
    compute_bin_rows,
    read_meta_txt,
    write_meta_latex,
    write_meta_txt,
)


def test_bin_rows_average_episode_relative_improvements_and_error_differences():
    [row] = compute_bin_rows(
        bins=[(1.0, 10.0)],
        split=[0, 0],
        pid_steps=[1.0, 100.0],
        policy_steps=[1.0, 50.0],
        oracle_steps=[1.0, 25.0],
        pid_errors=[-1.0, -2.0],
        policy_errors=[-0.5, -3.0],
        oracle_errors=[-1.5, -2.5],
        labels=["trained"],
        trained=[True],
    )

    # Mean of episode-wise improvements is (0 + 0.5) / 2. A ratio of bin
    # means would be about 0.495 and silently change the reported statistic.
    assert row["ours_step_improvement"] == pytest.approx(0.25)
    assert row["oracle_step_improvement"] == pytest.approx(0.375)
    assert row["ours_step_gap_to_oracle"] == pytest.approx(12.5)
    assert row["ours_error_diff_vs_pid"] == pytest.approx(-0.25)
    assert row["oracle_error_diff_vs_pid"] == pytest.approx(-0.5)
    assert (row["bin_label"], row["n"], row["trained"]) == ("trained", 2, True)


def test_empty_bins_and_missing_optional_series_remain_unavailable():
    rows = compute_bin_rows(
        bins=[(1.0, 10.0), (10.0, 100.0)],
        split=[0],
        pid_steps=[3.0],
    )

    assert rows[0]["n"] == 1
    assert np.isnan(rows[0]["ours_steps"])
    assert np.isnan(rows[0]["ours_step_improvement"])
    assert rows[1]["n"] == 0
    assert np.isnan(rows[1]["pid_steps"])
    assert np.isnan(rows[1]["ours_error_diff_vs_pid"])


def test_bin_rows_reject_episode_series_with_different_lengths():
    with pytest.raises(ValueError, match="policy_steps has 1 values, expected 2"):
        compute_bin_rows(
            bins=[(1.0, 10.0)],
            split=[0, 0],
            pid_steps=[10.0, 20.0],
            policy_steps=[5.0],
        )


def test_aggregate_bin_rows_averages_matching_bins_and_ignores_nan():
    run_a = compute_bin_rows(
        bins=[(1.0, 10.0)],
        split=[0],
        pid_steps=[10.0],
        policy_steps=[5.0],
        labels=["same"],
        trained=[True],
    )
    run_b = compute_bin_rows(
        bins=[(1.0, 10.0)],
        split=[0],
        pid_steps=[20.0],
        policy_steps=[10.0],
        labels=["same"],
        trained=[True],
    )
    run_b[0]["oracle_steps"] = 4.0
    [row] = aggregate_bin_rows([run_a, run_b])

    assert row["bin_label"] == "same"
    assert row["n_seeds"] == 2
    assert row["trained"] is True
    assert row["pid_steps"] == pytest.approx(15.0)
    assert row["ours_step_improvement"] == pytest.approx(0.5)
    assert row["oracle_steps"] == pytest.approx(4.0)


def test_comparison_metadata_text_round_trips_numeric_and_missing_values(tmp_path):
    [row] = compute_bin_rows(
        bins=[(1.0, 10.0)],
        split=[0, 0],
        pid_steps=[10.0, 20.0],
        policy_steps=[8.0, 10.0],
        pid_errors=[-1.0, -2.0],
        policy_errors=[-0.5, -1.5],
        labels=["heldout"],
        trained=[False],
    )
    row["oracle_steps"] = float("nan")
    path = tmp_path / "nested" / "meta.txt"

    write_meta_txt([row], str(path), system="scalar_decay", n_seeds=3)
    [loaded] = read_meta_txt(path)

    assert loaded["bin_label"] == "heldout"
    assert loaded["trained"] is False
    assert loaded["n"] == 2.0
    assert loaded["ours_step_improvement"] == pytest.approx(0.35)
    assert loaded["ours_error_diff_vs_pid"] == pytest.approx(0.5)
    assert np.isnan(loaded["oracle_steps"])


def test_comparison_metadata_latex_escapes_labels_and_formats_missing_values(tmp_path):
    path = tmp_path / "meta.tex"
    write_meta_latex(
        [
            {
                "bin_label": "trained_bin_1%",
                "trained": None,
                "pid_steps": 10.0,
                "ours_steps": float("nan"),
                "oracle_steps": None,
                "ours_step_improvement": 0.25,
                "ours_error_diff_vs_pid": -0.125,
                "oracle_error_diff_vs_pid": float("nan"),
            }
        ],
        str(path),
        system="van_der_pol",
        n_seeds=2,
    )
    rendered = path.read_text()

    assert "van\\_der\\_pol" in rendered
    assert "trained\\_bin\\_1\\%" in rendered
    assert "25.0\\%" in rendered
    assert rendered.count("---") >= 3
