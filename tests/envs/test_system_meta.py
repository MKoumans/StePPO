import numpy as np


def test_compute_bin_rows_reports_average_relative_steps_and_error_delta():
    from research.ode.post_run_analysis.compare import compute_bin_rows

    rows = compute_bin_rows(
        bins=[(1.0, 2.0), (2.0, 4.0)],
        split=np.array([0, 0, 1, 1]),
        pid_steps=np.array([100.0, 200.0, 50.0, 100.0]),
        policy_steps=np.array([50.0, 100.0, 75.0, 50.0]),
        oracle_steps=np.array([25.0, 50.0, 25.0, 25.0]),
        pid_errors=np.array([-2.0, -1.0, -3.0, -2.0]),
        policy_errors=np.array([-1.0, -2.0, -2.0, -1.0]),
        oracle_errors=np.array([-3.0, -2.0, -4.0, -3.0]),
    )

    # Mean of per-episode ratios: (0.5 + 0.5) / 2, not ratio of means.
    assert rows[0]["ours_step_improvement"] == 0.5
    assert rows[1]["ours_step_improvement"] == 0.0
    assert rows[0]["ours_error_diff_vs_pid"] == 0.0
    assert rows[1]["ours_error_diff_vs_pid"] == 1.0
    assert rows[0]["oracle_step_improvement"] == 0.75
    assert rows[1]["oracle_error_diff_vs_pid"] == -1.0


def test_aggregate_bin_rows_averages_seed_metrics():
    from research.ode.post_run_analysis.compare import aggregate_bin_rows

    run_a = [
        {
            "bin_label": "[1,2]",
            "bin_lo": 1.0,
            "bin_hi": 2.0,
            "n": 4,
            "ours_step_improvement": 0.2,
            "ours_error_diff_vs_pid": 0.4,
        }
    ]
    run_b = [
        {
            "bin_label": "[1,2]",
            "bin_lo": 1.0,
            "bin_hi": 2.0,
            "n": 4,
            "ours_step_improvement": 0.6,
            "ours_error_diff_vs_pid": -0.2,
        }
    ]

    rows = aggregate_bin_rows([run_a, run_b])
    assert rows[0]["n_seeds"] == 2
    assert rows[0]["ours_step_improvement"] == 0.4
    assert rows[0]["ours_error_diff_vs_pid"] == 0.1


def test_write_meta_latex_contains_bin_table_and_escaped_labels(tmp_path):
    from research.ode.post_run_analysis.compare import write_meta_latex

    path = tmp_path / "system_meta.tex"
    write_meta_latex(
        [
            {
                "bin_label": "train_10%",
                "bin_lo": 1.0,
                "bin_hi": 2.0,
                "n": 8,
                "pid_steps": 100.0,
                "ours_steps": 80.0,
                "oracle_steps": 60.0,
                "ours_step_improvement": 0.2,
                "oracle_step_improvement": 0.4,
                "ours_error_diff_vs_pid": -0.3,
                "oracle_error_diff_vs_pid": -0.5,
            }
        ],
        path,
        system="van_der_pol",
        n_seeds=2,
    )
    content = path.read_text()
    assert "\\toprule" in content
    assert "train\\_10\\%" in content
    assert "20.0\\%" in content
    assert "-0.300" in content
