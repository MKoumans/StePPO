import numpy as np


def test_average_seeds_accepts_one_explicit_run(tmp_path):
    from research.ode.post_run_analysis.learning_statistics import run_averaging

    run = tmp_path / "run"
    run.mkdir()
    (run / "l2_error.txt").write_text(
        "# subset: train\n"
        "iteration policy_err_mean policy_err_std policy_err_integrated_mean policy_err_integrated_std "
        "pid_err_mean pid_err_std pid_err_integrated_mean pid_err_integrated_std\n"
        "0 1 99 3 99 5 99 7 99\n"
    )

    out = tmp_path / "average"
    run_averaging([str(run)], str(out))

    row = (out / "l2_error_avg.txt").read_text().splitlines()[2].split()
    assert np.allclose([float(value) for value in row], [0, 1, 0, 3, 0, 5, 0, 7, 0])
