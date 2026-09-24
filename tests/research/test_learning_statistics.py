from pathlib import Path

import numpy as np


def _write_l2(path: Path, offset: float) -> None:
    path.write_text(
        "# subset: train\n"
        "iteration policy_err_mean policy_err_std policy_err_integrated_mean policy_err_integrated_std "
        "pid_err_mean pid_err_std pid_err_integrated_mean pid_err_integrated_std\n"
        f"0 {1 + offset} 99 {3 + offset} 99 {5 + offset} 99 {7 + offset} 99\n"
        "\n"
        "# subset: test\n"
        "iteration policy_err_mean policy_err_std policy_err_integrated_mean policy_err_integrated_std "
        "pid_err_mean pid_err_std pid_err_integrated_mean pid_err_integrated_std\n"
        f"0 {2 + offset} 99 {4 + offset} 99 {6 + offset} 99 {8 + offset} 99\n"
    )


def _data_rows(text: str, subset: str) -> list[list[float]]:
    lines = text.splitlines()
    start = lines.index(f"# subset: {subset}") + 2
    rows = []
    for line in lines[start:]:
        if not line.strip() or line.startswith("#"):
            break
        if line.split()[0].isdigit():
            rows.append([float(value) for value in line.split()])
    return rows


def test_seed_average_l2_preserves_blocks_and_uses_cross_seed_std(tmp_path):
    from research.ode.post_run_analysis.learning_statistics import aggregate_l2_error

    runs = []
    for offset in (0, 2):
        run = tmp_path / f"run{offset}"
        run.mkdir()
        _write_l2(run / "l2_error.txt", offset)
        runs.append(str(run))

    out = tmp_path / "average"
    aggregate_l2_error(runs, str(out))

    text = (out / "l2_error_avg.txt").read_text()
    assert "# subset: train" in text
    assert "# subset: test" in text
    assert (out / "l2_error_avg.png").stat().st_size > 0
    assert np.allclose(_data_rows(text, "train")[0], [0, 2, 1, 4, 1, 6, 1, 8, 1])
    assert np.allclose(_data_rows(text, "test")[0], [0, 3, 1, 5, 1, 7, 1, 9, 1])
