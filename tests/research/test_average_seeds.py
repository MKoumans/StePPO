def _write_steps_table(path, offset):
    path.write_text(
        "# split: train\n"
        "bin_center pid_mean pid_std pid_count policy_mean policy_std policy_count oracle_mean oracle_std oracle_count\n"
        f"1  {10 + offset}  1  2  {8 + offset}  1  2  {7 + offset}  1  2\n"
        "\n"
        "# split: val\n"
        "bin_center pid_mean pid_std pid_count policy_mean policy_std policy_count oracle_mean oracle_std oracle_count\n"
        f"1  {20 + offset}  1  2  {18 + offset}  1  2  {17 + offset}  1  2\n"
        "\n"
        "# split: test\n"
        "bin_center pid_mean pid_std pid_count policy_mean policy_std policy_count oracle_mean oracle_std oracle_count\n"
        f"1  {30 + offset}  1  2  {28 + offset}  1  2  {27 + offset}  1  2\n"
    )


def test_seed_average_steps_vs_mu_preserves_all_splits(tmp_path):
    from research.ode.post_run_analysis.learning_statistics import aggregate_steps_vs_mu

    runs = []
    for seed in (0, 1):
        run = tmp_path / f"run{seed}"
        run.mkdir()
        _write_steps_table(run / "steps_vs_mu.txt", seed)
        runs.append(str(run))

    out = tmp_path / "average"
    out.mkdir()
    aggregate_steps_vs_mu(runs, str(out))

    text = (out / "steps_vs_mu_avg.txt").read_text()
    assert "# split: train" in text
    assert "# split: val" in text
    assert "# split: test" in text
    assert "policy_mean" in text
    assert "oracle_mean" in text
    assert (out / "steps_vs_mu_avg.png").stat().st_size > 0


def test_seed_average_efficiency_preserves_all_subsets(tmp_path):
    from research.ode.post_run_analysis.learning_statistics import aggregate_efficiency

    row = "iteration mean std median mean_pid_steps std_pid_steps mean_policy_steps std_policy_steps\n"
    content = "\n".join(
        [
            "# subset: train",
            row,
            "0 0.1 0.01 0.1 100 1 90 1",
            "",
            "# subset: val",
            row,
            "0 0.2 0.01 0.2 100 1 90 1",
            "",
            "# subset: test",
            row,
            "0 0.3 0.01 0.3 100 1 90 1",
            "",
            "# subset: noise",
            row,
            "0 0.4 0.01 0.4 100 1 90 1",
            "",
        ]
    )
    runs = []
    for seed in (0, 1):
        run = tmp_path / f"run{seed}"
        run.mkdir()
        (run / "efficiency.txt").write_text(content)
        runs.append(str(run))

    out = tmp_path / "average"
    out.mkdir()
    aggregate_efficiency(runs, str(out))

    efficiency_text = (out / "efficiency_avg.txt").read_text()
    assert "# subset: train" in efficiency_text
    assert "# subset: val" in efficiency_text
    assert "# subset: test" in efficiency_text
    assert (out / "efficiency_avg.png").stat().st_size > 0


def test_seed_average_z_sensitivity_accepts_root_file(tmp_path):
    from research.ode.post_run_analysis.learning_statistics import aggregate_z_sensitivity

    content = (
        "mean_noise\n"
        "alpha return length\n"
        "0 1 2\n"
        "1 0 1\n"
        "\n"
        "logvar_noise\n"
        "alpha return length\n"
        "0 1 2\n"
        "1 0 1\n"
    )
    runs = []
    for seed in (0, 1):
        run = tmp_path / f"run{seed}"
        run.mkdir()
        (run / "z_sensitivity.txt").write_text(content)
        runs.append(str(run))

    out = tmp_path / "average"
    out.mkdir()
    aggregate_z_sensitivity(runs, str(out))

    assert (out / "z_sensitivity_avg.txt").is_file()
    assert (out / "z_sensitivity_avg.png").stat().st_size > 0


def test_seed_average_ignores_bins_a_seed_could_not_solve(tmp_path):
    """A bin whose solves all aborted is written as nan (it has no mean),
    and one such seed must not erase the bin for every other seed — the
    average is over the seeds that have a number there."""
    import numpy as np

    from research.ode.post_run_analysis.learning_statistics import aggregate_steps_vs_mu

    header = (
        "bin_center pid_mean pid_std pid_count pid_completion "
        "policy_mean policy_std policy_count policy_completion\n"
    )
    runs = []
    for seed, pid_mean in enumerate(("100.0", "nan")):
        run = tmp_path / f"run{seed}"
        run.mkdir()
        (run / "steps_vs_mu.txt").write_text(
            "# split: test\n" + header + f"1  {pid_mean}  1  2  1.0  50.0  1  2  1.0\n"
        )
        runs.append(str(run))

    out = tmp_path / "average"
    out.mkdir()
    aggregate_steps_vs_mu(runs, str(out))

    lines = [line.strip() for line in (out / "steps_vs_mu_avg.txt").read_text().splitlines()]
    rows = [line.split() for line in lines if line and not line.startswith(("#", "bin_center"))]
    avg_header = [line.split() for line in lines if line.startswith("bin_center")][0]
    pid_mean_col = avg_header.index("pid_mean")
    assert np.isclose(float(rows[0][pid_mean_col]), 100.0)
