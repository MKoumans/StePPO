import numpy as np

from steppo.utils.figures.ode import bin_stats


def test_bin_stats_excludes_solves_that_did_not_reach_t_end():
    """An aborted solve returns a small step count (chua_smooth's PID at
    m>=1.2 stopped at t=0.01 of 20.0 after ~30 steps). Averaging those in
    drags the curve down to where it reads as a cheap solve — below the
    oracle — so only completing episodes count towards the mean."""
    data = {
        "mu": np.array([1.0, 1.0, 1.0]),
        "steps": np.array([100.0, 98.0, 30.0]),
        "completed": np.array([True, True, False]),
    }
    means, stds, counts, completion = bin_stats(data, np.array([0.5, 1.5]))

    assert counts == [2]
    assert np.allclose(means, [99.0])
    assert np.allclose(stds, [1.0])
    assert np.allclose(completion, [2 / 3])


def test_bin_stats_reports_nan_for_a_bin_with_nothing_to_average():
    """Empty bins used to report 0.0, which plots as a real measurement of
    zero solver steps — the cliff at the edge of the training range. A bin
    with no completing episode has no mean; it must read as a gap."""
    empty = {"mu": np.array([5.0]), "steps": np.array([10.0]), "completed": np.array([True])}
    all_failed = {
        "mu": np.array([1.0]),
        "steps": np.array([30.0]),
        "completed": np.array([False]),
    }
    edges = np.array([0.5, 1.5])

    means, stds, counts, completion = bin_stats(empty, edges)
    assert counts == [0]
    assert np.isnan(means).all() and np.isnan(stds).all()
    assert np.isnan(completion).all()

    means, stds, counts, completion = bin_stats(all_failed, edges)
    assert counts == [0]
    assert np.isnan(means).all()
    assert np.allclose(completion, [0.0])


def test_bin_stats_treats_a_series_without_completion_as_complete():
    """Series that cannot fail to complete (a pure post-hoc aggregate, or a
    legacy caller) carry no mask and are averaged whole."""
    data = {"mu": np.array([1.0, 1.0]), "steps": np.array([4.0, 6.0])}
    means, _, counts, completion = bin_stats(data, np.array([0.5, 1.5]))
    assert counts == [2]
    assert np.allclose(means, [5.0])
    assert np.allclose(completion, [1.0])


def _series(mu, steps, completed=None):
    out = {"mu": np.asarray(mu, dtype=float), "steps": np.asarray(steps, dtype=float)}
    if completed is not None:
        out["completed"] = np.asarray(completed, dtype=bool)
    return out


def test_splits_txt_writes_a_completion_column_per_series(tmp_path):
    """The table is what seed-averaging and the paper tables read, so the
    completion fraction has to survive into it — otherwise a column of
    means silently mixes solved and unsolved parameter ranges."""
    from steppo.utils.figures.ode import save_steps_vs_mu_splits_txt

    split_data = {
        "test": {
            "pid": _series([1.0, 2.0], [100.0, 30.0], [True, False]),
            "policy": _series([1.0, 2.0], [120.0, 121.0], [True, True]),
            "oracle": _series([1.0, 2.0], [54.0, 54.0], [True, True]),
        }
    }
    out = tmp_path / "steps_vs_mu.txt"
    save_steps_vs_mu_splits_txt(split_data, str(out), mu_min=1.0, mu_max=2.0)

    header = [line for line in out.read_text().splitlines() if line.startswith("bin_center")][0]
    assert "pid_completion" in header.split()
    assert "policy_completion" in header.split()
    assert "oracle_completion" in header.split()


def test_splits_txt_records_a_fully_failed_bin_as_a_gap_not_a_zero(tmp_path):
    """A bin where nothing solved must not report 0.0000 steps — that is the
    number that made PID look four times cheaper than the oracle."""
    from steppo.utils.figures.ode import save_steps_vs_mu_splits_txt

    split_data = {
        "test": {"pid": _series([1.0, 2.0], [100.0, 30.0], [True, False])},
    }
    out = tmp_path / "steps_vs_mu.txt"
    save_steps_vs_mu_splits_txt(split_data, str(out), mu_min=1.0, mu_max=2.0)

    rows = [
        line.split()
        for line in out.read_text().splitlines()
        if line and not line.startswith(("#", "bin_center"))
    ]
    failed = [row for row in rows if row[3] == "0"]
    assert failed, "expected at least one bin with no usable episode"
    assert all(row[1].lower() == "nan" for row in failed)


def test_plot_steps_vs_mu_splits_renders_with_completion_aware_series(tmp_path):
    """End-to-end guard that the plotting path consumes the new per-bin
    tuple — a mismatch here is what would silently stop producing figures."""
    from steppo.utils.figures.ode import plot_steps_vs_mu_splits

    split_data = {
        "test": {
            "pid": _series([1.0, 2.0], [100.0, 30.0], [True, False]),
            "policy": _series([1.0, 2.0], [120.0, 121.0], [True, True]),
            "oracle": _series([1.0, 2.0], [54.0, 54.0], [True, True]),
            "pid_unlimited": _series([1.0, 2.0], [100.0, 30.0], [True, False]),
        }
    }
    out = tmp_path / "steps_vs_mu.png"
    plot_steps_vs_mu_splits(split_data, 1.0, 2.0, str(out), train_bins=[(1.0, 2.0)])

    assert out.stat().st_size > 0


def test_incomplete_spans_brackets_contiguous_runs_of_failed_bins():
    """The figure marks the parameter ranges where the baseline did not
    solve, so a reader sees "PID fails past here" instead of a line that
    simply bends downwards. Bins with no episodes at all (NaN completion)
    are not failures and must not be shaded."""
    from steppo.utils.figures.ode import incomplete_spans

    centers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    completion = np.array([1.0, 0.7, 0.0, np.nan, 1.0])

    assert incomplete_spans(centers, completion) == [(2.0, 3.0)]


def test_incomplete_spans_is_empty_when_everything_solved():
    from steppo.utils.figures.ode import incomplete_spans

    centers = np.array([1.0, 2.0])
    assert incomplete_spans(centers, np.array([1.0, 1.0])) == []
    assert incomplete_spans(centers, np.array([np.nan, np.nan])) == []


def test_eval_dataset_writer_passes_completion_into_its_series(tmp_path):
    """generate_pid_eval_dataset builds its own PID/oracle series dicts from
    the cache rather than going through build_steps_vs_mu_data, so it needs
    the masks wired in independently — otherwise the dataset's own figure
    keeps averaging aborted solves."""
    from steppo.envs.ode.generate_pid_eval_dataset import build_eval_steps_vs_mu_series

    data = {
        "task_params": np.array([1.0, 2.0]),
        "pid_steps": np.array([100.0, 30.0]),
        "oracle_steps": np.array([54.0, 54.0]),
        "pid_completed": np.array([True, False]),
        "oracle_completed": np.array([True, True]),
    }

    pid, oracle = build_eval_steps_vs_mu_series(data)

    assert np.array_equal(pid["completed"], [True, False])
    assert np.array_equal(oracle["completed"], [True, True])


def test_convergence_table_reports_over_completing_episodes_only(tmp_path):
    """The single-task table is the degenerate-range counterpart of the
    binned one and needs the same masking, plus the completion fraction —
    a row reading "30.00±2.00 (n=64)" for a method that never solved the
    problem is the most misleading form of this bug."""
    from steppo.utils.figures.ode import save_convergence_table

    split_data = {
        "test": {
            "pid": _series([1.0, 1.0, 1.0], [100.0, 98.0, 30.0], [True, True, False]),
            "policy": _series([1.0, 1.0, 1.0], [120.0, 121.0, 122.0], [True, True, True]),
        }
    }
    out = tmp_path / "steps_vs_mu.txt"
    save_convergence_table(split_data, str(out))

    text = out.read_text()
    pid_row = [line for line in text.splitlines() if line.startswith("PID")][0]
    assert "99.00±1.00 (n=2)" in pid_row
    assert "completion" in text
    assert "0.67" in pid_row


def test_convergence_table_marks_a_method_that_never_completed(tmp_path):
    from steppo.utils.figures.ode import save_convergence_table

    split_data = {
        "test": {"pid": _series([1.0, 1.0], [30.0, 31.0], [False, False])},
    }
    out = tmp_path / "steps_vs_mu.txt"
    save_convergence_table(split_data, str(out))

    pid_row = [line for line in out.read_text().splitlines() if line.startswith("PID")][0]
    assert "n/a" in pid_row
    assert "0.00" in pid_row


def test_error_split_data_preserves_the_masks_the_error_plots_bin_on():
    """error_vs_mu goes through the same binning as steps_vs_mu, so the
    remap into pid/policy/oracle shape has to carry each error series'
    completion mask with it — an error graded against a reference that
    stopped at t~0.02 must not be averaged in."""
    from steppo.utils.figures.ode import build_error_split_data

    split_data = {
        "test": {
            "pid_error": {
                "mu": np.array([1.0, 2.0]),
                "err": np.array([-3.0, -0.1]),
                "completed": np.array([True, False]),
            },
            "policy_error": None,
            "oracle_error": None,
        }
    }

    remapped = build_error_split_data(split_data, "err")

    assert np.array_equal(remapped["test"]["pid"]["completed"], [True, False])
