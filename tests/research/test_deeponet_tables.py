"""Tests for the data alignment and table helpers used by DeepONet figures."""

import numpy as np
import pytest

from research.baselines.deeponet import overleaf_tables as tables


def test_train_bin_arguments_parse_repeated_global_and_per_system_ranges():
    assert tables.parse_train_bins(["1,5", "10,20"]) == [[1.0, 5.0], [10.0, 20.0]]
    assert tables.parse_train_bins(None) is None
    assert tables.parse_train_bins_by_system(
        [
            "van_der_pol:1,5",
            "van_der_pol:10,20",
            "brusselator:2,4",
        ]
    ) == {
        "van_der_pol": [[1.0, 5.0], [10.0, 20.0]],
        "brusselator": [[2.0, 4.0]],
    }


def test_split_and_flat_table_readers_preserve_named_columns(tmp_path):
    split_path = tmp_path / "split.txt"
    split_path.write_text("# split: train\nmu mean\n1 2\n\n# split: test\nmu mean\n3 4\n5 6\n")
    flat_path = tmp_path / "flat.txt"
    flat_path.write_text("mu mean\n1 2\n3 4\n")

    train = tables.parse_split_table(str(split_path), split="train")
    test = tables.parse_split_table(str(split_path), split="test")
    flat = tables.parse_flat_table(str(flat_path))

    assert np.array_equal(train["mu"], [1.0])
    assert np.array_equal(test["mean"], [4.0, 6.0])
    assert np.array_equal(flat["mu"], [1.0, 3.0])
    assert tables.load_eval_table(str(tmp_path / "missing.txt"), missing_ok=True) is None
    with pytest.raises(ValueError, match="No '# split: val'"):
        tables.parse_split_table(str(split_path), split="val")


def test_oracle_testset_selects_largest_split_with_reference_times(tmp_path):
    testset = tmp_path / "testset"
    testset.mkdir()
    np.savez(testset / "small.npz", task_params=np.arange(2), ref_ts=np.ones(2))
    np.savez(testset / "large.npz", task_params=np.arange(5), ref_ts=np.ones(5))
    np.savez(testset / "larger_without_reference.npz", task_params=np.arange(7))

    selected = tables.load_oracle_testset(str(tmp_path))

    assert np.array_equal(selected["task_params"], np.arange(5))
    with pytest.raises(FileNotFoundError, match="No test-set file with ref_ts"):
        tables.load_oracle_testset(str(tmp_path / "empty"))


def test_bin_stats_preserves_empty_bins_and_counts_observations():
    means, stds, counts = tables.bin_stats(
        np.array([0.1, 0.2, 1.2, 1.8]),
        np.array([1.0, np.nan, 3.0, 5.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
    )

    assert np.allclose(means[:2], [1.0, 4.0])
    assert np.isnan(means[2]) and np.isnan(stds[2])
    assert np.array_equal(counts, [2, 2, 0])


def test_trajectory_alignment_interpolates_and_optionally_masks_after_early_stop():
    ref_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    times = np.array([0.0, 1.0, 2.0, np.inf])
    values = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0], [np.inf, np.inf]])

    endpoint_filled = tables.align_to_reference(ref_times, times, values)
    masked = tables.align_to_reference(ref_times, times, values, mask_beyond_end=True)

    assert np.allclose(endpoint_filled[:5], [[0, 0], [1, 2], [2, 4], [3, 6], [4, 8]])
    assert np.array_equal(endpoint_filled[-1], [4.0, 8.0])
    assert np.isnan(masked[-1]).all()
    assert np.isnan(
        tables.align_to_reference(
            ref_times,
            np.array([0.0]),
            np.array([[1.0, 2.0]]),
        )
    ).all()


def test_trajectory_column_names_cover_scalar_and_vector_state_shapes():
    assert tables.trajectory_column_names("ref", 1) == ["ref_y"]
    assert tables.trajectory_column_names("pred", 3) == ["pred_y0", "pred_y1", "pred_y2"]


def test_wallclock_lookup_prefers_requested_tag_and_reads_metadata(tmp_path):
    old = tmp_path / "wallclock_rl_van_der_pol_old_single.npz"
    new = tmp_path / "wallclock_rl_van_der_pol_new_single.npz"
    np.savez(old, mus=np.array([1.0]), rl_ms=np.array([9.0]))
    np.savez(
        new, mus=np.array([2.0]), rl_ms=np.array([3.0]), tag="new", mode="single", device="cpu"
    )

    mus, timings, tag, mode, device = tables.find_rl_wallclock(
        str(tmp_path),
        "van_der_pol",
        "single",
        tag="new",
    )

    assert np.array_equal(mus, [2.0])
    assert np.array_equal(timings, [3.0])
    assert (tag, mode, device) == ("new", "single", "cpu")
    assert tables.find_rl_wallclock(str(tmp_path), "van_der_pol", "batched") == (
        None,
        None,
        None,
        None,
        None,
    )
