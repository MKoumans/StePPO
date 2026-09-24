"""Persistence and cache-key contracts for experiment-comparison results."""

from types import SimpleNamespace

import numpy as np

from research.ode.execute_comparison import (
    _load_run_data,
    _run_data_fingerprint,
    _save_run_data,
)


def _fingerprint_inputs():
    run = {"checkpoint_dir": "/runs/e1/checkpoint_4", "config_path": "/runs/e1/config.yaml"}
    args = SimpleNamespace(
        num_step_diff_episodes=8,
        seed=17,
        ref_tol_factor=3.0,
        num_bin_episodes=12,
    )
    return run, args, [1.0, 10.0, 100.0]


def _with_args(args, **changes):
    return SimpleNamespace(**{**vars(args), **changes})


def test_run_data_fingerprint_is_stable_and_covers_computation_inputs():
    run, args, points = _fingerprint_inputs()
    original = _run_data_fingerprint(run, args, points)

    assert _run_data_fingerprint(dict(run), _with_args(args), list(points)) == original
    assert _run_data_fingerprint(run, args, [1.0, 20.0, 100.0]) != original
    assert _run_data_fingerprint(run, _with_args(args, seed=18), points) != original
    assert (
        _run_data_fingerprint(run, _with_args(args, num_step_diff_episodes=9), points) != original
    )
    assert _run_data_fingerprint(run, _with_args(args, num_bin_episodes=13), points) != original
    assert _run_data_fingerprint(run, _with_args(args, ref_tol_factor=4.0), points) != original
    assert (
        _run_data_fingerprint({**run, "checkpoint_dir": "/runs/e1/checkpoint_5"}, args, points)
        != original
    )
    assert (
        _run_data_fingerprint({**run, "config_path": "/runs/e1/changed.yaml"}, args, points)
        != original
    )


def test_run_data_cache_round_trip_preserves_latents_splits_and_bin_results(tmp_path):
    path = tmp_path / "comparison.npz"
    latent = {
        1: {
            "mu": np.arange(6, dtype=np.float32).reshape(3, 2),
            "var": np.full((3, 2), 0.25, dtype=np.float32),
        },
        3: {"mu": np.ones((2, 1), dtype=np.float32), "var": np.zeros((2, 1), dtype=np.float32)},
    }
    step_diff = {
        "test": {
            "diff": np.array([1.0, -2.0]),
            "rel_diff": np.array([0.1, -0.2]),
            "err": np.array([0.01, 0.02]),
            "pid_err": np.array([0.03, 0.04]),
            "pid_steps": np.array([10, 20]),
            "bin_idx": np.array([0, 1]),
        },
    }
    bin_result = {
        "bins": [(1.0, 10.0), (10.0, 100.0)],
        "improvement": np.array([0.25, -0.5]),
        "trained": [True, False],
        "scheme": "binned",
    }

    _save_run_data(str(path), latent, step_diff, bin_result)
    loaded_latent, loaded_splits, loaded_bins = _load_run_data(str(path))

    assert set(loaded_latent) == {1, 3}
    assert np.array_equal(loaded_latent[1]["mu"], latent[1]["mu"])
    assert np.array_equal(loaded_latent[1]["var"], latent[1]["var"])
    assert loaded_latent[1]["latent_dim"] == 2
    assert loaded_latent[3]["latent_dim"] == 1
    for key, values in step_diff["test"].items():
        assert np.array_equal(loaded_splits["test"][key], values)
    assert loaded_bins["bins"] == bin_result["bins"]
    assert np.array_equal(loaded_bins["improvement"], bin_result["improvement"])
    assert loaded_bins["trained"] == bin_result["trained"]
    assert loaded_bins["scheme"] == bin_result["scheme"]


def test_run_data_cache_without_bin_comparison_loads_none(tmp_path):
    path = tmp_path / "comparison.npz"
    _save_run_data(str(path), {}, {}, None)

    assert _load_run_data(str(path)) == ({}, {}, None)
