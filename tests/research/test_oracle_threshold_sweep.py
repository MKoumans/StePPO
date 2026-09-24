from types import SimpleNamespace

import numpy as np
import pytest

from research.ode.misc import oracle_maxiters_sweep as sweep


def test_build_threshold_grid_is_log_spaced_and_includes_endpoints():
    thresholds = sweep.build_threshold_grid(1, 10, 5)

    assert np.allclose(thresholds, np.geomspace(1, 10, 5))
    assert thresholds[0] == pytest.approx(1)
    assert thresholds[-1] == pytest.approx(10)
    assert np.allclose(np.diff(np.log(thresholds)), np.diff(np.log(thresholds))[0])


def test_summarize_threshold_averages_per_environment_relative_improvement():
    summary = sweep.summarize_threshold(
        np.array([10.0, 20.0, 1.0]),
        np.array([5.0, 30.0, 2.0]),
    )

    expected = np.array([0.5, -0.5, -1.0])
    assert summary["mean_relative_improvement"] == pytest.approx(float(expected.mean()))
    assert summary["std_relative_improvement"] == pytest.approx(float(expected.std()))
    assert summary["mean_pid_steps"] == pytest.approx(31.0 / 3.0)
    assert summary["mean_oracle_steps"] == pytest.approx(37.0 / 3.0)


def test_run_threshold_sweep_reuses_fixed_tasks_for_each_threshold(monkeypatch):
    calls = []
    pid_steps = np.array([10.0, 20.0])
    task_params = np.array([5.0, 25.0], dtype=np.float32)
    episode_keys = np.array([[1, 2], [3, 4]], dtype=np.uint32)
    config = SimpleNamespace(rtol=1e-3, atol=1e-6, dt_min=1e-4, dt_max=1.0)

    def fake_pid(config_arg, controller, tasks_arg, keys_arg, max_steps):
        assert config_arg is config
        assert np.array_equal(tasks_arg, task_params)
        assert np.array_equal(keys_arg, episode_keys)
        assert max_steps == 50
        return {"steps": pid_steps}

    def fake_oracle(config_arg, tasks_arg, keys_arg, max_steps, threshold, max_iters):
        calls.append((np.array(tasks_arg), np.array(keys_arg), threshold, max_iters))
        return {"steps": pid_steps - threshold}

    monkeypatch.setattr(sweep, "solve_pid_batch", fake_pid)
    monkeypatch.setattr(sweep, "solve_oracle_batch", fake_oracle)

    result = sweep.run_threshold_sweep(
        config,
        task_params,
        episode_keys,
        50,
        np.array([1e-3, 1e-1]),
        10,
    )

    assert len(calls) == 2
    for tasks_arg, keys_arg, threshold, max_iters in calls:
        assert np.array_equal(tasks_arg, task_params)
        assert np.array_equal(keys_arg, episode_keys)
        assert max_iters == 10
    assert np.allclose(result["threshold"], [1e-3, 1e-1])
    assert result["mean_relative_improvement"].shape == (2,)


def test_run_max_iters_sweep_uses_search_budget(monkeypatch):
    calls = []
    pid_steps = np.array([10.0, 20.0])
    task_params = np.array([5.0, 25.0], dtype=np.float32)
    episode_keys = np.array([[1, 2], [3, 4]], dtype=np.uint32)
    config = SimpleNamespace(rtol=1e-3, atol=1e-6, dt_min=1e-4, dt_max=1.0)

    def fake_pid(config_arg, tasks_arg, keys_arg, max_steps):
        assert config_arg is config
        assert np.array_equal(tasks_arg, task_params)
        assert np.array_equal(keys_arg, episode_keys)
        assert max_steps == 50
        return {"steps": pid_steps}

    def fake_oracle(config_arg, tasks_arg, keys_arg, max_steps, max_iters):
        calls.append(max_iters)
        return {"steps": pid_steps - max_iters}

    monkeypatch.setattr(sweep, "solve_pid_baseline_batch", fake_pid)
    monkeypatch.setattr(sweep, "solve_oracle_batch", fake_oracle)

    result = sweep.run_max_iters_sweep(
        config,
        task_params,
        episode_keys,
        50,
        np.array([0, 1, 3]),
    )

    assert calls == [1, 3]
    assert np.array_equal(result["max_iters"], [0, 1, 3])
    assert result["mean_relative_improvement"].shape == (3,)


def test_writers_create_nonempty_png_and_csv(tmp_path):
    result = {
        "threshold": np.array([1e-3, 1e0]),
        "mean_relative_improvement": np.array([0.4, 0.1]),
        "std_relative_improvement": np.array([0.2, 0.3]),
        "mean_pid_steps": np.array([20.0, 20.0]),
        "mean_oracle_steps": np.array([12.0, 18.0]),
    }
    csv_path = tmp_path / "sweep.csv"
    png_path = tmp_path / "sweep.png"

    sweep.save_threshold_sweep_csv(result, csv_path)
    sweep.plot_threshold_sweep(result, "scalar_decay", 2, 10, png_path)

    assert csv_path.is_file() and csv_path.stat().st_size > 0
    assert png_path.is_file() and png_path.stat().st_size > 0
    assert csv_path.read_text().splitlines()[0].split(",") == [
        "threshold",
        "mean_relative_improvement",
        "std_relative_improvement",
        "mean_pid_steps",
        "mean_oracle_steps",
    ]
