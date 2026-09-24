"""Deterministic tests for benchmark timing and diagnostic output helpers."""

from pathlib import Path

import numpy as np
import pytest

from research.ode.diagnostics import utils


def test_time_solve_fn_single_mode_reports_mean_and_repeat_samples(monkeypatch):
    ticks = iter([0.0, 0.010, 1.0, 1.030, 2.0, 2.020, 3.0, 3.060])
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(ticks))
    calls = []

    def solve(mu_batch, keys):
        calls.append((np.asarray(mu_batch).copy(), np.asarray(keys).shape))
        return np.asarray(mu_batch)

    means, samples = utils.time_solve_fn(
        solve,
        np.array([1.0, 10.0]),
        mode="single",
        repeats=2,
        return_samples=True,
    )

    assert np.allclose(samples, [[10.0, 30.0], [20.0, 60.0]])
    assert np.allclose(means, [20.0, 40.0])
    assert len(calls) == 7  # three warmups plus two repeats per task
    assert calls[0][0].tolist() == [1.0]
    assert calls[0][1] == (1, 2)


def test_time_solve_fn_batched_mode_amortizes_elapsed_batch_time(monkeypatch):
    ticks = iter([0.0, 1.0, 1.010, 2.0, 2.030])
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(ticks))
    calls = []

    def solve(mu_batch, keys):
        calls.append((np.asarray(mu_batch).shape, np.asarray(keys).shape))
        return np.asarray(mu_batch)

    values = utils.time_solve_fn(
        solve,
        np.array([1.0, 10.0]),
        mode="batched",
        repeats=2,
    )

    # Batch timings are 10ms and 30ms; this mode reports their mean per task.
    assert np.allclose(values, [10.0, 10.0])
    assert calls == [((2,), (2, 2))] * 5


def test_time_solve_fn_batch_report_repeats_each_mu_and_returns_raw_batch_ms(monkeypatch):
    ticks = iter([0.0, 1.0, 1.05, 2.0, 2.02, 3.0, 3.03, 4.0, 4.04])
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(ticks))
    calls = []

    def solve(mu_batch, keys):
        calls.append((np.asarray(mu_batch).copy(), np.asarray(keys).shape))
        return np.asarray(mu_batch)

    means, samples = utils.time_solve_fn(
        solve,
        np.array([7.0, 9.0]),
        mode="batched",
        repeats=2,
        report_batch_time=True,
        batch_size=3,
        return_samples=True,
    )

    assert len(calls) == 7
    assert all(mu_batch.tolist() == [7.0, 7.0, 7.0] for mu_batch, _ in calls[:5])
    assert all(mu_batch.tolist() == [9.0, 9.0, 9.0] for mu_batch, _ in calls[5:])
    assert all(key_shape == (3, 2) for _, key_shape in calls)
    assert np.allclose(samples, [[50.0, 20.0], [30.0, 40.0]])
    assert np.allclose(means, [35.0, 35.0])


def test_time_solve_fn_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode 'parallel'"):
        utils.time_solve_fn(lambda *_: None, np.array([1.0]), mode="parallel")


def test_write_simple_outputs_keeps_summary_and_per_task_values(tmp_path):
    stem = tmp_path / "timing"
    utils.write_simple_outputs(
        str(stem) + ".npz",
        np.array([1.0, 10.0]),
        pid_ms=np.array([2.0, 4.0]),
        rl_ms=np.array([1.0, 2.0]),
        metadata={"system": "scalar_decay", "device": "gpu", "mode": "batched"},
    )

    report = Path(str(stem) + ".txt").read_text()
    assert Path(str(stem) + ".png").is_file()
    assert "system: scalar_decay" in report
    assert "mean PID: 3.000 ms" in report
    assert "mean speedup (PID / RL): 2.000x" in report
    assert "1\t2.000000\t1.000000" in report
