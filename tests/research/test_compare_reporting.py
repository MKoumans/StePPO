"""Regression tests for comparison timing and exported summary values."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from research.ode.post_run_analysis import compare


def test_timed_lam_batches_preserve_episode_order_and_partial_tail():
    calls = []

    def run(lams, keys):
        lams = np.asarray(lams)
        keys = np.asarray(keys)
        calls.append((lams.copy(), keys.copy()))
        return {"value": lams * 100 + keys}

    repeats, timings = compare._run_timed_lams(
        run,
        lams=np.array([1, 2, 3, 4, 5]),
        keys=np.array([9, 8, 7, 6, 5]),
        batch_size=2,
        repeats=2,
    )

    assert len(calls) == 6  # two repeats, each with a final one-item batch
    assert [len(lams) for lams, _ in calls] == [2, 2, 1, 2, 2, 1]
    assert [result["value"].tolist() for result in repeats] == [[109, 208, 307, 406, 505]] * 2
    assert len(timings) == 2
    assert all(seconds >= 0.0 for seconds in timings)


@pytest.mark.parametrize(
    ("max_steps", "timing_key", "method"),
    [(4, "budget", "diffrax PID (budget=4)"), (20, "unlimited", "diffrax PID (budget=20)")],
)
def test_diffrax_pid_selects_cached_timing_by_budget(
    monkeypatch,
    max_steps,
    timing_key,
    method,
):
    observed = {}
    timing = {"success_rate": 0.75, "mean_steps": 12.0, "ms_per_ep": 3.5}

    def fake_load(config, **kwargs):
        observed.update(config=config, **kwargs)
        return {"timing": {"budget": timing, "unlimited": {**timing, "ms_per_ep": 8.0}}}

    monkeypatch.setattr(compare, "load_cached_pid_batch", fake_load)
    config = SimpleNamespace(env=object(), rollout_steps=4)

    result = compare.eval_diffrax_pid(
        config,
        bins=[(1.0, 10.0)],
        num_episodes=7,
        max_steps=max_steps,
    )

    assert observed["config"] is config.env
    assert observed["split"] == "test"
    assert observed["num_envs"] == 1024
    assert observed["max_steps"] == config.rollout_steps
    assert result["method"] == method
    assert result["ms_per_ep"] == (3.5 if timing_key == "budget" else 8.0)


def test_summary_csv_scales_success_rate_and_preserves_missing_values(tmp_path):
    path = tmp_path / "summary.csv"
    compare.export_summary_csv(
        [
            {
                "method": "Policy, held out",
                "mean_steps": 14.25,
                "success_rate": 0.875,
                "ms_per_ep": float("nan"),
                "mean_return": -1.25,
            }
        ],
        path,
    )

    with path.open(newline="") as f:
        [header, row] = list(csv.reader(f))

    assert header == ["Method", "Steps", "Wallclock (ms/ep)", "Success %", "Return"]
    assert row == ["Policy, held out", "14.2", "", "87.5", "-1.2"]


def test_controller_table_formats_proportions_and_nan_as_unavailable():
    lines = compare._build_table_lines(
        [
            {
                "method": "StePPO",
                "success_rate": 0.75,
                "mean_steps": 12.0,
                "ms_per_ep": float("nan"),
            }
        ],
        t_end=100.0,
    )

    assert any("StePPO" in line and "75.0%" in line for line in lines)
    assert any("—" in line for line in lines)
    assert any("t_end = 100.0" in line for line in lines)
