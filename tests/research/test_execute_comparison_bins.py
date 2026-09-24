"""Tests for the cached and fallback paths in per-bin comparisons."""

from types import SimpleNamespace

import numpy as np

from research.ode import execute_comparison


def _run_and_config():
    run = {
        "run_uid": "run-1",
        "exp_name": "van_der_pol_e1",
        "config_path": "config.yaml",
        "checkpoint_dir": "checkpoint_10",
    }
    config = SimpleNamespace(
        env=SimpleNamespace(
            test_bins=((1.0, 2.0), (2.0, 4.0)),
            train_bins=((1.0, 2.0),),
            task_sample_scheme="binned",
        ),
        rollout_steps=50,
    )
    return run, config


def test_collect_bin_comparison_reuses_matching_test_split(monkeypatch):
    run, config = _run_and_config()
    monkeypatch.setattr(execute_comparison, "load_config_from_yaml", lambda *_: config)
    monkeypatch.setattr(
        execute_comparison,
        "load_cached_pid_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected cache read")),
    )
    monkeypatch.setattr(
        execute_comparison,
        "solve_pid_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected solve")),
    )
    cached_test = {
        "pid_steps": np.array([100.0, 200.0, 100.0]),
        "diff": np.array([50.0, -20.0, 20.0]),
        "bin_idx": np.array([0, 1, 1]),
    }

    result = execute_comparison.collect_bin_comparison(
        [run],
        num_episodes_per_bin=3,
        seed=5,
        step_diff_traces={"run-1": {"test": cached_test}},
        step_diff_num_episodes=3,
    )["run-1"]

    assert np.allclose(result["improvement"], [0.5, 0.05])
    assert result["trained"] == [True, False]
    assert result["bins"] == list(config.env.test_bins)


def test_collect_bin_comparison_solves_when_episode_count_does_not_match(monkeypatch):
    run, config = _run_and_config()
    monkeypatch.setattr(execute_comparison, "load_config_from_yaml", lambda *_: config)
    expected_batch = {
        "task_params": np.array([1.5, 3.0, 3.5]),
        "episode_keys": np.array([11, 12, 13]),
        "pid_steps": np.array([10.0, 20.0, 30.0]),
        "split": np.array([0, 1, 1]),
    }
    calls = {}

    monkeypatch.setattr(
        execute_comparison,
        "load_cached_pid_batch",
        lambda env, **kwargs: calls.update(env=env, kwargs=kwargs) or expected_batch,
    )
    monkeypatch.setattr(execute_comparison, "ODEEnv", lambda env, steps: (env, steps))
    monkeypatch.setattr(
        execute_comparison.LearnedController,
        "from_checkpoint",
        lambda checkpoint, cfg, env: (checkpoint, cfg, env),
    )

    def fake_solve(env, controller, tasks, keys, max_steps):
        calls["solve"] = (env, controller, tasks, keys, max_steps)
        return {"steps": np.array([5.0, 30.0, 15.0])}

    monkeypatch.setattr(execute_comparison, "solve_pid_batch", fake_solve)

    result = execute_comparison.collect_bin_comparison(
        [run],
        num_episodes_per_bin=3,
        seed=7,
        step_diff_traces={"run-1": {"test": {"pid_steps": [1], "diff": [0], "bin_idx": [0]}}},
        step_diff_num_episodes=1,
    )["run-1"]

    assert calls["kwargs"] == {
        "num_envs": 3,
        "seed": 7,
        "max_steps": 50,
        "bins": [(1.0, 2.0), (2.0, 4.0)],
        "split": "test",
    }
    assert calls["solve"][0] is config.env
    assert calls["solve"][1][0] == run["checkpoint_dir"]
    assert np.array_equal(calls["solve"][2], expected_batch["task_params"])
    assert np.array_equal(calls["solve"][3], expected_batch["episode_keys"])
    assert calls["solve"][4] == 50
    assert np.allclose(result["improvement"], [0.5, 0.0])
