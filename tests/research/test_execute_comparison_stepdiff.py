"""Split alignment tests for cached PID-versus-policy comparison traces."""

from types import SimpleNamespace

import numpy as np

from research.ode import execute_comparison as comparison


def test_step_diff_traces_keep_each_split_and_local_bin_indices_aligned(monkeypatch):
    env_config = SimpleNamespace(
        system="scalar_decay",
        train_bins=((1.0, 2.0), (2.0, 3.0)),
        val_bins=(),
        test_bins=((4.0, 5.0),),
        atol=1e-6,
    )
    config = SimpleNamespace(env=env_config, rollout_steps=20)
    run = {
        "run_uid": "run-a",
        "exp_name": "scalar_decay_e1_r1",
        "config_path": "config.yaml",
        "checkpoint_dir": "checkpoint_7",
    }
    cached = {
        "train": {
            "task_params": np.array([1.5, 2.5]),
            "episode_keys": np.array([[1, 2], [3, 4]], dtype=np.uint32),
            "split": np.array([0, 1], dtype=np.int8),
            "pid_steps": np.array([10.0, 20.0]),
            "pid_final_y": np.array([[3.0], [5.0]]),
            "ref_final_y": np.array([[2.0], [2.0]]),
            "oracle_steps": None,
            "oracle_final_y": None,
        },
        "test": {
            "task_params": np.array([4.5, 4.8]),
            "episode_keys": np.array([[5, 6], [7, 8]], dtype=np.uint32),
            "split": np.array([0, 0], dtype=np.int8),
            "pid_steps": np.array([8.0, 20.0]),
            "pid_final_y": np.array([[7.0], [10.0]]),
            "ref_final_y": np.array([[2.0], [2.0]]),
            "oracle_steps": None,
            "oracle_final_y": None,
        },
    }
    cache_calls = []

    def fake_cache(cfg, **kwargs):
        cache_calls.append((cfg, kwargs))
        return cached[kwargs["split"]]

    monkeypatch.setattr(comparison, "load_config_from_yaml", lambda *_args: config)
    monkeypatch.setattr(comparison, "load_cached_pid_batch", fake_cache)
    monkeypatch.setattr(comparison, "ODEEnv", lambda *_args: "env")
    monkeypatch.setattr(
        comparison.LearnedController,
        "from_checkpoint",
        lambda checkpoint, cfg, env: (checkpoint, cfg, env),
    )
    probe_seen = {}

    def fake_controller(_cfg, controller, probe, refs, max_steps):
        probe_seen.update(controller=controller, probe=probe, refs=refs, max_steps=max_steps)
        return {
            "train": {"steps": np.array([5.0, 12.0]), "err": np.array([-2.0, -1.0])},
            "test": {"steps": np.array([3.0, 10.0]), "err": np.array([-3.0, -2.0])},
        }

    monkeypatch.setattr(comparison, "collect_controller_steps_and_errors", fake_controller)

    traces = comparison.collect_step_diff_traces([run], num_episodes=2, seed=9)["run-a"]

    assert [kwargs["split"] for _, kwargs in cache_calls] == ["train", "test"]
    assert all(kwargs["num_envs"] == 2 and kwargs["seed"] == 9 for _, kwargs in cache_calls)
    assert probe_seen["max_steps"] == 20
    assert probe_seen["controller"][0] == run["checkpoint_dir"]
    assert np.array_equal(probe_seen["probe"]["train"][0], cached["train"]["task_params"])
    assert np.array_equal(probe_seen["probe"]["test"][1], cached["test"]["episode_keys"])
    assert np.array_equal(probe_seen["refs"]["train"], cached["train"]["ref_final_y"])
    assert np.array_equal(probe_seen["refs"]["test"], cached["test"]["ref_final_y"])

    assert np.allclose(traces["train"]["diff"], [5.0, 8.0])
    assert np.allclose(traces["train"]["rel_diff"], [0.5, 0.4])
    assert np.allclose(traces["train"]["pid_err"], np.log10([0.5, 1.5]))
    assert np.allclose(traces["train"]["err"], [-2.0, -1.0])
    assert np.array_equal(traces["train"]["bin_idx"], [0, 1])
    assert np.allclose(traces["test"]["diff"], [5.0, 10.0])
    assert np.allclose(traces["test"]["rel_diff"], [0.625, 0.5])
    assert np.allclose(traces["test"]["pid_err"], np.log10([2.5, 4.0]))
    assert np.array_equal(traces["test"]["bin_idx"], [0, 0])


def test_step_diff_skips_runs_with_no_configured_splits(monkeypatch, capsys):
    config = SimpleNamespace(
        env=SimpleNamespace(train_bins=(), val_bins=(), test_bins=()),
        rollout_steps=10,
    )
    run = {"run_uid": "empty", "exp_name": "empty_e1", "config_path": "config.yaml"}
    monkeypatch.setattr(comparison, "load_config_from_yaml", lambda *_args: config)
    monkeypatch.setattr(comparison, "ODEEnv", lambda *_args: "env")
    monkeypatch.setattr(
        comparison,
        "load_cached_pid_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no split is configured")),
    )

    assert comparison.collect_step_diff_traces([run], num_episodes=1, seed=0) == {}
    assert "no train/val/test_bins configured" in capsys.readouterr().out
