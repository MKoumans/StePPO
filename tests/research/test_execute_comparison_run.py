"""Run orchestration tests without loading policies or solving ODEs."""

from types import SimpleNamespace

import numpy as np
import pytest

from research.ode import execute_comparison as comparison


def _inputs(tmp_path, run_uids=("run-a",)):
    config = SimpleNamespace(
        env=SimpleNamespace(
            system="scalar_decay",
            lam_min=1.0,
            lam_max=100.0,
            train_bins=((1.0, 50.0),),
            val_bins=(),
            test_bins=((1.0, 10.0), (10.0, 100.0)),
        ),
    )
    runs = [
        {
            "run_uid": uid,
            "exp_name": f"scalar_decay_e1_r{idx + 1}",
            "config_path": f"/checkpoints/{uid}/config.yaml",
            "checkpoint_dir": f"/checkpoints/{uid}/checkpoint_5",
        }
        for idx, uid in enumerate(run_uids)
    ]
    args = SimpleNamespace(
        num_task_points=2,
        num_step_diff_episodes=3,
        seed=17,
        ref_tol_factor=2.0,
        num_bin_episodes=3,
    )
    return config, runs, args, str(tmp_path / "comparison")


def _step_diff(uid):
    return {
        uid: {
            "test": {
                "diff": np.array([1.0, -2.0]),
                "rel_diff": np.array([0.1, -0.2]),
                "err": np.array([0.01, 0.02]),
                "pid_err": np.array([0.03, 0.04]),
                "pid_steps": np.array([10.0, 20.0]),
                "bin_idx": np.array([0, 1]),
            }
        }
    }


def test_run_and_save_computes_once_then_reuses_all_run_cache(tmp_path, monkeypatch):
    config, runs, args, out_dir = _inputs(tmp_path)
    monkeypatch.setattr(comparison, "load_config_from_yaml", lambda *_args: config)
    latent_calls = []

    def collect_latent(_runs, task_value, _num_envs, _seed):
        latent_calls.append(task_value)
        return {
            "run-a": {
                "mu": np.array([[task_value, task_value + 1]], dtype=np.float32),
                "var": np.array([[0.25, 0.5]], dtype=np.float32),
                "latent_dim": 2,
            }
        }

    monkeypatch.setattr(comparison, "collect_latent_traces_for_task", collect_latent)
    monkeypatch.setattr(
        comparison,
        "collect_step_diff_traces",
        lambda *_args, **_kwargs: _step_diff("run-a"),
    )
    bins = {
        "run-a": {
            "bins": list(config.env.test_bins),
            "improvement": np.array([0.1, -0.1]),
            "trained": [True, False],
            "scheme": "binned",
            "exp_name": runs[0]["exp_name"],
        }
    }
    monkeypatch.setattr(
        comparison,
        "collect_bin_comparison",
        lambda *_args, **_kwargs: bins,
    )

    first = comparison.run_and_save(runs, args, out_dir)

    assert latent_calls == [1.0, 100.0]
    assert first["test_points"] == [1.0, 100.0]
    assert first["param_label"] == "λ"
    assert first["latent_traces_by_task"][2]["run-a"]["latent_dim"] == 2
    assert first["step_diff_traces"]["run-a"]["test"]["diff"].tolist() == [1.0, -2.0]
    assert first["bin_results"]["run-a"]["exp_name"] == runs[0]["exp_name"]

    monkeypatch.setattr(
        comparison,
        "collect_latent_traces_for_task",
        lambda *_args: pytest.fail("complete cache should avoid latent rollout"),
    )
    monkeypatch.setattr(
        comparison,
        "collect_step_diff_traces",
        lambda *_args, **_kwargs: pytest.fail("complete cache should avoid controller solves"),
    )
    second = comparison.run_and_save(runs, args, out_dir)

    assert np.array_equal(
        second["latent_traces_by_task"][1]["run-a"]["mu"],
        first["latent_traces_by_task"][1]["run-a"]["mu"],
    )
    assert second["bin_results"]["run-a"]["bins"] == first["bin_results"]["run-a"]["bins"]
    assert np.array_equal(
        second["bin_results"]["run-a"]["improvement"],
        first["bin_results"]["run-a"]["improvement"],
    )


def test_run_and_save_recomputes_every_run_after_a_partial_cache_miss(
    tmp_path,
    monkeypatch,
):
    config, runs, args, out_dir = _inputs(tmp_path, run_uids=("run-a", "run-b"))
    monkeypatch.setattr(comparison, "load_config_from_yaml", lambda *_args: config)
    test_points = [1.0, 100.0]
    data_dir = tmp_path / "comparison" / "data"
    data_dir.mkdir(parents=True)
    first_cache = data_dir / (
        f"run-a_{comparison._run_data_fingerprint(runs[0], args, test_points)}.npz"
    )
    comparison._save_run_data(str(first_cache), {}, {}, None)

    latent_run_sets = []
    monkeypatch.setattr(
        comparison,
        "collect_latent_traces_for_task",
        lambda all_runs, task, *_args: (
            latent_run_sets.append((task, [r["run_uid"] for r in all_runs]))
            or {
                uid: {"mu": np.zeros((1, 1)), "var": np.ones((1, 1)), "latent_dim": 1}
                for uid in ("run-a", "run-b")
            }
        ),
    )
    step_diff_calls = []
    monkeypatch.setattr(
        comparison,
        "collect_step_diff_traces",
        lambda all_runs, *_args, **_kwargs: (
            step_diff_calls.append([r["run_uid"] for r in all_runs])
            or {uid: {} for uid in ("run-a", "run-b")}
        ),
    )
    monkeypatch.setattr(comparison, "collect_bin_comparison", lambda *_args, **_kwargs: {})

    result = comparison.run_and_save(runs, args, out_dir)

    assert latent_run_sets == [(1.0, ["run-a", "run-b"]), (100.0, ["run-a", "run-b"])]
    assert step_diff_calls == [["run-a", "run-b"]]
    assert set(result["latent_traces_by_task"][1]) == {"run-a", "run-b"}
