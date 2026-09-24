"""Tests for steppo.training.eval_metrics.eval_metrics — the fused
per-iteration efficiency + L2-error computation (issue #40).

Every test mocks at the same boundary tests/test_steps_vs_mu_errors.py
already mocks at: collect_policy_errors (the one dense policy solve) and
collect_pid_errors (the cache-only PID read) are monkeypatched at their
point of use inside steppo.training.eval_metrics, so no real diffrax solve
happens in this file, and solve_pid_batch (the shared solve primitive) is
monkeypatched directly where a test needs to prove no live solve occurs.
"""

import numpy as np

from steppo.envs.ode import (
    ODEEnv,  # noqa: F401 — import before eval_metrics, see ode_env.py's circular import
)


def _fake_cached(task_params, pid_steps):
    """Minimal cached-dataset dict — only the fields eval_metrics actually
    reads (task_params/episode_keys/pid_steps/ref_*) need to be present;
    collect_policy_errors/collect_pid_errors are mocked in these tests, so
    ref_final_y/ref_ts/ref_ys are never dereferenced, just passed through."""
    n = len(task_params)
    return {
        "task_params": np.asarray(task_params, dtype=np.float32),
        "episode_keys": np.arange(n),
        "pid_steps": np.asarray(pid_steps, dtype=np.float64),
        "ref_final_y": np.zeros((n, 1)),
        "ref_ts": np.zeros((n, 1)),
        "ref_ys": np.zeros((n, 1, 1)),
        "pid_completed": np.ones(n, dtype=bool),
        "ref_completed": np.ones(n, dtype=bool),
    }


def test_eval_metrics_matches_hand_computed_efficiency_and_l2_error(monkeypatch):
    from steppo.configs.base_config import ODEEnvConfig

    task_params = np.array([1.0, 2.0, 3.0])
    pid_steps = np.array([100.0, 200.0, 50.0])
    cached = _fake_cached(task_params, pid_steps)
    datasets = {"test": cached}

    fake_policy_result = {
        "mu": task_params,
        "err": np.array([-1.0, -2.0, -3.0]),
        "err_integrated": np.array([-1.5, -2.5, -3.5]),
        "steps": np.array([80.0, 250.0, 40.0]),
    }
    fake_pid_result = {
        "mu": task_params,
        "err": np.array([-0.5, -1.5, -2.5]),
        "err_integrated": np.array([-0.8, -1.8, -2.8]),
    }

    monkeypatch.setattr(
        "steppo.training.eval_metrics.collect_policy_errors",
        lambda *args, **kwargs: fake_policy_result,
    )
    monkeypatch.setattr(
        "steppo.training.eval_metrics.collect_pid_errors",
        lambda *args, **kwargs: fake_pid_result,
    )

    from steppo.training.eval_metrics import eval_metrics

    cfg = ODEEnvConfig(system="scalar_decay", atol=1e-8)
    result = eval_metrics(
        cfg,
        policy_controller=object(),
        datasets=datasets,
        num_envs=3,
        max_steps=10,
    )

    improvement = (pid_steps - fake_policy_result["steps"]) / np.maximum(pid_steps, 1.0)
    expected_efficiency = {
        "mean": float(np.mean(improvement)),
        "std": float(np.std(improvement)),
        "median": float(np.median(improvement)),
        "mean_pid_steps": float(np.mean(pid_steps)),
        "std_pid_steps": float(np.std(pid_steps)),
        "mean_policy_steps": float(np.mean(fake_policy_result["steps"])),
        "std_policy_steps": float(np.std(fake_policy_result["steps"])),
    }
    expected_l2_error = {
        "policy_err_mean": float(np.mean(fake_policy_result["err"])),
        "policy_err_std": float(np.std(fake_policy_result["err"])),
        "policy_err_integrated_mean": float(np.mean(fake_policy_result["err_integrated"])),
        "policy_err_integrated_std": float(np.std(fake_policy_result["err_integrated"])),
        "pid_err_mean": float(np.mean(fake_pid_result["err"])),
        "pid_err_std": float(np.std(fake_pid_result["err"])),
        "pid_err_integrated_mean": float(np.mean(fake_pid_result["err_integrated"])),
        "pid_err_integrated_std": float(np.std(fake_pid_result["err_integrated"])),
    }

    assert set(result.keys()) == {"efficiency", "l2_error"}
    assert set(result["efficiency"].keys()) == {"test"}
    assert set(result["l2_error"].keys()) == {"test"}
    for key, expected_val in expected_efficiency.items():
        assert np.isclose(result["efficiency"]["test"][key], expected_val), key
    for key, expected_val in expected_l2_error.items():
        assert np.isclose(result["l2_error"]["test"][key], expected_val), key


def test_eval_metrics_calls_collect_policy_errors_exactly_once_per_split(monkeypatch):
    from steppo.configs.base_config import ODEEnvConfig

    task_params = np.array([1.0, 2.0])
    pid_steps = np.array([10.0, 20.0])
    datasets = {
        "train": _fake_cached(task_params, pid_steps),
        "test": _fake_cached(task_params, pid_steps),
    }

    call_count = {"n": 0}

    def fake_collect_policy_errors(*args, **kwargs):
        call_count["n"] += 1
        return {
            "mu": task_params,
            "err": np.array([-1.0, -1.0]),
            "err_integrated": np.array([-1.0, -1.0]),
            "steps": np.array([5.0, 5.0]),
        }

    monkeypatch.setattr(
        "steppo.training.eval_metrics.collect_policy_errors",
        fake_collect_policy_errors,
    )
    monkeypatch.setattr(
        "steppo.training.eval_metrics.collect_pid_errors",
        lambda *args, **kwargs: {
            "mu": task_params,
            "err": np.array([-1.0, -1.0]),
            "err_integrated": np.array([-1.0, -1.0]),
        },
    )

    from steppo.training.eval_metrics import eval_metrics

    cfg = ODEEnvConfig(system="scalar_decay")
    eval_metrics(cfg, policy_controller=object(), datasets=datasets, num_envs=2, max_steps=10)

    assert call_count["n"] == len(datasets)


def test_eval_metrics_never_solves_pid_live(monkeypatch):
    """The PID side must read entirely from the cache — no live PID solve —
    even when eval_metrics is exercised end-to-end (real collect_pid_errors,
    only the policy side mocked out). Same style as
    tests/test_steps_vs_mu_errors.py::test_collect_pid_errors_reads_entirely_from_cache_no_live_solve."""
    from steppo.configs.base_config import ODEEnvConfig

    def fail_if_called(*args, **kwargs):
        raise AssertionError("eval_metrics must not solve PID live")

    monkeypatch.setattr("steppo.training.steps_vs_mu.solve_pid_batch", fail_if_called)

    task_params = np.array([5.0])
    pid_steps = np.array([10.0])
    cached = _fake_cached(task_params, pid_steps)
    # collect_pid_errors (real implementation) reads these cache fields directly.
    cached["pid_final_y"] = np.array([[2.5]])
    cached["ref_final_y"] = np.array([[2.0]])
    cached["pid_ts"] = np.array([[0.0, 1.0, np.inf]])
    cached["pid_local_err"] = np.array([[np.inf, 0.25, np.inf]])

    monkeypatch.setattr(
        "steppo.training.eval_metrics.collect_policy_errors",
        lambda *args, **kwargs: {
            "mu": task_params,
            "err": np.array([-1.0]),
            "err_integrated": np.array([-1.0]),
            "steps": np.array([8.0]),
        },
    )

    from steppo.training.eval_metrics import eval_metrics

    cfg = ODEEnvConfig(system="scalar_decay", atol=0.0)
    result = eval_metrics(
        cfg,
        policy_controller=object(),
        datasets={"test": cached},
        num_envs=1,
        max_steps=10,
    )

    assert "test" in result["efficiency"]
    assert "test" in result["l2_error"]


def test_eval_metrics_skips_missing_split(monkeypatch):
    from steppo.configs.base_config import ODEEnvConfig

    task_params = np.array([1.0, 2.0])
    pid_steps = np.array([10.0, 20.0])
    datasets = {
        "train": _fake_cached(task_params, pid_steps),
        "val": None,
        "test": _fake_cached(task_params, pid_steps),
    }

    monkeypatch.setattr(
        "steppo.training.eval_metrics.collect_policy_errors",
        lambda *args, **kwargs: {
            "mu": task_params,
            "err": np.array([-1.0, -1.0]),
            "err_integrated": np.array([-1.0, -1.0]),
            "steps": np.array([5.0, 5.0]),
        },
    )
    monkeypatch.setattr(
        "steppo.training.eval_metrics.collect_pid_errors",
        lambda *args, **kwargs: {
            "mu": task_params,
            "err": np.array([-1.0, -1.0]),
            "err_integrated": np.array([-1.0, -1.0]),
        },
    )

    from steppo.training.eval_metrics import eval_metrics

    cfg = ODEEnvConfig(system="scalar_decay")
    result = eval_metrics(
        cfg,
        policy_controller=object(),
        datasets=datasets,
        num_envs=2,
        max_steps=10,
    )

    assert set(result["efficiency"].keys()) == {"train", "test"}
    assert set(result["l2_error"].keys()) == {"train", "test"}
    assert "val" not in result["efficiency"]
    assert "val" not in result["l2_error"]


def test_eval_metrics_slices_reference_arrays_to_num_envs(monkeypatch):
    """When num_envs is smaller than the cached dataset's full size (the real
    production case: EvalMetricsConfig.num_envs defaults to 256 against a
    1024-episode cached dataset), ref_final_y/ref_ts/ref_ys passed into
    collect_policy_errors's dense solve must be sliced to match task_params/
    episode_keys — not passed through at the cache's full size, which would
    shape-mismatch inside the real collect_policy_errors (found by an
    end-to-end smoke test; every other test in this file happens to use
    num_envs == the full cached dataset size, which never exercises this)."""
    from steppo.configs.base_config import ODEEnvConfig

    full_n = 6
    task_params = np.arange(1.0, 1.0 + full_n)
    pid_steps = np.full(full_n, 10.0)
    cached = _fake_cached(task_params, pid_steps)
    # Distinct values per row so a wrong slice is caught by shape, not just luck.
    cached["ref_final_y"] = np.arange(full_n, dtype=np.float64).reshape(full_n, 1)
    cached["ref_ts"] = np.arange(full_n, dtype=np.float64).reshape(full_n, 1)
    cached["ref_ys"] = np.arange(full_n, dtype=np.float64).reshape(full_n, 1, 1)

    num_envs = 3
    seen = {}

    def fake_collect_policy_errors(
        env_config, controller, tp, ek, max_steps, ref_final_y, ref_ts, ref_ys, atol
    ):
        seen["task_params_len"] = len(tp)
        seen["ref_final_y_len"] = len(ref_final_y)
        seen["ref_ts_len"] = len(ref_ts)
        seen["ref_ys_len"] = len(ref_ys)
        return {
            "mu": tp,
            "err": np.zeros(len(tp)),
            "err_integrated": np.zeros(len(tp)),
            "steps": np.full(len(tp), 5.0),
        }

    monkeypatch.setattr(
        "steppo.training.eval_metrics.collect_policy_errors",
        fake_collect_policy_errors,
    )
    monkeypatch.setattr(
        "steppo.training.eval_metrics.collect_pid_errors",
        lambda *args, **kwargs: {
            "mu": task_params,
            "err": np.zeros(full_n),
            "err_integrated": np.zeros(full_n),
        },
    )

    from steppo.training.eval_metrics import eval_metrics

    cfg = ODEEnvConfig(system="scalar_decay")
    eval_metrics(
        cfg,
        policy_controller=object(),
        datasets={"test": cached},
        num_envs=num_envs,
        max_steps=10,
    )

    assert seen["task_params_len"] == num_envs
    assert seen["ref_final_y_len"] == num_envs
    assert seen["ref_ts_len"] == num_envs
    assert seen["ref_ys_len"] == num_envs


def test_eval_metrics_efficiency_formula_matches_old_definition(monkeypatch):
    """Hand-computed (pid_steps - policy_steps) / max(pid_steps, 1.0) mean/std/
    median must match eval_metrics's efficiency output exactly — proving the
    formula itself didn't change when the rollout mechanism switched from the
    native-env loop (eval.py's _eval_efficiency_subset) to the diffrax-native
    fast path (collect_policy_errors)."""
    from steppo.configs.base_config import ODEEnvConfig

    task_params = np.array([1.0, 2.0, 3.0, 4.0])
    pid_steps = np.array(
        [40.0, 0.0, 120.0, 60.0]
    )  # includes a zero to exercise max(pid_steps, 1.0)
    policy_steps = np.array([30.0, 5.0, 150.0, 60.0])
    cached = _fake_cached(task_params, pid_steps)

    monkeypatch.setattr(
        "steppo.training.eval_metrics.collect_policy_errors",
        lambda *args, **kwargs: {
            "mu": task_params,
            "err": np.zeros(4),
            "err_integrated": np.zeros(4),
            "steps": policy_steps,
        },
    )
    monkeypatch.setattr(
        "steppo.training.eval_metrics.collect_pid_errors",
        lambda *args, **kwargs: {
            "mu": task_params,
            "err": np.zeros(4),
            "err_integrated": np.zeros(4),
        },
    )

    from steppo.training.eval_metrics import eval_metrics

    cfg = ODEEnvConfig(system="scalar_decay")
    result = eval_metrics(
        cfg,
        policy_controller=object(),
        datasets={"test": cached},
        num_envs=4,
        max_steps=10,
    )

    # Independently re-derived old-definition formula (eval.py's _eval_efficiency_subset).
    improvement = (pid_steps - policy_steps) / np.maximum(pid_steps, 1.0)
    assert np.isclose(result["efficiency"]["test"]["mean"], float(np.mean(improvement)))
    assert np.isclose(result["efficiency"]["test"]["std"], float(np.std(improvement)))
    assert np.isclose(result["efficiency"]["test"]["median"], float(np.median(improvement)))
