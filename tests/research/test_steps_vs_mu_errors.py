import numpy as np

from steppo.envs.ode import (
    ODEEnv,  # noqa: F401 — import before steps_vs_mu, see ode_env.py's circular import
)


def test_collect_pid_errors_reads_entirely_from_cache_no_live_solve(monkeypatch):
    """collect_pid_errors must never call solve_pid_batch — it derives both
    error aggregates purely from the cache's pid_final_y/ref_final_y/pid_ts/
    pid_local_err fields (see error_dist.load_pid_batch)."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError("collect_pid_errors must not solve PID live")

    monkeypatch.setattr("steppo.training.steps_vs_mu.solve_pid_batch", fail_if_called)

    from steppo.training.steps_vs_mu import collect_pid_errors

    cached = {
        "task_params": np.array([5.0]),
        "pid_final_y": np.array([[2.5]]),
        "ref_final_y": np.array([[2.0]]),
        # PID's own dense solve, +inf-padded convention: two valid points.
        "pid_ts": np.array([[0.0, 1.0, np.inf]]),
        # Linear-space local error at those times: mirrors
        # tests/test_trajectory_compare.py's hand-computed local-error case.
        "pid_local_err": np.array([[np.inf, 0.25, np.inf]]),
        "pid_completed": np.array([True]),
        "ref_completed": np.array([True]),
    }

    result = collect_pid_errors(cached, atol=0.0)

    # last-step: relative_l2_error(2.5, 2.0, atol=0) = 0.25 -> log10(0.25)
    assert np.allclose(result["err"], [np.log10(0.25)])
    # time-integrated: time_average of [to_log10_clipped(inf), to_log10_clipped(0.25)]
    # over ts=[0, 1] -> trapz([CEIL, log10(0.25)], [0, 1]) / 1.0
    from steppo.training.trajectory_compare import LOG10_ERR_CEIL

    expected_integrated = (LOG10_ERR_CEIL + np.log10(0.25)) / 2.0
    assert np.allclose(result["err_integrated"], [expected_integrated])
    assert np.allclose(result["mu"], [5.0])


def test_collect_policy_errors_passes_the_given_controller_through_unmodified(monkeypatch):
    from steppo.configs.base_config import ODEEnvConfig

    ref_ts = np.array([[0.0, 1.0, 2.0]])
    ref_ys = np.array([[[0.0], [1.0], [2.0]]])
    ref_final_y = np.array([[2.0]])
    fake_dense = {
        "ts": np.array([[0.0, 1.0, 2.0]]),
        "ys": np.array([[[0.5], [1.5], [2.5]]]),
        "accepted": np.array([3]),
        "rejected": np.array([0]),
        "t_reached": np.array([2.0]),
        "completed": np.array([True]),
    }

    seen_controller = {}

    def fake_solve(
        env_config, controller, task_params, episode_keys, max_steps, save_steps, save_ys
    ):
        seen_controller["value"] = controller
        return fake_dense

    monkeypatch.setattr("steppo.training.steps_vs_mu.solve_pid_batch", fake_solve)

    from steppo.training.steps_vs_mu import collect_policy_errors

    cfg = ODEEnvConfig(system="scalar_decay", atol=0.0)
    sentinel_controller = object()
    result = collect_policy_errors(
        cfg,
        sentinel_controller,
        task_params=np.array([5.0]),
        episode_keys=np.array([0]),
        max_steps=10,
        ref_final_y=ref_final_y,
        ref_ts=ref_ts,
        ref_ys=ref_ys,
        atol=0.0,
    )

    assert seen_controller["value"] is sentinel_controller
    assert np.allclose(result["err"], [np.log10(0.25)])
    # time-integrated now averages log10-clipped local error in log space
    # (see test_trajectory_compare.py::test_log10_time_integrated_error_averages_local_error_in_log_space
    # for the derivation of this exact value): 0.44897, not log10(0.5).
    assert np.allclose(result["err_integrated"], [0.44897], atol=1e-4)
    # steps = accepted + rejected from the one dense solve already performed
    # above — no re-solve needed to also expose the step count (issue #40).
    assert np.array_equal(result["steps"], np.array([3]))


def _fake_cache(with_oracle=True):
    task_params = np.array([1.0, 2.0])
    return {
        "cache": "sentinel",
        "task_params": task_params,
        "episode_keys": np.array([0, 1]),
        "pid_steps": np.array([10.0, 20.0]),
        "pid_steps_unlimited": None,
        "oracle_steps": np.array([8.0, 18.0]) if with_oracle else None,
        "pid_final_y": np.array([[1.1], [2.2]]),
        "ref_final_y": np.array([[1.0], [2.0]]),
        "ref_ts": np.array([[0.0, 1.0], [0.0, 1.0]]),
        "ref_ys": np.array([[[0.0], [1.0]], [[0.0], [2.0]]]),
        "pid_ts": np.array([[0.0, 1.0], [0.0, 1.0]]),
        "pid_local_err": np.array([[0.05, 0.1], [0.05, 0.1]]),
        "oracle_final_y": np.array([[1.1], [2.1]]) if with_oracle else None,
        "pid_completed": np.array([True, False]),
        "ref_completed": np.array([True, True]),
        "oracle_completed": np.array([True, True]) if with_oracle else None,
        "pid_completed_unlimited": None,
    }


def test_build_steps_vs_mu_data_with_error_populates_pid_and_oracle_error(monkeypatch):
    from steppo.configs.base_config import ODEEnvConfig

    monkeypatch.setattr(
        "steppo.training.steps_vs_mu.load_cached_pid_batch",
        lambda *args, **kwargs: _fake_cache(),
    )
    seen_pid_call = {}

    def fake_collect_pid_errors(cached, atol):
        seen_pid_call.update(cached=cached, atol=atol)
        return {
            "mu": cached["task_params"],
            "err": np.array([-1.0, -2.0]),
            "err_integrated": np.array([-1.5, -2.5]),
        }

    monkeypatch.setattr("steppo.training.steps_vs_mu.collect_pid_errors", fake_collect_pid_errors)

    from steppo.training.steps_vs_mu import build_steps_vs_mu_data

    cfg = ODEEnvConfig(system="scalar_decay", atol=1e-8)
    result = build_steps_vs_mu_data(
        cfg,
        max_steps=10,
        bins=[(1.0, 10.0)],
        with_error=True,
    )

    assert seen_pid_call["atol"] == cfg.atol
    assert np.array_equal(seen_pid_call["cached"]["ref_final_y"], _fake_cache()["ref_final_y"])
    assert np.array_equal(result["pid_error"]["mu"], result["pid"]["mu"])
    assert np.allclose(result["pid_error"]["err"], [-1.0, -2.0])
    assert np.allclose(result["pid_error"]["err_integrated"], [-1.5, -2.5])
    assert np.allclose(
        result["oracle_error"]["err"],
        [
            np.log10(abs(1.1 - 1.0) / (abs(1.0) + cfg.atol)),
            np.log10(abs(2.1 - 2.0) / (abs(2.0) + cfg.atol)),
        ],
    )


def test_build_steps_vs_mu_data_without_error_leaves_error_fields_none(monkeypatch):
    from steppo.configs.base_config import ODEEnvConfig

    monkeypatch.setattr(
        "steppo.training.steps_vs_mu.load_cached_pid_batch",
        lambda *args, **kwargs: _fake_cache(),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("collect_pid_errors must not run when with_error=False")

    monkeypatch.setattr("steppo.training.steps_vs_mu.collect_pid_errors", fail_if_called)

    from steppo.training.steps_vs_mu import build_steps_vs_mu_data

    cfg = ODEEnvConfig(system="scalar_decay")
    result = build_steps_vs_mu_data(cfg, max_steps=10, bins=[(1.0, 10.0)])

    assert result["pid_error"] is None
    assert result["policy_error"] is None
    assert result["oracle_error"] is None


def test_build_steps_vs_mu_data_by_split_forwards_with_error(monkeypatch):
    from steppo.configs.base_config import ODEEnvConfig

    seen = []

    def fake_build(env_config, **kwargs):
        seen.append(kwargs.get("with_error"))
        return {"pid_error": kwargs.get("with_error")}

    monkeypatch.setattr("steppo.training.steps_vs_mu.build_steps_vs_mu_data", fake_build)

    from steppo.training.steps_vs_mu import build_steps_vs_mu_data_by_split

    cfg = ODEEnvConfig(system="scalar_decay", train_bins=((1.0, 5.0),), test_bins=((5.0, 10.0),))
    result = build_steps_vs_mu_data_by_split(
        cfg,
        max_steps=10,
        split_bins={"train": cfg.train_bins, "test": cfg.test_bins},
        with_error=True,
    )

    assert seen == [True, True]
    assert result["train"]["pid_error"] is True
    assert result["test"]["pid_error"] is True


def test_build_steps_vs_mu_data_carries_completion_onto_every_series(monkeypatch):
    """Each series dict pairs its step counts with the mask saying which of
    those counts came from a solve that actually reached t_end — the
    plotting layer aggregates from these dicts and has no other way to tell
    an aborted solve from a cheap one."""
    from steppo.configs.base_config import ODEEnvConfig

    monkeypatch.setattr(
        "steppo.training.steps_vs_mu.load_cached_pid_batch",
        lambda *args, **kwargs: _fake_cache(),
    )

    from steppo.training.steps_vs_mu import build_steps_vs_mu_data

    cfg = ODEEnvConfig(system="scalar_decay", atol=1e-8)
    result = build_steps_vs_mu_data(cfg, max_steps=10, bins=[(1.0, 10.0)])

    assert np.array_equal(result["pid"]["completed"], [True, False])
    assert np.array_equal(result["oracle"]["completed"], [True, True])


def test_build_steps_vs_mu_data_carries_policy_completion_from_the_live_solve(monkeypatch):
    """The policy series is solved live rather than read from the cache, so
    its completion mask has to come from that solve."""
    from steppo.configs.base_config import ODEEnvConfig

    monkeypatch.setattr(
        "steppo.training.steps_vs_mu.load_cached_pid_batch",
        lambda *args, **kwargs: _fake_cache(),
    )
    monkeypatch.setattr(
        "steppo.training.steps_vs_mu.solve_pid_batch",
        lambda *args, **kwargs: {
            "steps": np.array([6.0, 7.0]),
            "completed": np.array([False, True]),
            "final_y": np.array([[1.0], [2.0]]),
            "t_reached": np.array([0.5, 1.0]),
        },
    )

    from steppo.training.steps_vs_mu import build_steps_vs_mu_data

    cfg = ODEEnvConfig(system="scalar_decay", atol=1e-8)
    result = build_steps_vs_mu_data(
        cfg,
        max_steps=10,
        bins=[(1.0, 10.0)],
        policy_controller=object(),
    )

    assert np.array_equal(result["policy"]["steps"], [6.0, 7.0])
    assert np.array_equal(result["policy"]["completed"], [False, True])


def test_error_series_are_marked_incomplete_when_the_reference_did_not_finish(monkeypatch):
    """An error is only meaningful if both the graded solve and the
    reference it is graded against reached t_end. chua_smooth's cached
    reference stopped at t~0.02 for every m, so every error in that run
    was measured against a trajectory that never happened."""
    from steppo.configs.base_config import ODEEnvConfig

    cache = _fake_cache()
    cache["pid_completed"] = np.array([True, True])
    cache["ref_completed"] = np.array([True, False])
    monkeypatch.setattr(
        "steppo.training.steps_vs_mu.load_cached_pid_batch",
        lambda *a, **k: cache,
    )

    from steppo.training.steps_vs_mu import build_steps_vs_mu_data

    cfg = ODEEnvConfig(system="scalar_decay", atol=1e-8)
    result = build_steps_vs_mu_data(cfg, max_steps=10, bins=[(1.0, 10.0)], with_error=True)

    assert np.array_equal(result["pid_error"]["completed"], [True, False])
    assert np.array_equal(result["oracle_error"]["completed"], [True, False])
