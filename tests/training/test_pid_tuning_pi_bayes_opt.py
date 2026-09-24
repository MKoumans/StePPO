"""Tests for the Diffrax PID Bayesian-optimisation baseline (research/pid_tuning)."""

from types import SimpleNamespace

import numpy as np


def _config():
    return SimpleNamespace(
        system="scalar_decay",
        rtol=1e-3,
        atol=1e-6,
        dt_min=1e-8,
        dt_max=1.0,
        t_end=1.0,
    )


def _dataset():
    from research.pid_tuning.pi_bayes_opt import PIData

    return PIData(
        task_params=np.array([1.0, 2.0], dtype=np.float32),
        episode_keys=np.zeros((2, 2), dtype=np.uint32),
        ref_final_y=np.ones((2, 1), dtype=np.float64),
        fingerprint="fixed-train",
        split="train",
    )


def _fake_solver(_config, controller, task_params, keys, max_steps, save_steps=False):
    assert not save_steps
    kp = float(controller.pcoeff)
    ki = float(controller.icoeff)
    distance = (kp - 0.4) ** 2 + (ki - 1.2) ** 2
    steps = np.full(len(task_params), 20.0 + 10.0 * distance)
    final_y = np.ones((len(task_params), 1))
    return {
        "final_y": final_y,
        "steps": steps,
        "accepted": steps,
        "rejected": np.zeros_like(steps),
        "t_reached": np.full(len(task_params), _config.t_end),
    }


def test_pi_data_rejects_mismatched_episode_arrays():
    from research.pid_tuning.pi_bayes_opt import PIData

    try:
        PIData(
            task_params=np.ones(2),
            episode_keys=np.zeros((3, 2), dtype=np.uint32),
            ref_final_y=np.ones((2, 1)),
            fingerprint="x",
            split="train",
        )
    except ValueError as error:
        assert "same number of episodes" in str(error)
    else:
        raise AssertionError("PIData must reject mismatched episode arrays")


def test_load_split_uses_existing_cached_probe(monkeypatch):
    from research.pid_tuning import pi_bayes_opt

    config = _config()
    config.train_bins = ((1.0, 2.0),)
    expected = {
        "task_params": np.array([1.0]),
        "episode_keys": np.zeros((1, 2), dtype=np.uint32),
        "ref_final_y": np.ones((1, 1)),
        "fingerprint": "cached",
    }
    calls = []

    def load_cached(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(pi_bayes_opt, "load_cached_pid_batch", load_cached)
    data = pi_bayes_opt.load_split(config, "train", num_envs=1, max_steps=8)

    assert data.fingerprint == "cached"
    assert calls[0]["split"] == "train"
    assert calls[0]["bins"] == [(1.0, 2.0)]


def test_evaluate_pi_reports_steps_accuracy_and_completion():
    from research.pid_tuning.pi_bayes_opt import PIGains, evaluate_pi

    result = evaluate_pi(
        _config(),
        _dataset(),
        PIGains(0.4, 1.2),
        max_steps=100,
        solve_fn=_fake_solver,
    )

    assert result.gains == PIGains(0.4, 1.2)
    assert result.mean_steps == 20.0
    assert result.completion_rate == 1.0
    assert result.p95_error < -5.0
    assert result.feasible


def test_evaluate_pi_cache_avoids_second_solve(tmp_path):
    from research.pid_tuning.pi_bayes_opt import PIGains, evaluate_pi

    calls = []

    def solver(*args, **kwargs):
        calls.append(1)
        return _fake_solver(*args, **kwargs)

    kwargs = dict(
        env_config=_config(),
        dataset=_dataset(),
        gains=PIGains(0.4, 1.2),
        max_steps=100,
        solve_fn=solver,
        cache_dir=tmp_path,
    )
    first = evaluate_pi(**kwargs)
    second = evaluate_pi(**kwargs)

    assert len(calls) == 1
    assert second.mean_steps == first.mean_steps


def test_optimise_pi_selects_fastest_feasible_candidate_and_plots(tmp_path):
    from research.pid_tuning.pi_bayes_opt import PIGains, optimise_pi, plot_search

    search = optimise_pi(
        _config(),
        _dataset(),
        max_steps=100,
        initial_gains=(PIGains(0.0, 1.0), PIGains(0.4, 1.2), PIGains(1.0, 1.0)),
        iterations=0,
        solve_fn=_fake_solver,
        cache_dir=tmp_path / "cache",
    )

    assert search.best.gains == PIGains(0.4, 1.2)
    output = tmp_path / "pid-search.png"
    plot_search(search, output)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_evaluate_pi_passes_pid_coefficients():
    from research.pid_tuning.pi_bayes_opt import PIGains, evaluate_pi

    observed = {}

    def solver(_config, controller, task_params, keys, max_steps, save_steps=False):
        observed.update(
            p=float(controller.pcoeff), i=float(controller.icoeff), d=float(controller.dcoeff)
        )
        return {
            "final_y": np.ones((len(task_params), 1)),
            "steps": np.ones(len(task_params)),
            "t_reached": np.ones(len(task_params)),
        }

    evaluate_pi(_config(), _dataset(), PIGains(0.25, 1.5), max_steps=10, solve_fn=solver)

    assert observed == {"p": 0.25, "i": 1.5, "d": 0.0}


def test_evaluate_pi_passes_nonzero_derivative_gain():
    from research.pid_tuning.pi_bayes_opt import PIGains, evaluate_pi

    observed = {}

    def solver(_config, controller, task_params, keys, max_steps, save_steps=False):
        observed.update(d=float(controller.dcoeff))
        return {
            "final_y": np.ones((len(task_params), 1)),
            "steps": np.ones(len(task_params)),
            "t_reached": np.ones(len(task_params)),
        }

    evaluate_pi(_config(), _dataset(), PIGains(0.25, 1.5, 0.05), max_steps=10, solve_fn=solver)

    assert observed == {"d": 0.05}


def test_incomplete_candidate_is_infeasible_even_with_low_step_count():
    from research.pid_tuning.pi_bayes_opt import PIConstraints, PIGains, evaluate_pi

    def solver(_config, controller, task_params, keys, max_steps, save_steps=False):
        return {
            "final_y": np.ones((len(task_params), 1)),
            "steps": np.ones(len(task_params)),
            "t_reached": np.zeros(len(task_params)),
        }

    result = evaluate_pi(
        _config(),
        _dataset(),
        PIGains(0.5, 1.0),
        max_steps=10,
        constraints=PIConstraints(max_p95_error=-16.0, min_completion_rate=1.0),
        solve_fn=solver,
    )

    assert result.mean_steps == 1.0
    assert result.completion_rate == 0.0
    assert not result.feasible


def test_optimise_pi_searches_kd_when_bounds_given():
    from research.pid_tuning.pi_bayes_opt import PIGains, optimise_pi

    def solver(_env_config, controller, task_params, keys, max_steps, save_steps=False):
        kd = float(controller.dcoeff)
        steps = np.full(len(task_params), 20.0 + 10.0 * (kd - 0.3) ** 2)
        return {
            "final_y": np.ones((len(task_params), 1)),
            "steps": steps,
            "t_reached": np.full(len(task_params), _config().t_end),
        }

    search = optimise_pi(
        _config(),
        _dataset(),
        max_steps=100,
        kd_bounds=(0.0, 1.0),
        initial_gains=(PIGains(0.0, 1.0, 0.0), PIGains(0.0, 1.0, 0.3)),
        iterations=0,
        solve_fn=solver,
    )

    assert search.best.gains.kd == 0.3


def test_p95_slack_is_applied_in_raw_relative_error_space():
    from research.pid_tuning.pi_bayes_opt import PIGains, optimise_pi

    def solver(_env_config, controller, task_params, keys, max_steps, save_steps=False):
        assert not save_steps
        # Baseline raw relative error is 1.0. The two candidates are 5% and 20% worse.
        if controller.pcoeff == 0.0:
            raw_relative_error = 1.0
        else:
            raw_relative_error = 1.0 + (0.05 if controller.pcoeff < 0.75 else 0.20)
        return {
            "final_y": np.full((len(task_params), 1), 1.0 + raw_relative_error),
            "steps": np.ones(len(task_params)),
            "t_reached": np.full(len(task_params), _config().t_end),
        }

    search = optimise_pi(
        _config(),
        _dataset(),
        max_steps=10,
        initial_gains=(PIGains(0.5, 1.0), PIGains(1.0, 1.0)),
        iterations=0,
        p95_error_slack=0.10,
        solve_fn=solver,
    )

    expected_limit = search.baseline.p95_error + np.log10(1.10)
    assert np.isclose(search.constraints.max_p95_error, expected_limit)
    assert search.evaluations[1].feasible  # 5% raw error increase is allowed.
    assert not search.evaluations[2].feasible  # 20% raw error increase is not.


def test_optimise_pi_sends_realtime_progress_to_injected_logger():
    from research.pid_tuning.pi_bayes_opt import PIGains, optimise_pi

    def solver(_env_config, controller, task_params, keys, max_steps, save_steps=False):
        assert not save_steps
        steps = np.full(len(task_params), 3.0 + controller.pcoeff + controller.icoeff)
        return {
            "final_y": np.ones((len(task_params), 1)),
            "steps": steps,
            "t_reached": np.full(len(task_params), _config().t_end),
        }

    messages = []
    optimise_pi(
        _config(),
        _dataset(),
        max_steps=10,
        initial_gains=(PIGains(0.5, 1.0),),
        iterations=0,
        solve_fn=solver,
        log_fn=messages.append,
    )

    assert messages[0].startswith("[PID-BO] baseline start:")
    assert any("candidate 2/2 start" in message for message in messages)
    assert any("mean_steps=" in message and "elapsed=" in message for message in messages)
    assert messages[-1].startswith("[PID-BO] train optimum:")


def test_log_flag_prints_each_message_with_flush(monkeypatch):
    from research.pid_tuning.pi_bayes_opt import optimise_pi

    def solver(_env_config, controller, task_params, keys, max_steps, save_steps=False):
        assert not save_steps
        steps = np.full(len(task_params), 3.0 + controller.pcoeff + controller.icoeff)
        return {
            "final_y": np.ones((len(task_params), 1)),
            "steps": steps,
            "t_reached": np.full(len(task_params), _config().t_end),
        }

    calls = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: calls.append((args, kwargs)))
    optimise_pi(_config(), _dataset(), max_steps=10, iterations=0, solve_fn=solver, log=True)

    assert calls
    assert all(call[1].get("flush") is True for call in calls)
