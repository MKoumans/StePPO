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


def _data():
    from research.pid_tuning.pi_bayes_opt import PIData

    return PIData(
        task_params=np.array([1.0]),
        episode_keys=np.zeros((1, 2), dtype=np.uint32),
        ref_final_y=np.ones((1, 1)),
        fingerprint="logging",
        split="train",
    )


def _solver(_config, controller, task_params, keys, max_steps, save_steps=False):
    assert not save_steps
    steps = np.full(len(task_params), 3.0 + controller.pcoeff + controller.icoeff)
    return {
        "final_y": np.ones((len(task_params), 1)),
        "steps": steps,
        "t_reached": np.full(len(task_params), _config.t_end),
    }


def test_optimise_pi_sends_realtime_progress_to_injected_logger():
    from research.pid_tuning.pi_bayes_opt import PIGains, optimise_pi

    messages = []
    optimise_pi(
        _config(),
        _data(),
        max_steps=10,
        initial_gains=(PIGains(0.5, 1.0),),
        iterations=0,
        solve_fn=_solver,
        log_fn=messages.append,
    )

    assert messages[0].startswith("[PID-BO] baseline start:")
    assert any("candidate 2/2 start" in message for message in messages)
    assert any("mean_steps=" in message and "elapsed=" in message for message in messages)
    assert messages[-1].startswith("[PID-BO] train optimum:")


def test_log_flag_prints_each_message_with_flush(monkeypatch):
    from research.pid_tuning.pi_bayes_opt import optimise_pi

    calls = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: calls.append((args, kwargs)))
    optimise_pi(_config(), _data(), max_steps=10, iterations=0, solve_fn=_solver, log=True)

    assert calls
    assert all(call[1].get("flush") is True for call in calls)
