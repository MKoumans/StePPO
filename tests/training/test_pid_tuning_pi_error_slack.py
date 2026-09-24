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
        fingerprint="p95-slack",
        split="train",
    )


def _solver(_config, controller, task_params, keys, max_steps, save_steps=False):
    assert not save_steps
    # Baseline raw relative error is 1.0. The two candidates are 5% and 20% worse.
    if controller.pcoeff == 0.0:
        raw_relative_error = 1.0
    else:
        raw_relative_error = 1.0 + (0.05 if controller.pcoeff < 0.75 else 0.20)
    return {
        "final_y": np.full((len(task_params), 1), 1.0 + raw_relative_error),
        "steps": np.ones(len(task_params)),
        "t_reached": np.full(len(task_params), _config.t_end),
    }


def test_p95_slack_is_applied_in_raw_relative_error_space():
    from research.pid_tuning.pi_bayes_opt import PIGains, optimise_pi

    search = optimise_pi(
        _config(),
        _data(),
        max_steps=10,
        initial_gains=(PIGains(0.5, 1.0), PIGains(1.0, 1.0)),
        iterations=0,
        p95_error_slack=0.10,
        solve_fn=_solver,
    )

    expected_limit = search.baseline.p95_error + np.log10(1.10)
    assert np.isclose(search.constraints.max_p95_error, expected_limit)
    assert search.evaluations[1].feasible  # 5% raw error increase is allowed.
    assert not search.evaluations[2].feasible  # 20% raw error increase is not.
