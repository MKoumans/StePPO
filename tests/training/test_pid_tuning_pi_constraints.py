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
        task_params=np.array([1.0], dtype=np.float32),
        episode_keys=np.zeros((1, 2), dtype=np.uint32),
        ref_final_y=np.ones((1, 1)),
        fingerprint="constraints",
        split="train",
    )


def test_evaluate_pi_passes_pi_coefficients_and_zero_derivative():
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

    evaluate_pi(_config(), _data(), PIGains(0.25, 1.5), max_steps=10, solve_fn=solver)

    assert observed == {"p": 0.25, "i": 1.5, "d": 0.0}


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
        _data(),
        PIGains(0.5, 1.0),
        max_steps=10,
        constraints=PIConstraints(max_p95_error=-16.0, min_completion_rate=1.0),
        solve_fn=solver,
    )

    assert result.mean_steps == 1.0
    assert result.completion_rate == 0.0
    assert not result.feasible
