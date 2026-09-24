import dataclasses

import diffrax
import jax
import numpy as np

from steppo.configs.base_config import ODEEnvConfig
from steppo.envs.ode import (
    ODEEnv,  # noqa: F401 — import before pid_solve, see ode_env.py's circular import
)
from steppo.training.error_dist import sample_eval_tasks
from steppo.training.pid_solve import solve_pid_batch


def _controller(cfg):
    return diffrax.PIDController(
        rtol=cfg.rtol,
        atol=cfg.atol,
        dtmin=cfg.dt_min,
        dtmax=cfg.dt_max,
        force_dtmin=False,
    )


def test_float64_config_produces_float64_dense_trajectory():
    """The eval solve path must honor ODEEnvConfig.precision without relying
    on an ODEEnv instance having enabled JAX x64 first."""
    from steppo.training.pid_solve import solve_pid_batch

    cfg = ODEEnvConfig(
        system="scalar_decay",
        precision="float64",
        t_end=1.0,
        dt0=0.1,
        rtol=1e-4,
        atol=1e-6,
        dt_min=1e-8,
        dt_max=1.0,
        lam_min=1.0,
        lam_max=10.0,
    )
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        dense = solve_pid_batch(
            cfg,
            _controller(cfg),
            np.array([5.0]),
            np.array([[0, 1]], dtype=np.uint32),
            50,
            save_steps=True,
            save_ys=True,
        )
    finally:
        jax.config.update("jax_enable_x64", previous_x64)

    assert dense["ts"].dtype == np.float64
    assert dense["ys"].dtype == np.float64


def test_save_steps_without_save_ys_does_not_return_ys():
    """reward_norm.py / eval.py call solve_pid_batch(save_steps=True) on a
    hot training-startup path and only ever read ts/accepted/rejected/
    t_reached — ys must stay opt-in (save_ys=True) so their existing calls
    don't pay for a host transfer of a value they never use."""
    cfg = ODEEnvConfig(system="scalar_decay", lam_min=1.0, lam_max=10.0)
    tasks = sample_eval_tasks(cfg, num_envs=4, seed=0, bins=[(1.0, 10.0)])
    mu, keys = tasks["task_params"], tasks["episode_keys"]

    dense = solve_pid_batch(cfg, _controller(cfg), mu, keys, 50, save_steps=True)

    assert "ys" not in dense


def test_save_ys_returns_trajectory_matching_final_y_from_a_final_state_only_solve():
    from steppo.training.trajectory_compare import final_state_from_trajectory

    cfg = ODEEnvConfig(system="scalar_decay", lam_min=1.0, lam_max=10.0)
    tasks = sample_eval_tasks(cfg, num_envs=4, seed=0, bins=[(1.0, 10.0)])
    mu, keys = tasks["task_params"], tasks["episode_keys"]
    max_steps = 50

    dense = solve_pid_batch(
        cfg, _controller(cfg), mu, keys, max_steps, save_steps=True, save_ys=True
    )
    final_only = solve_pid_batch(cfg, _controller(cfg), mu, keys, max_steps, save_steps=False)

    assert "ys" in dense
    assert dense["ys"].shape == dense["ts"].shape + (final_only["final_y"].shape[-1],)
    final_from_dense = final_state_from_trajectory(dense["ts"], dense["ys"])
    assert np.allclose(final_from_dense, final_only["final_y"], atol=1e-5)


def test_default_path_reports_completion_for_solves_that_reach_t_end():
    """The cheap (save_steps=False) path is what feeds the steps_vs_mu
    PID/policy series. A solve that stops early — dt_min reached, budget
    exhausted — still returns a step count, so without a completion flag
    an aborted solve is plotted as a cheap one (see the chua_smooth
    m>=1.2 collapse). Callers must be able to tell the two apart."""
    cfg = ODEEnvConfig(system="scalar_decay", lam_min=1.0, lam_max=10.0)
    tasks = sample_eval_tasks(cfg, num_envs=4, seed=0, bins=[(1.0, 10.0)])
    mu, keys = tasks["task_params"], tasks["episode_keys"]

    out = solve_pid_batch(cfg, _controller(cfg), mu, keys, 200)

    assert out["completed"].shape == out["steps"].shape
    assert out["completed"].all()
    assert np.allclose(out["t_reached"], cfg.t_end)


def test_default_path_marks_budget_exhausted_solves_incomplete():
    """A solve truncated by its step budget reports completed=False and a
    t_reached short of t_end, rather than a small step count that reads as
    a fast solve."""
    cfg = ODEEnvConfig(system="scalar_decay", lam_min=1.0, lam_max=10.0)
    tasks = sample_eval_tasks(cfg, num_envs=4, seed=0, bins=[(1.0, 10.0)])
    mu, keys = tasks["task_params"], tasks["episode_keys"]

    out = solve_pid_batch(cfg, _controller(cfg), mu, keys, 2)

    assert not out["completed"].any()
    assert np.all(out["t_reached"] < cfg.t_end)


def test_dense_path_reports_completion_consistently_with_default_path():
    """The PID eval dataset is built from the dense (save_steps=True) solve,
    so the completion flag has to exist on that path too — and agree with
    the cheap path on the same episodes, since the two are compared
    directly (pid_steps vs pid_steps_unlimited)."""
    cfg = ODEEnvConfig(system="scalar_decay", lam_min=1.0, lam_max=10.0)
    tasks = sample_eval_tasks(cfg, num_envs=4, seed=0, bins=[(1.0, 10.0)])
    mu, keys = tasks["task_params"], tasks["episode_keys"]

    for max_steps in (200, 2):
        dense = solve_pid_batch(cfg, _controller(cfg), mu, keys, max_steps, save_steps=True)
        cheap = solve_pid_batch(cfg, _controller(cfg), mu, keys, max_steps)

        assert dense["completed"].shape == cheap["completed"].shape
        assert np.array_equal(dense["completed"], cheap["completed"])


def _chua_config(**overrides):
    """chua_smooth at the Case-2 training IC — the system that exposed the
    coupling between controller and root-finder tolerances."""
    cfg = dict(
        system="chua_smooth",
        precision="float64",
        t_end=20.0,
        dt0=1e-1,
        rtol=1e-3,
        atol=1e-6,
        dt_min=1e-5,
        dt_max=2.0,
        chua_v1_0=0.15264,
        chua_v2_0=-0.02281,
        chua_i_0=0.38127,
        chua_eps=1e-3,
        m_min=0.1,
        m_max=2.0,
        sample_m=True,
    )
    cfg.update(overrides)
    return ODEEnvConfig(**cfg)


def test_root_finder_tolerance_is_separable_from_controller_tolerance():
    """Tightening rtol/atol on the config tightens *both* the step-size
    controller and Kvaerno5's VeryChord root finder (envs/ode_env.py's
    _make_solver). On chua_smooth the tightened root finder needs steps
    below dt_min, so the reference solve dies at t~0.02 for every m. A
    caller must be able to tighten the controller alone."""
    env = _chua_config()
    tight = dataclasses.replace(env, rtol=env.rtol * 1e-2, atol=env.atol * 1e-2)
    mu = np.array([0.5, 0.8, 1.0], dtype=np.float32)
    keys = jax.random.split(jax.random.PRNGKey(0), len(mu))

    coupled = solve_pid_batch(tight, _controller(tight), mu, keys, 2000)
    decoupled = solve_pid_batch(
        tight,
        _controller(tight),
        mu,
        keys,
        2000,
        solver_tols=(env.rtol, env.atol),
    )

    assert not coupled["completed"].any()
    assert decoupled["completed"].all()
    # A reference solve is only useful if it is finer than the PID solve it
    # grades: more steps, same endpoint.
    baseline = solve_pid_batch(env, _controller(env), mu, keys, 2000)
    assert np.all(decoupled["steps"] > baseline["steps"])


def test_reference_batch_grades_pid_where_pid_completes():
    """The tight-tolerance reference every error metric is measured against
    must actually reach t_end. Built by scaling the config's tolerances
    wholesale it did not: on chua_smooth it stopped at t~0.02 for every m,
    silently poisoning every error number derived from it."""
    from steppo.training.pid_solve import solve_reference_batch

    env = _chua_config()
    mu = np.array([0.5, 0.8, 1.0], dtype=np.float32)
    keys = jax.random.split(jax.random.PRNGKey(0), len(mu))

    pid = solve_pid_batch(env, _controller(env), mu, keys, 200)
    ref = solve_reference_batch(env, mu, keys, 2000, ref_tol_factor=1e-2, save_steps=True)

    assert pid["completed"].all()
    assert ref["completed"].all()
    assert np.all(ref["accepted"] + ref["rejected"] > pid["steps"])


def test_reference_batch_reports_incompletion_rather_than_a_junk_trajectory():
    """Where the reference itself cannot be solved under the configured
    dt_min (chua_smooth at m>=1.2), it must say so, so the error series
    can drop those episodes instead of grading against a trajectory that
    stopped at t~0.01."""
    from steppo.training.pid_solve import solve_reference_batch

    env = _chua_config()
    mu = np.array([1.5, 2.0], dtype=np.float32)
    keys = jax.random.split(jax.random.PRNGKey(0), len(mu))

    ref = solve_reference_batch(env, mu, keys, 2000, ref_tol_factor=1e-2, save_steps=True)

    assert not ref["completed"].any()
    assert np.all(ref["t_reached"] < env.t_end)
