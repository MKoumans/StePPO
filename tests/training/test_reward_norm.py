"""Regression tests for PID-derived task reward scaling."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import steppo.training.reward_norm as reward_norm


def test_pid_scale_grid_extrapolates_truncated_trajectories_and_centers_logs(monkeypatch):
    observed = {}

    def fake_baseline(_env_config, *, mus, num_repeats, max_steps, **kwargs):
        observed.update(mus=list(mus), num_repeats=num_repeats, max_steps=max_steps)
        return {
            float(mus[0]): {"steps": 10.0, "t_reached": 10.0},
            float(mus[1]): {"steps": 20.0, "t_reached": 5.0},
        }

    monkeypatch.setattr(reward_norm, "compute_pid_baseline_steps", fake_baseline)
    grid, log_steps, log_center = reward_norm.build_pid_scale_grid(
        SimpleNamespace(t_end=10.0),
        1.0,
        100.0,
        grid_points=2,
        num_repeats=3,
        max_steps=50,
    )

    assert observed == {"mus": [1.0, 100.0], "num_repeats": 3, "max_steps": 50}
    assert np.allclose(grid, np.log([1.0, 100.0]))
    # The second rollout stopped halfway through t_end: scale its observed 20
    # PID steps up to an estimated 40 before computing the geometric center.
    assert np.allclose(log_steps, np.log([10.0, 40.0]))
    assert float(log_center) == pytest.approx(np.log(20.0))


def test_reward_scale_interpolates_in_log_space():
    # A straight line in log(mu), log(steps) is a power law in the raw values.
    scales = reward_norm.reward_scale_for_mu(
        jnp.exp(jnp.array([0.0, 0.5, 1.0])),
        jnp.array([0.0, 1.0]),
        jnp.array([-1.0, 1.0]),
        jnp.array(0.0),
    )

    assert np.allclose(scales, np.exp([-1.0, 0.0, 1.0]), rtol=1e-6)


def test_reward_scale_clips_both_extremes_including_out_of_grid_tasks():
    scales = reward_norm.reward_scale_for_mu(
        jnp.exp(jnp.array([-1.0, 2.0])),
        jnp.array([0.0, 1.0]),
        jnp.array([-2.0, 2.0]),
        jnp.array(0.0),
    )

    assert np.allclose(scales, [reward_norm.SCALE_MIN, reward_norm.SCALE_MAX])


def test_warp_grid_extrapolates_truncated_step_times_to_t_end(monkeypatch):
    calls = []

    def fake_solve(_env_config, _controller, mu_values, _keys, max_steps, *, save_steps):
        calls.append((np.asarray(mu_values), max_steps, save_steps))
        return {
            "ts": np.array([[2.0, 4.0, 6.0, np.inf]], dtype=np.float32),
            "accepted": np.array([3]),
            "t_reached": np.array([6.0]),
        }

    monkeypatch.setattr("steppo.training.pid_solve.solve_pid_batch", fake_solve)
    from steppo.configs.base_config import ODEEnvConfig

    log_mu, t_knots, fractions = reward_norm.build_pid_warp_grid(
        ODEEnvConfig(system="scalar_decay", t_end=10.0),
        1.0,
        100.0,
        grid_points=2,
        num_repeats=1,
        max_steps=4,
        num_knots=4,
        use_cache=False,
    )

    assert len(calls) == 2
    assert all(max_steps == 4 and save_steps for _, max_steps, save_steps in calls)
    assert np.allclose(log_mu, np.log([1.0, 100.0]))
    assert np.allclose(fractions, [0.25, 0.5, 0.75, 1.0])
    # Three accepted steps reached t=6. The remaining two estimated steps
    # stretch linearly to t_end, yielding monotone quarter-effort times.
    assert np.allclose(t_knots, [[2.5, 5.0, 7.5, 10.0]] * 2)


def test_warp_grid_uses_linear_time_when_pid_dies_immediately(monkeypatch):
    monkeypatch.setattr(
        "steppo.training.pid_solve.solve_pid_batch",
        lambda *_args, **_kwargs: {
            "ts": np.array([[0.05, np.inf, np.inf]], dtype=np.float32),
            "accepted": np.array([1]),
            "t_reached": np.array([0.05]),
        },
    )
    from steppo.configs.base_config import ODEEnvConfig

    _, t_knots, fractions = reward_norm.build_pid_warp_grid(
        ODEEnvConfig(system="scalar_decay", t_end=10.0),
        1.0,
        1.0,
        grid_points=1,
        num_repeats=1,
        max_steps=3,
        num_knots=4,
        use_cache=False,
    )

    assert np.allclose(t_knots[0], fractions * 10.0)
    assert np.all(np.diff(t_knots[0]) >= 0)


def test_warp_grid_cache_is_written_then_reused(tmp_path, monkeypatch):
    from steppo.configs.base_config import ODEEnvConfig
    from steppo.training import pid_solve

    monkeypatch.setattr(pid_solve, "cache_dir", lambda *_args: str(tmp_path))
    solve_calls = []

    def fake_solve(*_args, **_kwargs):
        solve_calls.append(1)
        return {
            "ts": np.array([[1.0, 2.0, np.inf]], dtype=np.float32),
            "accepted": np.array([2]),
            "t_reached": np.array([2.0]),
        }

    monkeypatch.setattr(pid_solve, "solve_pid_batch", fake_solve)
    config = ODEEnvConfig(system="scalar_decay", t_end=2.0)
    kwargs = dict(
        grid_points=1,
        num_repeats=1,
        max_steps=3,
        num_knots=2,
        use_cache=True,
    )

    first = reward_norm.build_pid_warp_grid(config, 1.0, 1.0, **kwargs)
    assert len(solve_calls) == 1
    assert list(tmp_path.glob("warp_grid_*.npz"))

    monkeypatch.setattr(
        pid_solve,
        "solve_pid_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    second = reward_norm.build_pid_warp_grid(config, 1.0, 1.0, **kwargs)

    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(first, second))
