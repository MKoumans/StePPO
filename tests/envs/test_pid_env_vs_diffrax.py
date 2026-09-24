"""Parity checks between the repo's closed-form PID replica
(`pid_controller.pid_action_from_obs`, applied via `ODEEnv.step`'s
`dt_scaled = dt * exp(action * dt_log_gain)`) and diffrax's native
`PIDController`, which it's meant to mirror.

The scalar-decay parity case runs in the regular suite. Set
`STEPPO_RUN_SLOW_SOLVER_TESTS=1` to run the full multi-system solver sweep.

Three bugs found and fixed while writing these tests:
1. `pid_action_from_obs` reused dt_log_gain (the RL action bound) as the PID
   controller's own growth cap, making pid_min_factor/pid_max_factor dead
   config. dt_log_gain is now purely a unit conversion; policy-facing
   callers clip to [-1, 1] themselves. See `TestGrowthBoundedByPidMaxFactor`.
2. Unlike diffrax, an accepted step could still shrink dt near tolerance.
   Fixed by making accept/reject clamping symmetric (accept floors the
   growth factor at 1, mirroring diffrax's own floor). See
   `TestAcceptedStepNeverShrinks`.
3. `pid_action_from_obs` read keep_step from the wrong obs index
   (offset+3 = normalized time, not offset+2 = last_keep_step), which could
   spuriously freeze dt near 1x indefinitely. See
   `TestKeepStepReadFromCorrectObsIndex`.

With all three fixed, `TestEndToEndStepCountDivergence` shows the env-driven
replica within ~10% of diffrax's step count on scalar_decay (was ~55x
before). The small residual gap is pid_max_factor (5.0) vs diffrax's
factormax (10.0) — an intentional, independent config choice, not a bug.
"""

import os

import diffrax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

from steppo.configs.base_config import ODEEnvConfig
from steppo.envs.ode import (
    ODEEnv,  # noqa: F401 — import before ode_env submodules, see ode_env.py's circular import
)
from steppo.envs.ode.systems.obs_factory import base_obs_factory
from steppo.training.error_dist import sample_eval_tasks, solve_pid_baseline_batch
from steppo.training.pid_controller import pid_action_from_obs
from steppo.training.pid_solve import solve_pid_batch
from steppo.utils.task_params import task_bounds


def _full_range_bins(cfg):
    """[(lo, hi)] over cfg's own task bounds, for tests needing *some*
    representative task sample rather than a specific split."""
    return [task_bounds(cfg)]


RTOL, ATOL, SAFETY = 1e-3, 1e-6, 0.9
PID_ORDER = 4  # WarmstartConfig.pid_order default
ERROR_ORDER = PID_ORDER + 1  # matches Kvaerno5's diffrax error_order (verified separately)
MIN_FACTOR, MAX_FACTOR = 0.2, 5.0  # WarmstartConfig.pid_{min,max}_factor defaults
DT_LOG_GAIN = 0.693147181  # ODEEnvConfig.dt_log_gain default (ln 2)
RUN_SLOW_SOLVER_TESTS = os.environ.get("STEPPO_RUN_SLOW_SOLVER_TESTS") == "1"


def _diffrax_factor(scaled_error):
    """(keep_step, dt_next/dt_prev) diffrax.PIDController.adapt_step_size
    applies for one step at the given scalar RMS scaled error."""
    y0 = jnp.array(1.0)
    scale = ATOL + RTOL * 1.0
    y_err = jnp.array(float(scaled_error) * scale)
    controller = diffrax.PIDController(rtol=RTOL, atol=ATOL, safety=SAFETY, error_order=ERROR_ORDER)
    t0, t1 = jnp.float32(0.0), jnp.float32(1e-3)
    _, state0 = controller.init(
        terms=None,
        t0=t0,
        t1=t1,
        y0=y0,
        dt0=(t1 - t0),
        args=None,
        func=None,
        error_order=ERROR_ORDER,
    )
    keep_step, next_t0, next_t1, _, _, _ = controller.adapt_step_size(
        t0,
        t1,
        y0,
        y0,
        args=None,
        y_error=y_err,
        error_order=ERROR_ORDER,
        controller_state=state0,
    )
    return bool(keep_step), float((next_t1 - next_t0) / (t1 - t0))


def _repo_factor(
    scaled_error, keep_step, min_factor=MIN_FACTOR, max_factor=MAX_FACTOR, dt_log_gain=DT_LOG_GAIN
):
    """dt_next/dt_prev from pid_action_from_obs's action via ODEEnv.step()'s
    formula, unclipped -- matching how reference-PID callers use it."""
    obs = jnp.zeros(3)
    obs = obs.at[0].set(jnp.log(jnp.asarray(scaled_error, jnp.float32)) / 5.0)
    obs = obs.at[2].set(jnp.float32(keep_step))
    action = pid_action_from_obs(
        obs,
        step_context_offset=0,
        q=PID_ORDER,
        safety=SAFETY,
        min_factor=min_factor,
        max_factor=max_factor,
        dt_log_gain=dt_log_gain,
    )
    return float(jnp.exp(action[0] * dt_log_gain))


class TestErrorNormMatchesDiffrax:
    """Validates the scaled-RMS-error formula against diffrax's own norm."""

    def test_rms_scaled_error_formula_matches_diffrax_rms_norm(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            y0 = jnp.asarray(rng.normal(size=4), dtype=jnp.float32)
            y1 = jnp.asarray(rng.normal(size=4), dtype=jnp.float32)
            y_err = jnp.asarray(rng.normal(size=4) * 1e-4, dtype=jnp.float32)
            scale = ATOL + jnp.maximum(jnp.abs(y0), jnp.abs(y1)) * RTOL

            repo_err = float(jnp.sqrt(jnp.mean((y_err / scale) ** 2)))
            diffrax_err = float(optx.rms_norm(y_err / scale))

            assert repo_err == pytest.approx(diffrax_err, rel=1e-6)


class TestGrowthBoundedByPidMaxFactor:
    """Regression guard for the dt_log_gain double-clip fix: the growth
    factor must be governed by pid_max_factor, never saturate at
    exp(dt_log_gain), and must agree with diffrax whenever neither side's
    cap is binding."""

    def test_moderate_error_matches_diffrax_exactly(self):
        """Below both controllers' growth caps, the formulas are identical
        and must agree."""
        _, diffrax_factor = _diffrax_factor(1e-3)
        repo_factor = _repo_factor(1e-3, keep_step=True)
        assert diffrax_factor == pytest.approx(3.583, rel=1e-2)
        assert repo_factor == pytest.approx(diffrax_factor, rel=1e-3)

    def test_tiny_error_saturates_at_pid_max_factor_not_dt_log_gain(self):
        """diffrax's factormax=10.0; this repo's pid_max_factor=5.0 is a
        deliberately more conservative, independent choice -- but growth
        must saturate AT pid_max_factor, not exp(dt_log_gain)=2.0."""
        diffrax_keep, diffrax_factor = _diffrax_factor(1e-6)
        repo_factor = _repo_factor(1e-6, keep_step=True)

        assert diffrax_keep is True
        assert diffrax_factor == pytest.approx(10.0, rel=1e-3)  # diffrax's factormax
        assert repo_factor == pytest.approx(MAX_FACTOR, rel=1e-3)  # pid_max_factor, not 2.0
        assert repo_factor != pytest.approx(float(jnp.exp(DT_LOG_GAIN)), rel=1e-2)


class TestAcceptedStepNeverShrinks:
    """diffrax's PIDController never shrinks dt after an accepted step;
    pid_action_from_obs applies the same floor."""

    @pytest.mark.parametrize("scaled_error", [0.5, 0.9, 0.99, 0.999])
    def test_accepted_step_matches_diffrax_near_threshold(self, scaled_error):
        diffrax_keep, diffrax_factor = _diffrax_factor(scaled_error)
        repo_factor = _repo_factor(scaled_error, keep_step=True)

        assert diffrax_keep is True
        assert diffrax_factor >= 1.0  # diffrax: never shrinks on accept
        assert repo_factor >= 1.0  # repo: matches now
        assert repo_factor == pytest.approx(diffrax_factor, rel=1e-3)

    def test_rejected_step_still_shrinks_or_holds(self):
        """Mirror-image case: a rejected step's factor is still capped at 1."""
        repo_factor = _repo_factor(2.0, keep_step=False)
        assert repo_factor <= 1.0


class TestKeepStepReadFromCorrectObsIndex:
    """pid_action_from_obs reads last_keep_step at offset+2."""

    def test_step_context_layout_puts_keep_step_at_index_2(self):
        """Ground truth: obs_factory._step_context's actual field order."""
        obs_fn, feature_dims = base_obs_factory(
            state_fn=lambda state, config: jnp.zeros(1),
            state_dim=1,
            y_dim=1,
        )
        assert feature_dims["step_context"] == 4

        class _Cfg:
            obs_features = ("step_context",)
            t_end = 1.0
            dt0 = 1.0

        class _State:
            t = jnp.float32(0.0)
            dt = jnp.float32(1.0)
            last_log_error = jnp.float32(0.0)
            log_error_delta = jnp.float32(0.42)  # distinctive, nonzero
            last_keep_step = jnp.bool_(True)
            solver_state_f = jnp.zeros(1)

        obs = obs_fn(_State(), _Cfg())
        # index 1 = log_error_delta/5, index 2 = last_keep_step
        assert obs[1] == pytest.approx(0.42 / 5.0, rel=1e-5)
        assert obs[2] == pytest.approx(1.0, rel=1e-5)

    def test_normalized_time_is_not_mistaken_for_keep_step(self):
        """A large time feature must not be mistaken for a rejection flag."""
        obs = jnp.zeros(4)
        obs = obs.at[0].set(jnp.log(1e-6) / 5.0)
        obs = obs.at[3].set(0.6)
        obs = obs.at[2].set(1.0)
        action = pid_action_from_obs(
            obs,
            step_context_offset=0,
            q=PID_ORDER,
            safety=SAFETY,
            min_factor=MIN_FACTOR,
            max_factor=MAX_FACTOR,
            dt_log_gain=DT_LOG_GAIN,
        )
        factor = float(jnp.exp(action[0] * DT_LOG_GAIN))
        assert factor > 1.0, (
            f"factor={factor}: a tiny error with last_keep_step=True should grow dt"
        )

    def test_log_error_delta_is_not_mistaken_for_keep_step(self):
        """A low error delta must not override the true acceptance flag."""
        obs = jnp.zeros(4)
        obs = obs.at[0].set(jnp.log(1e-6) / 5.0)
        obs = obs.at[1].set(0.02)
        obs = obs.at[2].set(1.0)
        action = pid_action_from_obs(
            obs,
            step_context_offset=0,
            q=PID_ORDER,
            safety=SAFETY,
            min_factor=MIN_FACTOR,
            max_factor=MAX_FACTOR,
            dt_log_gain=DT_LOG_GAIN,
        )
        factor = float(jnp.exp(action[0] * DT_LOG_GAIN))
        assert factor == pytest.approx(MAX_FACTOR, rel=1e-3)


class TestEndToEndStepCountDivergence:
    """Solves the same task batch with diffrax's native PIDController and
    with the real ODEEnv driven by pid_action_from_obs, both used elsewhere
    as "the PID baseline". The systems are parametrized because stiff and
    non-stiff trajectories can expose different integration regressions.

    Parametrized across every registered system at its own production
    config: the per-step factor formula is system-agnostic, but the
    trajectory of scaled errors it produces is not, so a single system
    can't catch a per-system discrepancy (e.g. a stiff system spending more
    time in Kvaerno5's rejection path). dt_log_gain=2.5 at every config --
    the value cause #1 (the dt_log_gain double-clip bug) was silently
    breaking in production.
    """

    _SYSTEM_CONFIGS = {
        # (env kwargs, num_envs, max_steps) -- max_steps has headroom over
        # the diffrax-side step count observed at num_envs=4, seed=0.
        "scalar_decay": (
            dict(
                system="scalar_decay",
                t_end=1.0,
                dt0=0.1,
                rtol=1e-5,
                atol=1e-8,
                dt_min=1e-8,
                dt_max=1.0,
                dt_log_gain=2.5,
                immediate_dt_action=True,
                lam_min=1.0,
                lam_max=100.0,
                sample_lam=True,
            ),
            4,
            2000,
        ),
        "van_der_pol": (
            dict(
                system="van_der_pol",
                t_end=50.0,
                dt0=1e-4,
                rtol=1e-3,
                atol=1e-6,
                dt_min=1e-4,
                dt_max=5.0,
                dt_log_gain=2.5,
                immediate_dt_action=True,
                mu_min=5.0,
                mu_max=100.0,
                sample_mu=True,
            ),
            4,
            4000,
        ),
        "brusselator": (
            dict(
                system="brusselator",
                t_end=100.0,
                dt0=1e-4,
                rtol=1e-3,
                atol=1e-6,
                dt_min=1e-4,
                dt_max=5.0,
                dt_log_gain=2.5,
                immediate_dt_action=True,
                B_min=2.0,
                B_max=50.0,
                sample_B=True,
            ),
            4,
            6000,
        ),
        "fitzhugh_nagumo": (
            dict(
                system="fitzhugh_nagumo",
                t_end=100.0,
                dt0=1e-4,
                rtol=1e-3,
                atol=1e-6,
                dt_min=1e-4,
                dt_max=5.0,
                dt_log_gain=2.5,
                immediate_dt_action=True,
                eps_min=0.005,
                eps_max=1.0,
                sample_eps=True,
            ),
            4,
            2000,
        ),
        "chemical_cascade": (
            dict(
                system="chemical_cascade",
                t_end=50.0,
                dt0=1e-4,
                rtol=1e-4,
                atol=1e-6,
                dt_min=1e-6,
                dt_max=1e3,
                dt_log_gain=2.5,
                immediate_dt_action=True,
                cc_lam_min=0.01,
                cc_lam_max=10.0,
                sample_cc_lam=True,
            ),
            4,
            1000,
        ),
        "robertson": (
            dict(
                system="robertson",
                t_end=100000.0,
                dt0=1e-2,
                rtol=1e-3,
                atol=1e-6,
                dt_min=1e-10,
                dt_max=1000.0,
                dt_log_gain=2.5,
                immediate_dt_action=True,
                k2_min=1e5,
                k2_max=1e9,
                sample_k2=True,
            ),
            4,
            2000,
        ),
    }

    @pytest.mark.parametrize(
        "system",
        [
            "scalar_decay",
            *[
                pytest.param(
                    system,
                    marks=pytest.mark.skipif(
                        not RUN_SLOW_SOLVER_TESTS,
                        reason="set STEPPO_RUN_SLOW_SOLVER_TESTS=1 for the full solver matrix",
                    ),
                    id=system,
                )
                for system in _SYSTEM_CONFIGS
                if system != "scalar_decay"
            ],
        ],
    )
    def test_env_driven_pid_is_close_to_diffrax(self, system):
        env_kwargs, num_envs, max_steps = self._SYSTEM_CONFIGS[system]
        cfg = ODEEnvConfig(**env_kwargs)
        tasks = sample_eval_tasks(cfg, num_envs=num_envs, seed=0, bins=_full_range_bins(cfg))
        mu, keys = tasks["task_params"], tasks["episode_keys"]

        sc = diffrax.PIDController(
            rtol=cfg.rtol, atol=cfg.atol, dtmin=cfg.dt_min, dtmax=cfg.dt_max, force_dtmin=False
        )
        diffrax_steps = solve_pid_batch(cfg, sc, mu, keys, max_steps)["steps"]
        env_steps = solve_pid_baseline_batch(cfg, mu, keys, max_steps)["steps"]

        diffrax_mean, env_mean = diffrax_steps.mean(), env_steps.mean()

        # A loose 2x bound regression-guards the fixed causes (pre-fix
        # ratios were 55x-100x on scalar_decay) without being brittle to the
        # residual pid_max_factor/factormax gap or per-system noise.
        assert env_mean < 2 * diffrax_mean, (
            f"{system}: diffrax_mean={diffrax_mean}, env_mean={env_mean}"
        )
        assert np.all(env_steps < max_steps), f"{system}: an episode exhausted the step budget"
        assert np.all(diffrax_steps < max_steps), (
            f"{system}: diffrax itself exhausted max_steps={max_steps} -- "
            "raise this system's budget in _SYSTEM_CONFIGS, unrelated to the "
            "env-driven replica's correctness"
        )


@pytest.mark.skipif(
    not RUN_SLOW_SOLVER_TESTS,
    reason="set STEPPO_RUN_SLOW_SOLVER_TESTS=1 for the difficult Brusselator regression",
)
class TestNoFailureSignalAtDtMin:
    """Regression guard that a difficult Brusselator task makes progress."""

    def _stuck_task(self):
        cfg = ODEEnvConfig(
            system="brusselator",
            t_end=100.0,
            dt0=1e-4,
            rtol=1e-3,
            atol=1e-6,
            dt_min=1e-4,
            dt_max=5.0,
            dt_log_gain=2.5,
            immediate_dt_action=True,
            B_min=2.0,
            B_max=50.0,
            sample_B=True,
        )
        # Same sampler and seed as TestEndToEndStepCountDivergence's
        # Brusselator case, focused on its first sampled task.
        tasks = sample_eval_tasks(cfg, num_envs=4, seed=0, bins=_full_range_bins(cfg))
        mu = tasks["task_params"][0:1]
        keys = tasks["episode_keys"][0:1]
        return cfg, mu, keys

    def test_env_driven_pid_does_not_silently_burn_the_full_budget(self):
        """A difficult task should finish without consuming the full step budget."""
        cfg, mu, keys = self._stuck_task()
        steps = solve_pid_baseline_batch(cfg, mu, keys, max_steps=20000)["steps"]
        assert steps[0] < 1000, (
            f"steps={steps[0]}: task deadlocked at dt_min and silently burned "
            "the step budget instead of failing fast like diffrax does"
        )
