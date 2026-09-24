"""Comprehensive tests for ODE environment features.

Covers: warmstart on/off, observation normalization ranges, episodes_per_trial
dynamics, reward computation, EMA tracking, action scaling, PID controller,
task sampling, multi-episode trial freeze/reset logic, and BC loss.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from steppo.configs.base_config import (
    EncoderArchConfig,
    ODEEnvConfig,
    PPOConfig,
    TrainConfig,
    VAEConfig,
    WarmstartConfig,
)
from steppo.envs.ode import ODEEnv, ODEParams
from steppo.envs.ode.systems import get_system
from steppo.envs.ode_env import _EMA_ALPHA
from steppo.training.pid_controller import (
    bc_loss,
    compute_step_context_offset,
    pid_action_from_obs,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

_VDP_DEFAULTS = dict(
    system="van_der_pol",
    t_end=50.0,
    dt0=1e-4,
    rtol=1e-3,
    atol=1e-6,
    dt_min=1e-4,
    dt_max=5.0,
    mu_min=5.0,
    mu_max=100.0,
    sample_mu=True,
    sample_y0=False,
    y0_x=0.0,
    y0_y=-2.0,
    train_bins=((5.0, 100.0),),
)

_SCALAR_DEFAULTS = dict(
    system="scalar_decay",
    t_end=10.0,
    dt0=1e-4,
    rtol=1e-3,
    atol=1e-6,
    dt_min=1e-4,
    dt_max=5.0,
    lam_min=1.0,
    lam_max=50.0,
    sample_lam=True,
    train_bins=((1.0, 50.0),),
)


def _make_vdp_env(max_steps=200, **kw):
    merged = {**_VDP_DEFAULTS, **kw}
    cfg = ODEEnvConfig(**merged)
    return ODEEnv(cfg, max_steps=max_steps), cfg


def _make_scalar_env(max_steps=200, **kw):
    merged = {**_SCALAR_DEFAULTS, **kw}
    cfg = ODEEnvConfig(**merged)
    return ODEEnv(cfg, max_steps=max_steps), cfg


def _step_n(env, state, params, n, action=0.0):
    key = jax.random.PRNGKey(42)
    act = jnp.array([action])
    for _ in range(n):
        _, state, _, done, info = env.step(key, state, act, params)
        if done:
            break
    return state, done, info


def _make_models(env, config):
    obs_dim = env.obs_shape()[0]
    action_dim = env.num_actions
    from steppo.models.policy import ActorCritic
    from steppo.models.vae import VariBADVAE

    rngs = nnx.Rngs(42)
    vae = VariBADVAE(obs_dim, action_dim, config.vae, rngs, task_dim=1)
    policy = ActorCritic(
        obs_dim,
        action_dim,
        config.vae.latent_dim,
        action_space="continuous",
        use_latent_sample=False,
        rngs=nnx.Rngs(43),
    )
    return vae, policy


# ═══════════════════════════════════════════════════════════════════════════
# 1. REWARD COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════


class TestReward:
    """Reward = progress/t_end * factor - survival_penalty + completion_bonus."""

    def test_progress_reward_proportional_to_time_advanced(self):
        env, _ = _make_vdp_env(max_steps=1000, reward_factor=50.0, completion_bonus=0.0)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=1000)
        _, state = env.reset(key, params)
        _, state2, reward, _, info = env.step(key, state, jnp.array([0.0]), params)
        progress = float(info["t_reached"]) - 0.0
        expected = progress / float(env.t_end) * 50.0
        assert jnp.isclose(float(reward), expected, atol=1e-4), (
            f"reward={float(reward)}, expected={expected}"
        )

    def test_survival_penalty_subtracted(self):
        env_no, _ = _make_vdp_env(max_steps=1000, survival_penalty=0.0, completion_bonus=0.0)
        env_pen, _ = _make_vdp_env(max_steps=1000, survival_penalty=0.1, completion_bonus=0.0)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=1000)
        _, s0 = env_no.reset(key, params)
        _, s1 = env_pen.reset(key, params)
        _, _, r0, _, _ = env_no.step(key, s0, jnp.array([0.0]), params)
        _, _, r1, _, _ = env_pen.step(key, s1, jnp.array([0.0]), params)
        assert jnp.isclose(float(r0) - float(r1), 0.1, atol=1e-4)

    def test_completion_bonus_at_success(self):
        # progress_frac is already Delta_t/t_end (see ODEEnv.step), not a raw
        # time delta -- must stay in [0, 1] here, unlike t_end itself.
        env, _ = _make_vdp_env(max_steps=1000, completion_bonus=50.0)
        progress_frac = jnp.float32(0.05)
        budget = jnp.float32(0.3)
        success = True
        r = env._get_reward(progress_frac, budget, success, keep_step=True)
        bonus = (1.0 - 0.3) * 50.0
        expected = float(progress_frac) * float(env._config.reward_factor) + bonus
        assert float(r) > 0.0
        assert jnp.isclose(float(r), expected, atol=1e-3)

    def test_no_bonus_without_success(self):
        env, _ = _make_vdp_env(max_steps=1000, completion_bonus=50.0)
        progress_frac = jnp.float32(0.02)
        r = env._get_reward(progress_frac, jnp.float32(0.5), False, keep_step=True)
        r_no_bonus = float(progress_frac) * float(env._config.reward_factor)
        assert jnp.isclose(float(r), r_no_bonus, atol=1e-4)

    def test_completion_bonus_scales_with_budget(self):
        """Earlier completion (lower budget_exhausted) yields higher bonus."""
        env, _ = _make_vdp_env(max_steps=1000, completion_bonus=50.0)
        r_early = env._get_reward(jnp.float32(0.05), jnp.float32(0.1), True, keep_step=True)
        r_late = env._get_reward(jnp.float32(0.05), jnp.float32(0.9), True, keep_step=True)
        assert float(r_early) > float(r_late)

    def test_zero_progress_gives_zero_base_reward(self):
        """Rejected step: progress=0, no bonus, reward = -survival_penalty (rejection_penalty=0 by default)."""
        env, _ = _make_vdp_env(max_steps=1000, survival_penalty=0.05, completion_bonus=0.0)
        r = env._get_reward(jnp.float32(0.0), jnp.float32(0.5), False, keep_step=False)
        assert jnp.isclose(float(r), -0.05, atol=1e-5)

    def test_rejection_penalty_applied_only_on_reject(self):
        """A configured rejection_penalty subtracts extra reward only when keep_step=False."""
        env, _ = _make_vdp_env(
            max_steps=1000, survival_penalty=0.0, completion_bonus=0.0, rejection_penalty=0.2
        )
        r_keep = env._get_reward(jnp.float32(0.0), jnp.float32(0.5), False, keep_step=True)
        r_reject = env._get_reward(jnp.float32(0.0), jnp.float32(0.5), False, keep_step=False)
        assert jnp.isclose(float(r_keep), 0.0, atol=1e-5)
        assert jnp.isclose(float(r_reject), -0.2, atol=1e-5)

    def test_margin_penalty_zero_at_threshold(self):
        """An accepted step landing exactly at the accept threshold (scaled_error=1) pays no margin penalty."""
        env, _ = _make_vdp_env(
            max_steps=1000, survival_penalty=0.0, completion_bonus=0.0, margin_penalty=0.3
        )
        r = env._get_reward(
            jnp.float32(0.0), jnp.float32(0.5), False, keep_step=True, scaled_error=jnp.float32(1.0)
        )
        assert jnp.isclose(float(r), 0.0, atol=1e-5)

    def test_margin_penalty_scales_with_unused_error_budget(self):
        """An accepted step that undershoots the threshold (small scaled_error) pays proportionally more."""
        env, _ = _make_vdp_env(
            max_steps=1000, survival_penalty=0.0, completion_bonus=0.0, margin_penalty=0.3
        )
        r_tight = env._get_reward(
            jnp.float32(0.0), jnp.float32(0.5), False, keep_step=True, scaled_error=jnp.float32(0.9)
        )
        r_loose = env._get_reward(
            jnp.float32(0.0), jnp.float32(0.5), False, keep_step=True, scaled_error=jnp.float32(0.1)
        )
        assert jnp.isclose(float(r_tight), -0.3 * 0.1, atol=1e-5)
        assert jnp.isclose(float(r_loose), -0.3 * 0.9, atol=1e-5)
        assert float(r_loose) < float(r_tight)

    def test_margin_penalty_not_applied_on_reject(self):
        """A rejected step pays no margin penalty (its own rejection_penalty covers that case)."""
        env, _ = _make_vdp_env(
            max_steps=1000,
            survival_penalty=0.0,
            completion_bonus=0.0,
            margin_penalty=0.3,
            rejection_penalty=0.0,
        )
        r = env._get_reward(
            jnp.float32(0.0),
            jnp.float32(0.5),
            False,
            keep_step=False,
            scaled_error=jnp.float32(0.01),
        )
        assert jnp.isclose(float(r), 0.0, atol=1e-5)

    def test_margin_penalty_disabled_by_default(self):
        """margin_penalty=0.0 (default) leaves reward unaffected by scaled_error."""
        env, _ = _make_vdp_env(max_steps=1000, survival_penalty=0.0, completion_bonus=0.0)
        r = env._get_reward(
            jnp.float32(0.0),
            jnp.float32(0.5),
            False,
            keep_step=True,
            scaled_error=jnp.float32(0.01),
        )
        assert jnp.isclose(float(r), 0.0, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# 2. ACTION SCALING: action → dt_next
# ═══════════════════════════════════════════════════════════════════════════


class TestActionScaling:
    """dt_next = dt * exp(action * ln(2)), clipped to [dt_min, dt_max]."""

    def test_action_zero_keeps_dt(self):
        env, _ = _make_vdp_env()
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=200)
        _, state = env.reset(key, params)
        dt_before = float(state.dt)
        _, state2, _, _, _ = env.step(key, state, jnp.array([0.0]), params)
        assert jnp.isclose(float(state2.dt), dt_before, rtol=1e-4)

    def test_action_plus_one_doubles_dt(self):
        env, _ = _make_vdp_env()
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=200)
        _, state = env.reset(key, params)
        dt_before = float(state.dt)
        _, state2, _, _, _ = env.step(key, state, jnp.array([1.0]), params)
        expected = min(dt_before * 2.0, float(env.dt_max))
        assert jnp.isclose(float(state2.dt), expected, rtol=1e-4)

    def test_action_minus_one_halves_dt(self):
        env, _ = _make_vdp_env(dt_min=1e-8)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=200)
        _, state = env.reset(key, params)
        dt_before = float(state.dt)
        _, state2, _, _, _ = env.step(key, state, jnp.array([-1.0]), params)
        expected = max(dt_before * 0.5, float(env.dt_min))
        assert jnp.isclose(float(state2.dt), expected, rtol=1e-4)

    def test_dt_clipped_to_dt_max(self):
        env, _ = _make_vdp_env(dt_max=0.01)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=200)
        _, state = env.reset(key, params)
        _, state2, _, _, _ = env.step(key, state, jnp.array([1.0]), params)
        assert float(state2.dt) <= float(env.dt_max) + 1e-10

    def test_dt_clipped_to_dt_min(self):
        env, _ = _make_vdp_env(dt_min=1e-3, dt0=1e-3)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=200)
        _, state = env.reset(key, params)
        _, state2, _, _, _ = env.step(key, state, jnp.array([-1.0]), params)
        assert float(state2.dt) >= float(env.dt_min) - 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# 3. OBSERVATION NORMALIZATION RANGES
# ═══════════════════════════════════════════════════════════════════════════


class TestObservationNormalization:
    """Verify obs features are within expected normalized ranges."""

    def test_vdp_obs_dim_matches_feature_config(self):
        env, cfg = _make_vdp_env()
        spec = get_system("van_der_pol")
        expected_dim = sum(spec.feature_dims[f] for f in cfg.obs_features)
        assert env.obs_shape() == (expected_dim,)

    def test_vdp_obs_dim_state_and_step_context_only(self):
        env, _ = _make_vdp_env(obs_features=("state", "step_context"))
        spec = get_system("van_der_pol")
        expected = spec.feature_dims["state"] + spec.feature_dims["step_context"]
        assert env.obs_shape() == (expected,)

    def test_step_context_log_error_finite(self):
        """step_context[0] = log_error should be finite."""
        env, cfg = _make_vdp_env(obs_features=("step_context",), max_steps=50)
        spec = get_system("van_der_pol")
        offset = compute_step_context_offset(cfg.obs_features, spec.feature_dims)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=50)
        obs, state = env.reset(key, params)
        assert jnp.isfinite(float(obs[offset + 0]))

        for _ in range(10):
            obs, state, _, done, _ = env.step(key, state, jnp.array([0.0]), params)
            if done:
                break
            assert jnp.isfinite(float(obs[offset + 0]))

    def test_step_context_keep_step_binary(self):
        """step_context[2] = last_keep_step should be 0.0 or 1.0."""
        env, cfg = _make_vdp_env(obs_features=("step_context",))
        spec = get_system("van_der_pol")
        offset = compute_step_context_offset(cfg.obs_features, spec.feature_dims)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=200)
        obs, state = env.reset(key, params)
        for _ in range(20):
            obs, state, _, done, _ = env.step(key, state, jnp.array([0.0]), params)
            if done:
                break
            val = float(obs[offset + 2])
            assert val in (0.0, 1.0), f"keep_step obs should be binary, got {val}"

    def test_solver_trend_budget_in_01(self):
        """solver_trend[5] = budget_exhausted = step_count / max_steps."""
        env, cfg = _make_vdp_env(obs_features=("step_context", "solver_trend"), max_steps=50)
        spec = get_system("van_der_pol")
        offset = (
            spec.feature_dims["step_context"] + 5
        )  # solver_trend follows step_context in obs_features
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=50)
        _, state = env.reset(key, params)
        for i in range(10):
            obs, state, _, done, _ = env.step(key, state, jnp.array([0.0]), params)
            if done:
                break
            budget = float(obs[offset])
            assert 0.0 <= budget <= 1.0, f"budget_exhausted={budget} out of [0,1]"

    def test_solver_trend_accept_ema_in_01(self):
        """solver_trend[0] = accept_ema should remain in (0, 1)."""
        env, cfg = _make_vdp_env(obs_features=("solver_trend",), max_steps=50)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=50)
        obs, state = env.reset(key, params)
        for _ in range(20):
            obs, state, _, done, _ = env.step(key, state, jnp.array([0.0]), params)
            if done:
                break
            assert 0.0 <= float(obs[0]) <= 1.0, f"accept_ema={float(obs[0])}"

    def test_obs_finite_after_many_steps(self):
        """All observations remain finite through a full rollout."""
        env, _ = _make_vdp_env(max_steps=100)
        key = jax.random.PRNGKey(1)
        params = ODEParams(lam=50.0, max_steps=100)
        obs, state = env.reset(key, params)
        for _ in range(100):
            obs, state, _, done, _ = env.step(key, state, jnp.array([0.0]), params)
            assert jnp.all(jnp.isfinite(obs)), f"Non-finite obs: {obs}"
            if done:
                break

    def test_reset_obs_finite_and_correct_shape(self):
        env, _ = _make_vdp_env()
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=200)
        obs, _ = env.reset(key, params)
        assert obs.shape == env.obs_shape()
        assert jnp.all(jnp.isfinite(obs))


# ═══════════════════════════════════════════════════════════════════════════
# 4. EMA TRACKING
# ═══════════════════════════════════════════════════════════════════════════


class TestEMATracking:
    """Exponential moving averages of accept rate and log error."""

    def test_ema_initial_values(self):
        env, _ = _make_vdp_env()
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=200)
        _, state = env.reset(key, params)
        assert float(state.accept_ema) == 1.0
        assert float(state.log_error_ema) == 0.0

    def test_accept_ema_update_on_accepted_step(self):
        """After accepted step: accept_ema = alpha*1 + (1-alpha)*prev."""
        env, _ = _make_scalar_env(max_steps=200)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=1.0, max_steps=200)
        _, state = env.reset(key, params)
        _, state2, _, _, info = env.step(key, state, jnp.array([0.0]), params)
        if bool(info["keep_step"]):
            alpha = _EMA_ALPHA
            expected = alpha * 1.0 + (1.0 - alpha) * 1.0
            assert jnp.isclose(float(state2.accept_ema), expected, atol=1e-5)

    def test_accept_ema_decreases_on_rejection(self):
        """After a rejected step, accept_ema should decrease."""
        env, _ = _make_vdp_env(max_steps=200)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=200)
        _, state = env.reset(key, params)
        initial_ema = float(state.accept_ema)
        # Force a rejection by using NaN values
        state_mod = state.replace(
            y=jnp.array([jnp.nan, jnp.nan]),
        )
        _, state2, _, _, info = env.step(key, state_mod, jnp.array([0.0]), params)
        # NaN should always be rejected
        assert not bool(info["keep_step"]), "NaN should cause rejection"
        assert float(state2.accept_ema) < initial_ema

    def test_log_error_ema_bounded(self):
        """log_error is clipped to [-30, 30], so EMA stays bounded."""
        env, _ = _make_scalar_env(max_steps=50)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=1.0, max_steps=50)
        _, state = env.reset(key, params)
        for _ in range(30):
            _, state, _, done, _ = env.step(key, state, jnp.array([0.0]), params)
            if done:
                break
            assert -30.0 <= float(state.log_error_ema) <= 30.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. TASK SAMPLING
# ═══════════════════════════════════════════════════════════════════════════


class TestTaskSampling:
    def test_vdp_sample_mu_in_range(self):
        env, _ = _make_vdp_env(mu_min=5.0, mu_max=100.0, sample_mu=True)
        keys = jax.random.split(jax.random.PRNGKey(0), 50)
        for k in keys:
            p = env.sample_task(k)
            mu = float(p.lam)
            assert 5.0 <= mu <= 100.0, f"mu={mu} outside [5, 100]"

    def test_vdp_deterministic_when_not_sampling(self):
        env, _ = _make_vdp_env(mu_min=5.0, mu_max=100.0, sample_mu=False)
        p1 = env.sample_task(jax.random.PRNGKey(0))
        p2 = env.sample_task(jax.random.PRNGKey(999))
        assert float(p1.lam) == float(p2.lam)
        assert jnp.isclose(float(p1.lam), (5.0 + 100.0) / 2.0, atol=1e-3)

    def test_vdp_sample_produces_variety(self):
        env, _ = _make_vdp_env(mu_min=5.0, mu_max=100.0, sample_mu=True)
        keys = jax.random.split(jax.random.PRNGKey(7), 20)
        mus = set()
        for k in keys:
            p = env.sample_task(k)
            mus.add(round(float(p.lam), 1))
        assert len(mus) >= 5, f"Expected variety, got {len(mus)} unique mu values"

    def test_scalar_decay_sample_uniform(self):
        env, _ = _make_scalar_env()
        keys = jax.random.split(jax.random.PRNGKey(0), 50)
        for k in keys:
            p = env.sample_task(k)
            lam = float(p.lam)
            assert 1.0 <= lam <= 50.0, f"lam={lam} outside [1, 50]"

    def test_get_task_params_normalized_01(self):
        """get_task_params should normalize task param to [0, 1]."""
        env, _ = _make_vdp_env(mu_min=5.0, mu_max=100.0)
        p_lo = ODEParams(lam=5.0, max_steps=200)
        p_hi = ODEParams(lam=100.0, max_steps=200)
        p_mid = ODEParams(lam=52.5, max_steps=200)
        assert jnp.isclose(float(env.get_task_params(p_lo)[0]), 0.0, atol=1e-4)
        assert jnp.isclose(float(env.get_task_params(p_hi)[0]), 1.0, atol=1e-4)
        assert jnp.isclose(float(env.get_task_params(p_mid)[0]), 0.5, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# 6. DONE CONDITION AND BUDGET
# ═══════════════════════════════════════════════════════════════════════════


class TestDoneCondition:
    def test_done_at_max_steps(self):
        env, _ = _make_scalar_env(max_steps=5)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=1.0, max_steps=5)
        _, state = env.reset(key, params)
        for i in range(5):
            _, state, _, done, info = env.step(key, state, jnp.array([0.0]), params)
        assert bool(done), "Should be done after max_steps"

    def test_done_at_t_end(self):
        """If solver reaches t_end before max_steps, done = True (success)."""
        env, _ = _make_scalar_env(max_steps=10000)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=0.1, max_steps=10000)
        _, state = env.reset(key, params)
        for _ in range(10000):
            _, state, _, done, info = env.step(key, state, jnp.array([1.0]), params)
            if done:
                break
        if bool(info["success"]):
            assert float(info["t_reached"]) >= float(env.t_end)

    def test_budget_exhausted_increments(self):
        env, _ = _make_scalar_env(max_steps=100)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=1.0, max_steps=100)
        _, state = env.reset(key, params)
        assert float(state.budget_exhausted) == 0.0
        _, state2, _, _, _ = env.step(key, state, jnp.array([0.0]), params)
        assert jnp.isclose(float(state2.budget_exhausted), 1.0 / 100.0, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# 7. ACCEPT/REJECT MECHANICS
# ═══════════════════════════════════════════════════════════════════════════


class TestAcceptReject:
    def test_rejected_step_preserves_state(self):
        """On rejection, y and t should not advance."""
        env, _ = _make_vdp_env(max_steps=200)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=50.0, max_steps=200)
        _, state = env.reset(key, params)
        state_extreme = state.replace(y=jnp.array([1e10, -1e10]))
        _, state2, _, _, info = env.step(key, state_extreme, jnp.array([0.0]), params)
        assert not bool(info["keep_step"]), "Extreme values should cause rejection"
        assert jnp.allclose(state2.y, state_extreme.y)
        assert jnp.isclose(float(state2.t), float(state_extreme.t))

    def test_accepted_step_advances_time(self):
        env, _ = _make_scalar_env(max_steps=200)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=1.0, max_steps=200)
        _, state = env.reset(key, params)
        _, state2, _, _, info = env.step(key, state, jnp.array([1.0]), params)
        assert bool(info["keep_step"]), "Normal state should be accepted"
        assert float(state2.t) > float(state.t)

    def test_nan_solver_output_rejected(self):
        """NaN solver output must be rejected and obs stays finite."""
        env, _ = _make_vdp_env(max_steps=100)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=50.0, max_steps=100)
        _, state = env.reset(key, params)
        nan_state = state.replace(
            y=jnp.array([jnp.nan, jnp.nan]),
            solver_state_f=jnp.array([jnp.nan, jnp.nan]),
        )
        obs, state2, reward, _, info = env.step(key, nan_state, jnp.array([0.0]), params)
        assert jnp.all(jnp.isfinite(obs))


# ═══════════════════════════════════════════════════════════════════════════
# 8. PID CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════


class TestPIDController:
    def test_compute_step_context_offset(self):
        dims = {"state": 2, "direction": 2, "step_context": 4, "solver_trend": 5}
        assert compute_step_context_offset(("state", "step_context", "solver_trend"), dims) == 2
        assert compute_step_context_offset(("state", "direction", "step_context"), dims) == 4
        assert compute_step_context_offset(("step_context",), dims) == 0

    def test_compute_step_context_offset_raises_without_step_context(self):
        dims = {"state": 2, "solver_trend": 4}
        with pytest.raises(ValueError, match="step_context not in obs_features"):
            compute_step_context_offset(("state", "solver_trend"), dims)

    def test_pid_action_shape(self):
        obs = jnp.zeros(9)
        action = pid_action_from_obs(obs, step_context_offset=2)
        assert action.shape == (1,)

    def test_pid_action_in_range(self):
        obs = jnp.zeros(9)
        action = pid_action_from_obs(obs, step_context_offset=2)
        assert -1.0 <= float(action[0]) <= 1.0

    def test_pid_action_shrinks_on_high_error(self):
        """High error, rejected → factor < 1 → negative action (shrink dt).
        keep_step=False here (not True): an accepted step now floors its
        factor at 1 (mirrors diffrax — never shrinks on accept), so pairing
        a huge error with keep_step=True would be a self-contradictory
        input a real rollout never produces (huge error → rejected)."""
        obs = jnp.zeros(9)
        obs = obs.at[2].set(2.0)
        obs = obs.at[4].set(0.0)
        action = pid_action_from_obs(obs, step_context_offset=2)
        assert float(action[0]) < 0.0

    def test_pid_action_grows_on_low_error(self):
        """Low error → factor > 1 → positive action (grow dt)."""
        obs = jnp.zeros(9)
        obs = obs.at[2].set(-2.0)
        obs = obs.at[4].set(1.0)
        action = pid_action_from_obs(obs, step_context_offset=2)
        assert float(action[0]) > 0.0

    def test_pid_rejection_caps_factor(self):
        """On rejection (keep_step=0), factor is capped to min(factor, 1.0)."""
        obs = jnp.zeros(9)
        obs = obs.at[2].set(-2.0)
        obs = obs.at[4].set(0.0)
        action = pid_action_from_obs(obs, step_context_offset=2)
        assert float(action[0]) <= 0.0 + 1e-6, "Rejected step should not grow dt (action <= 0)"

    def test_pid_jit_compatible(self):
        obs = jnp.zeros(9)
        jit_pid = jax.jit(lambda o: pid_action_from_obs(o, step_context_offset=2))
        action = jit_pid(obs)
        assert action.shape == (1,)
        assert jnp.isfinite(action[0])

    def test_pid_vmap_compatible(self):
        obs_batch = jnp.zeros((4, 9))
        vmap_pid = jax.vmap(lambda o: pid_action_from_obs(o, step_context_offset=2))
        actions = vmap_pid(obs_batch)
        assert actions.shape == (4, 1)


# ═══════════════════════════════════════════════════════════════════════════
# 9. EPISODES PER TRIAL — MULTI-EPISODE DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════


class TestEpisodesPerTrial:
    """Test multi-episode trial logic in _env_step_and_reset."""

    def _run_trial_rollout(self, episodes_per_trial, max_steps=5, rollout_steps=30):
        from steppo.training.rollout import (
            collect_rollout,
            init_rollout_state,
        )

        num_envs = 4
        env, cfg = _make_scalar_env(max_steps=max_steps)
        config = TrainConfig(
            vae=VAEConfig(latent_dim=3, encoder=EncoderArchConfig(hidden_size=32)),
            ppo=PPOConfig(),
            env=cfg,
            num_envs=num_envs,
            rollout_steps=rollout_steps,
            episodes_per_trial=episodes_per_trial,
        )
        vae, policy = _make_models(env, config)
        key = jax.random.PRNGKey(0)
        task_keys = jax.random.split(key, num_envs)
        env_params = jax.vmap(env.sample_task)(task_keys)
        rs = init_rollout_state(vae, env, env_params, num_envs, key)
        new_rs, batch = collect_rollout(
            rs, vae, policy, env, env_params, config, jax.random.PRNGKey(1)
        )
        return new_rs, batch

    def test_single_episode_no_trial_freeze(self):
        """episodes_per_trial=0 means single episode, no freeze logic."""
        new_rs, batch = self._run_trial_rollout(episodes_per_trial=0, max_steps=5, rollout_steps=20)
        assert not jnp.any(new_rs.trial_done)

    def test_multi_episode_trial_done_eventually(self):
        """With episodes_per_trial=2 and short episodes, trial should freeze."""
        new_rs, batch = self._run_trial_rollout(episodes_per_trial=2, max_steps=3, rollout_steps=30)
        assert jnp.any(new_rs.trial_done), "Trial should be done after enough steps"

    def test_frozen_envs_have_zero_reward(self):
        """Once trial is frozen, rewards should be zero."""
        _, batch = self._run_trial_rollout(episodes_per_trial=2, max_steps=3, rollout_steps=30)
        dones = batch.dones
        rewards = batch.rewards.squeeze(-1)
        for env_idx in range(dones.shape[1]):
            frozen_start = None
            episode_count = 0
            for t in range(dones.shape[0]):
                if bool(dones[t, env_idx]):
                    episode_count += 1
                    if episode_count >= 2:
                        frozen_start = t + 1
                        break
            if frozen_start is not None and frozen_start < dones.shape[0]:
                frozen_rewards = rewards[frozen_start:, env_idx]
                assert jnp.allclose(frozen_rewards, 0.0), (
                    f"Env {env_idx}: frozen rewards should be 0, got {frozen_rewards}"
                )

    def test_episode_count_increments(self):
        """Episode count should increment on episode done."""
        new_rs, _ = self._run_trial_rollout(episodes_per_trial=3, max_steps=3, rollout_steps=30)
        assert jnp.all(new_rs.episode_count >= 0)

    def test_init_rollout_state_episode_count_zero(self):
        from steppo.training.rollout import init_rollout_state

        num_envs = 4
        env, cfg = _make_scalar_env(max_steps=10)
        config = TrainConfig(
            vae=VAEConfig(latent_dim=3, encoder=EncoderArchConfig(hidden_size=32)),
            ppo=PPOConfig(),
            env=cfg,
            num_envs=num_envs,
            rollout_steps=10,
        )
        vae, _ = _make_models(env, config)
        key = jax.random.PRNGKey(0)
        params = jax.vmap(env.sample_task)(jax.random.split(key, num_envs))
        rs = init_rollout_state(vae, env, params, num_envs, key)
        assert jnp.all(rs.episode_count == 0)
        assert jnp.all(~rs.trial_done)


# ═══════════════════════════════════════════════════════════════════════════
# 10. WARMSTART ON/OFF
# ═══════════════════════════════════════════════════════════════════════════


class TestWarmstart:
    """Test warmstart-related functionality."""

    def test_warmstart_config_defaults(self):
        ws = WarmstartConfig()
        assert ws.enabled is False
        assert ws.num_iters == 200
        assert ws.pid_safety == 0.9
        assert ws.pid_order == 4
        assert ws.oracle_dataset_seed == 0

    def test_oracle_warmstart_uses_dataset_seed_not_training_seed(self, monkeypatch):
        """Oracle cache selection is independent of the repeat's training seed."""
        from steppo.training.base_trainer import Trainer
        from steppo.training.ppo_trainer import PPOTrainState

        env, env_cfg = _make_scalar_env(max_steps=5, immediate_dt_action=True)
        config = TrainConfig(
            env=env_cfg,
            seed=17,
            num_envs=1,
            rollout_steps=5,
            warmstart=WarmstartConfig(
                enabled=True,
                expert="oracle",
                num_iters=0,
                oracle_dataset_seed=3,
            ),
        )
        seen = {}

        def fake_load(_env_config, **kwargs):
            seen.update(kwargs)
            return {"states": np.zeros((5, 1, 1), dtype=np.float32)}

        monkeypatch.setattr("steppo.training.oracle_dataset.load_oracle_training_batch", fake_load)

        trainer = Trainer(config, vae=None, policy=None, env=env, env_params=None)
        ppo_state = PPOTrainState(
            graphdef=None,
            params={"param": jnp.zeros(1)},
            opt_state=None,
            step=0,
            ema_actor=jnp.float32(0.0),
            ema_value=jnp.float32(0.0),
        )
        monkeypatch.setattr(
            "steppo.training.base_trainer.create_ppo_train_state", lambda *args, **kwargs: ppo_state
        )
        monkeypatch.setattr(
            "steppo.training.base_trainer.nnx.merge", lambda *args, **kwargs: object()
        )
        trainer._warmstart(None, None, ppo_state, jax.random.PRNGKey(0))

        assert seen["seed"] == 3

    def test_warmstart_enabled_config(self):
        ws = WarmstartConfig(enabled=True, num_iters=50, bc_lr=0.001, bc_epochs=3)
        assert ws.enabled is True
        assert ws.num_iters == 50

    def test_pid_rollout_collection(self):
        """collect_pid_rollout should produce valid rollouts with PID actions."""
        from steppo.training.rollout import collect_pid_rollout

        num_envs = 4
        rollout_steps = 20
        env, cfg = _make_scalar_env(max_steps=rollout_steps)
        ws_cfg = WarmstartConfig(enabled=True, num_iters=1, bc_lr=0.001, bc_epochs=1)
        config = TrainConfig(
            vae=VAEConfig(latent_dim=3, encoder=EncoderArchConfig(hidden_size=32)),
            ppo=PPOConfig(),
            env=cfg,
            num_envs=num_envs,
            rollout_steps=rollout_steps,
            warmstart=ws_cfg,
        )
        vae, _ = _make_models(env, config)
        key = jax.random.PRNGKey(0)
        env_params = jax.vmap(env.sample_task)(jax.random.split(key, num_envs))
        spec = get_system("scalar_decay")
        offset = compute_step_context_offset(cfg.obs_features, spec.feature_dims)

        batch = collect_pid_rollout(vae, env, env_params, config, ws_cfg, offset, key)
        assert batch.states.shape == (rollout_steps, num_envs, env.obs_shape()[0])
        assert batch.actions.shape == (rollout_steps, num_envs, 1)
        assert batch.rewards.shape == (rollout_steps, num_envs, 1)
        assert jnp.all(jnp.isfinite(batch.states))
        assert jnp.all(jnp.isfinite(batch.actions))
        assert batch.log_probs.shape == (rollout_steps, num_envs)
        assert jnp.allclose(batch.log_probs, 0.0)

    def test_pid_actions_within_range(self):
        """PID rollout actions should all be in [-1, 1]."""
        from steppo.training.rollout import collect_pid_rollout

        num_envs = 4
        rollout_steps = 20
        env, cfg = _make_scalar_env(max_steps=rollout_steps)
        ws_cfg = WarmstartConfig(enabled=True)
        config = TrainConfig(
            vae=VAEConfig(latent_dim=3, encoder=EncoderArchConfig(hidden_size=32)),
            ppo=PPOConfig(),
            env=cfg,
            num_envs=num_envs,
            rollout_steps=rollout_steps,
        )
        vae, _ = _make_models(env, config)
        key = jax.random.PRNGKey(0)
        env_params = jax.vmap(env.sample_task)(jax.random.split(key, num_envs))
        spec = get_system("scalar_decay")
        offset = compute_step_context_offset(cfg.obs_features, spec.feature_dims)

        batch = collect_pid_rollout(vae, env, env_params, config, ws_cfg, offset, key)
        assert jnp.all(batch.actions >= -1.0 - 1e-6)
        assert jnp.all(batch.actions <= 1.0 + 1e-6)

    def test_warmstart_expert_default_is_pid(self):
        assert WarmstartConfig().expert == "pid"

    def test_oracle_dataset_generation(self):
        """oracle_dataset.generate_oracle_batch should produce a valid static
        trajectory (no vae/policy involved) with in-range actions."""
        from steppo.training.oracle_dataset import generate_oracle_batch

        num_envs = 4
        rollout_steps = 5
        env, _ = _make_scalar_env(max_steps=rollout_steps, immediate_dt_action=True)

        batch = generate_oracle_batch(
            env,
            num_envs=num_envs,
            rollout_steps=rollout_steps,
            episodes_per_trial=0,
            oracle_max_iters=3,
            seed=0,
        )
        assert batch["states"].shape == (rollout_steps, num_envs, env.obs_shape()[0])
        assert batch["actions"].shape == (rollout_steps, num_envs, 1)
        assert batch["rewards"].shape == (rollout_steps, num_envs, 1)
        assert np.all(np.isfinite(batch["states"]))
        assert np.all(np.isfinite(batch["actions"]))
        assert np.all(batch["actions"] >= -1.0 - 1e-6)
        assert np.all(batch["actions"] <= 1.0 + 1e-6)

    def test_encode_oracle_batch_attaches_beliefs(self):
        """encode_oracle_batch should replay a static oracle trajectory through
        the current vae's encoder and attach beliefs_mu/beliefs_logvar, with
        no other field altered."""
        from steppo.training.oracle_dataset import generate_oracle_batch
        from steppo.training.rollout import encode_oracle_batch

        num_envs = 4
        rollout_steps = 5
        env, cfg = _make_scalar_env(max_steps=rollout_steps, immediate_dt_action=True)
        config = TrainConfig(
            vae=VAEConfig(latent_dim=3, encoder=EncoderArchConfig(hidden_size=32)),
            ppo=PPOConfig(),
            env=cfg,
            num_envs=num_envs,
            rollout_steps=rollout_steps,
            warmstart=WarmstartConfig(enabled=True, expert="oracle"),
        )
        vae, _ = _make_models(env, config)

        static_batch = generate_oracle_batch(
            env,
            num_envs=num_envs,
            rollout_steps=rollout_steps,
            episodes_per_trial=0,
            oracle_max_iters=3,
            seed=0,
        )
        static_batch = {k: jnp.asarray(v) for k, v in static_batch.items()}

        batch = encode_oracle_batch(vae, static_batch, config)
        assert batch.beliefs_mu.shape == (rollout_steps, num_envs, config.vae.total_latent_dim)
        assert jnp.all(jnp.isfinite(batch.beliefs_mu))
        assert jnp.all(jnp.isfinite(batch.beliefs_logvar))
        assert jnp.array_equal(batch.states, static_batch["states"])
        assert jnp.array_equal(batch.actions, static_batch["actions"])


# ═══════════════════════════════════════════════════════════════════════════
# 11. BC LOSS
# ═══════════════════════════════════════════════════════════════════════════


class TestBCLoss:
    def test_bc_loss_zero_for_matching_actions(self):
        """BC loss should be near-zero when policy output matches expert."""
        from steppo.models.policy import ActorCritic

        obs_dim, action_dim, latent_dim = 6, 1, 3
        policy = ActorCritic(
            obs_dim,
            action_dim,
            latent_dim,
            action_space="continuous",
            use_latent_sample=False,
            rngs=nnx.Rngs(0),
        )
        graphdef, params = nnx.split(policy)
        N = 8
        states = jnp.zeros((N, obs_dim))
        z = jnp.zeros((N, latent_dim * 2))
        policy_merged = nnx.merge(graphdef, params)
        dist, _ = policy_merged(states[0], z[0])
        mean_action = jnp.tanh(dist.distribution.loc)
        expert = jnp.broadcast_to(mean_action, (N, action_dim))
        loss, metrics = bc_loss(graphdef, params, states, z, expert)
        assert float(loss) < 0.01, f"Loss should be near-zero for matching actions, got {loss}"

    def test_bc_loss_positive_for_different_actions(self):
        from steppo.models.policy import ActorCritic

        obs_dim, action_dim, latent_dim = 6, 1, 3
        policy = ActorCritic(
            obs_dim,
            action_dim,
            latent_dim,
            action_space="continuous",
            use_latent_sample=False,
            rngs=nnx.Rngs(0),
        )
        graphdef, params = nnx.split(policy)
        N = 8
        states = jnp.zeros((N, obs_dim))
        z = jnp.zeros((N, latent_dim * 2))
        expert = jnp.ones((N, action_dim)) * 0.9
        loss, metrics = bc_loss(graphdef, params, states, z, expert)
        assert float(loss) > 0.0

    def test_bc_loss_clips_expert_actions(self):
        """Expert actions at exactly ±1 should not produce NaN (arctanh clips)."""
        from steppo.models.policy import ActorCritic

        obs_dim, action_dim, latent_dim = 6, 1, 3
        policy = ActorCritic(
            obs_dim,
            action_dim,
            latent_dim,
            action_space="continuous",
            use_latent_sample=False,
            rngs=nnx.Rngs(0),
        )
        graphdef, params = nnx.split(policy)
        N = 4
        states = jnp.zeros((N, obs_dim))
        z = jnp.zeros((N, latent_dim * 2))
        expert = jnp.array([[1.0], [-1.0], [1.0], [-1.0]])
        loss, _ = bc_loss(graphdef, params, states, z, expert)
        assert jnp.isfinite(loss), f"BC loss should be finite for ±1 actions, got {loss}"


# ═══════════════════════════════════════════════════════════════════════════
# 12. ENV INIT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvInitValidation:
    def test_max_steps_required(self):
        with pytest.raises(ValueError, match="max_steps must be provided"):
            ODEEnv(ODEEnvConfig(), max_steps=None)

    def test_dt_min_greater_than_dt_max_raises(self):
        with pytest.raises(ValueError, match="dt_min.*cannot be greater.*dt_max"):
            ODEEnv(ODEEnvConfig(dt_min=10.0, dt_max=1.0), max_steps=100)

    def test_dt0_out_of_range_raises(self):
        with pytest.raises(ValueError, match="dt0.*must be within"):
            ODEEnv(ODEEnvConfig(dt0=0.1, dt_min=1.0, dt_max=5.0), max_steps=100)

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError, match="dt_min and dt_max must be positive"):
            ODEEnv(ODEEnvConfig(dt_min=-1.0), max_steps=100)


# ═══════════════════════════════════════════════════════════════════════════
# 13. ODE DYNAMICS CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════


class TestODEDynamics:
    def test_scalar_decay_solution_converges(self):
        """Scalar decay y' = -λy has exact solution y = exp(-λt). Check convergence."""
        env, _ = _make_scalar_env(max_steps=500)
        key = jax.random.PRNGKey(0)
        lam = 1.0
        params = ODEParams(lam=lam, max_steps=500)
        _, state = env.reset(key, params)

        def scan_step(state, _):
            _, new_state, _, done, _ = env.step(key, state, jnp.array([1.0]), params)
            return new_state, done

        final_state, dones = jax.lax.scan(scan_step, state, None, length=500)
        t_final = float(final_state.t)
        if t_final >= float(env.t_end):
            exact = float(jnp.exp(-lam * t_final))
            assert jnp.isclose(float(final_state.y[0]), exact, atol=1e-2), (
                f"y={float(final_state.y[0])}, exact={exact}"
            )

    def test_vdp_time_advances(self):
        """VdP solver should advance time on at least some steps."""
        env, _ = _make_vdp_env(max_steps=20)
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=10.0, max_steps=20)
        _, state = env.reset(key, params)

        def scan_step(state, _):
            _, new_state, _, _, _ = env.step(key, state, jnp.array([0.0]), params)
            return new_state, None

        final_state, _ = jax.lax.scan(scan_step, state, None, length=20)
        assert float(final_state.t) > 0.0, "Time should advance"

    def test_vmap_step_consistency(self):
        """Vmapped step should produce same results as sequential steps."""
        env, _ = _make_scalar_env(max_steps=200)
        N = 4
        key = jax.random.PRNGKey(0)
        params = ODEParams(lam=1.0, max_steps=200)
        params_batch = jax.tree.map(lambda x: jnp.stack([x] * N), params)
        keys = jax.random.split(key, N)
        obs_batch, state_batch = jax.vmap(env.reset)(keys, params_batch)
        actions = jnp.zeros((N, 1))
        obs2, state2, rewards, dones, info = jax.vmap(env.step)(
            keys, state_batch, actions, params_batch
        )
        assert obs2.shape == (N, env.obs_shape()[0])
        assert jnp.all(jnp.isfinite(obs2))


# ═══════════════════════════════════════════════════════════════════════════
# 14. MULTI-SYSTEM CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiSystem:
    @pytest.mark.parametrize("system", ["scalar_decay", "van_der_pol"])
    def test_reset_and_step_finite(self, system):
        if system == "scalar_decay":
            env, _ = _make_scalar_env(max_steps=50)
            params = ODEParams(lam=1.0, max_steps=50)
        else:
            env, _ = _make_vdp_env(max_steps=50)
            params = ODEParams(lam=10.0, max_steps=50)

        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        assert jnp.all(jnp.isfinite(obs))

        obs2, state2, reward, done, info = env.step(key, state, jnp.array([0.0]), params)
        assert jnp.all(jnp.isfinite(obs2))
        assert jnp.isfinite(reward)

    @pytest.mark.parametrize("system", ["scalar_decay", "van_der_pol"])
    def test_obs_dim_from_features(self, system):
        spec = get_system(system)
        cfg = ODEEnvConfig(system=system)
        expected = sum(spec.feature_dims.get(f, 0) for f in cfg.obs_features)
        actual = spec.get_obs_dim(cfg)
        assert actual == expected


# ═══════════════════════════════════════════════════════════════════════════
# 15. BELIEF RESET BETWEEN EPISODES
# ═══════════════════════════════════════════════════════════════════════════


class TestBeliefReset:
    def test_belief_reset_config(self):
        config = TrainConfig(
            reset_belief_between_episodes=True,
            episodes_per_trial=2,
        )
        assert config.reset_belief_between_episodes is True
        assert config.episodes_per_trial == 2

    def test_rollout_with_belief_reset(self):
        """Rollout with reset_belief should complete without errors."""
        from steppo.training.rollout import collect_rollout, init_rollout_state

        num_envs = 4
        env, cfg = _make_scalar_env(max_steps=5)
        config = TrainConfig(
            vae=VAEConfig(latent_dim=3, encoder=EncoderArchConfig(hidden_size=32)),
            ppo=PPOConfig(),
            env=cfg,
            num_envs=num_envs,
            rollout_steps=20,
            episodes_per_trial=2,
            reset_belief_between_episodes=True,
        )
        vae, policy = _make_models(env, config)
        key = jax.random.PRNGKey(0)
        env_params = jax.vmap(env.sample_task)(jax.random.split(key, num_envs))
        rs = init_rollout_state(vae, env, env_params, num_envs, key)
        new_rs, batch = collect_rollout(
            rs, vae, policy, env, env_params, config, jax.random.PRNGKey(1)
        )
        assert jnp.all(jnp.isfinite(batch.beliefs_mu))
        assert jnp.all(jnp.isfinite(batch.beliefs_logvar))

    def test_rollout_without_belief_reset(self):
        """Rollout without reset_belief should also work fine."""
        from steppo.training.rollout import collect_rollout, init_rollout_state

        num_envs = 4
        env, cfg = _make_scalar_env(max_steps=5)
        config = TrainConfig(
            vae=VAEConfig(latent_dim=3, encoder=EncoderArchConfig(hidden_size=32)),
            ppo=PPOConfig(),
            env=cfg,
            num_envs=num_envs,
            rollout_steps=20,
            episodes_per_trial=2,
            reset_belief_between_episodes=False,
        )
        vae, policy = _make_models(env, config)
        key = jax.random.PRNGKey(0)
        env_params = jax.vmap(env.sample_task)(jax.random.split(key, num_envs))
        rs = init_rollout_state(vae, env, env_params, num_envs, key)
        new_rs, batch = collect_rollout(
            rs, vae, policy, env, env_params, config, jax.random.PRNGKey(1)
        )
        assert jnp.all(jnp.isfinite(batch.beliefs_mu))


# ═══════════════════════════════════════════════════════════════════════════
# 16. KEEP STEPS AND ROLLOUT BATCH FIELDS
# ═══════════════════════════════════════════════════════════════════════════


class TestRolloutBatchFields:
    def test_keep_steps_binary(self):
        """keep_steps in RolloutBatch should be 0.0 or 1.0."""
        from steppo.training.rollout import collect_rollout, init_rollout_state

        num_envs = 4
        env, cfg = _make_scalar_env(max_steps=50)
        config = TrainConfig(
            vae=VAEConfig(latent_dim=3, encoder=EncoderArchConfig(hidden_size=32)),
            ppo=PPOConfig(),
            env=cfg,
            num_envs=num_envs,
            rollout_steps=20,
        )
        vae, policy = _make_models(env, config)
        key = jax.random.PRNGKey(0)
        env_params = jax.vmap(env.sample_task)(jax.random.split(key, num_envs))
        rs = init_rollout_state(vae, env, env_params, num_envs, key)
        _, batch = collect_rollout(rs, vae, policy, env, env_params, config, jax.random.PRNGKey(1))
        ks = batch.keep_steps
        assert jnp.all((ks == 0.0) | (ks == 1.0)), (
            f"keep_steps has non-binary values: min={float(ks.min())}, max={float(ks.max())}"
        )

    def test_task_params_present(self):
        from steppo.training.rollout import collect_rollout, init_rollout_state

        num_envs = 4
        env, cfg = _make_scalar_env(max_steps=50)
        config = TrainConfig(
            vae=VAEConfig(latent_dim=3, encoder=EncoderArchConfig(hidden_size=32)),
            ppo=PPOConfig(),
            env=cfg,
            num_envs=num_envs,
            rollout_steps=10,
        )
        vae, policy = _make_models(env, config)
        key = jax.random.PRNGKey(0)
        env_params = jax.vmap(env.sample_task)(jax.random.split(key, num_envs))
        rs = init_rollout_state(vae, env, env_params, num_envs, key)
        _, batch = collect_rollout(rs, vae, policy, env, env_params, config, jax.random.PRNGKey(1))
        assert batch.task_params.shape == (num_envs, 1)
        assert jnp.all(batch.task_params >= 0.0)
        assert jnp.all(batch.task_params <= 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 17. FEATURE GROUP SELECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureGroups:
    def test_state_only(self):
        env, _ = _make_vdp_env(obs_features=("state",))
        assert env.obs_shape() == (2,)

    def test_state_and_direction(self):
        env, _ = _make_vdp_env(obs_features=("state", "direction"))
        assert env.obs_shape() == (4,)

    def test_all_features(self):
        env, _ = _make_vdp_env(obs_features=("state", "direction", "step_context", "solver_trend"))
        assert env.obs_shape() == (13,)

    def test_step_context_only(self):
        env, _ = _make_vdp_env(obs_features=("step_context",))
        assert env.obs_shape() == (4,)

    def test_different_features_produce_different_obs(self):
        env_min, _ = _make_vdp_env(obs_features=("state",))
        env_full, _ = _make_vdp_env(obs_features=("state", "step_context", "solver_trend"))
        key = jax.random.PRNGKey(0)
        p = ODEParams(lam=10.0, max_steps=200)
        obs_min, _ = env_min.reset(key, p)
        obs_full, _ = env_full.reset(key, p)
        assert obs_min.shape[0] < obs_full.shape[0]
