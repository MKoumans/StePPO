"""Tests for steppo.envs.ode, ODE systems, and NaN/stability guards."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from steppo.configs.base_config import ODEEnvConfig
from steppo.envs import make_env
from steppo.envs.ode import ODEEnv

ODE_OBS_DIM = 10  # default obs_features=("state","step_context","solver_trend") -> 1+5+4
ODE_ACTION_DIM = 1


def _make_ode_env():
    from steppo.envs.ode import ODEParams

    # system="scalar_decay" matches ODEEnv's own config=None default (see ODEEnv.__init__);
    # train_bins matches its default lam_min/lam_max.
    config = ODEEnvConfig(system="scalar_decay", train_bins=((1.0, 100.0),))
    return ODEEnv(config, max_steps=200), ODEParams(lam=10.0)


# ── ODE env core ─────────────────────────────────────────────────────────────


def test_reset_shape():
    env, params = _make_ode_env()
    obs, state = env.reset(jax.random.PRNGKey(0), params)
    assert obs.shape == (ODE_OBS_DIM,)


def test_step_jit():
    env, params = _make_ode_env()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    step_jit = jax.jit(env.step)
    action = jnp.array([0.0])
    obs2, state2, reward, done, _ = step_jit(key, state, action, params)
    assert obs2.shape == (ODE_OBS_DIM,)
    assert jnp.isfinite(reward)


def test_vmap():
    env, params = _make_ode_env()
    N = 4
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    params_batch = jax.tree.map(lambda x: jnp.stack([x] * N), params)
    obs_batch, state_batch = jax.vmap(env.reset)(keys, params_batch)
    assert obs_batch.shape == (N, ODE_OBS_DIM)


def test_sample_task_varies():
    env, _ = _make_ode_env()
    keys = jax.random.split(jax.random.PRNGKey(7), 5)
    lams = set()
    for k in keys:
        p = env.sample_task(k)
        lams.add(round(float(p.lam), 2))
    assert len(lams) >= 3, f"Expected >=3 unique lambda values, got {lams}"


@pytest.mark.parametrize("scheme", ["binned", "log-binned"])
def test_sample_task_binned_scheme_never_falls_in_gaps(scheme):
    train_bins = ((1.0, 10.0), (20.0, 30.0), (60.0, 70.0))
    config = ODEEnvConfig(
        system="scalar_decay",
        lam_min=1.0,
        lam_max=100.0,
        sample_lam=True,
        task_sample_scheme=scheme,
        train_bins=train_bins,
    )
    env = ODEEnv(config, max_steps=200)
    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    lams = jax.vmap(lambda k: env.sample_task(k).lam)(keys)
    lams_np = jax.device_get(lams)
    in_some_bin = [any(lo <= v <= hi for lo, hi in train_bins) for v in lams_np]
    assert all(in_some_bin), (
        f"Sampled outside train_bins: {[v for v, ok in zip(lams_np, in_some_bin) if not ok]}"
    )


def test_sample_task_binned_scheme_requires_train_bins():
    config = ODEEnvConfig(system="scalar_decay", sample_lam=True, task_sample_scheme="binned")
    env = ODEEnv(config, max_steps=200)
    with pytest.raises(ValueError):
        env.sample_task(jax.random.PRNGKey(0))


def test_ode_env_config_with_train_bins_is_hashable():
    """Regression test for the load_config_from_dict nested-tuple coercion fix —
    ODEEnvConfig is @dataclass(unsafe_hash=True), so a bins field containing
    unhashable lists (rather than tuples) would break hash(config)."""
    config = ODEEnvConfig(system="scalar_decay", train_bins=((1.0, 10.0), (20.0, 30.0)))
    hash(config)  # must not raise


def test_ode_env_config_bins_must_fit_active_task_range():
    config = ODEEnvConfig(
        system="van_der_pol",
        mu_min=1.0,
        mu_max=200.0,
        train_bins=((5.0, 100.0),),
        val_bins=((50.0, 200.0),),
    )
    assert config.train_bins == ((5.0, 100.0),)

    with pytest.raises(ValueError, match=r"train_bins\[0\].*outside"):
        ODEEnvConfig(system="van_der_pol", mu_min=10.0, mu_max=200.0, train_bins=((5.0, 100.0),))

    with pytest.raises(ValueError, match=r"test_bins\[0\].*outside"):
        ODEEnvConfig(system="van_der_pol", mu_min=1.0, mu_max=200.0, test_bins=((50.0, 201.0),))


def test_done_at_max_steps():
    env, _ = _make_ode_env()
    from steppo.envs.ode import ODEParams

    params_short = ODEParams(lam=1.0, max_steps=2)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params_short)
    action = jnp.array([0.0])
    for _ in range(2):
        obs, state, reward, done, _ = env.step(key, state, action, params_short)
    assert bool(done)


def test_make_env_ode():
    env, params = make_env("ode")
    assert env is not None
    assert params is not None


# ── NaN / stability guards ───────────────────────────────────────────────────


def test_nan_solver_output_rejected():
    """If the solver returns NaN y1, the step must not propagate NaN to obs."""

    cfg = ODEEnvConfig(system="van_der_pol", mu_min=50.0, sample_mu=False)
    env = ODEEnv(cfg, max_steps=100)
    key = jax.random.PRNGKey(0)
    params = env.sample_task(key)
    _, state = env.reset(key, params)

    nan_state = state.replace(
        y=jnp.array([jnp.nan, jnp.nan]),
        solver_state_f=jnp.array([jnp.nan, jnp.nan]),
    )
    action = jnp.array([0.0])
    obs, new_state, reward, done, info = env.step(key, nan_state, action, params)
    assert jnp.all(jnp.isfinite(obs)), f"Obs contains NaN: {obs}"


def test_obs_nan_guard_on_extreme_state():
    """Observation NaN guard replaces non-finite values with zeros."""
    cfg = ODEEnvConfig(system="van_der_pol", mu_min=50.0, sample_mu=False)
    env = ODEEnv(cfg, max_steps=100)
    key = jax.random.PRNGKey(0)
    params = env.sample_task(key)
    _, state = env.reset(key, params)

    inf_state = state.replace(y=jnp.array([1e30, -1e30]))
    action = jnp.array([0.0])
    obs, _, _, _, _ = env.step(key, inf_state, action, params)
    assert jnp.all(jnp.isfinite(obs)), f"Obs not finite after inf state: {obs}"


def test_stiff_rollout_stays_finite():
    """Full rollout on VdP mu=50 must produce finite observations and beliefs."""
    from steppo.configs.base_config import (
        EncoderArchConfig,
        PPOConfig,
        TrainConfig,
        VAEConfig,
    )
    from steppo.models.policy import ActorCritic
    from steppo.models.vae import VariBADVAE
    from steppo.training.rollout import collect_rollout, init_rollout_state

    num_envs, rollout_steps = 8, 100
    env_cfg = ODEEnvConfig(
        system="van_der_pol",
        t_end=100.0,
        dt0=1e-4,
        rtol=1e-6,
        atol=1e-8,
        dt_min=1e-10,
        dt_max=5.0,
        mu_min=50.0,
        mu_max=200.0,
        sample_mu=False,
        sample_y0=False,
        y0_x=0.0,
        y0_y=-2.0,
    )
    env = ODEEnv(env_cfg, max_steps=rollout_steps)
    obs_dim = env.obs_shape()[0]
    action_dim = env.num_actions

    config = TrainConfig(
        vae=VAEConfig(
            latent_dim=5,
            encoder=EncoderArchConfig(hidden_size=64),
            state_loss_coeff=0.5,
            task_loss_coeff=0.05,
        ),
        ppo=PPOConfig(),
        env=env_cfg,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
    )
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

    key = jax.random.PRNGKey(99)
    task_keys = jax.random.split(key, num_envs)
    env_params_batch = jax.vmap(env.sample_task)(task_keys)

    rs = init_rollout_state(vae, env, env_params_batch, num_envs, key)
    _, batch = collect_rollout(
        rs, vae, policy, env, env_params_batch, config, jax.random.PRNGKey(100)
    )

    assert jnp.all(jnp.isfinite(batch.states)), "states contain NaN"
    assert jnp.all(jnp.isfinite(batch.next_states)), "next_states contain NaN"
    assert jnp.all(jnp.isfinite(batch.rewards)), "rewards contain NaN"
    assert jnp.all(jnp.isfinite(batch.beliefs_mu)), "beliefs_mu contain NaN"
    assert jnp.all(jnp.isfinite(batch.beliefs_logvar)), "beliefs_logvar contain NaN"
    assert jnp.all(jnp.isfinite(batch.log_probs)), "log_probs contain NaN"
    assert jnp.all(jnp.isfinite(batch.values)), "values contain NaN"


# ── terminated vs truncated (info["success"]) ────────────────────────────────


def test_step_reaches_t_end_naturally_marks_success():
    """A simple accept/reject bang-bang controller (grow dt on acceptance, shrink
    on rejection — driving dt to its unconditional max every step gets stuck in a
    permanent reject streak instead, since scalar_decay's error tolerance is
    exceeded well before dt_max) reaches t_end well within budget — done=True with
    info['success']=True (terminated, not truncated)."""
    from steppo.envs.ode import ODEParams

    config = ODEEnvConfig(system="scalar_decay", train_bins=((1.0, 100.0),))
    env = ODEEnv(config, max_steps=2000)
    params = ODEParams(lam=10.0, max_steps=2000)
    key = jax.random.PRNGKey(0)
    step_jit = jax.jit(env.step)
    obs, state = env.reset(key, params)
    action = jnp.array([1.0])
    done = False
    info = None
    for _ in range(2000):
        obs, state, reward, done, info = step_jit(key, state, action, params)
        action = jnp.array([1.0]) if bool(info["keep_step"]) else jnp.array([-1.0])
        if bool(done):
            break
    assert bool(done), "episode never finished within 2000 steps"
    assert bool(info["success"]), (
        "expected natural completion (t_end reached), not budget exhaustion"
    )


def test_step_truncated_by_max_steps_marks_not_success():
    """A budget far too small to reach t_end should still end the episode
    (done=True) but via truncation, not natural completion — info['success']=False."""
    from steppo.envs.ode import ODEParams

    config = ODEEnvConfig(system="scalar_decay", train_bins=((1.0, 100.0),))
    env = ODEEnv(config, max_steps=2)
    params = ODEParams(lam=10.0, max_steps=2)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    action = jnp.array([1.0])
    obs, state, reward, done, info = env.step(key, state, action, params)
    obs, state, reward, done, info = env.step(key, state, action, params)
    assert done, "expected done=True once max_steps is exhausted"
    assert not bool(info["success"]), (
        "t_end=500 cannot be reached in 2 steps — this must be truncation, not success"
    )
