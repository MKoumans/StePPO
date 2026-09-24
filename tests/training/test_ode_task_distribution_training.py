"""Task-distribution coverage for the ODE 'systems' registry.

scalar_decay and van_der_pol already have deep coverage elsewhere (see
test_envs_ode.py, test_ode_comprehensive.py, test_trainer.py); brusselator
does not appear in any of those, and none of them run a full VariBADTrainer
loop across more than one hand-picked system. This file adds that coverage
for all three systems without touching any src/ functionality:

  - sample_task actually spans the configured train_bins range (not a
    degenerate point), for scalar_decay, van_der_pol, and brusselator;
  - reset/step stay finite across several distinct sampled tasks;
  - a short VariBADTrainer run — which resamples a fresh task per env every
    rollout via env.sample_task (see Trainer.collect_rollout in
    base_trainer.py) — completes with finite metrics, i.e. each system is
    genuinely trained on a *distribution* of task parameters, not one fixed
    value.

brusselator's B range is kept modest (2-8) rather than the full production
default (2-50): B~15+ is known-stiff enough to occasionally stall the
implicit solver (see TestNoFailureSignalAtDtMin in
test_pid_env_vs_diffrax.py) — an unrelated, pre-existing solver-convergence
issue that these tests intentionally stay clear of.

Each system's tests carry an `xdist_group` mark (see _SYSTEM_PARAMS) so that,
run under pytest-xdist with the `loadgroup` scheduler, every test for a given
system lands on the same worker — and via conftest.py's TEST_GPU_POOL, that
worker (and hence that system) is pinned to one GPU for the whole run:

    TEST_GPU_POOL=0,1 uv run pytest -n 3 --dist=loadgroup \\
        tests/test_ode_task_distribution_training.py

With 3 systems and TEST_GPU_POOL's 2 entries, gw0/gw1/gw2 map to GPUs
0/1/0 (worker index % pool length) — pytest-xdist's scheduler decides which
system lands on which worker, so which system gets the shared GPU varies
run to run, not by name. For a fixed, deterministic system->GPU mapping
instead, run one `pytest -k <system>` invocation per system with
CUDA_VISIBLE_DEVICES set explicitly (conftest.py respects it if already
set), e.g.:

    CUDA_VISIBLE_DEVICES=0 uv run pytest tests/test_ode_task_distribution_training.py -k scalar_decay &
    CUDA_VISIBLE_DEVICES=1 uv run pytest tests/test_ode_task_distribution_training.py -k van_der_pol &
    CUDA_VISIBLE_DEVICES=0 uv run pytest tests/test_ode_task_distribution_training.py -k brusselator &
    wait
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from steppo.configs.base_config import (
    EncoderArchConfig,
    EvalMetricsConfig,
    ODEEnvConfig,
    PPOConfig,
    PrecomputeBaselineConfig,
    TrainConfig,
    VAEConfig,
)
from steppo.envs.ode import ODEEnv
from steppo.models.policy import ActorCritic
from steppo.models.vae import VariBADVAE
from steppo.training.trainer import VariBADTrainer

LATENT_DIM = 5

# Per-system env kwargs: modest, fast-to-integrate task ranges (same style as
# test_ode_comprehensive.py's _VDP_DEFAULTS/_SCALAR_DEFAULTS) with
# sample_*=True and train_bins spanning a genuine distribution.
_SYSTEM_KWARGS = {
    "scalar_decay": dict(
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
    ),
    "van_der_pol": dict(
        system="van_der_pol",
        t_end=50.0,
        dt0=1e-4,
        rtol=1e-3,
        atol=1e-6,
        dt_min=1e-4,
        dt_max=5.0,
        mu_min=5.0,
        mu_max=50.0,
        sample_mu=True,
        sample_y0=False,
        y0_x=0.0,
        y0_y=-2.0,
        train_bins=((5.0, 50.0),),
    ),
    "brusselator": dict(
        system="brusselator",
        t_end=50.0,
        dt0=1e-4,
        rtol=1e-3,
        atol=1e-6,
        dt_min=1e-4,
        dt_max=5.0,
        B_min=2.0,
        B_max=8.0,
        sample_B=True,
        train_bins=((2.0, 8.0),),
    ),
}

_SYSTEMS = tuple(_SYSTEM_KWARGS)
_SYSTEM_PARAMS = [
    pytest.param(system, marks=pytest.mark.xdist_group(name=system)) for system in _SYSTEMS
]


def _make_env(system, max_steps=100):
    cfg = ODEEnvConfig(**_SYSTEM_KWARGS[system])
    return ODEEnv(cfg, max_steps=max_steps), cfg


# ── sample_task actually covers a distribution, not a point ────────────────


@pytest.mark.parametrize("system", _SYSTEM_PARAMS)
def test_sample_task_spans_configured_range(system):
    env, _ = _make_env(system)
    lo, hi = _SYSTEM_KWARGS[system]["train_bins"][0]
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    lams = jax.vmap(lambda k: env.sample_task(k).lam)(keys)
    lams_np = jax.device_get(lams)

    assert all(lo - 1e-4 <= v <= hi + 1e-4 for v in lams_np), (
        f"{system}: sampled task params outside train_bins {lo, hi}: {lams_np}"
    )
    assert len(set(round(float(v), 3) for v in lams_np)) >= 8, (
        f"{system}: task samples look degenerate, expected real spread: {lams_np}"
    )


# ── reset/step stay finite across several distinct sampled tasks ───────────


@pytest.mark.parametrize("system", _SYSTEM_PARAMS)
def test_reset_and_step_finite_across_sampled_tasks(system):
    env, _ = _make_env(system, max_steps=50)
    task_keys = jax.random.split(jax.random.PRNGKey(1), 8)
    for tk in task_keys:
        params = env.sample_task(tk)
        obs, state = env.reset(tk, params)
        assert jnp.all(jnp.isfinite(obs)), f"{system}: non-finite reset obs (lam={params.lam})"

        obs2, state2, reward, done, info = env.step(tk, state, jnp.array([0.0]), params)
        assert jnp.all(jnp.isfinite(obs2)), f"{system}: non-finite step obs (lam={params.lam})"
        assert jnp.isfinite(reward), f"{system}: non-finite reward (lam={params.lam})"


# ── vmapped reset over a batch of sampled tasks, as rollout collection uses ─


@pytest.mark.parametrize("system", _SYSTEM_PARAMS)
def test_vmap_reset_over_task_batch(system):
    env, _ = _make_env(system, max_steps=50)
    n_envs = 6
    task_keys = jax.random.split(jax.random.PRNGKey(2), n_envs)
    params_batch = jax.vmap(env.sample_task)(task_keys)
    obs_batch, _ = jax.vmap(env.reset)(task_keys, params_batch)

    assert obs_batch.shape == (n_envs, env.obs_shape()[0])
    assert jnp.all(jnp.isfinite(obs_batch))


# ── short VariBAD training runs over the task distribution ─────────────────


def _make_trainer(system, total_iters=6, num_envs=4, rollout_steps=15):
    env_cfg = ODEEnvConfig(**_SYSTEM_KWARGS[system])
    config = TrainConfig(
        env=env_cfg,
        eval_metrics=EvalMetricsConfig(enabled=False),  # no pre-generated PID cache here
        precompute_baseline=PrecomputeBaselineConfig(enabled=False),
        vae=VAEConfig(
            latent_dim=LATENT_DIM,
            encoder=EncoderArchConfig(hidden_size=32),
            batch_size=4,
            num_updates_per_iter=1,
        ),
        ppo=PPOConfig(num_epochs=1, num_minibatches=2, lr=1e-3),
        total_iters=total_iters,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        eval_interval=0,
        num_eval_episodes=3,
    )
    env = ODEEnv(env_cfg, max_steps=config.rollout_steps)
    obs_dim = env.obs_shape()[0]
    action_dim = env.num_actions

    vae = VariBADVAE(obs_dim, action_dim, config.vae, nnx.Rngs(0))
    policy = ActorCritic(
        obs_dim,
        action_dim,
        LATENT_DIM,
        action_space="continuous",
        use_latent_sample=False,
        rngs=nnx.Rngs(1),
    )
    # Template params only (obs/shape bookkeeping) — every rollout below
    # resamples a fresh per-env task from train_bins via env.sample_task.
    params = env.sample_task(jax.random.PRNGKey(42))
    trainer = VariBADTrainer(config, vae, policy, env, params)
    return trainer, config


@pytest.mark.parametrize("system", _SYSTEM_PARAMS)
def test_training_on_task_distribution_runs(system):
    trainer, _ = _make_trainer(system, total_iters=6)
    result = trainer.train(jax.random.PRNGKey(7))
    assert "metrics" in result
    assert len(result["metrics"]) == 6


@pytest.mark.parametrize("system", _SYSTEM_PARAMS)
def test_training_on_task_distribution_finite_metrics(system):
    trainer, _ = _make_trainer(system, total_iters=6)
    result = trainer.train(jax.random.PRNGKey(8))
    for metrics in result["metrics"]:
        for k, v in metrics.items():
            try:
                assert jnp.isfinite(jnp.array(float(v))), f"{system}: NaN/Inf in {k}={v}"
            except (TypeError, ValueError):
                pass
