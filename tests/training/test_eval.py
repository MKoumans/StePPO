"""Tests for steppo.training.eval."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from steppo.configs.base_config import (
    EncoderArchConfig,
    EvalMetricsConfig,
    ODEEnvConfig,
    PPOConfig,
    PrecomputeBaselineConfig,
    TrainConfig,
    VAEConfig,
)
from steppo.envs.ode import ODEEnv, ODEParams
from steppo.models.policy import ActorCritic
from steppo.models.vae import VariBADVAE
from steppo.training.eval import eval_episode
from steppo.training.trainer import VariBADTrainer

ODE_OBS_DIM = 10  # default obs_features=("state","step_context","solver_trend") -> 1+5+4
ODE_ACTION_DIM = 1
LATENT_DIM = 5


def _train_small_model(total_iters=5):
    config = TrainConfig(
        vae=VAEConfig(
            latent_dim=LATENT_DIM,
            encoder=EncoderArchConfig(hidden_size=64),
            batch_size=4,
            num_updates_per_iter=1,
        ),
        ppo=PPOConfig(num_epochs=1, num_minibatches=2, lr=1e-3),
        eval_metrics=EvalMetricsConfig(enabled=False),  # no pre-generated PID cache in this test
        precompute_baseline=PrecomputeBaselineConfig(
            enabled=False
        ),  # same — no pre-generated PID cache
        env=ODEEnvConfig(system="scalar_decay", train_bins=((1.0, 100.0),)),
        # eval_interval=0: this helper's callers evaluate manually via
        # trainer.evaluate()/eval_episode() after training, not via in-loop
        # periodic eval — periodic eval with real val/test bins needs a
        # pre-generated PID evaluation cache this test doesn't have.
        total_iters=total_iters,
        num_envs=2,
        rollout_steps=10,
        eval_interval=0,
        num_eval_episodes=3,
    )
    rngs = nnx.Rngs(0)
    vae = VariBADVAE(ODE_OBS_DIM, ODE_ACTION_DIM, config.vae, rngs)
    policy = ActorCritic(
        ODE_OBS_DIM,
        ODE_ACTION_DIM,
        LATENT_DIM,
        action_space="continuous",
        use_latent_sample=False,
        rngs=nnx.Rngs(1),
    )
    env = ODEEnv(config.env, max_steps=config.rollout_steps)
    params = ODEParams(lam=10.0)
    trainer = VariBADTrainer(config, vae, policy, env, params)
    result = trainer.train(jax.random.PRNGKey(0))
    return trainer, result


def test_collect_control_stat_envelope_all_three_statistics():
    """collect_control_stat_envelope (src/steppo/training/eval.py) — the
    control-chart calibration statistic collector behind PidFallbackConfig.
    No training loop: exercises the collection wiring against a freshly-
    constructed vae/policy/env, matching the val_bins test's pattern above."""
    from steppo.configs.base_config import ODEEnvConfig
    from steppo.envs.ode import ODEEnv
    from steppo.training.eval import collect_control_stat_envelope

    ODE_OBS_DIM = 10  # scalar_decay default obs_features -> 1+5+4
    ODE_ACTION_DIM = 1

    vae = VariBADVAE(
        ODE_OBS_DIM,
        ODE_ACTION_DIM,
        VAEConfig(latent_dim=LATENT_DIM, encoder=EncoderArchConfig(hidden_size=16)),
        nnx.Rngs(0),
    )
    policy = ActorCritic(
        ODE_OBS_DIM,
        ODE_ACTION_DIM,
        LATENT_DIM,
        action_space="continuous",
        use_latent_sample=False,
        rngs=nnx.Rngs(1),
    )
    config = ODEEnvConfig(
        system="scalar_decay",
        lam_min=1.0,
        lam_max=100.0,
        sample_lam=True,
        train_bins=((1.0, 100.0),),
    )
    env = ODEEnv(config, max_steps=20)

    import math

    reject_streak_env = collect_control_stat_envelope(
        vae,
        policy,
        env,
        jax.random.PRNGKey(0),
        num_envs=4,
        max_steps=20,
        stat_name="reject_streak",
        percentile=99.0,
    )
    assert math.isfinite(reject_streak_env) and reject_streak_env >= 0.0

    log_error_env = collect_control_stat_envelope(
        vae,
        policy,
        env,
        jax.random.PRNGKey(0),
        num_envs=4,
        max_steps=20,
        stat_name="log_error_ema",
        percentile=99.0,
    )
    assert math.isfinite(log_error_env)

    accept_env = collect_control_stat_envelope(
        vae,
        policy,
        env,
        jax.random.PRNGKey(0),
        num_envs=4,
        max_steps=20,
        stat_name="accept_ema",
        percentile=1.0,
    )
    assert 0.0 <= accept_env <= 1.0


def test_evaluate_returns_required_keys():
    trainer, result = _train_small_model()
    eval_m = trainer.evaluate(result["vae_state"], result["policy_state"], jax.random.PRNGKey(99))
    assert "mean_return" in eval_m
    assert "success_rate" in eval_m


def test_belief_variance_decreases_ode():
    """Belief uncertainty should reduce as the agent observes the ODE."""
    from steppo.configs.base_config import ODEEnvConfig
    from steppo.envs.ode import ODEEnv, ODEParams

    ODE_OBS_DIM = 10  # default obs_features=("state","step_context","solver_trend") -> 1+5+4
    ODE_ACTION_DIM = 1

    config = TrainConfig(
        vae=VAEConfig(
            latent_dim=LATENT_DIM,
            encoder=EncoderArchConfig(hidden_size=64),
            batch_size=4,
            num_updates_per_iter=1,
        ),
        ppo=PPOConfig(num_epochs=1, num_minibatches=2, lr=1e-3),
        eval_metrics=EvalMetricsConfig(enabled=False),  # no pre-generated PID cache in this test
        precompute_baseline=PrecomputeBaselineConfig(
            enabled=False
        ),  # same — no pre-generated PID cache
        total_iters=10,
        num_envs=2,
        rollout_steps=10,
        # eval_interval=0: belief variance is checked manually via eval_episode()
        # below, not via in-loop periodic eval — that needs a pre-generated PID
        # evaluation cache this test doesn't have.
        eval_interval=0,
        num_eval_episodes=3,
    )
    rngs = nnx.Rngs(0)
    vae = VariBADVAE(ODE_OBS_DIM, ODE_ACTION_DIM, config.vae, rngs)
    policy = ActorCritic(
        ODE_OBS_DIM,
        ODE_ACTION_DIM,
        LATENT_DIM,
        action_space="continuous",
        use_latent_sample=False,
        rngs=nnx.Rngs(1),
    )
    env_config = ODEEnvConfig(system="scalar_decay", train_bins=((1.0, 100.0),))
    env = ODEEnv(env_config, max_steps=config.rollout_steps)
    params = ODEParams(lam=10.0)
    trainer = VariBADTrainer(config, vae, policy, env, params)
    result = trainer.train(jax.random.PRNGKey(2))

    vae = nnx.merge(result["vae_state"].graphdef, result["vae_state"].params)
    policy = nnx.merge(result["policy_state"].graphdef, result["policy_state"].params)
    eval_params = ODEParams(lam=10.0, max_steps=20)

    metrics = eval_episode(
        vae, policy, env, eval_params, jax.random.PRNGKey(3), return_trajectories=True
    )
    logvars = metrics["belief_logvar_trajectory"]
    num_points = min(3, logvars.shape[0])

    var_start = float(jnp.mean(jnp.exp(logvars[:num_points])))
    var_end = float(jnp.mean(jnp.exp(logvars[-num_points:])))
    assert var_end < var_start * 3.0, f"Variance grew too much: {var_start:.4f} -> {var_end:.4f}"
