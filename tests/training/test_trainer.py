"""Tests for steppo.training.trainer.VariBADTrainer (integration tests)."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from steppo.configs.base_config import (
    EncoderArchConfig,
    PPOConfig,
    PrecomputeBaselineConfig,
    TrainConfig,
    VAEConfig,
)
from steppo.models.policy import ActorCritic
from steppo.models.vae import VariBADVAE
from steppo.training.trainer import VariBADTrainer

LATENT_DIM = 5


def _make_ode_trainer(
    total_iters=20, eval_interval=0, num_envs=2, rollout_steps=10, vae_batch_size=4
):
    from steppo.configs.base_config import EvalMetricsConfig, ODEEnvConfig
    from steppo.envs.ode import ODEEnv, ODEParams

    ODE_OBS_DIM = 10  # default obs_features=("state","step_context","solver_trend") -> 1+4+5
    ODE_ACTION_DIM = 1

    config = TrainConfig(
        env=ODEEnvConfig(
            system="scalar_decay", train_bins=((1.0, 100.0),), val_bins=(), test_bins=()
        ),
        eval_metrics=EvalMetricsConfig(enabled=False),  # no pre-generated PID cache in this test
        precompute_baseline=PrecomputeBaselineConfig(
            enabled=False
        ),  # same — no pre-generated PID cache
        vae=VAEConfig(
            latent_dim=LATENT_DIM,
            encoder=EncoderArchConfig(hidden_size=64),
            batch_size=vae_batch_size,
            num_updates_per_iter=1,
        ),
        ppo=PPOConfig(num_epochs=1, num_minibatches=2, lr=1e-3),
        total_iters=total_iters,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        eval_interval=eval_interval,
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
    return trainer, config


def test_eval_metrics_appear_in_returned_metrics(monkeypatch):
    """Evaluation output is merged into the returned and accumulated metrics."""
    import steppo.training.base_trainer as base_trainer

    trainer, config = _make_ode_trainer(total_iters=6, eval_interval=5)
    (vae_state, _), (policy_state, _), logger = trainer._init(config)
    trainer._efficiency_datasets = {"val": object()}
    monkeypatch.setattr(
        base_trainer,
        "eval_policy",
        lambda *args, **kwargs: {
            "mean_return": 1.5,
            "success_rate": 0.5,
        },
    )

    metrics = trainer._log_and_evaluate(
        info={"iteration": 5},
        states={"vae_state": vae_state, "policy_state": policy_state},
        keys={"key_eval": jax.random.PRNGKey(3)},
        logger=logger,
        cfg=config,
    )

    assert metrics["eval/val/mean_return"] == 1.5
    assert metrics["eval/val/success_rate"] == 0.5
    assert trainer._all_metrics[-1] == metrics


def test_vae_buffer_wait_for_enough_data():
    """VAE training should not start until buffer has enough trajectories."""
    trainer, config = _make_ode_trainer(
        total_iters=3,
        eval_interval=0,
        rollout_steps=5,
        vae_batch_size=100,
    )
    result = trainer.train(jax.random.PRNGKey(7))
    early_vae = [m.get("vae/total_loss") for m in result["metrics"][:3]]
    assert all(v is None for v in early_vae), f"VAE trained before buffer was ready: {early_vae}"


# ── ODE training loop ───────────────────────────────────────────────────────


def test_ode_training_runs():
    trainer, _ = _make_ode_trainer(total_iters=15)
    result = trainer.train(jax.random.PRNGKey(0))
    assert "metrics" in result
    assert len(result["metrics"]) == 15


def test_ode_training_finite_metrics():
    trainer, _ = _make_ode_trainer(total_iters=15)
    result = trainer.train(jax.random.PRNGKey(1))
    for metrics in result["metrics"]:
        for k, v in metrics.items():
            try:
                assert jnp.isfinite(jnp.array(float(v))), f"NaN/Inf in {k}={v}"
            except (TypeError, ValueError):
                pass
