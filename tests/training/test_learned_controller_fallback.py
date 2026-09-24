"""Tests for the disabled control-chart PID fallback contract.

The fallback is intentionally hard-disabled in the current controller. This
file preserves that contract without testing the retired trigger logic."""

import dataclasses

import flax.nnx as nnx
import jax.numpy as jnp

from steppo.configs.base_config import (
    EncoderArchConfig,
    ODEEnvConfig,
    PidFallbackConfig,
    PPOConfig,
    TrainConfig,
    VAEConfig,
)
from steppo.envs.ode import ODEEnv
from steppo.envs.ode.learned_controller import LearnedController
from steppo.models.policy import ActorCritic
from steppo.models.vae import VariBADVAE

OBS_DIM = 10  # scalar_decay default obs_features -> 1+5+4
ACTION_DIM = 1
LATENT_DIM = 5


def _make_controller(control_envelopes=None, multipliers=None):
    vae = VariBADVAE(
        OBS_DIM,
        ACTION_DIM,
        VAEConfig(latent_dim=LATENT_DIM, encoder=EncoderArchConfig(hidden_size=16)),
        nnx.Rngs(0),
    )
    policy = ActorCritic(
        OBS_DIM,
        ACTION_DIM,
        LATENT_DIM,
        action_space="continuous",
        use_latent_sample=False,
        rngs=nnx.Rngs(1),
    )
    pid_fallback = PidFallbackConfig(enabled=bool(control_envelopes))
    if multipliers:
        pid_fallback = dataclasses.replace(
            pid_fallback, **{f"{k}_multiplier": v for k, v in multipliers.items()}
        )
    config = TrainConfig(
        vae=VAEConfig(latent_dim=LATENT_DIM, encoder=EncoderArchConfig(hidden_size=16)),
        ppo=PPOConfig(),
        env=ODEEnvConfig(system="scalar_decay", lam_min=1.0, lam_max=100.0, sample_lam=True),
        rollout_steps=20,
        pid_fallback=pid_fallback,
    )
    env = ODEEnv(config.env, max_steps=20)
    return LearnedController.from_models(
        vae, policy, config, env, control_envelopes=control_envelopes
    )


def _init_and_force_reject(controller, prior_reject_streak=0, prior_active=False):
    """Runs init() then adapt_step_size() once with a huge y_error, guaranteeing
    a rejected step (keep_step=False) so new_reject_streak = prior + 1 —
    a fully controllable way to drive the reject_streak statistic."""
    t0 = jnp.float32(0.0)
    y0 = jnp.zeros((1,), dtype=jnp.float32)
    dt0 = jnp.float32(0.1)
    args = jnp.array([10.0, 0.0], dtype=jnp.float32)  # [lam, pulse_phase]

    _, state0 = controller.init(
        terms=None,
        t0=t0,
        t1=None,
        y0=y0,
        dt0=dt0,
        args=args,
        func=None,
        error_order=None,
    )
    state0 = dataclasses.replace(
        state0,
        reject_streak=jnp.int32(prior_reject_streak),
        pid_fallback_active=jnp.bool_(prior_active),
    )

    huge_error = jnp.array(
        [1.0], dtype=jnp.float32
    )  # scale ~= atol (tiny) -> scaled_error huge -> reject
    keep_step, new_t0, new_t1, made_jump, new_state, results = controller.adapt_step_size(
        t0=t0,
        t1=t0 + dt0,
        y0=y0,
        y1_candidate=y0,
        args=args,
        y_error=huge_error,
        error_order=jnp.float32(4.0),
        controller_state=state0,
    )
    assert not bool(keep_step)  # sanity: the step really was rejected
    return new_state


def test_configured_envelopes_do_not_activate_disabled_pid_fallback():
    """PID fallback is currently disabled even when control envelopes trip."""
    controller = _make_controller(
        control_envelopes={"reject_streak": 0.0},
        multipliers={"reject_streak": 1.0},
    )
    state = _init_and_force_reject(controller, prior_reject_streak=0)

    assert not bool(state.pid_fallback_active)
