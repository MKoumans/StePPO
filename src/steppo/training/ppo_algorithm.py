"""On-policy PPO implementation of the policy-algorithm interface."""

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

from steppo.configs.base_config import TrainConfig
from steppo.models.policy import ActorCritic
from steppo.training.policy_algos import PolicyAlgorithm
from steppo.training.ppo import compute_gae
from steppo.training.ppo_trainer import PPOTrainState, create_ppo_train_state, ppo_train_step
from steppo.training.reward_norm import reward_scale_for_mu
from steppo.utils.policy_buffer import PolicyBuffer


def total_ppo_steps_for(cfg: TrainConfig) -> int:
    """Return the total optimizer steps implied by the training config."""
    return cfg.total_iters * cfg.ppo.num_epochs * cfg.ppo.num_minibatches


class PPOAlgorithm(PolicyAlgorithm):
    """On-policy PPO: consumes one just-collected rollout per update, then
    discards it (see `PolicyBuffer`)."""

    name = "ppo"
    checkpoint_prefix = "ppo"
    supports_warmstart = True

    @staticmethod
    def build_model(
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        task_dim: int,
        rngs,
        cfg: TrainConfig,
        use_latent_sample: bool,
    ) -> ActorCritic:
        """Build the continuous-action actor-critic used by PPO."""
        return ActorCritic(
            obs_dim,
            action_dim,
            latent_dim,
            arch=cfg.ppo.policy,
            action_space="continuous",
            use_latent_sample=use_latent_sample,
            task_dim=task_dim,
            rngs=rngs,
        )

    @staticmethod
    def init_state(policy, cfg: TrainConfig) -> tuple[PPOTrainState, PolicyBuffer]:
        """Create PPO optimizer state and its per-rollout policy buffer."""
        total_ppo_steps = total_ppo_steps_for(cfg)
        policy_state = create_ppo_train_state(policy, cfg.ppo, cfg.training, total_ppo_steps)
        policy_buffer = PolicyBuffer()
        return policy_state, policy_buffer

    @staticmethod
    def update(
        trainer: Any,
        cfg: TrainConfig,
        ppo_state: PPOTrainState,
        policy_buffer: PolicyBuffer,
        key_ppo: PRNGKey,
        batch: Any,
        iteration: int,
    ) -> tuple[PPOTrainState, dict]:
        """Update PPO from one rollout and return metrics from its last step."""
        total_ppo_steps = total_ppo_steps_for(cfg)

        t_ppo_start = time.perf_counter()
        p = trainer.profiler

        with p.section("ppo.gae_prep"):
            if cfg.episodes_per_trial > 0:
                frozen = (batch.keep_steps < 0.5) & (batch.dones > 0.5)
                frozen_next = jnp.concatenate(
                    [frozen[1:], jnp.zeros((1, frozen.shape[1]), dtype=bool)], axis=0
                )
                dones_for_gae = (frozen | (batch.dones.astype(bool) & frozen_next)).astype(
                    batch.dones.dtype
                )
                active_mask = ~frozen
            else:
                dones_for_gae = batch.dones
                active_mask = None

        with p.section("ppo.compute_gae"):
            rewards = batch.rewards.squeeze(-1)
            reward_scale = None
            if trainer._reward_norm is not None and batch.task_params.shape[-1] == 1:
                # task_params holds μ normalized to [0,1]; invert to raw μ and
                # scale each env's rewards by PID_steps(μ)/C so per-step reward
                # density is comparable across task regimes.
                lo, hi = trainer.env._task_bounds()
                mu_raw = lo + batch.task_params[:, 0] * (hi - lo)
                reward_scale = reward_scale_for_mu(mu_raw, *trainer._reward_norm)
                rewards = rewards * reward_scale[None, :]

            advantages, returns = compute_gae(
                rewards,
                jnp.concatenate([batch.values, batch.bootstrap_value[None]], axis=0),
                dones_for_gae,
                cfg.ppo.gamma,
                cfg.ppo.gae_lambda,
            )

        # Accumulate per-env advantage magnitude between eval checkpoints.
        # Captured here while advantages still have shape (T, num_envs) and
        # task_params carries the μ value for each env.
        if batch.task_params.shape[-1] > 0:
            trainer._adv_stats_buffer[0].append(np.asarray(jnp.std(advantages, axis=0)))
            trainer._adv_stats_buffer[1].append(np.asarray(batch.task_params[:, 0]))

        with p.section("ppo.buffer_store"):
            T = batch.states.shape[0]
            task_broadcast = jnp.broadcast_to(
                batch.task_params[None], (T,) + batch.task_params.shape
            )
            policy_buffer.store(
                batch.states,
                jnp.concatenate([batch.beliefs_mu, batch.beliefs_logvar], axis=-1),
                task_broadcast,
                batch.actions,
                batch.log_probs,
                advantages,
                returns,
                active_mask=active_mask,
            )

        ppo_metrics_iter = {}
        target_kl = cfg.ppo.target_kl
        kl_exceeded = False
        epochs_run = 0
        for epoch in range(cfg.ppo.num_epochs):
            if kl_exceeded:
                break
            key_epoch = jax.random.fold_in(key_ppo, epoch)
            with p.section("ppo.get_minibatches"):
                minibatches = list(
                    policy_buffer.get_minibatches(cfg.ppo.num_minibatches, key_epoch)
                )
            for mb in minibatches:
                with p.section("ppo.train_step"):
                    ppo_state, ppo_metrics_iter = ppo_train_step(
                        ppo_state, mb, cfg.ppo, key_epoch, cfg.training, total_ppo_steps
                    )
                if target_kl > 0 and float(ppo_metrics_iter["approx_kl"]) > target_kl:
                    kl_exceeded = True
                    break
            epochs_run += 1
        p.block_and_record("ppo.train_step", ppo_state.params)
        ppo_metrics_iter["kl_early_stopped"] = float(kl_exceeded)
        ppo_metrics_iter["ppo_epochs_run"] = float(epochs_run)

        if reward_scale is not None:
            ppo_metrics_iter["reward_scale_mean"] = jnp.mean(reward_scale)
            ppo_metrics_iter["reward_scale_max"] = jnp.max(reward_scale)
            ppo_metrics_iter["norm_return_mean"] = jnp.mean(jnp.sum(rewards, axis=0))

        trainer.timings.update(ppo_time=time.perf_counter() - t_ppo_start)

        return ppo_state, ppo_metrics_iter

    @staticmethod
    def checkpoint_fields(policy_state: PPOTrainState) -> dict:
        """Return PPO state fields in the checkpoint schema."""
        prefix = PPOAlgorithm.checkpoint_prefix
        return {
            f"{prefix}_params": policy_state.params,
            f"{prefix}_opt_state": policy_state.opt_state,
            f"{prefix}_step": policy_state.step,
        }
