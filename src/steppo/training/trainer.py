"""Trainer subclass for the variational VariBAD belief model."""

import time
from dataclasses import dataclass
from typing import Any

import jax
from jax.random import PRNGKey

from steppo.configs.base_config import TrainConfig
from steppo.training.base_trainer import Trainer
from steppo.training.vae_trainer import VAETrainState, create_vae_train_state, vae_train_step
from steppo.utils.replay_buffer import VAEReplayBuffer, get_trajectory_batch


@dataclass
class VariBADTrainer(Trainer):
    """VariBAD backbone: GRU encoder + reconstruction/KL (ELBO) training."""

    def _init_belief_model(self, cfg: TrainConfig):
        """Create the VariBAD VAE state and trajectory replay buffer."""
        self.total_vae_steps = cfg.total_iters * cfg.vae.num_updates_per_iter

        vae_state = create_vae_train_state(self.vae, cfg.vae, cfg.training, self.total_vae_steps)

        obs_dim = self.env.obs_shape(self.env_params)[0]
        # Buffer width must match whatever rollout.py's _task_batch_for actually produces
        # (env.task_dim whenever the env exposes get_task_params), independent of whether
        # task_loss_coeff/encoder_inputs/policy_inputs happen to consume it downstream —
        # otherwise VAEReplayBuffer.insert's task_params assignment shape-mismatches.
        task_dim = getattr(self.env, "task_dim", 0)
        buf_cap = max(1024, cfg.vae.batch_size * cfg.vae.buffer_capacity_multiplier)
        vae_buffer = VAEReplayBuffer(
            capacity=buf_cap,
            trajectory_len=cfg.rollout_steps,
            state_dim=obs_dim,
            action_dim=self.policy.action_dim,
            task_dim=task_dim,
        )
        return vae_state, vae_buffer

    def _update_belief_model(
        self,
        cfg: TrainConfig,
        vae_state: VAETrainState,
        vae_buffer: VAEReplayBuffer,
        key_vae: PRNGKey,
        batch: Any,
    ) -> tuple[VAETrainState, dict]:
        """Insert the rollout and perform the configured VAE updates."""
        p = self.profiler

        with p.section("vae.get_trajectory_batch"):
            traj = get_trajectory_batch(batch, self.policy.action_dim)

        with p.section("vae.buffer_insert"):
            vae_buffer.insert(traj)

        if vae_buffer.size < cfg.vae.batch_size:
            return vae_state, {}

        t_vae_start = time.perf_counter()

        vae_metrics_iter = {}
        for vae_upd in range(cfg.vae.num_updates_per_iter):
            key_vae_u = jax.random.fold_in(key_vae, vae_upd)
            with p.section("vae.buffer_sample"):
                traj_batch = vae_buffer.sample(cfg.vae.batch_size, key_vae_u)
            with p.section("vae.train_step"):
                vae_state, vae_metrics_iter = vae_train_step(
                    vae_state, traj_batch, key_vae_u, cfg.vae, cfg.training, self.total_vae_steps
                )
        p.block_and_record("vae.train_step", vae_state.params)

        self.timings.update(vae_time=time.perf_counter() - t_vae_start)
        return vae_state, vae_metrics_iter

    def _belief_checkpoint_fields(self, vae_state: VAETrainState) -> dict:
        """Return VariBAD state fields in the checkpoint schema."""
        return {
            "vae_params": vae_state.params,
            "vae_opt_state": vae_state.opt_state,
            "vae_step": vae_state.step,
        }
