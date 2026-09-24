"""Training state and update routines for the VariBAD VAE."""

from functools import partial
from typing import Any, Optional, Tuple

import flax.nnx as nnx
import flax.struct
import jax
import jax.numpy as jnp
import optax
from jax import Array
from jax.random import PRNGKey

from steppo.configs.base_config import TrainingConfig, VAEConfig
from steppo.training.vae_loss import elbo_loss
from steppo.utils.debug import maybe_jit
from steppo.utils.optim import current_lr
from steppo.utils.replay_buffer import TrajectoryBatch


@flax.struct.dataclass
class VAETrainState:
    """Pytree state for VAE parameters, optimizer, and loss EMAs."""

    graphdef: Any = flax.struct.field(pytree_node=False)  # graphdef is a
    params: Any  #
    opt_state: Any  #
    step: int  #
    ema_rew: Array  #
    ema_state: Array  #
    ema_kl: Array  #
    ema_task: Array  #
    ema_accept: Array  #
    ema_end_reward: Array  #


def _vae_lr_schedule(config: VAEConfig, total_vae_steps: int):
    """Creates a learning rate schedule for the VAE optimizer."""
    if config.lr_end > 0 and total_vae_steps > 0:
        return optax.cosine_decay_schedule(
            init_value=config.lr,
            decay_steps=total_vae_steps,
            alpha=config.lr_end / config.lr,
        )
    return config.lr


def _build_vae_optimizer(
    config: VAEConfig, training_config: TrainingConfig, total_vae_steps: int = 0
):
    """Build the VAE's gradient-clipping and Adam optimizer chain."""
    return optax.chain(
        optax.clip_by_global_norm(training_config.grad_clip_vae),
        optax.adam(_vae_lr_schedule(config, total_vae_steps)),
    )


def create_vae_train_state(
    vae,
    config: VAEConfig,
    training_config: Optional[TrainingConfig] = None,
    total_vae_steps: int = 0,
) -> VAETrainState:
    """Create initialized VAE parameters, optimizer state, and loss EMAs."""
    if training_config is None:
        training_config = TrainingConfig()

    optimizer = _build_vae_optimizer(config, training_config, total_vae_steps)
    graphdef, params = nnx.split(
        vae
    )  # Necessary to NNX and JAX transformations (see https://flax.readthedocs.io/en/latest/flax.nnx.html#flax.nnx.split)
    opt_state = optimizer.init(params)

    return VAETrainState(
        graphdef=graphdef,
        params=params,
        opt_state=opt_state,
        step=0,
        ema_rew=jnp.float32(0.0),
        ema_state=jnp.float32(0.0),
        ema_kl=jnp.float32(0.0),
        ema_task=jnp.float32(0.0),
        ema_accept=jnp.float32(0.0),
        ema_end_reward=jnp.float32(0.0),
    )


@partial(maybe_jit, static_argnums=(3, 4, 5))
def vae_train_step(
    train_state: VAETrainState,
    trajectories: TrajectoryBatch,
    rng_key: PRNGKey,
    config: VAEConfig,
    training_config: Optional[TrainingConfig] = None,
    total_vae_steps: int = 0,
) -> Tuple[VAETrainState, dict]:
    """Apply one ELBO update and return the updated state and metrics."""
    if training_config is None:
        training_config = TrainingConfig()
    optimizer = _build_vae_optimizer(config, training_config, total_vae_steps)

    # Bias-corrected EMAs (Adam-style): ema starts at 0, bias correction makes
    # the estimate correct from step 1 without a slow burn-in from a 1.0 init.

    alpha = training_config.ema_alpha
    t = jnp.float32(train_state.step + 1)
    bc = 1.0 - jnp.power(jnp.float32(alpha), t)
    ema = (
        train_state.ema_rew / jnp.maximum(bc, 1e-8),
        train_state.ema_state / jnp.maximum(bc, 1e-8),
        train_state.ema_kl / jnp.maximum(bc, 1e-8),
        train_state.ema_task / jnp.maximum(bc, 1e-8),
        train_state.ema_accept / jnp.maximum(bc, 1e-8),
        train_state.ema_end_reward / jnp.maximum(bc, 1e-8),
    )

    if training_config.kl_anneal_frac > 0 and total_vae_steps > 0:
        kl_anneal_iters = training_config.kl_anneal_frac * total_vae_steps
        kl_anneal_factor = jnp.minimum(t / kl_anneal_iters, 1.0)
    else:
        kl_anneal_factor = 1.0

    def loss_fn(params):
        vae = nnx.merge(train_state.graphdef, params)
        loss, metrics = elbo_loss(
            vae, trajectories, rng_key, config, training_config, ema, kl_anneal_factor
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    metrics["grad_norm"] = optax.global_norm(grads)
    metrics["grad_norm_post_clip"] = jnp.minimum(
        metrics["grad_norm"], training_config.grad_clip_vae
    )
    metrics["lr"] = current_lr(_vae_lr_schedule(config, total_vae_steps), train_state.step)

    updates, new_opt_state = optimizer.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    # Update EMA with the raw (pre-normalisation) loss values.
    new_ema_rew = alpha * train_state.ema_rew + (1.0 - alpha) * metrics["rew_loss"]
    new_ema_state = alpha * train_state.ema_state + (1.0 - alpha) * metrics["state_loss"]
    new_ema_kl = alpha * train_state.ema_kl + (1.0 - alpha) * metrics["kl_loss"]
    new_ema_task = alpha * train_state.ema_task + (1.0 - alpha) * metrics["task_loss"]
    new_ema_accept = alpha * train_state.ema_accept + (1.0 - alpha) * metrics["accept_loss"]
    new_ema_end_reward = (
        alpha * train_state.ema_end_reward + (1.0 - alpha) * metrics["end_reward_loss"]
    )

    new_state = VAETrainState(
        graphdef=train_state.graphdef,
        params=new_params,
        opt_state=new_opt_state,
        step=train_state.step + 1,
        ema_rew=new_ema_rew,
        ema_state=new_ema_state,
        ema_kl=new_ema_kl,
        ema_task=new_ema_task,
        ema_accept=new_ema_accept,
        ema_end_reward=new_ema_end_reward,
    )

    return new_state, metrics
