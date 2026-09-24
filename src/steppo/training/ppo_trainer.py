"""Optimizer state and update routines for PPO policy training."""

from functools import partial
from typing import Any, Optional

import flax.nnx as nnx
import flax.struct
import jax
import jax.numpy as jnp
import optax
from jax import Array
from jax.random import PRNGKey

from steppo.configs.base_config import PPOConfig, TrainingConfig
from steppo.training.ppo import PPOBatch, ppo_loss
from steppo.utils.debug import maybe_jit
from steppo.utils.optim import current_lr


@flax.struct.dataclass
class PPOTrainState:
    """Pytree state for PPO parameters, optimizer, and loss EMAs."""

    graphdef: Any = flax.struct.field(pytree_node=False)
    params: Any
    opt_state: Any
    step: int
    ema_actor: Array
    ema_value: Array


def _ppo_lr_schedule(config: PPOConfig, total_ppo_steps: int):
    """Build the configured constant or cosine-decay learning-rate schedule."""
    if config.lr_end > 0 and total_ppo_steps > 0:
        return optax.cosine_decay_schedule(
            init_value=config.lr,
            decay_steps=total_ppo_steps,
            alpha=config.lr_end / config.lr,
        )
    return config.lr


def _build_ppo_optimizer(
    config: PPOConfig, training_config: TrainingConfig, total_ppo_steps: int = 0
):
    """Build PPO's gradient-clipping and Adam optimizer chain."""
    return optax.chain(
        optax.clip_by_global_norm(training_config.grad_clip_ppo),
        optax.adam(_ppo_lr_schedule(config, total_ppo_steps)),
    )


def create_ppo_train_state(
    policy,
    config: PPOConfig,
    training_config: Optional[TrainingConfig] = None,
    total_ppo_steps: int = 0,
) -> PPOTrainState:
    """Create initialized PPO parameters, optimizer state, and loss EMAs."""
    if training_config is None:
        training_config = TrainingConfig()
    optimizer = _build_ppo_optimizer(config, training_config, total_ppo_steps)
    graphdef, params = nnx.split(policy)
    opt_state = optimizer.init(params)
    return PPOTrainState(
        graphdef=graphdef,
        params=params,
        opt_state=opt_state,
        step=0,
        ema_actor=jnp.float32(0.0),
        ema_value=jnp.float32(0.0),
    )


@partial(maybe_jit, static_argnums=(2, 4, 5))
def ppo_train_step(
    train_state: PPOTrainState,
    batch: PPOBatch,
    config: PPOConfig,
    rng_key: PRNGKey,
    training_config: Optional[TrainingConfig] = None,
    total_ppo_steps: int = 0,
) -> tuple[PPOTrainState, dict]:
    """Apply one PPO gradient update and return metrics for that update."""
    if training_config is None:
        training_config = TrainingConfig()
    optimizer = _build_ppo_optimizer(config, training_config, total_ppo_steps)

    alpha = training_config.ema_alpha
    t = jnp.float32(train_state.step + 1)
    bc = 1.0 - jnp.power(jnp.float32(alpha), t)
    ema = (
        train_state.ema_actor / jnp.maximum(bc, 1e-8),
        train_state.ema_value / jnp.maximum(bc, 1e-8),
    )

    ppo_step = train_state.step

    def loss_fn(params):
        policy = nnx.merge(train_state.graphdef, params)
        return ppo_loss(policy, batch, config, rng_key, training_config, ema)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)

    def grad_update(_grads, _train_state, _training_config):
        _metrics = {}
        _metrics["grad_norm"] = optax.global_norm(_grads)
        if training_config.log_grad_norms:
            for term in ("norm_actor_loss", "norm_value_loss", "norm_entropy_loss"):
                term_grads = jax.grad(lambda p, t=term: loss_fn(p)[1][t])(_train_state.params)
                _metrics[f"grad_norm_{term.removeprefix('norm_').removesuffix('_loss')}"] = (
                    optax.global_norm(term_grads)
                )
        _metrics["grad_norm_post_clip"] = jnp.minimum(
            _metrics["grad_norm"], _training_config.grad_clip_ppo
        )
        _metrics["lr"] = current_lr(_ppo_lr_schedule(config, total_ppo_steps), ppo_step)

        updates, new_opt_state = optimizer.update(
            _grads, _train_state.opt_state, _train_state.params
        )
        new_params = optax.apply_updates(_train_state.params, updates)

        return _metrics, new_params, new_opt_state

    grad_metrics, new_params, new_opt_state = grad_update(grads, train_state, training_config)
    metrics = {**metrics, **grad_metrics}

    cap = training_config.ema_input_cap
    ema_floor = jnp.float32(1e-2)
    capped_actor = jnp.minimum(
        jnp.abs(metrics["actor_loss"]), cap * jnp.maximum(train_state.ema_actor, ema_floor)
    )
    capped_value = jnp.minimum(
        metrics["value_loss"], cap * jnp.maximum(train_state.ema_value, ema_floor)
    )
    new_ema_actor = alpha * train_state.ema_actor + (1.0 - alpha) * capped_actor
    new_ema_value = alpha * train_state.ema_value + (1.0 - alpha) * capped_value

    new_state = PPOTrainState(
        graphdef=train_state.graphdef,
        params=new_params,
        opt_state=new_opt_state,
        step=train_state.step + 1,
        ema_actor=new_ema_actor,
        ema_value=new_ema_value,
    )

    return new_state, metrics
