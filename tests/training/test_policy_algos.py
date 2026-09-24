"""Tests for steppo.training.policy_algos / ppo_algorithm.py.

Covers the policy-algorithm registry (mirrors test coverage of
src/steppo/models/backbones.py's belief-model registry) and PPOAlgorithm's
build_model -> init_state -> update -> checkpoint_fields round trip.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from steppo.configs.base_config import PPOConfig, TrainConfig
from steppo.models.policy import ActorCritic
from steppo.training.policy_algos import PolicyAlgoSpec, get_policy_algo
from steppo.training.ppo_algorithm import PPOAlgorithm
from steppo.training.ppo_trainer import PPOTrainState
from steppo.training.rollout import RolloutBatch
from steppo.utils.logger import TimingProfiler, Timings
from steppo.utils.policy_buffer import PolicyBuffer

STATE_DIM = 2
ACTION_DIM = 3
LATENT_DIM = 5
Z_DIM = LATENT_DIM * 2
T = 6
NUM_ENVS = 4


class _FakeEnv:
    def _task_bounds(self):
        return 1.0, 100.0


class _FakeTrainer:
    """Minimal stand-in exposing exactly what PPOAlgorithm.update reads off
    the Trainer (profiler, timings, env, reward-norm/adv-stats bookkeeping)."""

    def __init__(self):
        self.profiler = TimingProfiler(enabled=False)
        self.timings = Timings()
        self.env = _FakeEnv()
        self._reward_norm = None
        self._adv_stats_buffer = ([], [])


def _make_batch(task_dim=0):
    key = jax.random.PRNGKey(0)
    ks = jax.random.split(key, 8)
    return RolloutBatch(
        states=jax.random.normal(ks[0], (T, NUM_ENVS, STATE_DIM)),
        actions=jax.random.normal(ks[1], (T, NUM_ENVS, ACTION_DIM)),
        rewards=jax.random.normal(ks[2], (T, NUM_ENVS, 1)),
        next_states=jax.random.normal(ks[3], (T, NUM_ENVS, STATE_DIM)),
        dones=jnp.zeros((T, NUM_ENVS)),
        beliefs_mu=jnp.zeros((T, NUM_ENVS, LATENT_DIM)),
        beliefs_logvar=jnp.zeros((T, NUM_ENVS, LATENT_DIM)),
        log_probs=jax.random.normal(ks[4], (T, NUM_ENVS)),
        values=jax.random.normal(ks[5], (T, NUM_ENVS)),
        bootstrap_value=jax.random.normal(ks[6], (NUM_ENVS,)),
        task_params=jnp.zeros((NUM_ENVS, task_dim)),
        keep_steps=jnp.ones((T, NUM_ENVS)),
    )


# ── registry ─────────────────────────────────────────────────────────────────


def test_get_policy_algo_ppo():
    spec = get_policy_algo("ppo")
    assert isinstance(spec, PolicyAlgoSpec)
    assert spec.name == "ppo"
    assert spec.algo_cls is PPOAlgorithm
    assert spec.config_attr == "ppo"
    assert spec.checkpoint_prefix == "ppo"


def test_get_policy_algo_unknown_raises():
    with pytest.raises(ValueError, match="Unknown policy algorithm"):
        get_policy_algo("bogus")


def test_trainconfig_algo_defaults_to_ppo():
    assert TrainConfig().algo == "ppo"


# ── PPOAlgorithm round trip ───────────────────────────────────────────────────


def test_ppo_algorithm_build_model_shape():
    model = PPOAlgorithm.build_model(
        STATE_DIM,
        ACTION_DIM,
        LATENT_DIM,
        0,
        rngs=nnx.Rngs(0),
        cfg=TrainConfig(),
        use_latent_sample=False,
    )
    assert isinstance(model, ActorCritic)
    assert model.action_dim == ACTION_DIM


def test_ppo_algorithm_init_state_matches_create_ppo_train_state_shape():
    policy = PPOAlgorithm.build_model(
        STATE_DIM,
        ACTION_DIM,
        LATENT_DIM,
        0,
        rngs=nnx.Rngs(0),
        cfg=TrainConfig(ppo=PPOConfig(num_epochs=1, num_minibatches=2)),
        use_latent_sample=False,
    )
    cfg = TrainConfig(ppo=PPOConfig(num_epochs=1, num_minibatches=2), total_iters=4)
    policy_state, policy_buffer = PPOAlgorithm.init_state(policy, cfg)

    assert isinstance(policy_state, PPOTrainState)
    assert isinstance(policy_buffer, PolicyBuffer)
    assert policy_state.step == 0
    _, expected_params = nnx.split(policy)
    assert jax.tree.structure(policy_state.params) == jax.tree.structure(expected_params)


def test_ppo_algorithm_update_changes_params_and_returns_metrics():
    cfg = TrainConfig(ppo=PPOConfig(num_epochs=1, num_minibatches=2, lr=1e-2))
    policy = PPOAlgorithm.build_model(
        STATE_DIM,
        ACTION_DIM,
        LATENT_DIM,
        0,
        rngs=nnx.Rngs(0),
        cfg=cfg,
        use_latent_sample=False,
    )
    policy_state, policy_buffer = PPOAlgorithm.init_state(policy, cfg)
    trainer = _FakeTrainer()
    batch = _make_batch()

    new_state, metrics = PPOAlgorithm.update(
        trainer, cfg, policy_state, policy_buffer, jax.random.PRNGKey(1), batch, iteration=0
    )

    assert isinstance(new_state, PPOTrainState)
    assert new_state.step > policy_state.step
    old_leaves = jax.tree.leaves(policy_state.params)
    new_leaves = jax.tree.leaves(new_state.params)
    assert any(not jnp.array_equal(o, n) for o, n in zip(old_leaves, new_leaves))
    assert "actor_loss" in metrics
    assert "kl_early_stopped" in metrics
    assert "ppo_epochs_run" in metrics


def test_ppo_algorithm_checkpoint_fields_uses_ppo_prefix():
    cfg = TrainConfig(ppo=PPOConfig(num_epochs=1, num_minibatches=2))
    policy = PPOAlgorithm.build_model(
        STATE_DIM,
        ACTION_DIM,
        LATENT_DIM,
        0,
        rngs=nnx.Rngs(0),
        cfg=cfg,
        use_latent_sample=False,
    )
    policy_state, _ = PPOAlgorithm.init_state(policy, cfg)
    fields = PPOAlgorithm.checkpoint_fields(policy_state)

    assert set(fields) == {"ppo_params", "ppo_opt_state", "ppo_step"}
    assert fields["ppo_step"] == policy_state.step
