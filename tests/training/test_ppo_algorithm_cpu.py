"""CPU-safe wiring tests for PPOAlgorithm.update."""

from contextlib import contextmanager
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from steppo.configs.base_config import PPOConfig, TrainConfig
from steppo.training import ppo_algorithm


class _Profiler:
    @contextmanager
    def section(self, name):
        yield

    @staticmethod
    def block_and_record(name, *outputs):
        del name, outputs


class _Buffer:
    def __init__(self):
        self.stored = None
        self.minibatch_counts = []

    def store(self, *args, **kwargs):
        self.stored = (args, kwargs)

    def get_minibatches(self, count, key):
        del key
        self.minibatch_counts.append(count)
        return [object() for _ in range(count)]


def test_update_scales_rewards_masks_frozen_episodes_and_stops_at_target_kl(monkeypatch):
    config = TrainConfig(
        total_iters=2,
        episodes_per_trial=2,
        ppo=PPOConfig(num_epochs=2, num_minibatches=2, target_kl=0.1),
    )
    trainer = SimpleNamespace(
        profiler=_Profiler(),
        timings=SimpleNamespace(update=lambda **_kwargs: None),
        env=SimpleNamespace(_task_bounds=lambda: (1.0, 10.0)),
        _reward_norm=(jnp.log(jnp.array([1.0, 10.0])), jnp.log(jnp.array([1.0, 10.0])), 0.0),
        _adv_stats_buffer=([], []),
    )
    batch = SimpleNamespace(
        states=jnp.zeros((3, 2, 1)),
        actions=jnp.zeros((3, 2, 1)),
        rewards=jnp.ones((3, 2, 1)),
        dones=jnp.array([[1, 1], [1, 0], [0, 1]], dtype=jnp.float32),
        keep_steps=jnp.array([[0, 1], [1, 0], [1, 1]], dtype=jnp.float32),
        values=jnp.zeros((3, 2)),
        bootstrap_value=jnp.zeros((2,)),
        task_params=jnp.array([[0.0], [1.0]]),
        beliefs_mu=jnp.zeros((3, 2, 2)),
        beliefs_logvar=jnp.zeros((3, 2, 2)),
        log_probs=jnp.zeros((3, 2)),
    )
    buffer = _Buffer()
    gae_inputs = {}
    step_calls = []

    def fake_compute_gae(rewards, values, dones, gamma, gae_lambda):
        gae_inputs.update(
            rewards=rewards, values=values, dones=dones, gamma=gamma, gae_lambda=gae_lambda
        )
        advantages = jnp.ones_like(rewards)
        return advantages, advantages * 2.0

    def fake_train_step(state, minibatch, ppo_cfg, key, training_cfg, total_steps):
        del minibatch, ppo_cfg, key, training_cfg
        step_calls.append(total_steps)
        return state, {"approx_kl": jnp.array(0.2), "actor_loss": jnp.array(1.0)}

    monkeypatch.setattr(ppo_algorithm, "compute_gae", fake_compute_gae)
    monkeypatch.setattr(ppo_algorithm, "ppo_train_step", fake_train_step)

    state, metrics = ppo_algorithm.PPOAlgorithm.update(
        trainer,
        config,
        ppo_state=SimpleNamespace(params=jnp.array([0.0])),
        policy_buffer=buffer,
        key_ppo=jax.random.PRNGKey(6),
        batch=batch,
        iteration=0,
    )

    assert state is not None
    assert np.array_equal(gae_inputs["rewards"], [[1.0, 4.0]] * 3)
    assert np.array_equal(gae_inputs["dones"], [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert gae_inputs["gamma"] == config.ppo.gamma
    assert gae_inputs["gae_lambda"] == config.ppo.gae_lambda

    stored_args, stored_kwargs = buffer.stored
    task_broadcast = stored_args[2]
    active_mask = stored_kwargs["active_mask"]
    assert task_broadcast.shape == (3, 2, 1)
    assert np.array_equal(task_broadcast[:, :, 0], [[0.0, 1.0]] * 3)
    assert np.array_equal(active_mask, [[False, True], [True, True], [True, True]])

    assert buffer.minibatch_counts == [2]
    assert step_calls == [8]
    assert metrics["kl_early_stopped"] == 1.0
    assert metrics["ppo_epochs_run"] == 1.0
    assert metrics["reward_scale_mean"] == 2.5
    assert metrics["reward_scale_max"] == 4.0
    assert metrics["norm_return_mean"] == 7.5
    assert len(trainer._adv_stats_buffer[0]) == 1
    assert np.array_equal(trainer._adv_stats_buffer[1][0], [0.0, 1.0])
