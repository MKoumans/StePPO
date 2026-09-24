"""Small orchestration regressions for Trainer's reward setup and metrics."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from steppo.configs.base_config import (
    EvalMetricsConfig,
    ODEEnvConfig,
    RewardNormalizationConfig,
    TrainConfig,
    TrainingConfig,
)
from steppo.training import base_trainer


class _FakeEnv:
    t_end = 4.0
    task_dim = 1

    def __init__(self):
        self.warp = None

    def _task_bounds(self):
        return 2.0, 20.0

    def set_progress_warp(self, log_mu_grid, t_knots, fracs):
        self.warp = (log_mu_grid, t_knots, fracs)


def _make_trainer(config=None, env=None):
    trainer = base_trainer.Trainer(
        config or TrainConfig(),
        vae=None,
        policy=None,
        env=env or _FakeEnv(),
        env_params=None,
        policy_algo=SimpleNamespace(name="test"),
    )
    trainer._progress_warp = False
    trainer._reward_norm = None
    return trainer


def test_return_buffer_aggregates_across_rollouts_and_clears_after_flush():
    trainer = _make_trainer()
    trainer._return_buffer = []
    trainer._record_rollout_returns(
        SimpleNamespace(rewards=jnp.array([[[1.0], [3.0]], [[2.0], [4.0]], [[3.0], [5.0]]]))
    )
    trainer._record_rollout_returns(
        SimpleNamespace(rewards=jnp.array([[[-1.0], [1.0]], [[0.0], [1.0]]]))
    )

    assert trainer._flush_return_buffer() == {
        "train/mean_return": 4.75,
        "train/min_return": -1.0,
        "train/max_return": 12.0,
    }
    assert trainer._return_buffer == []
    assert trainer._flush_return_buffer() is None


def test_progress_warp_installs_grid_with_configured_task_range_and_budget(monkeypatch):
    config = TrainConfig(
        env=ODEEnvConfig(system="scalar_decay", progress_warp=True),
        training=TrainingConfig(
            reward_normalization=RewardNormalizationConfig(
                grid_points=3,
                num_repeats=5,
                max_steps=17,
            )
        ),
        rollout_steps=40,
    )
    env = _FakeEnv()
    trainer = _make_trainer(config, env)
    grid = (np.array([0.7, 1.5]), np.ones((2, 4)), np.linspace(0.25, 1.0, 4))
    observed = {}

    def fake_build(env_config, mu_lo, mu_hi, **kwargs):
        observed.update(env_config=env_config, mu_lo=mu_lo, mu_hi=mu_hi, **kwargs)
        return grid

    monkeypatch.setattr(base_trainer, "build_pid_warp_grid", fake_build)
    trainer._setup_progress_warp(config)

    assert trainer._progress_warp is True
    assert all(np.array_equal(actual, expected) for actual, expected in zip(env.warp, grid))
    assert observed == {
        "env_config": config.env,
        "mu_lo": 2.0,
        "mu_hi": 20.0,
        "grid_points": 3,
        "num_repeats": 5,
        "max_steps": 17,
        "require_cache": True,
        "verbose": 0,
    }


def test_progress_warp_falls_back_to_rollout_budget_and_skips_non_scalar_env(
    monkeypatch,
    capsys,
):
    config = TrainConfig(
        env=ODEEnvConfig(system="scalar_decay", progress_warp=True),
        rollout_steps=11,
    )
    env = SimpleNamespace(t_end=4.0, task_dim=2)
    trainer = _make_trainer(config, env)
    monkeypatch.setattr(
        base_trainer,
        "build_pid_warp_grid",
        lambda *args, **kwargs: pytest.fail("non-scalar env must not build a warp"),
    )

    trainer._setup_progress_warp(config, verbose=1)

    assert trainer._progress_warp is False
    assert "not a scalar-task ODE env" in capsys.readouterr().out


def test_reward_normalization_does_not_build_a_second_grid_after_progress_warp(
    monkeypatch,
):
    config = TrainConfig(
        env=ODEEnvConfig(system="scalar_decay"),
        training=TrainingConfig(reward_normalization=RewardNormalizationConfig(enabled=True)),
    )
    trainer = _make_trainer(config)
    trainer._progress_warp = True
    monkeypatch.setattr(
        base_trainer,
        "build_pid_scale_grid",
        lambda *args, **kwargs: pytest.fail("warp already normalizes per-task reward"),
    )

    trainer._per_mu_reward_normalization(config)

    assert trainer._reward_norm is None


def test_reward_normalization_propagates_missing_required_cache(monkeypatch):
    config = TrainConfig(
        env=ODEEnvConfig(system="scalar_decay"),
        training=TrainingConfig(reward_normalization=RewardNormalizationConfig(enabled=True)),
        rollout_steps=19,
    )
    trainer = _make_trainer(config)
    monkeypatch.setattr(
        base_trainer,
        "build_pid_scale_grid",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("cache missing")),
    )

    with pytest.raises(FileNotFoundError, match="cache missing"):
        trainer._per_mu_reward_normalization(config)


@struct.dataclass
class _TrainState:
    params: dict
    opt_state: dict
    step: jnp.ndarray


def test_restore_training_states_remaps_checkpoint_leaves_and_restores_iteration(
    monkeypatch,
):
    trainer = _make_trainer()
    vae_template = _TrainState(
        params={"bias": jnp.zeros((1,)), "weight": jnp.zeros((2,))},
        opt_state={"momentum": jnp.zeros((2,))},
        step=jnp.array(0),
    )
    policy_template = _TrainState(
        params={"weight": jnp.zeros((2,))},
        opt_state={"momentum": jnp.zeros((2,))},
        step=jnp.array(0),
    )
    monkeypatch.setattr(
        trainer,
        "_belief_checkpoint_fields",
        lambda _state: {"vae_params": None, "vae_opt_state": None, "vae_step": None},
    )
    trainer.policy_algo.checkpoint_fields = lambda _state: {
        "policy_params": None,
        "policy_opt_state": None,
        "policy_step": None,
    }
    checkpoint = {
        # The source trees intentionally have different structure; restore must
        # map their leaves into the fresh model's expected tree shape.
        "vae_params": [jnp.array([7.0]), jnp.array([2.0, 3.0])],
        "vae_opt_state": [jnp.array([4.0, 5.0])],
        "vae_step": jnp.array(8),
        "policy_params": [jnp.array([6.0, 9.0])],
        "policy_opt_state": [jnp.array([1.0, 2.0])],
        "policy_step": jnp.array(8),
        "iteration": 12,
        "control_envelopes": {"reject_streak": jnp.array(3), "accept_ema": jnp.array(0.75)},
    }
    monkeypatch.setattr(trainer, "load_checkpoint", lambda _path: checkpoint)

    vae, policy, iteration = trainer._restore_training_states(
        vae_template,
        policy_template,
        "checkpoint_12",
    )

    assert vae.params["bias"].tolist() == [7.0]
    assert vae.params["weight"].tolist() == [2.0, 3.0]
    assert vae.opt_state["momentum"].tolist() == [4.0, 5.0]
    assert int(vae.step) == 8
    assert policy.params["weight"].tolist() == [6.0, 9.0]
    assert policy.opt_state["momentum"].tolist() == [1.0, 2.0]
    assert int(policy.step) == 8
    assert iteration == 12
    assert trainer._control_envelopes == {"reject_streak": 3.0, "accept_ema": 0.75}


def test_restore_training_states_rejects_missing_required_checkpoint_field(monkeypatch):
    trainer = _make_trainer()
    state = _TrainState(params={"weight": jnp.zeros((1,))}, opt_state={}, step=jnp.array(0))
    monkeypatch.setattr(
        trainer,
        "_belief_checkpoint_fields",
        lambda _state: {"vae_params": None},
    )
    trainer.policy_algo.checkpoint_fields = lambda _state: {"policy_step": None}
    monkeypatch.setattr(
        trainer,
        "load_checkpoint",
        lambda _path: {"vae_params": [jnp.ones((1,))]},
    )

    with pytest.raises(KeyError, match=r"missing policy fields: \['policy_step'\]"):
        trainer._restore_training_states(state, state, "checkpoint_1")


def test_setup_efficiency_datasets_loads_only_nonempty_splits(monkeypatch):
    from steppo.training import error_dist

    env_config = ODEEnvConfig(
        system="scalar_decay",
        train_bins=((1.0, 2.0),),
        val_bins=((2.0, 3.0),),
        test_bins=(),
    )
    config = TrainConfig(
        env=env_config,
        eval_interval=5,
        rollout_steps=37,
        eval_metrics=EvalMetricsConfig(enabled=False),
    )
    trainer = _make_trainer(config, SimpleNamespace(t_end=4.0, _config=env_config))
    calls = []

    def fake_load(cfg, **kwargs):
        calls.append((cfg, kwargs))
        return {"split": kwargs["split"]}

    monkeypatch.setattr(error_dist, "load_cached_pid_batch", fake_load)
    trainer._setup_efficiency_datasets(config)

    assert trainer._efficiency_datasets == {"train": {"split": "train"}, "val": {"split": "val"}}
    assert [kwargs["split"] for _, kwargs in calls] == ["train", "val"]
    assert all(cfg is env_config for cfg, _ in calls)
    assert all(kwargs["max_steps"] == 37 for _, kwargs in calls)
    assert all(kwargs["num_envs"] == error_dist.PID_EVAL_NUM_ENVS for _, kwargs in calls)
    assert all(kwargs["seed"] == error_dist.PID_EVAL_SEED for _, kwargs in calls)


def test_setup_efficiency_datasets_rejects_enabled_eval_without_heldout_bins():
    env_config = ODEEnvConfig(system="scalar_decay", train_bins=((1.0, 2.0),))
    config = TrainConfig(env=env_config, eval_interval=1)
    trainer = _make_trainer(config, SimpleNamespace(t_end=4.0, _config=env_config))

    with pytest.raises(ValueError, match="env.val_bins and env.test_bins are both empty"):
        trainer._setup_efficiency_datasets(config)
