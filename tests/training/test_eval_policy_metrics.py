"""Fast regressions for eval_policy's batching and metric aggregation."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from steppo.training import eval as eval_module


def _metrics_for_tasks(task_values):
    tasks = np.asarray(task_values, dtype=np.float32)
    n = len(tasks)
    return {
        "total_return": jnp.asarray(tasks),
        "episode_length": jnp.asarray(tasks + 2),
        "final_logvar": jnp.tile(jnp.log(jnp.array([[1.0, 4.0]])), (n, 1)),
        "sigma_reduction_ratio": jnp.asarray(tasks / 10.0),
        "ep_t_reached": jnp.asarray(np.stack((tasks / 2.0, tasks / 2.0 + 5.0), axis=1)),
        "num_episodes": jnp.full((n,), 2),
        "task_embedding_mae": jnp.asarray(tasks / 10.0),
        "budget_exhausted": jnp.asarray(tasks >= 20.0),
        "task_true": jnp.asarray(tasks[:, None]),
        "final_mu": jnp.asarray(np.stack((tasks, -tasks), axis=1)),
    }


def test_cached_eval_caps_to_dataset_and_aggregates_optional_metrics(monkeypatch):
    tasks = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    keys = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint32)
    dataset = {"task_params": tasks, "episode_keys": keys}
    batches = []

    def fake_run(_vae, _policy, _env, batch_tasks, batch_keys, *_args):
        batches.append((np.asarray(batch_tasks), np.asarray(batch_keys)))
        return _metrics_for_tasks(batch_tasks)

    monkeypatch.setattr(eval_module, "_run_eval_dataset_batch", fake_run)
    env = SimpleNamespace(t_end=15.0)

    result = eval_module.eval_policy(
        None,
        None,
        env,
        None,
        None,
        num_episodes=5,
        num_envs=2,
        max_steps=10,
        dataset=dataset,
    )

    assert [batch_tasks.tolist() for batch_tasks, _ in batches] == [[10.0, 20.0], [30.0]]
    assert [batch_keys.tolist() for _, batch_keys in batches] == [
        keys[:2].tolist(),
        keys[2:].tolist(),
    ]
    assert result["mean_return"] == pytest.approx(20.0)
    assert result["std_return"] == pytest.approx(np.std(tasks))
    assert result["mean_length"] == pytest.approx(22.0)
    assert result["belief_final_variance"] == pytest.approx(2.5)
    assert result["sigma_reduction_ratio"] == pytest.approx(2.0)
    assert result["t_per_episode"] == "10.0-15.0"
    assert result["success_per_episode"] == pytest.approx([1 / 3, 2 / 3])
    assert result["success_rate"] == pytest.approx(1 / 3)
    assert result["task_embedding_mae"] == pytest.approx(2.0)
    assert result["mean_step_count"] == pytest.approx(20 / 3)
    assert result["task_belief_corr_per_dim"] == pytest.approx([1.0, -1.0])
    assert result["task_belief_corr_sum"] == pytest.approx(2.0)


def test_cached_eval_rejects_empty_dataset_before_running_batches(monkeypatch):
    monkeypatch.setattr(
        eval_module,
        "_run_eval_dataset_batch",
        lambda *_args: pytest.fail("empty cache must fail before a rollout"),
    )

    with pytest.raises(ValueError, match="Cached evaluation dataset is empty"):
        eval_module.eval_policy(
            None,
            None,
            SimpleNamespace(t_end=1.0),
            None,
            None,
            num_episodes=4,
            dataset={"task_params": np.array([]), "episode_keys": np.empty((0, 2))},
        )


def test_sampled_non_ode_eval_batches_and_omits_ode_only_metrics(monkeypatch):
    batch_sizes = []

    def fake_run(_vae, _policy, _env, keys, *_args):
        batch_sizes.append(len(keys))
        n = len(keys)
        # eval_policy always needs these fields; ODE-only fields stay absent.
        return {
            "total_return": jnp.arange(n, dtype=jnp.float32),
            "episode_length": jnp.ones((n,)),
            "final_logvar": jnp.zeros((n, 2)),
            "sigma_reduction_ratio": jnp.zeros((n,)),
        }

    monkeypatch.setattr(eval_module, "_run_eval_batch", fake_run)
    result = eval_module.eval_policy(
        None,
        None,
        SimpleNamespace(),
        None,
        jax.random.PRNGKey(0),
        num_episodes=3,
        num_envs=2,
    )

    assert batch_sizes == [2, 1]
    assert result["mean_return"] == pytest.approx(1 / 3)
    assert result["success_rate"] == 0.0
    assert "t_per_episode" not in result
    assert "task_belief_corr_sum" not in result
