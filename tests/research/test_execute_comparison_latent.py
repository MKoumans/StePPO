"""Latent trace aggregation without initializing models or running an ODE."""

from types import SimpleNamespace

import numpy as np

from research.ode import execute_comparison as comparison


def test_latent_traces_average_envs_and_clip_log_variance(monkeypatch):
    config = SimpleNamespace(
        env=object(),
        rollout_steps=2,
        backbone="varibad",
        algo="ppo",
    )
    run = {
        "run_uid": "run-a",
        "config_path": "config.yaml",
        "checkpoint_dir": "checkpoint_7",
    }
    seen = {}
    monkeypatch.setattr(comparison, "load_config_from_yaml", lambda *_args: config)
    monkeypatch.setattr(comparison, "ODEEnv", lambda env_config, steps: (env_config, steps))
    monkeypatch.setattr(
        comparison,
        "build_models",
        lambda cfg, env, seed: seen.update(build=(cfg, env, seed)) or ("vae", "policy"),
    )
    monkeypatch.setattr(
        comparison,
        "try_load_checkpoint",
        lambda vae, policy, checkpoint, **kwargs: ("loaded-vae", "loaded-policy"),
    )
    mu = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        ]
    )
    logvar = np.array(
        [
            [[-20.0, 0.0], [0.0, 10.0], [20.0, 20.0]],
            [[0.0, -20.0], [10.0, 0.0], [20.0, 20.0]],
        ]
    )

    def fake_rollout(vae, policy, env, env_params, num_envs, steps, key):
        seen.update(
            rollout=(vae, policy, env, num_envs, steps),
            task_values=np.asarray(env_params.lam),
            budgets=env_params.max_steps,
            seed_key=np.asarray(key),
        )
        return {"belief_mu": mu, "belief_logvar": logvar}

    monkeypatch.setattr(comparison, "collect_analysis_rollout", fake_rollout)

    result = comparison.collect_latent_traces_for_task(
        [run],
        task_value=7.5,
        num_envs=3,
        seed=42,
    )["run-a"]

    assert seen["build"] == (config, (config.env, 2), 42)
    assert seen["rollout"] == ("loaded-vae", "loaded-policy", (config.env, 2), 3, 2)
    assert np.allclose(seen["task_values"], [7.5, 7.5, 7.5])
    assert seen["budgets"] == 2
    assert seen["seed_key"].shape == (2,)
    assert np.allclose(result["mu"], mu.mean(axis=1))
    expected_var = np.exp(np.clip(logvar, -10.0, 10.0)).mean(axis=1)
    assert np.allclose(result["var"], expected_var)
    assert result["latent_dim"] == 2
