import json
from dataclasses import asdict

import flax.nnx as nnx
import jax
import orbax.checkpoint as ocp
import pytest
import yaml

from steppo.configs.base_config import TrainConfig
from steppo.envs.ode import ODEEnv
from steppo.models.huggingface import (
    export_checkpoint_artifact,
    export_model_artifact,
    load_model_artifact,
)
from steppo.utils.checkpoint import build_models


def test_model_artifact_round_trip_returns_inference_bundle(tmp_path):
    config = TrainConfig()
    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, config.seed)

    artifact_dir = export_model_artifact(
        vae,
        policy,
        config,
        tmp_path / "model-artifact",
        source_checkpoint_iteration=17,
    )

    assert artifact_dir.is_dir()
    assert {path.name for path in artifact_dir.iterdir()} == {
        "README.md",
        "config.yaml",
        "manifest.json",
        "policy.safetensors",
        "vae.safetensors",
    }

    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    assert manifest["format_version"] == 1
    assert manifest["backbone"] == "varibad"
    assert manifest["source_checkpoint_iteration"] == 17
    assert manifest["includes_control_envelopes"] is False

    loaded = load_model_artifact(artifact_dir)

    assert loaded.config.backbone == "varibad"
    assert loaded.env.task_dim == env.task_dim
    assert loaded.controller.encoder.task_embed is not None

    _, original_vae_params = nnx.split(vae)
    _, loaded_vae_params = nnx.split(loaded.vae)
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: (left == right).all(), original_vae_params, loaded_vae_params
        )
    )


def test_model_artifact_rejects_modified_configuration(tmp_path):
    config = TrainConfig()
    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, config.seed)
    artifact_dir = export_model_artifact(vae, policy, config, tmp_path / "model-artifact")

    (artifact_dir / "config.yaml").write_text((artifact_dir / "config.yaml").read_text() + "\n")

    with pytest.raises(ValueError, match="config hash"):
        load_model_artifact(artifact_dir)


def test_model_artifact_rejects_missing_weights(tmp_path):
    config = TrainConfig()
    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, config.seed)
    artifact_dir = export_model_artifact(vae, policy, config, tmp_path / "model-artifact")
    (artifact_dir / "policy.safetensors").unlink()

    with pytest.raises(ValueError, match="policy weights"):
        load_model_artifact(artifact_dir)


def test_checkpoint_artifact_export_resolves_run_directory(tmp_path):
    config = TrainConfig()
    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, config.seed)
    _, vae_params = nnx.split(vae)
    _, policy_params = nnx.split(policy)

    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint_7"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_dir, {"vae_params": vae_params, "ppo_params": policy_params})
    checkpointer.wait_until_finished()

    artifact_dir = export_checkpoint_artifact(run_dir, tmp_path / "exported")

    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    assert manifest["source_checkpoint_iteration"] == 7
    assert load_model_artifact(artifact_dir).config.env.system == config.env.system
