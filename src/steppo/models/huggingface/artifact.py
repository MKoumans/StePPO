"""Export and load deployable VariBAD ODE model artifacts.

The artifact format intentionally contains model parameters and the resolved
configuration, but no optimizer or training state. Hugging Face transport is
implemented separately so this module can validate artifacts without network
access.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from safetensors.flax import load_file, save_file

from steppo.configs.base_config import TrainConfig, load_config_from_dict, load_config_from_yaml
from steppo.envs.ode import ODEEnv
from steppo.envs.ode.learned_controller import LearnedController
from steppo.utils.checkpoint import build_models, load_checkpoint, resolve_checkpoint

ARTIFACT_FORMAT_VERSION = 1
_CONFIG_FILE = "config.yaml"
_MANIFEST_FILE = "manifest.json"
_README_FILE = "README.md"
_COMPONENT_FILES = {"vae": "vae.safetensors", "policy": "policy.safetensors"}


@dataclass(frozen=True)
class LoadedModelArtifact:
    """The live inference objects reconstructed from one model artifact."""

    vae: Any
    policy: Any
    config: TrainConfig
    env: ODEEnv
    controller: LearnedController
    artifact_dir: Path


def _config_bytes(config: TrainConfig) -> bytes:
    return yaml.safe_dump(asdict(config), sort_keys=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _component_metadata(params) -> tuple[list[Any], dict[str, Any]]:
    leaves = list(jax.tree_util.tree_leaves(params))
    tree = repr(jax.tree_util.tree_structure(params))
    metadata = {
        "tree": tree,
        "tree_sha256": _sha256(tree.encode("utf-8")),
        "tensor_names": [f"leaf_{index:06d}" for index in range(len(leaves))],
        "leaves": [
            {"shape": list(np.asarray(leaf).shape), "dtype": str(np.asarray(leaf).dtype)}
            for leaf in leaves
        ],
    }
    return leaves, metadata


def _write_component(path: Path, params) -> dict[str, Any]:
    leaves, metadata = _component_metadata(params)
    save_file(
        {name: np.asarray(leaf) for name, leaf in zip(metadata["tensor_names"], leaves)},
        str(path),
    )
    return metadata


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read model artifact manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("Model artifact manifest must contain a JSON object")
    return value


def _model_card(manifest: dict[str, Any], config: TrainConfig) -> str:
    iteration = manifest.get("source_checkpoint_iteration")
    iteration_text = "unknown" if iteration is None else str(iteration)
    return f"""# VariBAD ODE model artifact

This repository contains a deployable VariBAD ODE model artifact.

- Backbone: `{manifest["backbone"]}`
- ODE system: `{config.env.system}`
- Source training iteration: `{iteration_text}`
- Artifact format version: `{manifest["format_version"]}`
- Weights: `vae.safetensors`, `policy.safetensors`
- Configuration: `config.yaml`
- Training optimizer state and PID control-envelope calibration are not included.

Load the artifact with the project's `load_model_artifact` Python interface or
the `steppo-hf download` command. The loader reconstructs the VAE, policy,
ODE environment, and `LearnedController` from the packaged configuration.
"""


def export_model_artifact(
    vae,
    policy,
    config: TrainConfig,
    output_dir: str | Path,
    *,
    source_checkpoint_iteration: int | None = None,
) -> Path:
    """Write a validated local deployable model artifact."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Model artifact output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.backbone != "varibad":
        raise ValueError(f"Only the 'varibad' backbone is supported, got {config.backbone!r}")

    config_bytes = _config_bytes(config)
    (output_dir / _CONFIG_FILE).write_bytes(config_bytes)

    _, vae_params = nnx.split(vae)
    _, policy_params = nnx.split(policy)
    components = {}
    for name, params in (("vae", vae_params), ("policy", policy_params)):
        components[name] = _write_component(output_dir / _COMPONENT_FILES[name], params)

    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "backbone": config.backbone,
        "config_sha256": _sha256(config_bytes),
        "files": {
            "config": _CONFIG_FILE,
            "manifest": _MANIFEST_FILE,
            **_COMPONENT_FILES,
            "readme": _README_FILE,
        },
        "source_checkpoint_iteration": source_checkpoint_iteration,
        "includes_control_envelopes": False,
        "components": components,
    }
    (output_dir / _MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / _README_FILE).write_text(_model_card(manifest, config))
    return output_dir


def export_checkpoint_artifact(
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Export a deployable artifact from an Orbax step or run directory."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = Path(resolve_checkpoint(str(checkpoint_path)))
    config_path = checkpoint_path / _CONFIG_FILE
    if not config_path.is_file():
        config_path = checkpoint_dir.parent / _CONFIG_FILE
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Could not find the resolved training config next to {checkpoint_dir}"
        )

    config = load_config_from_yaml(TrainConfig, str(config_path))
    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, config.seed)
    vae, policy = load_checkpoint(
        vae, policy, str(checkpoint_dir), backbone=config.backbone, algo=config.algo
    )
    try:
        iteration = int(checkpoint_dir.name.rsplit("_", 1)[-1])
    except ValueError:
        iteration = None
    return export_model_artifact(
        vae,
        policy,
        config,
        output_dir,
        source_checkpoint_iteration=iteration,
    )


def _required_files(manifest: dict[str, Any]) -> set[str]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Model artifact manifest is missing its files mapping")
    required = {_CONFIG_FILE, _MANIFEST_FILE, _README_FILE, *_COMPONENT_FILES.values()}
    try:
        valid = set(files.values()) == required and all(
            isinstance(value, str) for value in files.values()
        )
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("Model artifact manifest has an unexpected file set")
    return required


def load_model_artifact(artifact_dir: str | Path) -> LoadedModelArtifact:
    """Load and strictly validate a local deployable model artifact."""
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / _MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model artifact manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported model artifact format version: {manifest.get('format_version')!r}"
        )
    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != set(_COMPONENT_FILES):
        raise ValueError("Model artifact manifest is missing component metadata")
    if any(not isinstance(components[name], dict) for name in _COMPONENT_FILES):
        raise ValueError("Model artifact component metadata is malformed")
    if manifest.get("backbone") != "varibad":
        raise ValueError(
            f"Only the 'varibad' backbone is supported, got {manifest.get('backbone')!r}"
        )
    _required_files(manifest)

    config_path = artifact_dir / _CONFIG_FILE
    if not config_path.is_file():
        raise FileNotFoundError(f"Model artifact config not found: {config_path}")
    config_bytes = config_path.read_bytes()
    if _sha256(config_bytes) != manifest.get("config_sha256"):
        raise ValueError("Model artifact config hash does not match the manifest")
    try:
        config_data = yaml.safe_load(config_bytes) or {}
        config = load_config_from_dict(TrainConfig, config_data, strict=True)
    except Exception as error:
        raise ValueError(f"Could not load model artifact configuration: {config_path}") from error

    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, config.seed)
    vae_graphdef, vae_params = nnx.split(vae)
    policy_graphdef, policy_params = nnx.split(policy)
    vae = nnx.merge(
        vae_graphdef,
        jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(vae_params),
            _loaded_component_leaves(
                artifact_dir / _COMPONENT_FILES["vae"],
                vae_params,
                components["vae"],
                "vae",
            ),
        ),
    )
    policy = nnx.merge(
        policy_graphdef,
        jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(policy_params),
            _loaded_component_leaves(
                artifact_dir / _COMPONENT_FILES["policy"],
                policy_params,
                components["policy"],
                "policy",
            ),
        ),
    )
    controller = LearnedController.from_models(vae, policy, config, env)
    return LoadedModelArtifact(vae, policy, config, env, controller, artifact_dir)


def _loaded_component_leaves(
    path: Path,
    params,
    expected: dict[str, Any],
    component_name: str,
) -> list[Any]:
    if not all(key in expected for key in ("tree_sha256", "tensor_names", "leaves")):
        raise ValueError(f"{component_name} parameter metadata is malformed")
    leaves, actual = _component_metadata(params)
    for key in ("tree_sha256", "tensor_names", "leaves"):
        if actual[key] != expected.get(key):
            raise ValueError(
                f"{component_name} parameter tree does not match the artifact manifest"
            )
    try:
        tensors = load_file(str(path))
    except Exception as error:
        raise ValueError(f"Could not load {component_name} weights from {path}") from error
    expected_names = expected["tensor_names"]
    if set(tensors) != set(expected_names):
        raise ValueError(f"{component_name} weight names do not match the artifact manifest")
    loaded_leaves = []
    for name, expected_leaf in zip(expected_names, expected["leaves"]):
        loaded = np.asarray(tensors[name])
        if (
            list(loaded.shape) != expected_leaf["shape"]
            or str(loaded.dtype) != expected_leaf["dtype"]
        ):
            raise ValueError(
                f"{component_name} tensor {name} shape or dtype does not match the manifest"
            )
        loaded_leaves.append(jnp.asarray(loaded))
    return loaded_leaves
