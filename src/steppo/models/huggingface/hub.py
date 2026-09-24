"""Hugging Face transport for validated VariBAD model artifacts."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from huggingface_hub import HfApi

from steppo.models.huggingface.artifact import (
    LoadedModelArtifact,
    export_checkpoint_artifact,
    load_model_artifact,
)


def _get_api(api: Any | None) -> Any:
    """Use an injected client in tests, otherwise the standard HF client."""
    return api if api is not None else HfApi()


def upload_model(
    checkpoint_path: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    revision: str = "main",
    commit_message: str | None = None,
    api: Any | None = None,
) -> Any:
    """Export, validate, and upload a checkpoint as a Hub model artifact.

    Authentication is delegated to ``huggingface_hub``.  In particular, the
    standard ``HF_TOKEN`` environment variable is honored by the default
    client without storing credentials in project configuration.
    """
    client = _get_api(api)
    with TemporaryDirectory(prefix="steppo-hf-upload-") as temporary_dir:
        artifact_dir = export_checkpoint_artifact(checkpoint_path, temporary_dir)
        load_model_artifact(artifact_dir)

        client.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        upload_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "folder_path": str(artifact_dir),
            "repo_type": "model",
            "revision": revision,
        }
        if commit_message is not None:
            upload_kwargs["commit_message"] = commit_message
        return client.upload_folder(**upload_kwargs)


def download_model(
    repo_id: str,
    *,
    revision: str = "main",
    local_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    api: Any | None = None,
    local_files_only: bool = False,
) -> LoadedModelArtifact:
    """Download and strictly reconstruct a Hub model artifact.

    `local_files_only` loads the cached snapshot without contacting the Hub
    (raises huggingface_hub's LocalEntryNotFoundError when it is not cached).
    """
    client = _get_api(api)
    download_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
    }
    if local_files_only:
        download_kwargs["local_files_only"] = True
    if local_dir is not None:
        download_kwargs["local_dir"] = str(local_dir)
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(cache_dir)
    artifact_dir = client.snapshot_download(**download_kwargs)
    return load_model_artifact(Path(artifact_dir))
