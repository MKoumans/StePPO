"""Portable Hugging Face model artifacts for VariBAD ODE inference."""

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "LoadedModelArtifact",
    "export_checkpoint_artifact",
    "export_model_artifact",
    "load_model_artifact",
    "download_model",
    "upload_model",
]


def __getattr__(name):
    """Resolve public implementations lazily to keep CLI help lightweight."""
    if name in {
        "ARTIFACT_FORMAT_VERSION",
        "LoadedModelArtifact",
        "export_checkpoint_artifact",
        "export_model_artifact",
        "load_model_artifact",
    }:
        from steppo.models.huggingface import artifact

        return getattr(artifact, name)
    if name in {"download_model", "upload_model"}:
        from steppo.models.huggingface import hub

        return getattr(hub, name)
    raise AttributeError(name)
