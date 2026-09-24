from pathlib import Path

import pytest

from steppo.models.huggingface import hub


class FakeHubClient:
    def __init__(self, downloaded_dir=None):
        self.downloaded_dir = downloaded_dir
        self.calls = []

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        return "commit-info"

    def snapshot_download(self, **kwargs):
        self.calls.append(("snapshot_download", kwargs))
        return str(self.downloaded_dir)


def test_upload_validates_before_creating_or_updating_repository(tmp_path, monkeypatch):
    client = FakeHubClient()
    exported = []

    def fake_export(checkpoint_path, output_dir):
        exported.append((checkpoint_path, output_dir))
        Path(output_dir, "manifest.json").write_text("{}")
        return Path(output_dir)

    monkeypatch.setattr(hub, "export_checkpoint_artifact", fake_export)
    monkeypatch.setattr(hub, "load_model_artifact", lambda path: object())

    result = hub.upload_model(
        tmp_path / "checkpoint",
        "org/controller",
        private=False,
        revision="release-1",
        commit_message="publish release",
        api=client,
    )

    assert result == "commit-info"
    assert len(exported) == 1
    assert client.calls == [
        (
            "create_repo",
            {
                "repo_id": "org/controller",
                "private": False,
                "repo_type": "model",
                "exist_ok": True,
            },
        ),
        (
            "upload_folder",
            {
                "repo_id": "org/controller",
                "folder_path": str(exported[0][1]),
                "repo_type": "model",
                "revision": "release-1",
                "commit_message": "publish release",
            },
        ),
    ]


def test_upload_does_not_touch_hub_when_local_validation_fails(tmp_path, monkeypatch):
    client = FakeHubClient()
    monkeypatch.setattr(hub, "export_checkpoint_artifact", lambda *_args: tmp_path / "missing")
    monkeypatch.setattr(
        hub,
        "load_model_artifact",
        lambda _path: (_ for _ in ()).throw(ValueError("invalid artifact")),
    )

    with pytest.raises(ValueError, match="invalid artifact"):
        hub.upload_model(tmp_path / "checkpoint", "org/controller", api=client)

    assert client.calls == []


def test_download_pins_revision_and_reconstructs_bundle(tmp_path, monkeypatch):
    client = FakeHubClient(tmp_path / "downloaded")
    expected = object()
    loaded = []
    monkeypatch.setattr(hub, "load_model_artifact", lambda path: loaded.append(path) or expected)

    result = hub.download_model(
        "org/controller",
        revision="release-1",
        local_dir=tmp_path / "local-artifact",
        cache_dir=tmp_path / "cache",
        api=client,
    )

    assert result is expected
    assert loaded == [tmp_path / "downloaded"]
    assert client.calls == [
        (
            "snapshot_download",
            {
                "repo_id": "org/controller",
                "repo_type": "model",
                "revision": "release-1",
                "local_dir": str(tmp_path / "local-artifact"),
                "cache_dir": str(tmp_path / "cache"),
            },
        )
    ]


def test_download_local_files_only_is_passed_to_the_hub(tmp_path, monkeypatch):
    client = FakeHubClient(tmp_path / "downloaded")
    monkeypatch.setattr(hub, "load_model_artifact", lambda path: path)

    hub.download_model("org/controller", api=client, local_files_only=True)

    assert client.calls[0][1]["local_files_only"] is True


def test_paper_model_uses_the_cache_and_downloads_only_when_missing(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    from steppo import cli

    calls = []

    def fake_download(repo_id, revision="main", local_files_only=False):
        calls.append(local_files_only)
        if local_files_only and not cached:
            raise LocalEntryNotFoundError("not cached")
        return repo_id

    monkeypatch.setattr(hub, "download_model", fake_download)

    cached = True
    assert cli.load_paper_model("van_der_pol") == cli.PAPER_MODELS["van_der_pol"]
    assert calls == [True]

    calls.clear()
    cached = False
    cli.load_paper_model("van_der_pol")
    assert calls == [True, False]

    calls.clear()
    cli.load_paper_model("van_der_pol", refresh=True)
    assert calls == [False]
