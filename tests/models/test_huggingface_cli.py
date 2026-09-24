from pathlib import Path

import pytest

from steppo.models.huggingface import cli


def test_upload_cli_forwards_checkpoint_repository_and_visibility(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "upload_model",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "commit-info",
    )

    assert (
        cli.main(
            [
                "upload",
                "--checkpoint",
                "/runs/controller",
                "--repo-id",
                "org/controller",
                "--public",
                "--revision",
                "release-1",
                "--commit-message",
                "publish",
            ]
        )
        == 0
    )

    assert calls == [
        (
            (Path("/runs/controller"), "org/controller"),
            {
                "private": False,
                "revision": "release-1",
                "commit_message": "publish",
            },
        )
    ]
    assert "Uploaded" in capsys.readouterr().out


def test_download_cli_protects_non_empty_directory(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "artifact"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep")
    monkeypatch.setattr(
        cli, "download_model", lambda *_args, **_kwargs: pytest.fail("Hub contacted")
    )

    assert (
        cli.main(
            [
                "download",
                "--repo-id",
                "org/controller",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 1
    )
    assert "not empty" in capsys.readouterr().err
