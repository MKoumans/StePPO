"""Command-line interface for VariBAD Hugging Face model artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def upload_model(*args, **kwargs):
    """Forward to the Hub upload implementation."""
    from steppo.models.huggingface.hub import upload_model as operation

    return operation(*args, **kwargs)


def download_model(*args, **kwargs):
    """Forward to the Hub download implementation."""
    from steppo.models.huggingface.hub import download_model as operation

    return operation(*args, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="steppo-hf")
    commands = parser.add_subparsers(dest="command", required=True)

    upload = commands.add_parser("upload", help="export and upload a checkpoint")
    upload.add_argument("--checkpoint", type=Path, required=True)
    upload.add_argument("--repo-id", required=True)
    upload.add_argument("--public", action="store_true", help="make the Hub repository public")
    upload.add_argument("--revision", default="main")
    upload.add_argument("--commit-message")

    download = commands.add_parser("download", help="download and validate a model artifact")
    download.add_argument("--repo-id", required=True)
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--cache-dir", type=Path)
    download.add_argument("--revision", default="main")
    download.add_argument("--overwrite", action="store_true")
    return parser


def _check_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Download output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ValueError(f"Download output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the model-artifact command-line interface."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "upload":
            result = upload_model(
                args.checkpoint,
                args.repo_id,
                private=not args.public,
                revision=args.revision,
                commit_message=args.commit_message,
            )
            print(f"Uploaded {args.repo_id}@{args.revision}: {result}")
            return 0

        _check_output_dir(args.output_dir, args.overwrite)
        download_model(
            args.repo_id,
            revision=args.revision,
            local_dir=args.output_dir,
            cache_dir=args.cache_dir,
        )
        print(f"Downloaded and validated {args.repo_id}@{args.revision} to {args.output_dir}")
        return 0
    except Exception as error:
        print(f"steppo-hf: error: {error}", file=sys.stderr)
        return 1
