"""Tests for stable and explicit diagnostic output-directory naming."""

from dataclasses import dataclass
from pathlib import Path

from research.ode.output_dirs import resolve_output_dir


@dataclass
class _Config:
    system: str
    seed: int
    checkpoint_dir: str


def test_fingerprint_path_is_stable_and_excludes_consumer_only_fields(tmp_path):
    first = _Config(system="scalar_decay", seed=1, checkpoint_dir="checkpoint_a")
    same_analysis = _Config(system="scalar_decay", seed=1, checkpoint_dir="checkpoint_b")
    changed_analysis = _Config(system="scalar_decay", seed=2, checkpoint_dir="checkpoint_a")

    first_dir = resolve_output_dir(
        "diagnostics", first, exclude=("checkpoint_dir",), root=str(tmp_path)
    )
    same_dir = resolve_output_dir(
        "diagnostics", same_analysis, exclude=("checkpoint_dir",), root=str(tmp_path)
    )
    changed_dir = resolve_output_dir(
        "diagnostics", changed_analysis, exclude=("checkpoint_dir",), root=str(tmp_path)
    )

    assert first_dir == same_dir
    assert first_dir != changed_dir
    assert (tmp_path / "diagnostics").is_dir()


def test_explicit_name_is_used_as_directory_slug(tmp_path):
    config = _Config(system="van_der_pol", seed=3, checkpoint_dir="checkpoint_2")

    path = resolve_output_dir("inference", config, name="paper-figure", root=str(tmp_path))
    parts = Path(path).parts

    assert parts[-3] == "inference"
    assert parts[-1] == "paper-figure"
    assert Path(path).is_dir()
