"""Pure file-discovery and naming tests for the factorial ODE analysis CLI."""

import json
import sys
from pathlib import Path

import pytest

_ODE_RESEARCH_DIR = Path(__file__).resolve().parents[2] / "research" / "ode"
sys.path.insert(0, str(_ODE_RESEARCH_DIR))

from analyse_experiment import (  # noqa: E402
    EXPERIMENT_DESIGN,
    discover_runs,
    exp_label,
    find_checkpoint,
    load_metrics,
    short_label,
)


def test_factorial_labels_cover_each_design_and_baseline_is_explicit():
    labels = [exp_label(eid) for eid in EXPERIMENT_DESIGN]
    short_labels = [short_label(eid) for eid in EXPERIMENT_DESIGN]

    assert len(set(labels)) == len(EXPERIMENT_DESIGN)
    assert len(set(short_labels)) == len(EXPERIMENT_DESIGN)
    assert short_label("e1") == "e1 (baseline)"
    assert short_label("e8") == "e8 (D,T,E)"
    assert "dir+task+ema" in exp_label("e8")


def test_discover_runs_uses_matching_run_directories_and_latest_run(tmp_path):
    old = tmp_path / "run_001"
    new = tmp_path / "run_002"
    unrelated = tmp_path / "not_a_run"
    for run_dir in (old, new, unrelated):
        run_dir.mkdir()
    (old / "scalar_decay_e1_metrics.json").write_text("[]")
    (new / "scalar_decay_e1_metrics.json").write_text("[]")
    (new / "scalar_decay_e9_metrics.json").write_text("[]")
    (unrelated / "scalar_decay_e2_metrics.json").write_text("[]")

    assert discover_runs(str(tmp_path), "scalar_decay_e") == {"e1": str(new)}


def test_load_metrics_prefers_requested_name_then_uses_single_file_fallback(tmp_path):
    preferred = [{"iteration": 1, "return": 10.0}]
    other = [{"iteration": 1, "return": -1.0}]
    (tmp_path / "scalar_decay_e1_metrics.json").write_text(json.dumps(preferred))
    (tmp_path / "fallback_metrics.json").write_text(json.dumps(other))

    assert load_metrics(str(tmp_path), "scalar_decay_e1") == preferred

    (tmp_path / "scalar_decay_e1_metrics.json").unlink()
    (tmp_path / "fallback_metrics.json").write_text(json.dumps(other))
    assert load_metrics(str(tmp_path), "scalar_decay_e1") == other


def test_load_metrics_returns_none_for_missing_run_metrics(tmp_path, capsys):
    assert load_metrics(str(tmp_path), "scalar_decay_e1") is None
    assert "No metrics JSON found" in capsys.readouterr().out


def test_find_checkpoint_uses_latest_zero_padded_iteration_with_run_metrics(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    run_dir = checkpoints / "run_001"
    (run_dir / "checkpoint_0001").mkdir(parents=True)
    (run_dir / "checkpoint_0010").mkdir()
    metrics = tmp_path / "outputs" / "run_001" / "scalar_decay_e1_metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("[]")

    assert find_checkpoint(str(checkpoints), "scalar_decay_e1") == str(run_dir / "checkpoint_0010")


@pytest.mark.xfail(
    strict=True,
    reason="find_checkpoint currently sorts checkpoint suffixes lexically, so 9 outranks 10",
)
def test_find_checkpoint_selects_highest_numeric_iteration(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    run_dir = checkpoints / "run_001"
    (run_dir / "checkpoint_9").mkdir(parents=True)
    (run_dir / "checkpoint_10").mkdir()
    metrics = tmp_path / "outputs" / "run_001" / "scalar_decay_e1_metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("[]")

    assert find_checkpoint(str(checkpoints), "scalar_decay_e1") == str(run_dir / "checkpoint_10")


def test_find_checkpoint_ignores_unrelated_runs_and_returns_none(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "run_001" / "checkpoint_0005").mkdir(parents=True)

    assert find_checkpoint(str(checkpoints), "scalar_decay_e1") is None


@pytest.mark.parametrize("argv", [[], ["--runs", "one", "--discover", "two"]])
def test_discovery_arguments_require_exactly_one_run_source(monkeypatch, argv):
    from analyse_experiment import parse_args

    monkeypatch.setattr(sys, "argv", ["analyse_experiment.py", *argv])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2
