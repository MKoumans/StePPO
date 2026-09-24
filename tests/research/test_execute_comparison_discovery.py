"""Filesystem-level tests for experiment run discovery."""

from research.ode.execute_comparison import discover_runs


def _make_run(outputs_root, checkpoints_root, name, *, checkpoint_steps=(2, 10), config=True):
    relative_run = f"batch/{name}"
    output_dir = outputs_root / relative_run
    output_dir.mkdir(parents=True)
    (output_dir / f"{name}_metrics.json").write_text("{}")

    checkpoint_dir = checkpoints_root / relative_run
    checkpoint_dir.mkdir(parents=True)
    if config:
        (checkpoint_dir / "config.yaml").write_text("env: {}\n")
    for step in checkpoint_steps:
        (checkpoint_dir / f"checkpoint_{step}").mkdir()
    return output_dir, checkpoint_dir


def test_discover_runs_sorts_by_experiment_and_seed_and_uses_latest_numeric_checkpoint(tmp_path):
    outputs = tmp_path / "outputs"
    checkpoints = tmp_path / "checkpoints"
    outputs.mkdir()
    checkpoints.mkdir()
    _make_run(outputs, checkpoints, "van_der_pol_e2")
    _make_run(outputs, checkpoints, "van_der_pol_e1_r10")
    _make_run(outputs, checkpoints, "van_der_pol_e1_r2")

    runs = discover_runs(str(checkpoints), str(outputs))

    assert [(run["eid"], run["seed"]) for run in runs] == [(1, 2), (1, 10), (2, None)]
    assert runs[0]["run_uid"] == "van_der_pol_e1_r2"
    assert runs[0]["checkpoint_dir"].endswith("checkpoint_10")
    assert runs[0]["config_path"].endswith("config.yaml")


def test_discover_runs_skips_unparseable_incomplete_runs_with_diagnostics(tmp_path, capsys):
    outputs = tmp_path / "outputs"
    checkpoints = tmp_path / "checkpoints"
    outputs.mkdir()
    checkpoints.mkdir()
    _make_run(outputs, checkpoints, "malformed", checkpoint_steps=())
    _make_run(outputs, checkpoints, "model_e3", config=False)
    _make_run(outputs, checkpoints, "model_e4", checkpoint_steps=())

    runs = discover_runs(str(checkpoints), str(outputs))

    assert runs == []
    output = capsys.readouterr().out
    assert "could not parse experiment id" in output
    assert "no config.yaml" in output
    assert "no checkpoint_* dirs" in output
