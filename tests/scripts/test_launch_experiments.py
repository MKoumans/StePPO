"""Tests for scripts/launch_experiments.sh's config-discovery error paths.

These don't launch any training; they only check that missing/unknown
experiment configs are reported with a clear message and non-zero exit,
rather than failing silently or with a confusing GPU/generation error.
"""

import datetime
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
SCRIPT = "scripts/launch_experiments.sh"

UID_RE = re.compile(r"Experiment UID: ([0-9a-f]{8})")


def _run(*args, workdir=None):
    # Disable the local-mode auto-detach (setsid+nohup) so the error paths
    # under test run synchronously and their stdout/exit code are captured
    # directly, instead of the parent exiting 0 immediately after forking.
    env = {**os.environ, "_LAUNCH_DETACHED": "1"}
    if workdir is not None:
        env["SLURM_SUBMIT_DIR"] = str(workdir)
    return subprocess.run(
        ["bash", SCRIPT, *args],
        cwd=workdir or REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def _cleanup(result, workdir=REPO_ROOT):
    match = UID_RE.search(result.stdout)
    if not match:
        return
    date = datetime.date.today().strftime("%Y%m%d")
    for base in ("checkpoints/experiments", "outputs/experiments"):
        shutil.rmtree(f"{workdir}/{base}/{date}/{match.group(1)}", ignore_errors=True)


def test_missing_configs_reports_clear_error_and_fails(tmp_path):
    # brusselator has neither hand-authored configs/envs/ode/brusselator/brusselator_e*.yaml
    # nor a *_experiment.yml sweep spec to auto-default to (only *_default.yaml) -- this is
    # the exact case that broke silently for a user running the script without --base/--sweep.
    # (van_der_pol now ships both *_default.yaml and *_experiment.yml, so it no longer hits
    # this path -- it auto-generates configs from them instead.)
    # Run in a minimal writable repo-shaped directory so the shell script's
    # experiment-output setup stays isolated from the checkout.
    (tmp_path / "scripts/lib").mkdir(parents=True)
    shutil.copy(Path(REPO_ROOT) / SCRIPT, tmp_path / SCRIPT)
    shutil.copy(Path(REPO_ROOT) / "scripts/lib/device.sh", tmp_path / "scripts/lib/device.sh")
    (tmp_path / "configs/envs/ode/brusselator").mkdir(parents=True)
    result = _run("brusselator", "0", workdir=tmp_path)
    try:
        assert result.returncode != 0
        assert "No experiment configs found matching" in result.stdout
        assert "Generate them first with research/ode/generate_experiments.py" in result.stdout
    finally:
        _cleanup(result, tmp_path)


def test_unknown_system_fails_before_creating_any_experiment_dir():
    result = _run("not_a_real_system", "0")
    assert result.returncode != 0
    assert "Unknown system: not_a_real_system" in result.stdout
    # Should fail before the UID/checkpoint/output setup even runs.
    assert "Experiment UID:" not in result.stdout


def test_missing_system_arg_fails_with_usage():
    result = _run()
    assert result.returncode != 0
    assert "Usage:" in (result.stdout + result.stderr)


def test_dataset_preflight_diagnostics_do_not_pollute_missing_count():
    source = Path(SCRIPT).read_text()

    assert (
        "run_py src/steppo/envs/ode/check_oracle_dataset.py "
        '--config "${CONFIG_DIR}/${CFG}" >&2 || MISSING_CFGS+=("${CFG}")'
    ) in source


def test_explicit_gpu_is_bound_during_serial_dataset_generation():
    source = Path(SCRIPT).read_text()

    assert 'GENERATION_GPU="${GPU_ARG%%,*}"' in source
    assert source.index('GENERATION_GPU="${GPU_ARG%%,*}"') < source.index("run_py()")
    assert 'CUDA_VISIBLE_DEVICES="${GENERATION_GPU}" PYTHONPATH="${WORKDIR}" python "$@"' in source
