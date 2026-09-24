"""Contract tests for the GPU selection options exposed by shell entry points."""

import os
import subprocess
from pathlib import Path

import pytest

GPU_BACKED_SCRIPTS = (
    "scripts/run.sh",
    "scripts/run-all-ode.sh",
    "scripts/generate-pid-dataset.sh",
    "scripts/generate-pid-datasets-all-ode.sh",
    "scripts/launch_experiments.sh",
    "scripts/ode-post-run-analysis.sh",
    "scripts/ode-experiment-analysis.sh",
    "scripts/ode-average-seeds.sh",
)


REPO_ROOT = Path(__file__).resolve().parents[2]


HELP_CONTRACT_SCRIPTS = GPU_BACKED_SCRIPTS + ("scripts/stop_experiments.sh",)


@pytest.mark.parametrize("script", GPU_BACKED_SCRIPTS)
def test_gpu_backed_script_help_advertises_gpus(script):
    result = subprocess.run(
        ["bash", script, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    help_text = result.stdout + result.stderr
    assert "--gpus" in help_text
    assert "--gpus=" in help_text


def test_shared_gpu_argument_extractor_preserves_other_arguments():
    command = 'source scripts/lib/device.sh\nGPU_ARG=""\nFORWARD_ARGS=()\ndevice_extract_gpu_args GPU_ARG FORWARD_ARGS "$@"\nprintf \'gpu=%s\\n\' "$GPU_ARG"\nprintf \'args=%s\\n\' "${FORWARD_ARGS[*]}"'
    result = subprocess.run(
        ["bash", "-c", command, "script", "--gpus=4,5", "--rollout_steps", "400"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "gpu=4,5",
        "args=--rollout_steps 400",
    ]


@pytest.mark.parametrize("script", HELP_CONTRACT_SCRIPTS)
def test_script_help_stays_concise(script):
    result = subprocess.run(
        ["bash", script, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) <= 20


def test_shared_gpu_argument_extractor_does_not_use_cuda_visible_devices():
    command = 'source scripts/lib/device.sh\nGPU_ARG=""\nFORWARD_ARGS=()\ndevice_extract_gpu_args GPU_ARG FORWARD_ARGS "$@"\nprintf "gpu=%s\\n" "$GPU_ARG"'
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    result = subprocess.run(
        ["bash", "-c", command, "script"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "gpu=\n"


def test_shared_gpu_runner_applies_optional_device():
    command = 'source scripts/lib/device.sh\ndevice_run_with_gpu 4 bash -c \'printf "gpu=%s" "$CUDA_VISIBLE_DEVICES"\'\n'
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "gpu=4"


def test_run_all_ode_advertises_dataset_generation():
    result = subprocess.run(
        ["bash", "scripts/run-all-ode.sh", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "--generate-datasets" in result.stdout


def test_run_all_ode_contains_dataset_generation_preflight():
    source = Path("scripts/run-all-ode.sh").read_text()
    assert "check_oracle_dataset.py" in source
    assert "generate_pid_training_dataset.py" in source
    assert "generate_pid_eval_dataset.py" in source
    assert "--print-eval-args" in source


def test_pid_batch_gpu_argument_overrides_cuda_visible_devices(tmp_path):
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text("#!/bin/sh\nexit 0\n")
    fake_sbatch.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "3"
    env["SLURM_SUBMIT_DIR"] = str(tmp_path)
    config = tmp_path / "configs/envs/ode/van_der_pol"
    config.mkdir(parents=True)
    (config / "van_der_pol_default.yaml").write_text("env: {}\n")
    result = subprocess.run(
        ["bash", "scripts/generate-pid-datasets-all-ode.sh", "--gpus", "4,5", "van_der_pol"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "GPU=4" in result.stdout
    assert "GPU=3" not in result.stdout


@pytest.mark.parametrize("script", GPU_BACKED_SCRIPTS)
def test_help_works_when_slurm_submit_dir_is_scripts(script):
    env = os.environ.copy()
    env["SLURM_SUBMIT_DIR"] = str(REPO_ROOT / "scripts")
    result = subprocess.run(
        ["bash", script, "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "script",
    ("scripts/run.sh", "scripts/generate-pid-dataset.sh"),
)
def test_single_job_scripts_export_selected_gpu(script):
    source = Path(script).read_text()
    assert 'export CUDA_VISIBLE_DEVICES="${GPU_ARG}"' in source
