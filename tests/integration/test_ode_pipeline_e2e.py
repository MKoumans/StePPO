"""E2e tests driving scripts/run-all-ode.sh and scripts/launch_experiments.sh
for real against a tiny scalar_decay run, checking actual dir/file shapes.

Marked `pipeline` and excluded from the default run (pyproject.toml addopts).
    pytest -m pipeline tests/test_ode_pipeline_e2e.py
"""

import datetime
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = str(Path(__file__).resolve().parents[2])

pytestmark = pytest.mark.pipeline

# Shrunk scalar_decay_default.yaml (cheapest ODE system) for a fast smoke test.
_TINY_BASE_CONFIG = """\
model: sd_1
exp_name: "pipeline_e2e_test"
seed: 0
total_iters: 2
num_envs: 4
num_eval_episodes: 2
rollout_steps: 8
eval_interval: 1
log_interval: 1

vae:
  kl_weight: 0.05
  rew_loss_coeff: 0.40
  state_loss_coeff: 0.35
  task_loss_coeff: 0.25
  lr: 0.001
  num_updates_per_iter: 1
  batch_size: 4

ppo:
  lr: 0.0007
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  entropy_coeff: 0.05
  value_coeff: 0.5
  num_epochs: 1
  num_minibatches: 1

training:
  ema_norm: true
  grad_clip_vae: 10.0
  grad_clip_ppo: 0.5
  reward_normalization:
    enabled: true

env:
  system: scalar_decay
  t_end: 1.0
  dt0: 0.1
  rtol: 1.0e-4
  atol: 1.0e-6
  dt_min: 1.0e-8
  dt_max: 1.0
  dt_log_gain: 2.5
  immediate_dt_action: true
  survival_penalty: 0.0
  lam_min: 1.0
  lam_max: 100.0
  sample_lam: true
  obs_features: [state, step_context]
"""

# Sweeps one cheap field into 2 configs to exercise factorial config generation.
_TINY_SWEEP_SPEC = """\
ppo.entropy_coeff: 0.05, 0.1
"""


def _run(cmd, env_extra=None, timeout=300):
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


class TestRunAllOdePipeline:
    """Mimics a Slurm-less `run-all-ode.sh` run + the per-run post-run-analysis suite."""

    RUN_LOG_RE = re.compile(r"- log: .*/ode_scalar_decay_([0-9a-f]{8})\.log")

    def test_config_checkpoint_outputs_and_analysis(self):
        run_date = datetime.date.today().strftime("%Y%m%d")
        result = _run(
            [
                "bash",
                "scripts/run-all-ode.sh",
                "scalar_decay",
                "--gpus",
                "0",
                "--total_iters",
                "2",
                "--num_envs",
                "4",
                "--rollout_steps",
                "8",
                "--eval_interval",
                "1",
                "--log_interval",
                "1",
                "--num_eval_episodes",
                "2",
                "--vae.num_updates_per_iter",
                "1",
                "--vae.batch_size",
                "4",
                "--ppo.num_epochs",
                "1",
                "--ppo.num_minibatches",
                "1",
            ],
            env_extra={"_RUN_ALL_DETACHED": "1"},
            timeout=300,
        )
        match = self.RUN_LOG_RE.search(result.stdout)
        run_uid = match.group(1) if match else None
        # TrainConfig.run_dir = runs_dir/run_date/env_family/env.system/run_uid
        run_dir = (
            os.path.join(REPO_ROOT, "outputs", "runs", run_date, "ode", "scalar_decay", run_uid)
            if run_uid
            else None
        )

        try:
            assert result.returncode == 0, result.stdout + result.stderr
            assert run_uid, f"Could not find run log line in stdout:\n{result.stdout}"
            assert os.path.isdir(run_dir), f"Run dir missing: {run_dir}"

            # -- Checkpoint creation --
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            assert os.path.isfile(os.path.join(ckpt_dir, "config.yaml"))
            assert os.path.isdir(os.path.join(ckpt_dir, "checkpoint_2")), (
                "final checkpoint (total_iters=2) missing"
            )
            assert os.path.isfile(os.path.join(ckpt_dir, "checkpoint_2", "_CHECKPOINT_METADATA"))

            # -- Outputs structure --
            out_dir = os.path.join(run_dir, "outputs")
            assert os.path.isfile(os.path.join(out_dir, "benchmark_summary.txt"))
            assert os.path.isfile(os.path.join(out_dir, "scalar_decay_default_metrics.json"))
            assert os.path.isfile(os.path.join(out_dir, "scalar_decay_default_metrics.png"))

            # -- Post-run analysis (scripts/ode-post-run-analysis.sh --run-dir) --
            analysis_dir = os.path.join(out_dir, "post_run_analysis")
            assert os.path.isdir(os.path.join(analysis_dir, "compare"))
            assert os.path.isfile(os.path.join(analysis_dir, "steps_vs_mu.png"))
            assert os.path.isdir(os.path.join(analysis_dir, "analyze_rollout"))
            assert os.path.isfile(os.path.join(analysis_dir, "analyze_rollout", "belief_mu.png"))
            assert os.path.isdir(os.path.join(analysis_dir, "collapse_vs_task"))
            assert os.path.isfile(
                os.path.join(analysis_dir, "collapse_vs_task", "collapse_onset_vs_mu.png")
            )
        finally:
            if run_uid:
                shutil.rmtree(
                    os.path.join(
                        REPO_ROOT, "outputs", "runs", run_date, "ode", "scalar_decay", run_uid
                    ),
                    ignore_errors=True,
                )


class TestLaunchExperimentsPipeline:
    """Mimics `launch_experiments.sh`: generates a factorial config sweep, runs
    one experiment per config grouped under one UID, then aggregates via
    scripts/ode-experiment-analysis.sh."""

    UID_RE = re.compile(r"Experiment UID: ([0-9a-f]{8})")
    TMP_DIR = os.path.join(REPO_ROOT, "tests", "tmp")

    def test_config_generation_checkpoints_outputs_and_analysis(self):
        os.makedirs(self.TMP_DIR, exist_ok=True)
        base_config = os.path.join(self.TMP_DIR, "scalar_decay_tiny_base.yaml")
        sweep_spec = os.path.join(self.TMP_DIR, "scalar_decay_tiny_sweep.yml")
        with open(base_config, "w") as f:
            f.write(_TINY_BASE_CONFIG)
        with open(sweep_spec, "w") as f:
            f.write(_TINY_SWEEP_SPEC)

        result = _run(
            [
                "bash",
                "scripts/launch_experiments.sh",
                "scalar_decay",
                "0",
                "--base",
                str(base_config),
                "--sweep",
                str(sweep_spec),
                "--post-run-analysis",
                "false",  # aggregated analysis exercised separately below
            ],
            env_extra={"_LAUNCH_DETACHED": "1"},
            timeout=300,
        )
        match = self.UID_RE.search(result.stdout)
        experiment_uid = match.group(1) if match else None
        experiment_date = datetime.date.today().strftime("%Y%m%d")
        experiment_dir = (
            os.path.join(REPO_ROOT, "outputs", "experiments", experiment_date, experiment_uid)
            if experiment_uid
            else None
        )

        try:
            assert result.returncode == 0, result.stdout + result.stderr
            assert experiment_uid, f"Could not find 'Experiment UID:' in stdout:\n{result.stdout}"
            assert os.path.isdir(experiment_dir)

            # -- Config creation: factorial sweep wrote 2 generated configs --
            config_dir = os.path.join(experiment_dir, "configs")
            e1_config = os.path.join(config_dir, "scalar_decay_e1.yaml")
            e2_config = os.path.join(config_dir, "scalar_decay_e2.yaml")
            assert os.path.isfile(e1_config)
            assert os.path.isfile(e2_config)
            assert "entropy_coeff: 0.05" in open(e1_config).read()
            assert "entropy_coeff: 0.1" in open(e2_config).read()

            # -- Checkpoint creation: one subdir per run, grouped by experiment UID --
            checkpoints_root = os.path.join(experiment_dir, "checkpoints")
            run_uids = [
                d
                for d in os.listdir(checkpoints_root)
                if os.path.isdir(os.path.join(checkpoints_root, d))
            ]
            assert len(run_uids) == 2, f"Expected 2 runs, found {run_uids}"
            for run_uid in run_uids:
                run_ckpt_dir = os.path.join(checkpoints_root, run_uid)
                assert os.path.isfile(os.path.join(run_ckpt_dir, "config.yaml"))
                assert os.path.isdir(os.path.join(run_ckpt_dir, "checkpoint_2"))

            # -- Outputs structure: one subdir per run, grouped by experiment UID --
            outputs_root = os.path.join(experiment_dir, "outputs")
            for run_uid in run_uids:
                run_out_dir = os.path.join(outputs_root, run_uid)
                assert os.path.isfile(os.path.join(run_out_dir, "benchmark_summary.txt"))
                metrics_jsons = [f for f in os.listdir(run_out_dir) if f.endswith("_metrics.json")]
                assert len(metrics_jsons) == 1

            # -- Post-run-analysis, run separately (per-run suite + aggregate) --
            analysis_result = _run(
                [
                    "bash",
                    "scripts/ode-experiment-analysis.sh",
                    "--experiment_uid",
                    experiment_uid,
                    "--experiment_date",
                    experiment_date,
                    "--num_envs",
                    "4",
                    "--num_task_points",
                    "2",
                    "--num_step_diff_episodes",
                    "8",
                    "--step_diff_bins",
                    "4",
                ],
                timeout=300,
            )
            assert analysis_result.returncode == 0, analysis_result.stdout + analysis_result.stderr

            for run_uid in run_uids:
                run_analysis_dir = os.path.join(outputs_root, run_uid)
                assert os.path.isdir(os.path.join(run_analysis_dir, "compare"))
                assert os.path.isfile(os.path.join(run_analysis_dir, "steps_vs_mu.png"))

            comparison_dir = os.path.join(outputs_root, "comparison")
            assert os.path.isdir(comparison_dir)
            assert len(os.listdir(comparison_dir)) > 0
        finally:
            if experiment_uid:
                shutil.rmtree(experiment_dir, ignore_errors=True)
            os.remove(base_config) if os.path.exists(base_config) else None
            os.remove(sweep_spec) if os.path.exists(sweep_spec) else None
