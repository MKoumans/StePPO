import os
import sys
import tempfile
from glob import glob

# Prefer the checkout containing this conftest over an editable install that
# may point at another worktree (common when pytest is launched from an IDE).
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _repo_path in (os.path.join(_repo_root, "src"), _repo_root):
    if _repo_path not in sys.path:
        sys.path.insert(0, _repo_path)

# Must run before any test module imports jax, or it'll grab memory on every
# visible GPU. Respects GPUS/CUDA_VISIBLE_DEVICES if already set (e.g. in CI).
#
# Under pytest-xdist (`-n N`), each worker can be pinned to its own GPU by
# setting TEST_GPU_POOL to a comma-separated list of device indices (e.g.
# "0,1,2"); worker gwK gets pool[K % len(pool)]. Unset by default so a plain
# `pytest` run keeps today's single-GPU behavior.
_xdist_worker = os.environ.get("PYTEST_XDIST_WORKER")  # "gw0", "gw1", ... or None
_gpu_pool = os.environ.get("TEST_GPU_POOL")
if _gpu_pool:
    _pool = _gpu_pool.split(",")
    if _xdist_worker is None:
        # The controller's setting is inherited by workers. Workers below
        # replace it with their own pool entry before importing JAX.
        os.environ["CUDA_VISIBLE_DEVICES"] = _pool[0]
    else:
        _idx = int(_xdist_worker.removeprefix("gw"))
        os.environ["CUDA_VISIBLE_DEVICES"] = _pool[_idx % len(_pool)]

os.environ.setdefault("GPUS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ["GPUS"])
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".2")
_gpu_mode = os.environ.get("STEPPO_REQUIRE_GPU", "auto").lower()
if _gpu_mode not in {"auto", "0", "1"}:
    raise ValueError("STEPPO_REQUIRE_GPU must be 'auto', '0', or '1'")
if _gpu_mode == "1":
    # GPU CI can opt into a hard failure if device passthrough is misconfigured.
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
elif _gpu_mode == "0":
    # Explicit CPU runs must not initialize CUDA or train slowly by accident.
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
elif not glob("/dev/nvidia[0-9]*"):
    # Avoid a failed CUDA plugin probe in containers that have no device nodes.
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax  # noqa: E402
import pytest  # noqa: E402

try:
    _jax_backend = jax.default_backend()
except Exception as exc:
    if _gpu_mode == "1":
        raise RuntimeError(
            "STEPPO_REQUIRE_GPU=1 but pytest cannot access a JAX GPU; check "
            "device passthrough or use STEPPO_REQUIRE_GPU=auto"
        ) from exc
    raise
if _gpu_mode == "1" and _jax_backend != "gpu":
    raise RuntimeError(
        f"STEPPO_REQUIRE_GPU=1 but JAX selected {_jax_backend!r}; "
        "check device passthrough or use STEPPO_REQUIRE_GPU=auto"
    )

_gpu_available = _jax_backend == "gpu"

# Persist compiled XLA executables in a writable location so re-running the
# suite (or hitting the same jitted shape from another test) can reuse
# compiles. The worktree may be mounted read-only inside a test sandbox.
jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "steppo-jax-cache"),
    ),
)


def pytest_collection_modifyitems(items):
    """Skip numerical training/solver integration tests when no GPU is exposed.

    These cases exercise JAX compilation and long rollout/training loops. Let
    the pure Python, config, cache, and algorithm tests run on CPU in sandboxes.
    GPU CI can set STEPPO_REQUIRE_GPU=1 to make missing hardware a hard error.
    """
    if _gpu_available:
        return

    gpu_only_modules = (
        # These files exercise policy training, full rollouts, checkpoint I/O,
        # or real adaptive ODE solves rather than isolated CPU-safe logic.
        "tests/integration/",
        "tests/training/test_eval.py",
        "tests/training/test_pid_solve.py",
        "tests/training/test_rollout.py",
        "tests/training/test_trainer.py",
    )
    gpu_only_cases = {
        "tests/envs/test_envs_ode.py": {
            "test_step_jit",
            "test_done_at_max_steps",
            "test_nan_solver_output_rejected",
            "test_obs_nan_guard_on_extreme_state",
            "test_stiff_rollout_stays_finite",
            "test_step_reaches_t_end_naturally_marks_success",
            "test_step_truncated_by_max_steps_marks_not_success",
        },
        "tests/envs/test_pid_env_vs_diffrax.py": {
            "test_env_driven_pid_is_close_to_diffrax",
            "test_env_driven_pid_does_not_silently_burn_the_full_budget",
        },
        "tests/envs/test_ode_comprehensive.py": {
            # Environment.step performs a real Diffrax attempt in these cases.
            "TestReward::test_progress_reward_proportional_to_time_advanced",
            "TestReward::test_survival_penalty_subtracted",
            "TestActionScaling::test_action_zero_keeps_dt",
            "TestActionScaling::test_action_plus_one_doubles_dt",
            "TestActionScaling::test_action_minus_one_halves_dt",
            "TestActionScaling::test_dt_clipped_to_dt_max",
            "TestActionScaling::test_dt_clipped_to_dt_min",
            "TestObservationNormalization::test_step_context_log_error_finite",
            "TestObservationNormalization::test_step_context_keep_step_binary",
            "TestObservationNormalization::test_solver_trend_budget_in_01",
            "TestObservationNormalization::test_solver_trend_accept_ema_in_01",
            "TestObservationNormalization::test_obs_finite_after_many_steps",
            "TestEMATracking::test_accept_ema_update_on_accepted_step",
            "TestEMATracking::test_accept_ema_decreases_on_rejection",
            "TestEMATracking::test_log_error_ema_bounded",
            "TestDoneCondition::test_done_at_max_steps",
            "TestDoneCondition::test_done_at_t_end",
            "TestDoneCondition::test_budget_exhausted_increments",
            "TestAcceptReject::test_rejected_step_preserves_state",
            "TestAcceptReject::test_accepted_step_advances_time",
            "TestAcceptReject::test_nan_solver_output_rejected",
            "TestEpisodesPerTrial::",
            "TestWarmstart::test_pid_rollout_collection",
            "TestWarmstart::test_pid_actions_within_range",
            "TestWarmstart::test_oracle_dataset_generation",
            "TestWarmstart::test_encode_oracle_batch_attaches_beliefs",
            "TestODEDynamics::",
            "TestMultiSystem::test_reset_and_step_finite",
            "TestBeliefReset::test_rollout_with_belief_reset",
            "TestBeliefReset::test_rollout_without_belief_reset",
            "TestRolloutBatchFields::",
        },
        "tests/training/test_learned_controller.py": {
            "test_task_conditioned_model_runs_through_diffrax_controller",
        },
        "tests/training/test_ode_task_distribution_training.py": {
            "test_reset_and_step_finite_across_sampled_tasks",
            "test_vmap_reset_over_task_batch",
            "test_training_on_task_distribution_runs",
            "test_training_on_task_distribution_finite_metrics",
        },
        "tests/training/test_policy_algos.py": {
            "test_ppo_algorithm_update_changes_params_and_returns_metrics",
        },
        "tests/training/test_ppo.py": {
            "test_ppo_loss_decreases",
            "test_gradient_clipping_ppo",
        },
        "tests/training/test_vae_training.py": {
            "test_elbo_loss_jit_compilable",
            "test_elbo_loss_finite_on_random_data",
            "test_elbo_metrics_keys",
            "test_masks_zero_out_loss",
            "test_train_step_loss_decreases",
            "test_train_step_finite",
            "test_gradient_finite",
            "test_ode_vae_loss_finite_after_rollout",
        },
        "tests/models/test_model_artifact.py": {
            "test_checkpoint_artifact_export_resolves_run_directory",
        },
        "tests/utils/test_logging_export.py": {
            "test_checkpointing_save_and_restore",
        },
        "tests/utils/test_checkpoint.py": {
            "test_load_control_envelopes_round_trip",
            "test_load_control_envelopes_missing_key_returns_empty_dict",
            "test_load_checkpoint_falls_back_to_legacy_ppo_params_key",
        },
    }
    skip_gpu = pytest.mark.skip(
        reason="omitted from CPU-only runs to avoid costly solver, neural, or checkpoint work",
    )
    for item in items:
        nodeid = item.nodeid.replace("\\", "/")
        path, *test_path = nodeid.split("::")
        file_cases = gpu_only_cases.get(path, ())
        exact_case = bool(test_path and test_path[-1].split("[")[0] in file_cases)
        partial_case = any(case in nodeid for case in file_cases if "::" in case)
        if any(module in nodeid for module in gpu_only_modules) or exact_case or partial_case:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _reset_jax_x64_config():
    """`apply_precision()` (called by any trainer built over a float64-precision
    ODE env) flips jax_enable_x64 globally with no reset, so it leaks across
    test files depending on run order. Restore it after every test so tests
    relying on the float32 default aren't broken by an earlier test's trainer."""
    prev = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", prev)
