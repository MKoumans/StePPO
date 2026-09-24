"""Dataclass schemas for the diagnostics scripts' YAML configs (configs/diagnostics/...).

Loaded via steppo.configs.base_config.load_config_from_yaml, which gives these
the same strict-unknown-key checking and dotted-override support (via
apply_dotted_overrides) as the main TrainConfig — see that module for details.
"""

from dataclasses import dataclass, field


@dataclass
class WallclockVsMuConfig:
    """One wallclock_vs_mu.py run: PID/RL controller latency across a task sweep."""

    name: str | None = (
        None  # outputs/diagnostics/<date>/<name> dir; default: a content fingerprint of this config
    )
    env_config: str | None = (
        None  # TrainConfig YAML; not needed with hf_repo, or checkpoint with a bundled config.yaml
    )
    checkpoint: str | None = (
        None  # local RL checkpoint (see steppo.utils.checkpoint.resolve_checkpoint)
    )
    hf_repo: str | None = None  # HF repo-id with an RL model artifact (alternative to checkpoint)
    revision: str = "main"  # git branch/tag within hf_repo
    skip_pid: bool = False  # skip PID timing (e.g. timing several RL variants back to back)
    tag: str | None = (
        None  # RL output filename suffix (default: revision, or "ckpt" for checkpoint)
    )
    device: str | None = (
        None  # "cpu" | "gpu"; default: auto (cpu for mode=single, gpu for mode=batched)
    )
    mode: str = "single"  # "single" | "batched"
    n_sweep: int = 30
    batch_size: int = 30  # samples per parameter value when report_batch_time is set
    report_batch_time: bool = False  # report total elapsed batch time instead of amortized ms/mu
    param_min: float | None = None  # override lower end of the parameter sweep
    param_max: float | None = None  # override upper end of the parameter sweep
    save_repeats: bool = False  # save per-repeat timing arrays in the output NPZ
    max_steps: int = (
        8192  # step budget; 8192 = "unlimited" (matches compare.py's PID-unlimited row)
    )
    output: str | None = None  # output path for RL benchmark results
    pid_output: str | None = None  # output path for PID benchmark results
    repeats: int = 20  # timing repeats per mu (single) or per batch (batched)
    verbose: bool = False  # print per-repeat timing


@dataclass
class ModelSweepConfig:
    """One model_sweep.py run: e_int, steps and wall-clock vs mu for a list of models on one grid.

    `models` is a list of dicts (validated by model_sweep.py), each with a "type"
    ("pid", "pid_oracle", "pid_tuned", "checkpoint" or "deeponet"), a "label", and
    type-specific keys, e.g. {"type": "pid_tuned", "label": "PID (tuned)", "kp": 0.06,
    "ki": 1.9, "kd": 0.02}. `checkpoint_steppo`, when set, appends an RL (StePPO)
    checkpoint entry.
    """

    name: str | None = None  # outputs/diagnostics/<date>/<name> dir; default: a content fingerprint
    env_config: str = (
        ""  # TrainConfig YAML; supplies the system, solver tolerances, and default mu range
    )
    models: list = field(default_factory=list)
    checkpoint_steppo: str | None = None  # optional StePPO checkpoint path — see docstring above
    n_sweep: int = 30
    param_min: float | None = (
        None  # override lower end of the mu sweep (default: min of env_config.test_bins)
    )
    param_max: float | None = (
        None  # override upper end of the mu sweep (default: max of env_config.test_bins)
    )
    max_steps: int = 8192  # step budget for the dense (steps/e_int) solve; 8192 = "unlimited"
    num_repeats: int = 8  # distinct episodes (keys) solved per mu, for steps/e_int means
    ref_tol_factor: float = 0.01  # reference trajectory tolerance = env_config's rtol/atol * this
    wallclock_repeats: int = 20  # timing repeats per mu
    wallclock_batch_size: int = (
        16  # episodes solved together per timed batch (see utils.time_solve_fn)
    )
    seed: int = 0
    device: str | None = (
        None  # "cpu" | "gpu"; default: gpu (wallclock timing wants an uncontended device)
    )
    output: str | None = (
        None  # output PNG/txt path; default: outputs/diagnostics/<date>/<name>/model_sweep.png
    )


@dataclass
class WallclockBatchSizeSweepConfig:
    """One wallclock_batch_size_sweep.py run: sweeps batch sizes over wallclock_vs_mu.py and integrates AUC."""

    name: str | None = (
        None  # outputs/diagnostics/<date>/<name> dir; default: a content fingerprint of this config
    )
    system: str = "van_der_pol"  # "van_der_pol" | "brusselator"
    checkpoint: str = ""
    repeats: int = 3
    n_sweep: int = 30
    max_steps: int = 8192
    device: str = "gpu"  # "gpu" | "cpu"
    reuse: bool = False  # reuse existing per-batch-size NPZ outputs instead of re-running
