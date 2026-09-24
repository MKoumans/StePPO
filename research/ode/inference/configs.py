"""Dataclass schema for plot_custom_trajectories.py's YAML config (configs/inference/...).

Loaded via steppo.configs.base_config.load_config_from_yaml — see that module
for the strict-unknown-key checking and dotted-override support this inherits.
"""

from dataclasses import dataclass, field


@dataclass
class PlotCustomTrajectoriesConfig:
    """One plot_custom_trajectories.py run: rolls out a controller on explicit
    (task, y0) starting conditions and plots/exports the resulting trajectories.

    Exactly one of checkpoint / repo_id / model_dir must be set.
    """

    name: str | None = (
        None  # outputs/inference/<date>/<name> dir; default: a content fingerprint of this config
    )
    checkpoint: str | None = None  # path to a checkpoint_N dir, or a run dir containing one
    repo_id: str | None = None  # Hugging Face model repository ID
    model_dir: str | None = None  # local downloaded model-artifact directory
    env_config: str | None = (
        None  # TrainConfig YAML; defaults to config.yaml bundled next to checkpoint
    )
    # List of [task] or [task, [y0...]] entries, e.g.:
    #   [[1], [50], [100]]                          (task only)
    #   [[5, [0.0, 0.0]], [50, [2.0, 2.0]]]          (task, y0)
    conditions: list = field(default_factory=list)
    revision: str = "main"  # Hub revision to load when using repo_id
    step_interval: int = 1  # mark every Nth step; also the Excel export stride
    show_rejected: bool = False  # overlay a red dot on every rejected solver step
    max_steps: int = 2000  # upper bound on the diagnostic scan loop
    rollout_steps: int | None = (
        None  # override the env's episode step budget (config.rollout_steps)
    )
    seed: int = 0  # base RNG seed; each condition folds in its index
    out_dir: str | None = (
        None  # output directory for the PNG/XLSX (default: the checkpoint's run dir)
    )
    tag: str = "custom_trajectories"  # filename stem for the output PNG/XLSX
