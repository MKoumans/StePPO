"""Dataclass configuration schema and YAML/CLI loading helpers."""

import math
import os
from dataclasses import dataclass, field
from typing import ClassVar, Tuple

import jax
import jax.numpy as jnp

# src/steppo/configs/base_config.py -> repo root (used to locate configs/envs/,
# configs/models/, configs/envs/ode/ regardless of the caller's cwd).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


@dataclass(unsafe_hash=True)
class EncoderArchConfig:
    """Architecture for the trajectory encoder."""

    hidden_size: int = 64
    state_embed_dim: int = 10
    action_embed_dim: int = 10
    reward_embed_dim: int = 5
    # Embed width for the true hidden task parameter (e.g. ODEParams.lam), when
    # "task" is included in encoder_inputs. Unused (no embed layer built) otherwise.
    task_embed_dim: int = 10
    activation: str = "relu"  # "relu" | "tanh" | "gelu" | "elu" | "silu"
    normalize: str = "none"  # "none" | "layer_norm" | "instance_norm"
    encoder_type: str = "gru"
    # Hidden layer widths for the "linear" encoder's per-step MLP (ignored by "gru",
    # which uses `hidden_size` as its single GRU state dim instead).
    linear_hidden_dims: Tuple[int, ...] = (64, 64)
    # Per-step signals the encoder consumes: any subset of ("action", "state", "reward", "task").
    # "task" feeds the true task parameter to the encoder.
    encoder_inputs: Tuple[str, ...] = ("action", "state", "reward")


@dataclass(unsafe_hash=True)
class DecoderArchConfig:
    """Architecture for reward/state/task decoder MLPs."""

    hidden_dims: Tuple[int, ...] = (32, 32)
    activation: str = "relu"
    normalize: str = "none"  # "none" | "layer_norm" | "instance_norm"


@dataclass(unsafe_hash=True)
class PolicyArchConfig:
    """Architecture for the ActorCritic MLP."""

    hidden_dims: Tuple[int, ...] = (32, 32)
    activation: str = "tanh"
    normalize: str = "none"  # "none" | "layer_norm" | "instance_norm"
    # Policy/value inputs: "state" and "z", optionally "task" (true task parameter).
    policy_inputs: Tuple[str, ...] = ("state", "z")


@dataclass(unsafe_hash=True)
class VAEConfig:
    """VariBAD encoder, decoder, and ELBO optimization settings."""

    latent_dim: int = 5
    # Second, long-term latent for task identity; 0 disables it (task/end-reward
    # decoders then read the short-term latent).
    latent_dim_long: int = 0
    encoder: EncoderArchConfig = field(default_factory=EncoderArchConfig)
    decoder: DecoderArchConfig = field(default_factory=DecoderArchConfig)
    kl_weight: float = 1e-2
    # Free-bits floor per latent dim (nats): dims with KL below this contribute the
    # floor instead, removing the incentive to collapse them to the prior. 0 disables.
    kl_free_bits: float = 0.0
    rew_loss_coeff: float = 1.0
    state_loss_coeff: float = 0.0
    task_loss_coeff: float = 0.0
    accept_loss_coeff: float = 0.0
    end_reward_loss_coeff: float = 0.0
    lr: float = 1e-3
    lr_end: float = 0.0
    num_updates_per_iter: int = 3
    batch_size: int = 25
    buffer_capacity_multiplier: int = 16
    sequential_kl: bool = False
    # Let gradients flow into the previous posterior of the sequential KL (VariBAD
    # default); False stop-gradients it (the paper's "detached gradient" ablation).
    sequential_kl_propagate_gradient: bool = True
    # Weight of an extra KL(q_t || N(0, I)) anchor next to the sequential KL; 0 disables it.
    sequential_kl_anchor_weight: float = 0.0
    subsample_len: int = 0
    # Reward/state decoders predict (mean, logvar) and train with a Gaussian NLL instead of MSE.
    heteroscedastic_recon: bool = False
    # Ablation: use z = mu (no sampling) and drop the KL term.
    deterministic_latent: bool = False

    @property
    def total_latent_dim(self) -> int:
        """Encoder output width: short-term latent + long-term latent."""
        return self.latent_dim + self.latent_dim_long


@dataclass(unsafe_hash=True)
class PPOConfig:
    """PPO optimizer, objective, and policy-architecture settings."""

    lr: float = 7e-4
    lr_end: float = 0.0
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    entropy_coeff: float = 0.005  # constant entropy bonus weight
    value_coeff: float = 0.5
    num_epochs: int = 4
    num_minibatches: int = 4
    target_kl: float = 0.0
    policy: PolicyArchConfig = field(default_factory=PolicyArchConfig)


@dataclass(unsafe_hash=True)
class RewardNormalizationConfig:
    """Per-task reward scale PID_steps(μ) / geomean(PID_steps), clipped (see reward_norm.py)."""

    enabled: bool = False
    grid_points: int = 24  # log-spaced μ grid over env task bounds
    num_repeats: int = 8  # y0/task draws per grid point
    max_steps: int = (
        0  # PID budget per grid point; 0 = rollout_steps (censored points extrapolated)
    )
    # Per-step penalty applied after reward scaling (env.survival_penalty is applied before).
    post_scale_step_penalty: float = 0.0


@dataclass(unsafe_hash=True)
class TrainingConfig:
    """Training-process knobs — separate from model architecture (VAEConfig/PPOConfig)."""

    ema_norm: bool = False  # normalise each VAE and PPO loss term by its running EMA
    ema_alpha: float = 0.99  # EMA decay (higher = slower adaptation)
    ema_clamp: float = 5.0  # clamp normalised loss to [-C, C] * ema before dividing
    ema_input_cap: float = 10.0  # cap EMA update input to this × current EMA (spike protection)
    grad_clip_vae: float = 10.0  # global gradient norm clip for VAE optimiser
    grad_clip_ppo: float = 0.5  # global gradient norm clip for PPO optimiser
    kl_anneal_frac: float = 0.0  # fraction of total VAE steps to linearly ramp kl_weight from 0 to its configured value (0 = no annealing)
    log_grad_norms: bool = True  # per-term PPO gradient norms (actor/value/entropy); costs 3 extra backward passes per PPO step
    reward_normalization: RewardNormalizationConfig = field(
        default_factory=RewardNormalizationConfig
    )


@dataclass(unsafe_hash=True)
class ODEEnvConfig:
    """ODE environment settings; `system` selects the problem, other systems' fields are ignored."""

    system: str = "van_der_pol"  # registered ODE system name
    precision: str = "float64"  # "float32" | "float64" — solver arithmetic precision
    # Solver
    t_end: float = 500.0
    dt0: float = 1e-4
    rtol: float = 1e-6
    atol: float = 1e-8
    dt_min: float = 1e-10
    dt_max: float = 5.0
    dt_log_gain: float = 0.693147181  # ln(2) — max per-step dt multiplier is exp(dt_log_gain)
    immediate_dt_action: bool = False
    task_sample_scheme: str = (
        "binned"  # "binned" | "log-binned" — draws from train_bins, see ODEEnv.sample_task
    )
    train_bins: Tuple[
        Tuple[float, float], ...
    ] = ()  # task support sampled during training rollouts
    val_bins: Tuple[
        Tuple[float, float], ...
    ] = ()  # swept periodically during training (eval_metrics)
    test_bins: Tuple[
        Tuple[float, float], ...
    ] = ()  # swept periodically during training and by post-hoc analysis scripts
    # Van der Pol task distribution (log-uniform)
    mu_min: float = 50.0
    mu_max: float = 200.0
    sample_mu: bool = True
    # Scalar decay task distribution (uniform)
    lam_min: float = 1.0
    lam_max: float = 100.0
    sample_lam: bool = True
    # Robertson task distribution (k2, log-uniform)
    k2_min: float = 1e5
    k2_max: float = 1e9
    sample_k2: bool = True
    # Brusselator task distribution (B, log-uniform)
    B_min: float = 2.0
    B_max: float = 50.0
    sample_B: bool = True
    # FitzHugh-Nagumo task distribution (eps, log-uniform)
    eps_min: float = 0.005
    eps_max: float = 1.0
    sample_eps: bool = True
    # Chemical cascade task distribution (lambda, log-uniform uniform decay rate)
    cc_lam_min: float = 0.01
    cc_lam_max: float = 10.0
    sample_cc_lam: bool = True
    # FOSM task: forcing amplitude D, must stay below the gain k = 2 (see systems/fosm.py).
    D_min: float = 1.0
    D_max: float = 1.9
    sample_D: bool = True
    fosm_w0: float = 0.1  # initial sliding variable w(0); paper trains at 0.1, evals at 0.15
    fosm_eps: float = (
        1e-3  # tanh(w/eps) boundary-layer width for "fosm_smooth" (see systems/fosm.py);
    )
    # smaller = closer to true sign() but more steps -- 1e-2 leaves a visible
    # ~0.01 residual oscillation band, 1e-3 is visually flat at ~2x the step count
    # Chua task: diode gain m (see systems/chua.py).
    m_min: float = 0.8
    m_max: float = 0.8
    sample_m: bool = False
    chua_v1_0: float = 0.15264  # paper training IC
    chua_v2_0: float = -0.02281
    chua_i_0: float = 0.38127
    chua_eps: float = 1e-3  # tanh(V1/eps) boundary-layer width for "chua_smooth"
    # Chemical cascade periodic pulse-train forcing (smooth Gaussian bumps into
    # layer 1's first chemical). Off by default; does not touch other systems.
    cc_pulse_enabled: bool = False
    cc_pulse_period: float = 15.0
    cc_pulse_width: float = 0.3
    cc_pulse_amplitude: float = 2.0
    cc_pulse_dim: int = 0
    cc_pulse_random: float = 0.0
    fhn_pulse_enabled: bool = False
    fhn_pulse_period: float = 15.0
    fhn_pulse_width: float = 0.3
    fhn_pulse_amplitude: float = 1.0
    fhn_pulse_dim: int = 0
    fhn_pulse_random: float = 0.0  # see cc_pulse_random
    # Scalar decay periodic forcing (adds a Gaussian pulse train into the
    # single state dim, on top of the -lambda*y decay). Off by default.
    sd_pulse_enabled: bool = False
    sd_pulse_period: float = 0.15  # ~6-7 pulses over the default t_end=1.0 episode
    sd_pulse_width: float = 0.02
    sd_pulse_amplitude: float = (
        5.0  # impulse ~= amplitude*width*sqrt(2*pi) =~ 0.25, a visible jump given y in ~[0, 1]
    )
    sd_pulse_dim: int = 0
    sd_pulse_random: float = 0.0  # see cc_pulse_random
    # Initial conditions (shared)
    sample_y0: bool = False
    y0_x: float = 0.0
    y0_y: float = -2.0
    # Observation feature groups to include (system-specific "state"/"direction" dims;
    # shared "step_context"/"solver_trend" dims — see obs_factory.py)
    obs_features: Tuple[str, ...] = ("state", "step_context", "solver_trend")
    reward_factor: float = 50.0  # scales per-step progress reward
    progress_warp: bool = False
    completion_bonus: float = 25.0  # bonus at success, scaled by (1 - budget_exhausted)
    survival_penalty: float = 0.0  # fixed per-step penalty (0 = disabled)
    rejection_penalty: float = 0.0  # fixed penalty per rejected step (0 = disabled)

    margin_penalty: float = 0.0

    def __post_init__(self):
        """Require every train/val/test bin to lie inside the system's task range."""
        bounds_by_system = {
            "scalar_decay": ("lam_min", "lam_max"),
            "van_der_pol": ("mu_min", "mu_max"),
            "robertson": ("k2_min", "k2_max"),
            "brusselator": ("B_min", "B_max"),
            "fitzhugh_nagumo": ("eps_min", "eps_max"),
            "chemical_cascade": ("cc_lam_min", "cc_lam_max"),
            "fosm": ("D_min", "D_max"),
            "fosm_smooth": ("D_min", "D_max"),
            "chua": ("m_min", "m_max"),
            "chua_smooth": ("m_min", "m_max"),
        }
        lo_name, hi_name = bounds_by_system.get(self.system, ("lam_min", "lam_max"))
        domain_lo = float(getattr(self, lo_name))
        domain_hi = float(getattr(self, hi_name))
        if not (math.isfinite(domain_lo) and math.isfinite(domain_hi) and domain_lo <= domain_hi):
            raise ValueError(
                f"{self.system} task bounds must be finite and ordered: "
                f"[{lo_name}={domain_lo}, {hi_name}={domain_hi}]"
            )

        for split in ("train", "val", "test"):
            bins = getattr(self, f"{split}_bins")
            for index, bounds in enumerate(bins):
                if len(bounds) != 2:
                    raise ValueError(f"{split}_bins[{index}] must be a [lo, hi] pair")
                bin_lo, bin_hi = (float(value) for value in bounds)
                if not (math.isfinite(bin_lo) and math.isfinite(bin_hi) and bin_lo <= bin_hi):
                    raise ValueError(
                        f"{split}_bins[{index}] must contain finite ordered bounds, got {bounds}"
                    )
                if bin_lo < domain_lo or bin_hi > domain_hi:
                    raise ValueError(
                        f"{self.system} {split}_bins[{index}]={tuple(bounds)} lies outside "
                        f"the configured task range [{domain_lo}, {domain_hi}] "
                        f"({lo_name}, {hi_name})"
                    )


@dataclass(unsafe_hash=True)
class WarmstartConfig:
    """Behaviour-clone the policy on expert rollouts before RL fine-tuning.

    `expert="pid"` collects PID rollouts live; `expert="oracle"` reads a cached
    oracle dataset produced by generate_pid_training_dataset.py.
    """

    enabled: bool = False
    expert: str = "pid"  # "pid" | "oracle"
    num_iters: int = 200
    bc_lr: float = 1e-3
    bc_epochs: int = 5
    pid_safety: float = 0.9
    pid_order: int = 4
    pid_min_factor: float = 0.2
    pid_max_factor: float = 10.0
    pid_kp: float = 0.0
    pid_ki: float = 1.0
    pid_kd: float = 0.0
    oracle_max_iters: int = 6  # hill-climb iteration cap per trajectory step
    oracle_dataset_seed: int = (
        0  # fixed seed for the cached oracle corpus, independent of training repeats
    )


@dataclass(unsafe_hash=True)
class PrecomputeBaselineConfig:
    """PID baseline step-count precomputation, used as an eval diagnostic for ODE envs."""

    enabled: bool = True
    eval_mus: Tuple[float, ...] = (1, 5, 10, 20, 50, 100, 200)
    num_repeats: int = 16
    max_steps: int = 0  # 0 = derive from rollout_steps * 10


@dataclass(unsafe_hash=True)
class EvalMetricsConfig:
    """Periodic train/val/test evaluation: step count relative to PID and L2 error
    against the reference solution. PID numbers come from the cached evaluation
    datasets. Each evaluation re-jits the solve, so keep `interval` coarse.
    """

    enabled: bool = True
    num_envs: int = 256
    max_steps: int = (
        0  # 0 = derive from rollout_steps (matches the former ErrorDistConfig's budget,
    )
    # not EfficiencyMetricConfig's eval_max_steps — see base_trainer.py's usage)
    interval: int = 100  # cadence, in training iterations


@dataclass(unsafe_hash=True)
class LatentScatterConfig:
    """Periodic scatter of the final belief mean coloured by task parameter, stitched into a GIF."""

    enabled: bool = True
    num_samples: int = 300
    dims: Tuple[int, int] = (0, 1)  # which latent dims to plot on (x, y)
    max_steps: int = 0  # 0 = derive from eval_max_steps
    gif_fps: int = 4
    interval: int = 10  # snapshot cadence, in training iterations (decoupled from eval_interval)


@dataclass(unsafe_hash=True)
class PidFallbackConfig:
    """Control-chart PID fallback (off by default).

    Late in training, a percentile of each tracked solver statistic is EMA'd
    into a control limit. At inference, LearnedController switches to PID for
    the rest of the episode once a statistic crosses `multiplier x` its limit.
    """

    enabled: bool = False
    interval: int = 100  # iterations between calibration updates
    num_envs: int = 128  # episodes per calibration update (one "subgroup")
    start_frac: float = 0.8  # only accumulate once iteration/total_iters >= this
    ema_alpha: float = 0.99  # EMA smoothing of the control limit across updates
    max_steps: int = 0  # 0 = derive from eval_max_steps

    track_reject_streak: bool = True  #
    reject_streak_percentile: float = 99.0  # upper limit percentile
    reject_streak_multiplier: float = 1.5  #

    track_accept_ema: bool = True
    accept_ema_percentile: float = 1.0  # lower limit percentile
    accept_ema_multiplier: float = 1.5  # applied as limit / multiplier (see class docstring)

    track_log_error_ema: bool = (
        False  # off by default: noisier, not yet empirically validated as a signal
    )
    log_error_ema_percentile: float = 99.0
    log_error_ema_multiplier: float = 1.5


@dataclass(unsafe_hash=True)
class TrainConfig:
    """Top-level configuration for one training or evaluation run."""

    # Sections of removed features (TFT backbone, TQC algorithm) that configs saved with
    # older checkpoints and model artifacts still contain; ignored on load.
    RETIRED_KEYS: ClassVar[Tuple[str, ...]] = ("tft", "tqc")
    model_precision: str = "float32"  # "float32" | "float16" | "bfloat16" — neural network dtype
    backbone: str = (
        "varibad"  # selects belief-model architecture (see src/steppo/models/backbones.py)
    )
    algo: str = "ppo"  # selects policy-update algorithm (see src/steppo/training/policy_algos.py)
    # TrainConfig is the ODE-specific top-level config (env: ODEEnvConfig below) — the encoder
    # defaults to consuming the true task parameter here since every registered env exposes one.
    vae: VAEConfig = field(
        default_factory=lambda: VAEConfig(
            encoder=EncoderArchConfig(encoder_inputs=("action", "state", "reward", "task")),
        )
    )
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    env: ODEEnvConfig = field(default_factory=ODEEnvConfig)
    warmstart: WarmstartConfig = field(default_factory=WarmstartConfig)
    precompute_baseline: PrecomputeBaselineConfig = field(default_factory=PrecomputeBaselineConfig)
    eval_metrics: EvalMetricsConfig = field(default_factory=EvalMetricsConfig)
    latent_scatter: LatentScatterConfig = field(default_factory=LatentScatterConfig)
    pid_fallback: PidFallbackConfig = field(default_factory=PidFallbackConfig)
    episodes_per_trial: int = 0
    episode_max_steps: int = 0
    reset_belief_between_episodes: bool = False

    total_iters: int = 1_000
    num_envs: int = 16
    rollout_steps: int = 400
    eval_max_steps: int = 200  # lax.scan length for eval episodes; keep ≥ expected episode length
    seed: int = 0
    use_wandb: bool = False
    eval_interval: int = 100
    num_eval_episodes: int = 10
    log_interval: int = 10
    save_plots: bool = True
    save_json: bool = True
    exp_name: str = "experiment"
    note: str = ""  # Free-text message describing this run, saved to checkpoint_path/note.txt
    run_uid: str = ""
    run_date: str = ""  # YYYYMMDD, groups runs by day
    repeat_batch_uid: str = ""  # shared by all --repeat seeds from one run-all-ode.sh invocation
    env_family: str = "ode"  # e.g. "ode" — groups runs alongside env.system
    runs_dir: str = "outputs/runs"
    checkpoints_dir: str = ""
    outputs_dir: str = ""
    extensive_timing_logging: bool = False

    def __post_init__(self):
        """Assign run identity fields and validate reward settings."""
        import datetime

        if not self.run_uid:
            import uuid

            self.run_uid = uuid.uuid4().hex[:8]
        if not self.run_date:
            self.run_date = datetime.datetime.now().strftime("%Y%m%d")
        if self.env.margin_penalty > 0 and self.env.rejection_penalty <= 0:
            raise ValueError(
                f"env.margin_penalty={self.env.margin_penalty} is set but "
                f"env.rejection_penalty={self.env.rejection_penalty} (<= 0). With no "
                "rejection cost, always rejecting every step is a free, zero-reward "
                "policy that dominates any accepted step (which risks the margin "
                "penalty unless scaled_error lands almost exactly at 1.0) — PPO has "
                "no gradient to escape that plateau and training collapses (seen on "
                "the van_der_pol margin-penalty sweep, "
                "outputs/experiments/20260824/22193876). Set env.rejection_penalty > 0 "
                "alongside env.margin_penalty, or leave margin_penalty at 0."
            )

    @property
    def _run_subdir(self) -> str:
        """Return the relative path used to group this run's artifacts."""
        return os.path.join(self.run_date, self.env_family, self.env.system, self.run_uid)

    @property
    def run_dir(self) -> str:
        """Return the absolute root directory for this run."""
        return os.path.abspath(os.path.join(self.runs_dir, self._run_subdir))

    @property
    def checkpoint_path(self) -> str:
        """Return the directory containing this run's checkpoints."""
        if self.checkpoints_dir:
            return os.path.abspath(os.path.join(self.checkpoints_dir, self.run_uid))
        return os.path.join(self.run_dir, "checkpoints")

    @property
    def output_path(self) -> str:
        """Return the directory containing this run's analysis outputs."""
        if self.outputs_dir:
            return os.path.abspath(os.path.join(self.outputs_dir, self.run_uid))
        return os.path.join(self.run_dir, "outputs")


def load_config_from_dict(config_cls, data: dict, _path: str = "", strict: bool = True):
    """Build a nested config dataclass from a dict; unknown keys raise when `strict`."""
    import typing
    from dataclasses import fields, is_dataclass

    # Resolve annotations so forward-reference strings become real types.
    try:
        hints = typing.get_type_hints(config_cls)
    except Exception:
        hints = {}

    def _is_tuple_type(t) -> bool:
        return t is tuple or (hasattr(t, "__origin__") and t.__origin__ is tuple)

    def _coerce_tuple(val, resolved):
        """Convert nested lists to nested tuples so hashable configs stay hashable."""
        if not isinstance(val, list):
            return val
        args = typing.get_args(resolved)
        elem_type = args[0] if args and args[0] is not Ellipsis else None
        return tuple(_coerce_tuple(v, elem_type) if elem_type is not None else v for v in val)

    kwargs = {}
    known_names = set()
    for f in fields(config_cls):
        known_names.add(f.name)
        if f.name not in data:
            continue
        val = data[f.name]
        resolved = hints.get(f.name, f.type)
        if is_dataclass(resolved) and isinstance(val, dict):
            kwargs[f.name] = load_config_from_dict(
                resolved, val, _path=f"{_path}{f.name}.", strict=strict
            )
        elif _is_tuple_type(resolved) and isinstance(val, list):
            kwargs[f.name] = _coerce_tuple(val, resolved)
        else:
            kwargs[f.name] = val

    retired = getattr(config_cls, "RETIRED_KEYS", ())
    unknown = [k for k in data if k not in known_names and k not in retired]
    if unknown and strict:
        dotted = [f"{_path}{k}" for k in unknown]
        raise ValueError(
            f"Unrecognized config key(s) {dotted} for {config_cls.__name__} — "
            f"known fields: {sorted(known_names)}. If this field was renamed or "
            f"removed, update the YAML; if it's meant for a different nesting "
            f"level (e.g. a PPOConfig field placed at the top level instead of "
            f"under 'ppo:'), move it there."
        )
    return config_cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge 'override' onto 'base'. 'override' wins on conflicts."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _merge_extends(data: dict, yaml_path: str) -> dict:
    """Deep-merge the YAML named by a top-level 'extends' key underneath `data`.

    Looks in the parent directory first, then under configs/envs/.
    """
    import glob

    import yaml as _yaml

    base_name = data.pop("extends", None)
    if base_name is None:
        return data

    config_dir = os.path.dirname(os.path.abspath(yaml_path))
    base_path = os.path.join(os.path.dirname(config_dir), f"{base_name}.yaml")
    if not os.path.isfile(base_path):
        repo_root = _REPO_ROOT
        matches = glob.glob(
            os.path.join(repo_root, "configs", "envs", "**", f"{base_name}.yaml"), recursive=True
        )
        if not matches:
            raise FileNotFoundError(
                f"extends: base config '{base_name}.yaml' not found at {base_path} "
                f"or anywhere under {repo_root}/configs/envs/"
            )
        base_path = matches[0]
    with open(base_path, "r") as f:
        base_data = _yaml.safe_load(f) or {}
    base_data = _merge_extends(base_data, base_path)
    return _deep_merge(base_data, data)


def _merge_model_ref(data: dict, yaml_path: str) -> dict:
    """Merge the architecture YAML named by a top-level 'model' key; `data` takes precedence."""
    import yaml as _yaml

    model_name = data.pop("model", None)
    if model_name is None:
        return data

    config_dir = os.path.dirname(os.path.abspath(yaml_path))
    model_path = os.path.join(config_dir, "models", f"{model_name}.yaml")
    if not os.path.isfile(model_path):
        # Fall back to top-level configs/models/ directory
        repo_root = _REPO_ROOT
        model_path = os.path.join(repo_root, "configs", "models", f"{model_name}.yaml")
    with open(model_path, "r") as f:
        model_data = _yaml.safe_load(f)

    # Merge top-level scalar fields (model provides default, env config overrides)
    if "model_precision" in model_data and "model_precision" not in data:
        data["model_precision"] = model_data["model_precision"]

    # Which backbone this model yaml targets (default: current VariBAD VAE behaviour).
    backbone = model_data.get("backbone", "varibad")
    if "backbone" not in data:
        data["backbone"] = backbone

    # Merge arch fields into the VAE section (model = base, env config = override).
    target = data.setdefault("vae", {})
    for key in ("latent_dim", "latent_dim_long", "encoder", "decoder"):
        if key in model_data:
            if isinstance(model_data[key], dict):
                # deep merge: model provides base, env config overrides per-key
                merged = {**model_data[key], **target.get(key, {})}
                target[key] = merged
            elif key not in target:
                target[key] = model_data[key]

    # Merge policy arch into ppo section
    if "policy" in model_data:
        ppo = data.setdefault("ppo", {})
        if isinstance(model_data["policy"], dict):
            merged = {**model_data["policy"], **ppo.get("policy", {})}
            ppo["policy"] = merged
        elif "policy" not in ppo:
            ppo["policy"] = model_data["policy"]

    return data


def _merge_pulse_config(data: dict, yaml_path: str) -> dict:
    """Merge configs/envs/ode/<env.pulse_config>.yaml into `env`; values under env: take precedence."""
    import yaml as _yaml

    env_data = data.get("env")
    if not isinstance(env_data, dict):
        return data
    pulse_name = env_data.pop("pulse_config", None)
    if not pulse_name:
        return data

    repo_root = _REPO_ROOT
    pulse_path = os.path.join(repo_root, "configs", "envs", "ode", f"{pulse_name}.yaml")
    if not os.path.isfile(pulse_path):
        raise FileNotFoundError(f"env.pulse_config: '{pulse_name}.yaml' not found at {pulse_path}")
    with open(pulse_path, "r") as f:
        pulse_data = _yaml.safe_load(f) or {}

    data["env"] = _deep_merge(pulse_data, env_data)
    return data


def load_config_from_yaml(config_cls, yaml_path: str, strict: bool = True):
    """Load YAML, resolve inheritance, and instantiate a nested config dataclass."""
    import yaml

    if not yaml_path.endswith((".yaml", ".yml")):
        yaml_path = yaml_path + ".yaml"

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: list(loader.construct_sequence(node)),
    )
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=_Loader)
    data = _merge_extends(data, yaml_path)
    data = _merge_model_ref(data, yaml_path)
    data = _merge_pulse_config(data, yaml_path)
    data = _expand_dotted_keys(data)
    return load_config_from_dict(config_cls, data, strict=strict)


def _expand_dotted_keys(data: dict) -> dict:
    """Expand dotted keys ('latent_scatter.interval: 25') into nested dicts; nested sections win."""
    dotted = {}
    plain = {}
    for key, val in data.items():
        if isinstance(key, str) and "." in key:
            node = dotted
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = val
        else:
            plain[key] = val
    return _deep_merge(dotted, plain) if dotted else plain


def _coerce_override(raw: str, target_type):
    """Coerce a raw CLI string to a dataclass field's declared type."""
    import typing

    if target_type is bool:
        low = raw.strip().lower()
        if low in ("true", "1", "yes", "y"):
            return True
        if low in ("false", "0", "no", "n"):
            return False
        raise ValueError(f"cannot parse bool override from '{raw}' (use true/false)")

    origin = typing.get_origin(target_type)
    if origin is tuple:
        args = typing.get_args(target_type)
        elem_type = args[0] if args and args[0] is not Ellipsis else str
        raw = raw.strip()
        if not raw:
            return ()
        return tuple(_coerce_override(v.strip(), elem_type) for v in raw.split(","))

    if target_type in (int, float, str):
        return target_type(raw)

    return raw


def apply_dotted_overrides(config: "TrainConfig", overrides: dict) -> "TrainConfig":
    """Apply string overrides keyed by dotted path (e.g. {"env.t_end": "50"}), coercing to field types."""
    import typing
    from dataclasses import is_dataclass, replace

    for dotted_key, raw_value in overrides.items():
        parts = dotted_key.split(".")

        chain = []  # (ancestor_obj, field_name_holding_the_next_obj)
        obj = config
        for part in parts[:-1]:
            hints = typing.get_type_hints(type(obj))
            if part not in hints or not is_dataclass(hints[part]):
                raise ValueError(
                    f"Unknown config path segment '{part}' in override '--{dotted_key}'"
                )
            chain.append((obj, part))
            obj = getattr(obj, part)

        leaf_name = parts[-1]
        hints = typing.get_type_hints(type(obj))
        if leaf_name not in hints:
            raise ValueError(f"Unknown config field '{leaf_name}' in override '--{dotted_key}'")

        obj = replace(obj, **{leaf_name: _coerce_override(raw_value, hints[leaf_name])})
        for parent, field_name in reversed(chain):
            obj = replace(parent, **{field_name: obj})
        config = obj

    return config


PRECISION_DTYPES = {
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
    "float64": jnp.float64,
}


def apply_precision(cfg: TrainConfig):
    """Configure JAX precision and return environment and model dtypes."""
    env_prec = cfg.env.precision
    if env_prec == "float64":
        jax.config.update("jax_enable_x64", True)

    env_dtype = PRECISION_DTYPES[env_prec]
    model_dtype = PRECISION_DTYPES[cfg.model_precision]
    return env_dtype, model_dtype
