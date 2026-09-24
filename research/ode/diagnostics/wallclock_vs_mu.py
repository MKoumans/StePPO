"""Wall-clock time per task parameter for PID and, optionally, RL controllers.

Reuses steppo.training.pid_solve.solve_pid_batch (same solve fn as compare.py).

Usage: --config points to a WallclockVsMuConfig YAML (configs/diagnostics/
wallclock_vs_mu/*.yaml); any field can be overridden ad hoc, e.g. --repeats 5.
    uv run python research/ode/diagnostics/wallclock_vs_mu.py \\
        --config configs/diagnostics/wallclock_vs_mu/van_der_pol_batched.yaml

mode: "single" solves one mu per call; "batched" vmaps the whole sweep, or
(with report_batch_time) times each mu as its own batch of batch_size
samples without dividing by batch size.

device: defaults to cpu for mode=single / gpu for mode=batched — at
single-sample, GPU kernel-launch overhead isn't amortized by vmap and is far
slower than CPU. Must be resolved before jax is imported, so device/mode are
read straight off argv/the config YAML before the rest of the config loads
(see the scan below).

max_steps: defaults to 8192 ("unlimited", matches compare.py's PID-unlimited
row) rather than the training rollout_steps budget — PID often doesn't finish
within that budget while RL does, so timing at the training budget would
partly measure "PID giving up early". Pass rollout_steps explicitly to
reproduce that budget-limited comparison instead.
"""

import os

from early_args import scan_flag

# device/mode must be resolved before jax is imported (see setup_devices()) —
# so peek at both the CLI and the config YAML (if given) with plain argv/yaml
# parsing, ahead of the full config load (which happens after jax import).
_config_path = scan_flag("--config")
_yaml_mode = _yaml_device = None
if _config_path:
    import yaml as _early_yaml

    _cfg_file = _config_path if _config_path.endswith((".yaml", ".yml")) else _config_path + ".yaml"
    with open(_cfg_file) as _f:
        _early_data = _early_yaml.safe_load(_f) or {}
    _yaml_mode = _early_data.get("mode")
    _yaml_device = _early_data.get("device")

_mode = scan_flag("--mode", _yaml_mode or "single")
_device = scan_flag("--device", _yaml_device) or ("gpu" if _mode == "batched" else "cpu")
if _device == "cpu":
    os.environ["GPUS"] = "cpu"

from steppo.utils.device import setup_devices

setup_devices()

import numpy as np
from utils import time_controller, write_simple_outputs

from configs import WallclockVsMuConfig
from steppo.configs.base_config import TrainConfig, apply_dotted_overrides, load_config_from_yaml
from steppo.envs.ode import ODEEnv  # noqa: F401
from steppo.training.pid_solve import env_pid_controller
from steppo.utils.checkpoint import resolve_checkpoint
from steppo.utils.task_params import task_label

try:
    from ..cli_config import parse_config_args
    from ..output_dirs import resolve_output_dir
except ImportError:
    from research.ode.cli_config import parse_config_args
    from research.ode.output_dirs import resolve_output_dir


def parse_args():
    """Parse the diagnostics config path and any CLI overrides."""
    return parse_config_args(
        description=__doc__,
        config_help="WallclockVsMuConfig YAML, e.g. configs/diagnostics/wallclock_vs_mu/<name>.yaml. "
        "Omit to use defaults + overrides only (used by wallclock_batch_size_sweep.py).",
    )


def main():
    """Measure controller latency across task-parameter values."""
    args = parse_args()
    cfg = (
        load_config_from_yaml(WallclockVsMuConfig, args.config)
        if args.config
        else WallclockVsMuConfig()
    )
    cfg = apply_dotted_overrides(cfg, args.overrides)
    # device was already resolved (possibly auto) before jax import — reflect
    # that actual choice rather than cfg.device, which may still be None.
    cfg.device = _device

    if cfg.checkpoint and cfg.hf_repo:
        raise SystemExit("checkpoint and hf_repo are mutually exclusive")
    if not cfg.env_config and not cfg.checkpoint and not cfg.hf_repo:
        raise SystemExit("config needs one of env_config, checkpoint, or hf_repo")
    if cfg.skip_pid and not cfg.checkpoint and not cfg.hf_repo:
        raise SystemExit("skip_pid without checkpoint/hf_repo does nothing")

    rl_controller = None
    if cfg.checkpoint:
        from steppo.envs.ode.learned_controller import LearnedController

        ckpt_path = resolve_checkpoint(cfg.checkpoint)
        bundled = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
        config_path = bundled if os.path.isfile(bundled) else cfg.env_config
        if not config_path:
            raise SystemExit(f"no config.yaml next to {ckpt_path} and no env_config given")
        config = load_config_from_yaml(TrainConfig, config_path, strict=False)
        env = ODEEnv(config.env, config.rollout_steps)
        rl_controller = LearnedController.from_checkpoint(ckpt_path, config, env)
    elif cfg.hf_repo:
        from steppo.models.huggingface.hub import download_model

        print(f"[*] Downloading HF model: {cfg.hf_repo}@{cfg.revision} ...")
        artifact = download_model(cfg.hf_repo, revision=cfg.revision)
        config = artifact.config
        rl_controller = artifact.controller
    else:
        config = load_config_from_yaml(TrainConfig, cfg.env_config, strict=False)

    env_config = config.env
    max_steps = cfg.max_steps
    param_label = task_label(env_config.system)

    test_bins = list(env_config.test_bins)
    lo = cfg.param_min if cfg.param_min is not None else min(b[0] for b in test_bins)
    hi = cfg.param_max if cfg.param_max is not None else max(b[1] for b in test_bins)
    if lo <= 0 or hi <= lo:
        raise SystemExit("parameter range must satisfy 0 < param_min < param_max")
    if cfg.n_sweep < 1:
        raise SystemExit("n_sweep must be positive")
    if cfg.batch_size is not None and cfg.batch_size < 1:
        raise SystemExit("batch_size must be positive")
    if cfg.report_batch_time and cfg.mode != "batched":
        raise SystemExit("report_batch_time requires mode=batched")
    mus = np.logspace(np.log10(lo), np.log10(hi), cfg.n_sweep)

    pid_sc = env_pid_controller(env_config)
    print(
        f"[*] System={env_config.system}  param range=[{lo:.3g}, {hi:.3g}]  n_sweep={cfg.n_sweep}  "
        f"mode={cfg.mode}  device={cfg.device}  max_steps={max_steps}  repeats={cfg.repeats}  "
        f"report={'batch total' if cfg.report_batch_time else 'amortized ms/mu'}"
    )

    out_dir = resolve_output_dir(
        "diagnostics", cfg, name=cfg.name, exclude=("name", "output", "pid_output", "verbose")
    )

    pid_out_path = None
    rl_out_path = None

    pid_ms = None
    if not cfg.skip_pid:
        print("[*] Timing PID ...")
        pid_result = time_controller(
            env_config,
            pid_sc,
            mus,
            max_steps,
            mode=cfg.mode,
            repeats=cfg.repeats,
            verbose=cfg.verbose,
            progress_label="PID",
            report_batch_time=cfg.report_batch_time,
            batch_size=cfg.batch_size,
            return_samples=cfg.save_repeats,
        )
        pid_ms, pid_samples = pid_result if cfg.save_repeats else (pid_result, None)
        if cfg.report_batch_time:
            print(f"  PID batch time mean over sweep: {pid_ms.mean():.3f} ms")
        else:
            print(f"  PID mean over sweep: {pid_ms.mean():.3f} ms")
        pid_suffix = "batched_sweep" if cfg.report_batch_time else cfg.mode
        pid_out_path = cfg.pid_output or os.path.join(out_dir, f"wallclock_vs_mu_{pid_suffix}.npz")
        if cfg.report_batch_time:
            np.savez(
                pid_out_path,
                mus=mus,
                pid_batch_ms=pid_ms,
                batch_size=cfg.batch_size,
                sweep_size=len(mus),
                param_label=param_label,
                mode=cfg.mode,
                device=cfg.device,
                max_steps=max_steps,
                report="per_parameter_batch",
                param_min=lo,
                param_max=hi,
                repeats=cfg.repeats,
                **({"pid_batch_ms_repeats": pid_samples} if cfg.save_repeats else {}),
            )
        else:
            np.savez(
                pid_out_path,
                mus=mus,
                pid_ms=pid_ms,
                param_label=param_label,
                mode=cfg.mode,
                device=cfg.device,
                max_steps=max_steps,
                param_min=lo,
                param_max=hi,
                repeats=cfg.repeats,
                **({"pid_ms_repeats": pid_samples} if cfg.save_repeats else {}),
            )
        print(f"[+] Saved: {pid_out_path}")

    rl_ms = None
    if rl_controller is not None:
        print("[*] Timing RL ...")
        rl_result = time_controller(
            env_config,
            rl_controller,
            mus,
            max_steps,
            mode=cfg.mode,
            repeats=cfg.repeats,
            verbose=cfg.verbose,
            progress_label="RL",
            report_batch_time=cfg.report_batch_time,
            batch_size=cfg.batch_size,
            return_samples=cfg.save_repeats,
        )
        rl_ms, rl_samples = rl_result if cfg.save_repeats else (rl_result, None)
        if cfg.report_batch_time:
            print(f"  RL batch time mean over sweep: {rl_ms.mean():.3f} ms")
        else:
            print(f"  RL mean over sweep: {rl_ms.mean():.3f} ms")

        tag = cfg.tag or (cfg.revision if cfg.hf_repo else "ckpt")
        output_suffix = "batched_sweep" if cfg.report_batch_time else cfg.mode
        rl_out_path = cfg.output or os.path.join(
            out_dir, f"wallclock_rl_{env_config.system}_{tag}_{output_suffix}.npz"
        )
        if cfg.report_batch_time:
            np.savez(
                rl_out_path,
                mus=mus,
                rl_batch_ms=rl_ms,
                batch_size=cfg.batch_size,
                sweep_size=len(mus),
                param_label=param_label,
                mode=cfg.mode,
                device=cfg.device,
                max_steps=max_steps,
                tag=tag,
                report="per_parameter_batch",
                hf_repo=cfg.hf_repo or "",
                revision=cfg.revision,
                checkpoint=cfg.checkpoint or "",
                param_min=lo,
                param_max=hi,
                repeats=cfg.repeats,
                **({"rl_batch_ms_repeats": rl_samples} if cfg.save_repeats else {}),
            )
        else:
            np.savez(
                rl_out_path,
                mus=mus,
                rl_ms=rl_ms,
                param_label=param_label,
                mode=cfg.mode,
                device=cfg.device,
                max_steps=max_steps,
                tag=tag,
                hf_repo=cfg.hf_repo or "",
                revision=cfg.revision,
                checkpoint=cfg.checkpoint or "",
                param_min=lo,
                param_max=hi,
                repeats=cfg.repeats,
                **({"rl_ms_repeats": rl_samples} if cfg.save_repeats else {}),
            )
        print(f"[+] Saved: {rl_out_path}")

    report_path = pid_out_path or rl_out_path
    if report_path is not None:
        write_simple_outputs(
            report_path,
            mus,
            pid_ms=pid_ms,
            rl_ms=rl_ms,
            metadata={
                "system": env_config.system,
                "device": cfg.device,
                "mode": cfg.mode,
                "batch_size": cfg.batch_size,
                "max_steps": max_steps,
                "repeats": cfg.repeats,
            },
            report_batch_time=cfg.report_batch_time,
        )


if __name__ == "__main__":
    main()
