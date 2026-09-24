"""e_int, steps and wall-clock vs mu for a config-defined list of models on one
shared mu/episode grid (see configs.ModelSweepConfig for the model schema).

Usage:
    PYTHONPATH=. python research/ode/diagnostics/model_sweep.py \\
        --config configs/diagnostics/model_sweep/scalar_decay_default.yaml

The device (default gpu; timings are always batched) is read from argv or the
config before JAX is imported.
"""

import os

from early_args import scan_flag

_config_path = scan_flag("--config")
_yaml_device = None
if _config_path:
    import yaml as _early_yaml

    _cfg_file = _config_path if _config_path.endswith((".yaml", ".yml")) else _config_path + ".yaml"
    with open(_cfg_file) as _f:
        _early_data = _early_yaml.safe_load(_f) or {}
    _yaml_device = _early_data.get("device")

_device = scan_flag("--device", _yaml_device) or "gpu"
if _device == "cpu":
    os.environ["GPUS"] = "cpu"

from steppo.utils.device import setup_devices

setup_devices()

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from utils import time_solve_fn

from configs import ModelSweepConfig
from steppo.configs.base_config import TrainConfig, apply_dotted_overrides, load_config_from_yaml
from steppo.envs.ode import ODEEnv  # noqa: F401
from steppo.envs.ode.learned_controller import LearnedController
from steppo.training.error_dist import solve_oracle_batch
from steppo.training.pid_solve import env_pid_controller, solve_pid_batch
from steppo.training.trajectory_compare import log10_time_integrated_error
from steppo.utils.checkpoint import resolve_checkpoint
from steppo.utils.task_params import task_label

try:
    from ..cli_config import parse_config_args
    from ..output_dirs import resolve_output_dir
except ImportError:
    from research.ode.cli_config import parse_config_args
    from research.ode.output_dirs import resolve_output_dir
from research.ode.post_run_analysis.rollout_cache import load_or_compute
from research.pid_tuning.pi_bayes_opt import PIGains


def parse_args():
    """Parse the model_sweep config path and any CLI overrides."""
    return parse_config_args(
        description=__doc__,
        config_help="ModelSweepConfig YAML, e.g. configs/diagnostics/model_sweep/<name>.yaml.",
    )


@dataclasses.dataclass
class ModelHandle:
    """One model's solve entry points, all bound to that model's own env_config/controller.

    `cheap_solve(mus, keys) -> {"steps": (N,), ...}` is the no-save solve
    used for both step counting and wallclock timing. `dense_solve(mus, keys)
    -> {"ts", "ys", ...}` is the save_steps=True/save_ys=True solve used for
    e_int; `None` when the model has no dense trajectory (the oracle's own
    solve path doesn't produce one, see steps_vs_mu.py's build_steps_vs_mu_data).
    """

    label: str
    cheap_solve: Callable
    dense_solve: Callable | None


def _pid_controller(env_config, kp=0.0, ki=1.0, kd=0.0):
    return env_pid_controller(env_config, pcoeff=kp, icoeff=ki, dcoeff=kd)


def _build_pid(spec, top_env_config, top_config, max_steps):
    controller = _pid_controller(top_env_config)
    return ModelHandle(
        label=spec["label"],
        cheap_solve=lambda mus, keys: solve_pid_batch(
            top_env_config, controller, mus, keys, max_steps
        ),
        dense_solve=lambda mus, keys: solve_pid_batch(
            top_env_config, controller, mus, keys, max_steps, save_steps=True, save_ys=True
        ),
    )


def _build_pid_tuned(spec, top_env_config, top_config, max_steps):
    gains = PIGains(
        float(spec.get("kp", 0.0)), float(spec.get("ki", 1.0)), float(spec.get("kd", 0.0))
    )
    controller = _pid_controller(top_env_config, kp=gains.kp, ki=gains.ki, kd=gains.kd)
    return ModelHandle(
        label=spec["label"],
        cheap_solve=lambda mus, keys: solve_pid_batch(
            top_env_config, controller, mus, keys, max_steps
        ),
        dense_solve=lambda mus, keys: solve_pid_batch(
            top_env_config, controller, mus, keys, max_steps, save_steps=True, save_ys=True
        ),
    )


def _build_pid_oracle(spec, top_env_config, top_config, max_steps):
    max_iters = int(spec.get("max_iters", 6))
    return ModelHandle(
        label=spec["label"],
        cheap_solve=lambda mus, keys: solve_oracle_batch(
            top_env_config, mus, keys, max_steps, max_iters=max_iters
        ),
        dense_solve=None,
    )


def _build_checkpoint(spec, top_env_config, top_config, max_steps):
    ckpt_path = resolve_checkpoint(spec["checkpoint"])
    bundled = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    config_path = bundled if os.path.isfile(bundled) else spec.get("env_config")
    if config_path is None:
        raise SystemExit(
            f"no config.yaml next to {ckpt_path} and no env_config given for model {spec['label']!r}"
        )
    run_config = load_config_from_yaml(TrainConfig, config_path, strict=False)
    env = ODEEnv(run_config.env, run_config.rollout_steps)
    controller = LearnedController.from_checkpoint(ckpt_path, run_config, env)
    env_config = run_config.env
    return ModelHandle(
        label=spec["label"],
        cheap_solve=lambda mus, keys: solve_pid_batch(env_config, controller, mus, keys, max_steps),
        dense_solve=lambda mus, keys: solve_pid_batch(
            env_config, controller, mus, keys, max_steps, save_steps=True, save_ys=True
        ),
    )


def _build_deeponet(spec, top_env_config, top_config, max_steps):
    raise NotImplementedError(
        "model_sweep.py has no DeepONet evaluation path — remove the 'deeponet' "
        "model entry from the config."
    )


def resolve_models(cfg):
    """Return cfg.models, plus one checkpoint-type StePPO entry appended when
    cfg.checkpoint_steppo is set — see ModelSweepConfig's docstring."""
    models = list(cfg.models)
    if cfg.checkpoint_steppo:
        models.append(
            {"type": "checkpoint", "label": "RL (StePPO)", "checkpoint": cfg.checkpoint_steppo}
        )
    return models


MODEL_BUILDERS = {
    "pid": _build_pid,
    "pid_tuned": _build_pid_tuned,
    "pid_oracle": _build_pid_oracle,
    "checkpoint": _build_checkpoint,
    "deeponet": _build_deeponet,
}


def build_model(spec, top_env_config, top_config, max_steps):
    """Validate one model spec dict and dispatch to its MODEL_BUILDERS entry."""
    if "type" not in spec or spec["type"] not in MODEL_BUILDERS:
        raise SystemExit(
            f"unknown/missing model 'type' in {spec!r}; expected one of {sorted(MODEL_BUILDERS)}"
        )
    if "label" not in spec:
        raise SystemExit(f"model spec missing 'label': {spec!r}")
    return MODEL_BUILDERS[spec["type"]](spec, top_env_config, top_config, max_steps)


def sweep_one_model(spec, handle, *, mus, mu_grid, episode_keys, ref_out, cfg, atol):
    """Compute (steps, e_int, wallclock) per-mu series for one model, cached by spec+grid."""
    payload = {
        "kind": "model_sweep",
        "type": spec["type"],
        "spec": spec,
        "mus": mus.tolist(),
        "num_repeats": cfg.num_repeats,
        "seed": cfg.seed,
        "max_steps": cfg.max_steps,
        "ref_tol_factor": cfg.ref_tol_factor,
        "wallclock_repeats": cfg.wallclock_repeats,
        "wallclock_batch_size": cfg.wallclock_batch_size,
    }

    def compute():
        print("    steps solve ...", flush=True)
        cheap = handle.cheap_solve(mu_grid, episode_keys)
        steps = np.asarray(cheap["steps"]).reshape(cfg.n_sweep, cfg.num_repeats)

        err_int_mean = None
        if handle.dense_solve is not None:
            print("    e_int (dense) solve ...", flush=True)
            dense = handle.dense_solve(mu_grid, episode_keys)
            err_int = log10_time_integrated_error(
                ref_out["ts"], ref_out["ys"], dense["ts"], dense["ys"], atol
            )
            err_int_mean = np.asarray(err_int).reshape(cfg.n_sweep, cfg.num_repeats).mean(axis=1)

        print("    wallclock timing ...", flush=True)
        wallclock_ms = time_solve_fn(
            handle.cheap_solve,
            mus,
            mode="batched",
            report_batch_time=True,
            batch_size=cfg.wallclock_batch_size,
            repeats=cfg.wallclock_repeats,
            progress_label=spec["label"],
        )
        return {
            "steps_mean": steps.mean(axis=1),
            "err_int_mean": err_int_mean,
            "wallclock_ms": np.asarray(wallclock_ms),
        }

    return load_or_compute("model_sweep", payload, compute)


def plot_model_sweep(mus, results, out_path, param_label="mu", system="ODE"):
    """Write a 3-panel (steps, e_int, wallclock) comparison plot, one line per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_steps, ax_err, ax_wall) = plt.subplots(
        1, 3, figsize=(16, 4.6), constrained_layout=True
    )
    for label, r in results.items():
        ax_steps.plot(mus, r["steps_mean"], "o-", label=label, markersize=3)
        if r["err_int_mean"] is not None:
            ax_err.plot(mus, r["err_int_mean"], "o-", label=label, markersize=3)
        ax_wall.plot(mus, r["wallclock_ms"], "o-", label=label, markersize=3)

    ax_steps.set_ylabel("solver steps (accepted+rejected)")
    ax_err.set_ylabel("time-integrated log10 relative error (e_int)")
    ax_wall.set_ylabel("wallclock per batch (ms)")
    for ax in (ax_steps, ax_err, ax_wall):
        ax.set_xscale("log")
        ax.set_xlabel(param_label)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Model sweep vs {param_label} ({system})")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_model_sweep_txt(mus, results, out_path):
    """Write a tab-separated mu/steps/e_int/wallclock table, one column group per model."""
    header = ["mu"]
    for label in results:
        header += [f"{label}_steps", f"{label}_e_int", f"{label}_wallclock_ms"]
    lines = ["\t".join(header)]
    for i, mu in enumerate(mus):
        row = [f"{mu:.9g}"]
        for r in results.values():
            e_int = "" if r["err_int_mean"] is None else f"{r['err_int_mean'][i]:.6g}"
            row += [f"{r['steps_mean'][i]:.6g}", e_int, f"{r['wallclock_ms'][i]:.6g}"]
        lines.append("\t".join(row))
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved: {out_path}")


def main():
    """Sweep every configured model over one shared mu/episode grid and plot the result."""
    args = parse_args()
    cfg = (
        load_config_from_yaml(ModelSweepConfig, args.config) if args.config else ModelSweepConfig()
    )
    cfg = apply_dotted_overrides(cfg, args.overrides)
    cfg.device = _device
    cfg.models = resolve_models(cfg)

    if not cfg.env_config:
        raise SystemExit("env_config is required")
    if not cfg.models:
        raise SystemExit("models is empty — see configs.ModelSweepConfig's docstring")
    if cfg.n_sweep < 1:
        raise SystemExit("n_sweep must be positive")
    if cfg.num_repeats < 1:
        raise SystemExit("num_repeats must be positive")

    top_config = load_config_from_yaml(TrainConfig, cfg.env_config, strict=False)
    top_env_config = top_config.env
    max_steps = cfg.max_steps

    test_bins = list(top_env_config.test_bins)
    lo = cfg.param_min if cfg.param_min is not None else min(b[0] for b in test_bins)
    hi = cfg.param_max if cfg.param_max is not None else max(b[1] for b in test_bins)
    if lo <= 0 or hi <= lo:
        raise SystemExit("parameter range must satisfy 0 < param_min < param_max")
    mus = np.logspace(np.log10(lo), np.log10(hi), cfg.n_sweep)
    param_label = task_label(top_env_config.system)

    key_base = jax.random.PRNGKey(cfg.seed)
    mu_grid = np.repeat(mus, cfg.num_repeats)
    episode_keys = jax.vmap(lambda i: jax.random.fold_in(key_base, i))(jnp.arange(len(mu_grid)))

    labels = [spec.get("label", spec.get("type", "?")) for spec in cfg.models]
    print(
        f"[*] System={top_env_config.system}  mu range=[{lo:.3g}, {hi:.3g}]  n_sweep={cfg.n_sweep}  "
        f"num_repeats={cfg.num_repeats}  max_steps={max_steps}  device={cfg.device}"
    )
    print(f"[*] Models: {labels}")

    out_dir = resolve_output_dir("diagnostics", cfg, name=cfg.name, exclude=("name", "output"))

    print("[*] Solving shared tight-tolerance reference trajectory ...", flush=True)
    ref_config = dataclasses.replace(
        top_env_config,
        rtol=top_env_config.rtol * cfg.ref_tol_factor,
        atol=top_env_config.atol * cfg.ref_tol_factor,
    )
    ref_sc = env_pid_controller(ref_config)
    ref_out = solve_pid_batch(
        ref_config, ref_sc, mu_grid, episode_keys, max_steps * 10, save_steps=True, save_ys=True
    )

    results = {}
    for spec in cfg.models:
        print(f"[*] {spec.get('label', '?')} (type={spec.get('type', '?')}) ...", flush=True)
        handle = build_model(spec, top_env_config, top_config, max_steps)
        r = sweep_one_model(
            spec,
            handle,
            mus=mus,
            mu_grid=mu_grid,
            episode_keys=episode_keys,
            ref_out=ref_out,
            cfg=cfg,
            atol=top_env_config.atol,
        )
        results[spec["label"]] = r
        e_int_msg = "n/a" if r["err_int_mean"] is None else f"{r['err_int_mean'].mean():.3g}"
        print(
            f"    mean steps={r['steps_mean'].mean():.2f}  mean e_int={e_int_msg}  "
            f"mean wallclock={r['wallclock_ms'].mean():.3f} ms"
        )

    out_path = cfg.output or os.path.join(out_dir, "model_sweep.png")
    plot_model_sweep(mus, results, out_path, param_label=param_label, system=top_env_config.system)
    save_model_sweep_txt(mus, results, os.path.splitext(out_path)[0] + ".txt")
    print(f"[+] Saved: {out_path}")


if __name__ == "__main__":
    main()
