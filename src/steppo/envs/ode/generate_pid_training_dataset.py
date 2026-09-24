"""Generate the training-time PID datasets for a config: PID baseline step counts,
the reward-scale and progress-warp grids, and (for warmstart.expert: oracle) the oracle
dataset. Written under data/<system>/training/. Training only reads these, so run
this before training.

Usage:
    python generate_pid_training_dataset.py --config configs/envs/ode/van_der_pol/van_der_pol_default.yaml
    python generate_pid_training_dataset.py --config ... --force   # bust and rebuild existing entries
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
from pathlib import Path

import numpy as np

from steppo.configs.base_config import TrainConfig, apply_precision, load_config_from_yaml
from steppo.envs.ode import (
    ODEEnv,  # noqa: F401 — import before pid_solve, see ode_env.py's circular import
)
from steppo.training.eval import compute_pid_baseline_steps
from steppo.training.pid_solve import fingerprint, pid_env_payload
from steppo.training.reward_norm import build_pid_scale_grid, build_pid_warp_grid
from steppo.utils.figures.ode import get_param_range, plot_steps_vs_mu


def _generate_oracle_training_batch(config, args, lo, hi):
    from steppo.training.oracle_dataset import (
        DEFAULT_ORACLE_DATASET_EPISODES,
        generate_and_cache_oracle_training_batch,
        load_oracle_training_batch,
    )

    ws = config.warmstart
    oracle_episodes = args.oracle_episodes or DEFAULT_ORACLE_DATASET_EPISODES
    print(
        f"[*] Oracle training set: episodes={oracle_episodes}  rollout_steps={config.rollout_steps}  "
        f"episodes_per_trial={config.episodes_per_trial}  max_iters={ws.oracle_max_iters}  "
        f"dataset_seed={ws.oracle_dataset_seed}"
    )

    oracle_batch = None
    if not args.force:
        try:
            oracle_batch = load_oracle_training_batch(
                config.env,
                rollout_steps=config.rollout_steps,
                episodes_per_trial=config.episodes_per_trial,
                seed=ws.oracle_dataset_seed,
                oracle_max_iters=ws.oracle_max_iters,
            )
            print(f"[+] already cached: {oracle_batch['states'].shape[1]} episodes")
        except FileNotFoundError:
            pass
    if oracle_batch is None:
        oracle_batch = generate_and_cache_oracle_training_batch(
            config.env,
            num_envs=oracle_episodes,
            rollout_steps=config.rollout_steps,
            episodes_per_trial=config.episodes_per_trial,
            seed=ws.oracle_dataset_seed,
            oracle_max_iters=ws.oracle_max_iters,
        )
        print(
            f"[+] generated: {oracle_batch['states'].shape[1]} episodes "
            f"x {oracle_batch['states'].shape[0]} steps -> data/{config.env.system}/training/"
        )

    oracle_plot_data = _oracle_scale_grid_plot_data(oracle_batch, lo, hi)
    print(
        f"[+] oracle steps range: [{oracle_plot_data['steps'].min():.0f}, "
        f"{oracle_plot_data['steps'].max():.0f}]"
    )

    return oracle_plot_data


def _pid_generation_payload(config, lo, hi):
    """Return the canonical identity and resolved settings for this generation run."""
    pb = config.precompute_baseline
    rn = config.training.reward_normalization
    baseline_max_steps = pb.max_steps or config.rollout_steps * 10
    grid_max_steps = rn.max_steps or config.rollout_steps
    return (
        {
            "kind": "pid_training_generation",
            "env_config": pid_env_payload(config.env),
            "task_bounds": [float(lo), float(hi)],
            "baseline": {
                "mus": [float(mu) for mu in pb.eval_mus],
                "num_repeats": int(pb.num_repeats),
                "max_steps": int(baseline_max_steps),
            },
            "warp": {
                "grid_points": int(rn.grid_points),
                "num_repeats": int(rn.num_repeats),
                "max_steps": int(grid_max_steps),
                "num_knots": 64,
            },
            "scale": {
                "grid_points": int(rn.grid_points),
                "num_repeats": int(rn.num_repeats),
                "max_steps": int(grid_max_steps),
            },
        },
        baseline_max_steps,
        grid_max_steps,
    )


def pid_generation_fingerprint(config, lo, hi):
    """Return the stable output identity for a complete PID training generation run."""
    payload, _, _ = _pid_generation_payload(config, lo, hi)
    return fingerprint(payload)


def _scale_grid_plot_data(scale_grid, baseline_result=None):
    """Convert the cached scale grid into plot data and retain OOD baseline points."""
    log_mu_grid, log_pid_steps, _ = scale_grid
    mu = np.exp(np.asarray(log_mu_grid, dtype=float))
    steps = np.exp(np.asarray(log_pid_steps, dtype=float))
    if baseline_result:
        extras = [
            (float(mu_value), float(values["steps"]))
            for mu_value, values in baseline_result.items()
            if float(mu_value) < float(mu.min()) or float(mu_value) > float(mu.max())
        ]
        if extras:
            extra_mu, extra_steps = np.asarray(extras, dtype=float).T
            mu = np.concatenate([mu, extra_mu])
            steps = np.concatenate([steps, extra_steps])
    order = np.argsort(mu)
    return {"mu": mu[order], "steps": steps[order]}


def _oracle_scale_grid_plot_data(oracle_batch, lo, hi):
    """Per-episode oracle step counts and raw task values from the cached oracle rollout."""
    dones_ep = np.asarray(oracle_batch["dones_ep"], dtype=bool)  # (T, num_envs)
    task_params = np.asarray(
        oracle_batch["task_params"], dtype=float
    )  # (num_envs, task_dim), normalized [0,1]
    num_steps = dones_ep.shape[0]
    first_done = np.argmax(dones_ep, axis=0)
    never_done = ~dones_ep.any(axis=0)
    steps = np.where(never_done, num_steps, first_done + 1).astype(float)
    mu = task_params[:, 0] * (float(hi) - float(lo)) + float(lo)
    order = np.argsort(mu)
    return {"mu": mu[order], "steps": steps[order]}


def pid_training_figure_path(config, lo, hi, output_root="outputs/datasets"):
    """Return the training-calibration steps-vs-mu figure path."""
    return (
        Path(output_root)
        / config.env.system
        / "training"
        / pid_generation_fingerprint(config, lo, hi)
        / "steps_vs_mu.png"
    )


def main():
    """Generate the cached PID and optional oracle training datasets."""
    parser = argparse.ArgumentParser(description="Create PID training set.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true", help="Regenerate even if already generated")
    parser.add_argument(
        "--oracle_episodes",
        type=int,
        default=None,
        help="Episodes in the oracle training set (only used when config.warmstart.expert "
        "== 'oracle'; generation-time only, not part of the cache identity — see "
        "oracle_dataset.py). Default: oracle_dataset.DEFAULT_ORACLE_DATASET_EPISODES.",
    )
    args = parser.parse_args()

    config = load_config_from_yaml(TrainConfig, args.config)
    apply_precision(config)
    env = ODEEnv(config.env, max_steps=config.rollout_steps)
    lo, hi = env._task_bounds()
    rn = config.training.reward_normalization
    _, baseline_max_steps, grid_max_steps = _pid_generation_payload(config, lo, hi)

    print(f"[*] System        : {config.env.system}")
    print(f"[*] Task bounds   : [{float(lo):.4g}, {float(hi):.4g}]")
    print(f"[*] Force rebuild : {args.force}")

    pb = config.precompute_baseline
    print(
        f"[*] PID baseline  : mus={list(pb.eval_mus)}  num_repeats={pb.num_repeats}  max_steps={baseline_max_steps}"
    )
    result = compute_pid_baseline_steps(
        config.env,
        mus=list(pb.eval_mus),
        num_repeats=pb.num_repeats,
        max_steps=baseline_max_steps,
        force_regenerate=args.force,
        require_cache=False,
    )
    pid_str = ", ".join(
        f"μ={mu}: {v['steps']:.0f} ({v['accepted']:.0f}/{v['rejected']:.0f})"
        for mu, v in result.items()
    )
    print(f"[+] {pid_str}")

    print(
        f"[*] Progress warp : grid_points={rn.grid_points}  num_repeats={rn.num_repeats}  max_steps={grid_max_steps}"
    )
    log_mu_grid, t_knots, fracs = build_pid_warp_grid(
        config.env,
        float(lo),
        float(hi),
        grid_points=rn.grid_points,
        num_repeats=rn.num_repeats,
        max_steps=grid_max_steps,
        force_regenerate=args.force,
        require_cache=False,
    )
    print(f"[+] warp grid: {t_knots.shape[0]} μ pts × {t_knots.shape[1]} knots")

    print(
        f"[*] Reward-norm scale grid: grid_points={rn.grid_points}  num_repeats={rn.num_repeats}  max_steps={grid_max_steps}"
    )
    scale_grid = build_pid_scale_grid(
        config.env,
        float(lo),
        float(hi),
        grid_points=rn.grid_points,
        num_repeats=rn.num_repeats,
        max_steps=grid_max_steps,
        force_regenerate=args.force,
        require_cache=False,
    )

    oracle_plot_data = None
    if config.warmstart.expert == "oracle":
        oracle_plot_data = _generate_oracle_training_batch(config, args, lo, hi)

    plot_data = _scale_grid_plot_data(scale_grid, result)
    mu_values = plot_data["mu"]
    steps = plot_data["steps"]
    mu_min, mu_max, param_label = get_param_range(config)
    plot_mu_lo = min(float(lo), float(mu_values.min()))
    plot_mu_hi = max(float(hi), float(mu_values.max()))
    output_path = pid_training_figure_path(config, lo, hi)
    plot_steps_vs_mu(
        {"mu": mu_values, "steps": steps},
        oracle_plot_data,
        float(mu_min),
        float(mu_max),
        plot_mu_lo,
        plot_mu_hi,
        bin_width=5.0,
        num_bins=64,
        log_bins=plot_mu_lo > 0.0 and bool(np.all(mu_values > 0.0)),
        out_path=output_path,
        param_label=param_label,
        comparison_label="Oracle",
    )
    print(f"[+] cached -> data/{config.env.system}/training/")
    print(f"[+] figure -> {output_path}")

    print("[+] Done!")


if __name__ == "__main__":
    main()
