"""Histogram: solver steps vs μ, comparing PID (budget) vs RL (diffeqsolve).

Includes out-of-distribution μ values beyond the training range, with vertical
lines marking the in-distribution boundaries.

Usage:
    # PID only:
    python steps_vs_mu.py --config configs/envs/ode/van_der_pol/van_der_pol_default.yaml

    # PID vs RL:
    python steps_vs_mu.py --checkpoint checkpoints/run_.../checkpoint_0
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import dataclasses
import os
from pathlib import Path

import jax
import numpy as np

from research.ode.post_run_analysis.rollout_cache import load_or_compute
from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv
from steppo.envs.ode.learned_controller import LearnedController
from steppo.training.error_dist import PID_UNLIMITED_BUDGET_MULTIPLIER
from steppo.training.pid_solve import solve_pid_batch
from steppo.training.steps_vs_mu import build_steps_vs_mu_data_by_split
from steppo.utils.checkpoint import resolve_checkpoint
from steppo.utils.figures.ode import (
    get_param_range,
    plot_steps_vs_mu_splits,
    save_convergence_table,
    save_steps_vs_mu_splits_txt,
    write_steps_vs_mu_error_outputs,
)


def collect_learned_controller(
    config, checkpoint_or_controller, env, mu_values, max_steps, episode_keys=None
):
    """Return per-episode (mu, steps) for RL diffeqsolve controller with explicit μ values.

    `checkpoint_or_controller` accepts a checkpoint path or an already-built
    LearnedController (pass the latter to reuse one model build across batches).
    """
    sc = (
        checkpoint_or_controller
        if isinstance(checkpoint_or_controller, LearnedController)
        else LearnedController.from_checkpoint(checkpoint_or_controller, config, env)
    )
    if episode_keys is None:
        rng = jax.random.PRNGKey(0)
        episode_keys = jax.random.split(rng, len(mu_values))
    out = solve_pid_batch(config.env, sc, mu_values, episode_keys, max_steps)
    return {"mu": mu_values, "steps": out["steps"]}


def main():
    """Build and plot PID-versus-policy solver steps by task parameter."""
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

    parser = argparse.ArgumentParser(description="Steps vs μ histogram: PID vs RL")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("-n", "--num_episodes", type=int, default=1024)
    parser.add_argument("--bin_width", type=float, default=5.0, help="μ bin width (linear mode)")
    parser.add_argument("--num_bins", type=int, default=64, help="Number of bins (log mode)")
    parser.add_argument("--log_bins", action="store_true", help="Use logarithmic bin spacing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--oracle_max_iters",
        type=int,
        default=6,
        help="Max inner iterations for oracle PID controller",
    )
    parser.add_argument(
        "--budget_multiplier",
        type=int,
        default=PID_UNLIMITED_BUDGET_MULTIPLIER,
        help="Also plot PID solved at max_steps * this (same tolerances), so the "
        "PID line isn't truncated by the RL episode's own step budget",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    if args.checkpoint is not None:
        args.checkpoint = resolve_checkpoint(args.checkpoint)

    if args.config is None and args.checkpoint is not None:
        run_dir = os.path.dirname(args.checkpoint)
        bundled = os.path.join(run_dir, "config.yaml")
        if os.path.isfile(bundled):
            args.config = bundled
            print(f"[*] Auto-detected config: {bundled}")

    if args.config is None:
        parser.error("--config is required (no config.yaml found in checkpoint dir)")

    config = load_config_from_yaml(TrainConfig, args.config, strict=False)
    env = ODEEnv(config.env, config.rollout_steps)
    max_steps = config.rollout_steps
    mu_min, mu_max, param_label = get_param_range(config)

    if not config.env.test_bins:
        parser.error("config.env.test_bins is empty")
    split_bins = {
        "train": config.env.train_bins,
        "val": config.env.val_bins,
        "test": config.env.test_bins,
    }
    train_bins = config.env.train_bins or None

    print(f"System: {config.env.system}  |  splits: {split_bins}")
    bin_label = (
        f"num_bins={args.num_bins} (log)" if args.log_bins else f"bin_width={args.bin_width}"
    )
    print(f"Episodes: {args.num_episodes}  |  {bin_label}")
    print()

    policy_controller = None
    if args.checkpoint:
        print("Collecting Policy data on cached episodes ...", flush=True)
        policy_controller = LearnedController.from_checkpoint(args.checkpoint, config, env)
    print("Collecting PID/RL data for train/val/test splits ...", flush=True)
    cache_payload = {
        "checkpoint": args.checkpoint,
        "env_config": dataclasses.asdict(config.env),
        "max_steps": max_steps,
        "split_bins": {k: [[float(lo), float(hi)] for lo, hi in v] for k, v in split_bins.items()},
        "num_envs": args.num_episodes,
        "seed": args.seed,
        "oracle_max_iters": args.oracle_max_iters,
        "budget_multiplier": args.budget_multiplier,
        "with_error": True,
    }
    split_data = load_or_compute(
        "steps_vs_mu",
        cache_payload,
        lambda: build_steps_vs_mu_data_by_split(
            config.env,
            max_steps=max_steps,
            split_bins=split_bins,
            policy_controller=policy_controller,
            num_envs=args.num_episodes,
            seed=args.seed,
            oracle_max_iters=args.oracle_max_iters,
            budget_multiplier=args.budget_multiplier,
            with_error=True,
        ),
    )
    for split, series in split_data.items():
        pid_data = series["pid"]
        msg = (
            f"  {split}: {len(pid_data['mu'])} episodes, steps range: "
            f"[{pid_data['steps'].min()}, {pid_data['steps'].max()}]"
        )
        pid_unlimited = series.get("pid_unlimited")
        if pid_unlimited is not None:
            msg += (
                f"  |  unlimited (x{args.budget_multiplier}) range: "
                f"[{pid_unlimited['steps'].min()}, {pid_unlimited['steps'].max()}]"
            )
        print(msg)
        # Steps are only comparable across episodes that finished — flag the
        # ones that did not, since a solve aborted at dt_min reports a small
        # step count and is otherwise indistinguishable from a cheap one.
        for name in ("pid", "policy", "oracle"):
            data = series.get(name)
            if data is None or "completed" not in data:
                continue
            completion = float(np.mean(data["completed"]))
            if completion < 1.0:
                print(
                    f"    [!] {name}: only {completion:.1%} of {split} episodes reached "
                    f"t_end; the rest are excluded from the plotted means"
                )
        ref_completed = series["cache"].get("ref_completed")
        if ref_completed is not None and not np.all(ref_completed):
            print(
                f"    [!] reference: only {float(np.mean(ref_completed)):.1%} of {split} "
                "reference solves reached t_end; errors for the rest are not reported"
            )

    if args.output:
        out_path = args.output
    elif args.checkpoint:
        run_output_dir = Path(args.checkpoint.replace("checkpoints", "outputs", 1)).parent
        out_dir = run_output_dir / "compare"
        os.makedirs(out_dir, exist_ok=True)
        out_path = str(out_dir / "steps_vs_mu.png")
    else:
        out_path = "steps_vs_mu.png"

    if mu_min == mu_max:
        table_path = os.path.splitext(out_path)[0] + ".txt"
        save_convergence_table(split_data, table_path, param_label=param_label)
        return

    plot_steps_vs_mu_splits(
        split_data,
        mu_min,
        mu_max,
        out_path,
        param_label=param_label,
        train_bins=train_bins,
    )
    txt_path = os.path.splitext(out_path)[0] + ".txt"
    save_steps_vs_mu_splits_txt(split_data, txt_path, mu_min, mu_max)
    print(f"  → {txt_path}")

    for err_txt_path in write_steps_vs_mu_error_outputs(
        split_data,
        mu_min,
        mu_max,
        out_path,
        param_label=param_label,
        train_bins=train_bins,
    ):
        print(f"  → {err_txt_path}")


if __name__ == "__main__":
    main()
