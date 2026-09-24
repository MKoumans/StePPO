"""Generate the PID evaluation dataset (PID, reference and optionally oracle solves)
for one split of a config, under data/<system>/evaluation/<split>set/. Training
and analysis only read it, so run this (or scripts/generate-pid-datasets-all-ode.sh)
first.

Usage:
    python generate_pid_eval_dataset.py --config configs/envs/ode/van_der_pol/van_der_pol_default.yaml --split train
    python generate_pid_eval_dataset.py --config ... --split val --num_envs 2048
    python generate_pid_eval_dataset.py --config ... --split test --force   # bust and rebuild an existing entry
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
from pathlib import Path

import numpy as np

from steppo.configs.base_config import TrainConfig, apply_precision, load_config_from_yaml
from steppo.envs.ode import (
    ODEEnv,  # noqa: F401 — import before error_dist, see ode_env.py's circular import
)
from steppo.training.error_dist import (
    DEFAULT_PID_CACHE_DIR,
    PID_EVAL_REF_TOL_FACTOR,
    PID_UNLIMITED_BUDGET_MULTIPLIER,
    load_pid_batch,
    log10_rel_error,
)
from steppo.training.steps_vs_mu import collect_pid_errors
from steppo.utils.figures.ode import (
    get_param_range,
    plot_steps_vs_mu,
    write_steps_vs_mu_error_outputs,
)


def build_eval_steps_vs_mu_series(data):
    """Return (pid_series, oracle_series) with completion masks; oracle is None if absent."""
    task_params = np.asarray(data["task_params"], dtype=float)
    pid_data = {
        "mu": task_params,
        "steps": np.asarray(data["pid_steps"], dtype=float),
        "completed": np.asarray(data["pid_completed"], dtype=bool),
    }
    oracle_steps = data.get("oracle_steps")
    oracle_data = (
        None
        if oracle_steps is None
        else {
            "mu": task_params,
            "steps": np.asarray(oracle_steps, dtype=float),
            "completed": np.asarray(data["oracle_completed"], dtype=bool),
        }
    )
    return pid_data, oracle_data


def save_eval_steps_vs_mu(config, data, split: str, output_root="outputs/datasets"):
    """Save a PID-vs-oracle steps figure for one cached evaluation dataset."""
    task_params = np.asarray(data["task_params"], dtype=float)
    pid_data, oracle_data = build_eval_steps_vs_mu_series(data)

    mu_min, mu_max, param_label = get_param_range(config)
    mu_lo = min(float(mu_min), float(task_params.min()))
    mu_hi = max(float(mu_max), float(task_params.max()))
    log_bins = mu_lo > 0.0 and bool(np.all(task_params > 0.0))
    output_path = (
        Path(output_root)
        / config.env.system
        / "evaluation"
        / f"{split}set"
        / str(data["fingerprint"])
        / "steps_vs_mu.png"
    )
    plot_steps_vs_mu(
        pid_data,
        oracle_data,
        float(mu_min),
        float(mu_max),
        mu_lo,
        mu_hi,
        bin_width=5.0,
        num_bins=64,
        log_bins=log_bins,
        out_path=str(output_path),
        param_label=param_label,
        comparison_label="Oracle",
    )
    pid_error = collect_pid_errors(data, config.env.atol)
    oracle_error = None
    if data.get("oracle_final_y") is not None:
        oracle_error = {
            "mu": task_params,
            "err": log10_rel_error(
                np.asarray(data["oracle_final_y"]),
                np.asarray(data["ref_final_y"]),
                config.env.atol,
            ),
        }
    write_steps_vs_mu_error_outputs(
        {split: {"pid_error": pid_error, "oracle_error": oracle_error}},
        float(mu_min),
        float(mu_max),
        str(output_path),
        param_label=param_label,
        train_bins=getattr(config.env, "train_bins", None),
    )
    return output_path


def main():
    """Generate or pre-warm the requested PID evaluation dataset."""
    parser = argparse.ArgumentParser(
        description="Pre-warm the diffrax-native PID eval-dataset cache"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=("train", "val", "test"),
        help="Which of config.env.{train,val,test}_bins to sample from.",
    )
    parser.add_argument(
        "-n", "--num_envs", type=int, default=1024, help="Episodes per bin in the chosen split"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ref_tol_factor", type=float, default=PID_EVAL_REF_TOL_FACTOR)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Step budget for the cached solve (default: config.rollout_steps)",
    )
    parser.add_argument("--timing_n", type=int, default=128)
    parser.add_argument("--timing_repeats", type=int, default=8)
    parser.add_argument(
        "--oracle_max_iters",
        type=int,
        default=6,
        help="Oracle hill-climb iteration cap per trajectory step",
    )
    parser.add_argument(
        "--budget_multiplier",
        type=int,
        default=PID_UNLIMITED_BUDGET_MULTIPLIER,
        help="Also solve PID at max_steps * this, same tolerances, so plots can "
        "show PID's step count once it isn't truncated by the RL budget",
    )
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_PID_CACHE_DIR)
    parser.add_argument("--force", action="store_true", help="Regenerate even if already cached")
    args = parser.parse_args()

    config = load_config_from_yaml(TrainConfig, args.config)
    apply_precision(config)
    max_steps = args.max_steps or config.rollout_steps
    bins = list(getattr(config.env, f"{args.split}_bins"))
    if not bins:
        parser.error(f"--split {args.split} given but config.env.{args.split}_bins is empty")

    print(f"[*] System        : {config.env.system}")
    print(f"[*] Split         : {args.split}  bins={bins}")
    print(
        f"[*] Episodes      : {args.num_envs} per bin  |  max_steps={max_steps}  |  seed={args.seed}"
    )
    print(
        f"[*] Timing        : {'on (n=' + str(args.timing_n) + ', repeats=' + str(args.timing_repeats) + ')'}"
    )
    assert config.env.immediate_dt_action, (
        "--with_oracle requires config.env.immediate_dt_action=True"
    )
    print(f"[*] Oracle        : {'on (max_iters=' + str(args.oracle_max_iters) + ')'}")
    print(f"[*] Cache dir     : {args.cache_dir}")
    print(f"[*] Force rebuild : {args.force}")

    data = load_pid_batch(
        config.env,
        num_envs=args.num_envs,
        seed=args.seed,
        max_steps=max_steps,
        bins=bins,
        split=args.split,
        ref_tol_factor=args.ref_tol_factor,
        timing_n=args.timing_n,
        timing_repeats=args.timing_repeats,
        oracle_max_iters=args.oracle_max_iters,
        budget_multiplier=args.budget_multiplier,
        cache_dir=args.cache_dir,
        force_regenerate=args.force,
    )

    print(f"[+] fingerprint={data['fingerprint']}")
    print(
        f"[+] {len(data['task_params'])} total episodes  |  "
        f"pid_steps range=[{data['pid_steps'].min()}, {data['pid_steps'].max()}]"
    )
    if data.get("pid_steps_unlimited") is not None:
        u = data["pid_steps_unlimited"]
        print(
            f"[+] pid_steps_unlimited (budget x{args.budget_multiplier}) range=[{u.min()}, {u.max()}]"
        )
    if data["timing"]:
        b, u = data["timing"]["budget"], data["timing"]["unlimited"]
        print(
            f"[+] timing: budget ms/ep={b['ms_per_ep']:.2f}  |  unlimited ms/ep={u['ms_per_ep']:.2f}"
        )
    if data["oracle_steps"] is not None:
        print(
            f"[+] oracle_steps range=[{data['oracle_steps'].min()}, {data['oracle_steps'].max()}]  "
            f"(mean savings vs pid: {float((data['pid_steps'] - data['oracle_steps']).mean()):.2f} steps)"
        )
    figure_path = save_eval_steps_vs_mu(config, data, split=args.split)
    print(f"[+] figure -> {figure_path}")
    print(
        f"[+] cached -> {args.cache_dir}/{config.env.system}/evaluation/{args.split}set/{data['fingerprint']}.npz"
    )

    print("[+] Done!")


if __name__ == "__main__":
    main()
