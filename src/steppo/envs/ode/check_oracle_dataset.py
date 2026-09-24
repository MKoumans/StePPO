"""Check (read-only) that the datasets a config needs are cached, so a run fails fast.

Exit code 0: nothing needed, or already cached.
Exit code 1: cache miss (message printed, including the fix command).

Usage:
    python src/steppo/envs/ode/check_oracle_dataset.py --config <config>.yaml
"""

import argparse
import shlex

from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import (
    ODEEnv,  # noqa: F401 — import before pid_solve, see ode_env.py's circular import
)
from steppo.training.error_dist import (
    PID_BIN_EVAL_NUM_ENVS,
    PID_COMPARE_NUM_EPISODES,
    PID_COMPARE_SEED,
    PID_EVAL_NUM_ENVS,
    PID_EVAL_REF_TOL_FACTOR,
    PID_EVAL_SEED,
    load_cached_pid_batch,
)


def _check_oracle_dataset(config_path: str, config) -> bool:
    from steppo.training.oracle_dataset import load_oracle_training_batch

    ws = config.warmstart
    if not (ws.enabled and ws.expert == "oracle"):
        return True

    try:
        load_oracle_training_batch(
            config.env,
            rollout_steps=config.rollout_steps,
            episodes_per_trial=config.episodes_per_trial,
            seed=ws.oracle_dataset_seed,
            oracle_max_iters=ws.oracle_max_iters,
        )
    except FileNotFoundError as e:
        print(f"[-] {config_path}: {e}")
        return False

    print(f"[+] {config_path}: oracle training set already cached")
    return True


def _check_progress_warp_cache(config_path: str, config) -> bool:
    """Check the progress-warp grid cache, needed when env.progress_warp is on for a scalar task."""
    from steppo.training.reward_norm import build_pid_warp_grid

    if not getattr(config.env, "progress_warp", False):
        return True

    env = ODEEnv(config.env, max_steps=config.rollout_steps)
    if not (hasattr(env, "t_end") and getattr(env, "task_dim", 0) == 1):
        return True

    rn = config.training.reward_normalization
    lo, hi = env._task_bounds()
    try:
        build_pid_warp_grid(
            config.env,
            float(lo),
            float(hi),
            grid_points=rn.grid_points,
            num_repeats=rn.num_repeats,
            max_steps=rn.max_steps or config.rollout_steps,
            require_cache=True,
        )
    except FileNotFoundError as e:
        print(f"[-] {config_path}: {e}")
        return False

    print(f"[+] {config_path}: PID progress-warp grid cache already cached")
    return True


def _evaluation_requests(config):
    """One request per non-empty {train,val,test}_bins split (canonical num_envs/seed,
    shared with base_trainer's live eval and offline analysis scripts), plus an extra
    "test" request at PID_BIN_EVAL_NUM_ENVS if not already covered."""
    requests = []
    for split in ("train", "val", "test"):
        bins = getattr(config.env, f"{split}_bins")
        if not bins:
            continue
        requests.append(
            {
                "label": f"{split}_bins",
                "split": split,
                "num_envs": PID_EVAL_NUM_ENVS,
                "seed": PID_EVAL_SEED,
                "bins": list(bins),
                "generator_args": [
                    "--split",
                    split,
                    "--ref_tol_factor",
                    repr(PID_EVAL_REF_TOL_FACTOR),
                ],
            }
        )

    test_bins = config.env.test_bins
    if test_bins:
        extra = [
            ("bin_comparison", PID_BIN_EVAL_NUM_ENVS, PID_EVAL_SEED),
            ("compare_table", PID_COMPARE_NUM_EPISODES, PID_COMPARE_SEED),
        ]
        seen = {(r["num_envs"], r["seed"]) for r in requests if r["split"] == "test"}
        for label, num_envs, seed in extra:
            if (num_envs, seed) in seen:
                continue
            seen.add((num_envs, seed))
            requests.append(
                {
                    "label": label,
                    "split": "test",
                    "num_envs": num_envs,
                    "seed": seed,
                    "bins": list(test_bins),
                    "generator_args": [
                        "--split",
                        "test",
                        "--ref_tol_factor",
                        repr(PID_EVAL_REF_TOL_FACTOR),
                    ],
                }
            )
    return requests


def _generation_command(config_path, request):
    return shlex.join(
        [
            "PYTHONPATH=.",
            "python",
            "src/steppo/envs/ode/generate_pid_eval_dataset.py",
            "--config",
            config_path,
            "--num_envs",
            str(request["num_envs"]),
            "--seed",
            str(request["seed"]),
            *request["generator_args"],
            "--force",
        ]
    )


def eval_generator_args(config) -> list[list[str]]:
    """generate_pid_eval_dataset.py args (sans --config/--force) for every PID eval
    dataset this config requires; shared source of truth with
    scripts/generate-pid-dataset.sh's --print-eval-args mode."""
    return [
        [
            "--num_envs",
            str(request["num_envs"]),
            "--seed",
            str(request["seed"]),
            *request["generator_args"],
        ]
        for request in _evaluation_requests(config)
    ]


def _check_pid_eval_datasets(config_path: str, config) -> bool:
    all_cached = True
    for request in _evaluation_requests(config):
        try:
            load_cached_pid_batch(
                config.env,
                max_steps=config.rollout_steps,
                num_envs=request["num_envs"],
                seed=request["seed"],
                split=request["split"],
                bins=request["bins"],
            )
        except FileNotFoundError:
            all_cached = False
            print(f"[-] {config_path}: missing {request['label']} PID evaluation cache")
            print(f"    Run: {_generation_command(config_path, request)}")
        except Exception as exc:
            all_cached = False
            print(f"[-] {config_path}: invalid {request['label']} PID evaluation cache: {exc}")
            print(f"    Regenerate with: {_generation_command(config_path, request)}")
        else:
            print(f"[+] {config_path}: {request['label']} PID evaluation cache already cached")
    return all_cached


def check_oracle_dataset(config_path: str) -> bool:
    """Return True when all training and analysis caches for the config exist."""
    config = load_config_from_yaml(TrainConfig, config_path)
    oracle_ok = _check_oracle_dataset(config_path, config)
    warp_ok = _check_progress_warp_cache(config_path, config)
    pid_eval_ok = _check_pid_eval_datasets(config_path, config)
    return oracle_ok and warp_ok and pid_eval_ok


def main():
    """Validate the configured oracle and PID caches without generating them."""
    parser = argparse.ArgumentParser(
        description="Pre-flight: check (read-only, no GPU) whether the training and PID "
        "evaluation datasets this config needs are cached."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--print-eval-args",
        action="store_true",
        help="Print (shlex-quoted, one per line) generate_pid_eval_dataset.py "
        "args for every required PID eval dataset instead of checking.",
    )
    args = parser.parse_args()

    if args.print_eval_args:
        config = load_config_from_yaml(TrainConfig, args.config)
        for argv in eval_generator_args(config):
            print(shlex.join(argv))
        raise SystemExit(0)

    raise SystemExit(0 if check_oracle_dataset(args.config) else 1)


if __name__ == "__main__":
    main()
