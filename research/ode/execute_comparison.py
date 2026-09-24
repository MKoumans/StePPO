"""Compute half of the experiment-batch comparison: run discovery, belief traces,
PID-vs-RL step counts and errors per split, and trained-vs-held-out bins.

Results are cached per run under `<out_dir>/data/<run_uid>_<fingerprint>.npz`,
so replotting with unchanged runs and arguments does not recompute anything.
"""

from steppo.utils.device import setup_devices

setup_devices()

import glob
import os
import re

import jax
import jax.numpy as jnp
import numpy as np

from research.ode.post_run_analysis.analyse_rollout import (
    collect_analysis_rollout,
    try_load_checkpoint,
)
from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv, ODEParams
from steppo.envs.ode.learned_controller import LearnedController
from steppo.training.error_dist import (
    PID_EVAL_REF_TOL_FACTOR,
    collect_controller_steps_and_errors,
    load_cached_pid_batch,
    log10_rel_error,
)
from steppo.training.pid_solve import atomic_savez, fingerprint, solve_pid_batch
from steppo.utils.checkpoint import build_models
from steppo.utils.task_params import task_bounds, task_label

# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def discover_runs(checkpoints_root: str, outputs_root: str) -> list[dict]:
    """Find the runs of an experiment batch.

    A run directory contains a <prefix>_e<N>[_r<seed>]_metrics.json file; its
    checkpoint directory mirrors its path under outputs_root.
    """
    runs = []
    for metrics_path in sorted(
        glob.glob(os.path.join(outputs_root, "**", "*_metrics.json"), recursive=True)
    ):
        out_run_dir = os.path.dirname(metrics_path)
        exp_name = os.path.basename(metrics_path)[: -len("_metrics.json")]
        # launch_experiments.sh names --repeat runs "<prefix>_e<N>_r<seed>"; repeats share an eid.
        match = re.search(r"e(\d+)(?:_r(\d+))?$", exp_name)
        if not match:
            print(f"[!] Skipping {out_run_dir}: could not parse experiment id from '{exp_name}'")
            continue
        eid = int(match.group(1))
        seed = int(match.group(2)) if match.group(2) is not None else None

        run_uid = os.path.basename(out_run_dir)
        run_subdir = os.path.relpath(out_run_dir, outputs_root)
        ckpt_run_dir = os.path.join(checkpoints_root, run_subdir)
        config_path = os.path.join(ckpt_run_dir, "config.yaml")
        if not os.path.isfile(config_path):
            print(f"[!] Skipping {exp_name}: no config.yaml at {config_path}")
            continue

        ckpt_dirs = sorted(
            glob.glob(os.path.join(ckpt_run_dir, "checkpoint_*")),
            key=lambda p: int(p.rsplit("_", 1)[-1]),
        )
        if not ckpt_dirs:
            print(f"[!] Skipping {exp_name}: no checkpoint_* dirs in {ckpt_run_dir}")
            continue

        runs.append(
            {
                "eid": eid,
                "seed": seed,
                "exp_name": exp_name,
                "run_uid": run_uid,
                "config_path": config_path,
                "checkpoint_dir": ckpt_dirs[-1],
                "metrics_path": metrics_path,
            }
        )

    runs.sort(key=lambda r: (r["eid"], r["seed"] if r["seed"] is not None else -1))
    return runs


# ---------------------------------------------------------------------------
# 1. Latent (belief mu / variance) rollouts at fixed test tasks
# ---------------------------------------------------------------------------


def collect_latent_traces_for_task(
    runs: list[dict], task_value: float, num_envs: int, seed: int
) -> dict:
    """Roll out every run's policy at one task value, averaged over num_envs environments.

    Returns {run_uid: {"mu": (T, latent_dim), "var": (T, latent_dim), "latent_dim": int}}.
    """
    traces = {}
    for run in runs:
        config = load_config_from_yaml(TrainConfig, run["config_path"])
        env = ODEEnv(config.env, config.rollout_steps)
        vae, policy = build_models(config, env, seed)
        vae, policy = try_load_checkpoint(
            vae, policy, run["checkpoint_dir"], backbone=config.backbone, algo=config.algo
        )

        T = config.rollout_steps
        env_params_batch = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape) if hasattr(x, "shape") else x,
            ODEParams(lam=jnp.float32(task_value), max_steps=T),
        )
        rng = jax.random.PRNGKey(seed)
        tr = collect_analysis_rollout(vae, policy, env, env_params_batch, num_envs, T, rng)

        mu_all = np.array(tr["belief_mu"])  # (T, E, latent_dim)
        logvar_all = np.array(tr["belief_logvar"])
        var_all = np.exp(np.clip(logvar_all, -10, 10))

        traces[run["run_uid"]] = {
            "mu": mu_all.mean(axis=1),
            "var": var_all.mean(axis=1),
            "latent_dim": mu_all.shape[-1],
        }
    return traces


# ---------------------------------------------------------------------------
# 2. Step-count-vs-PID distribution comparison (per split: train / val / test)
# ---------------------------------------------------------------------------


def collect_step_diff_traces(
    runs: list[dict], num_episodes: int, seed: int, ref_tol_factor: float = PID_EVAL_REF_TOL_FACTOR
) -> dict:
    """Compare PID and each run's RL controller on every configured split.

    Tasks come from the split's own bins, via the cached PID evaluation dataset;
    only the RL side is solved here. Returns {run_uid: {split: {...}}} with
    "diff" = pid_steps - rl_steps (positive: RL is cheaper), "rel_diff" =
    diff / pid_steps, "err" and "pid_err" (log10 relative error at t_end against
    the reference), "pid_steps" and "bin_idx".
    """
    traces = {}

    for run in runs:
        config = load_config_from_yaml(TrainConfig, run["config_path"])
        env = ODEEnv(config.env, config.rollout_steps)
        max_steps = config.rollout_steps

        splits_bins = [
            (name, list(bins))
            for name, bins in (
                ("train", config.env.train_bins),
                ("val", config.env.val_bins),
                ("test", config.env.test_bins),
            )
            if bins
        ]
        if not splits_bins:
            print(
                f"[!] {run['exp_name']}: no train/val/test_bins configured, skipping step-diff traces"
            )
            continue

        batches = []
        split_of_bin = []
        split_bin_offset = {}
        bin_offset = 0
        for name, bins in splits_bins:
            cached = load_cached_pid_batch(
                config.env,
                num_envs=num_episodes,
                seed=seed,
                max_steps=max_steps,
                bins=list(bins),
                split=name,
                ref_tol_factor=ref_tol_factor,
            )
            cached["split"] = np.asarray(cached["split"], dtype=np.int8) + bin_offset
            batches.append(cached)
            split_of_bin.extend([name] * len(bins))
            split_bin_offset[name] = bin_offset
            bin_offset += len(bins)
        batch = dict(batches[0])
        for key in (
            "task_params",
            "episode_keys",
            "split",
            "pid_steps",
            "pid_final_y",
            "ref_final_y",
            "oracle_steps",
            "oracle_final_y",
        ):
            if batch.get(key) is not None:
                batch[key] = np.concatenate([item[key] for item in batches])
        split_names = np.asarray(split_of_bin)[batch["split"]]
        masks = {name: split_names == name for name, _ in splits_bins}

        task_params = jnp.asarray(batch["task_params"])
        keys = jnp.asarray(batch["episode_keys"])
        probe = {name: (task_params[mask], keys[mask]) for name, mask in masks.items()}
        refs = {name: batch["ref_final_y"][mask] for name, mask in masks.items()}
        pid_err = {
            name: log10_rel_error(batch["pid_final_y"][mask], refs[name], config.env.atol)
            for name, mask in masks.items()
        }
        pid_steps = {name: batch["pid_steps"][mask].astype(float) for name, mask in masks.items()}

        sc = LearnedController.from_checkpoint(run["checkpoint_dir"], config, env)
        rl_by_split = collect_controller_steps_and_errors(config.env, sc, probe, refs, max_steps)

        traces[run["run_uid"]] = {}
        for name, mask in masks.items():
            rl_steps = rl_by_split[name]["steps"].astype(float)
            diff = pid_steps[name] - rl_steps
            traces[run["run_uid"]][name] = {
                "diff": diff,
                "rel_diff": diff / pid_steps[name],
                "err": rl_by_split[name]["err"],
                "pid_err": pid_err[name],
                "pid_steps": pid_steps[name],
                "bin_idx": batch["split"][mask] - split_bin_offset[name],
            }
    return traces


# ---------------------------------------------------------------------------
# 3. Trained-vs-held-out bin comparison (task_sample_scheme sweeps only)
# ---------------------------------------------------------------------------


def _bin_is_trained(bin_lo: float, bin_hi: float, train_bins) -> bool:
    """True if the run trained on the whole range (no train_bins) or `bin_` lies in one of its train_bins."""
    if not train_bins:
        return True
    return any(t_lo <= bin_lo and bin_hi <= t_hi for t_lo, t_hi in train_bins)


def collect_bin_comparison(
    runs: list[dict],
    num_episodes_per_bin: int,
    seed: int,
    step_diff_traces: dict | None = None,
    step_diff_num_episodes: int | None = None,
) -> dict:
    """Per run, the mean relative PID-vs-RL improvement in each test bin, marked
    trained or held out for that run.

    Tasks are sampled uniformly within each bin for every run. When
    `step_diff_traces` already holds the run's test split with the same episode
    count, the per-bin numbers are derived from it instead of solving again.
    Returns {run_uid: {"bins", "improvement", "trained", "scheme", "exp_name"}}.
    """
    results = {}
    for run in runs:
        config = load_config_from_yaml(TrainConfig, run["config_path"])
        test_bins = config.env.test_bins
        if not test_bins:
            print(f"[!] {run['exp_name']}: config.env.test_bins is empty, skipping bin comparison")
            continue
        train_bins = config.env.train_bins

        reuse = (
            step_diff_traces is not None
            and num_episodes_per_bin == step_diff_num_episodes
            and "test" in step_diff_traces.get(run["run_uid"], {})
        )
        if reuse:
            test_data = step_diff_traces[run["run_uid"]]["test"]
            pid_steps_all = test_data["pid_steps"]
            rl_steps_all = pid_steps_all - test_data["diff"]
            bin_idx_all = test_data["bin_idx"]
        else:
            env = ODEEnv(config.env, config.rollout_steps)
            max_steps = config.rollout_steps
            batch = load_cached_pid_batch(
                config.env,
                num_envs=num_episodes_per_bin,
                seed=seed,
                max_steps=max_steps,
                bins=list(test_bins),
                split="test",
            )
            sc = LearnedController.from_checkpoint(run["checkpoint_dir"], config, env)
            rl_out = solve_pid_batch(
                config.env, sc, batch["task_params"], batch["episode_keys"], max_steps
            )
            pid_steps_all = batch["pid_steps"].astype(float)
            rl_steps_all = rl_out["steps"].astype(float)
            bin_idx_all = batch["split"]

        bin_improvement = []
        bin_trained = []
        for i, (bin_lo, bin_hi) in enumerate(test_bins):
            mask = bin_idx_all == i
            improvement = (pid_steps_all[mask] - rl_steps_all[mask]) / np.maximum(
                pid_steps_all[mask], 1.0
            )
            bin_improvement.append(float(np.mean(improvement)))
            bin_trained.append(_bin_is_trained(bin_lo, bin_hi, train_bins))

        results[run["run_uid"]] = {
            "bins": list(test_bins),
            "improvement": np.array(bin_improvement),
            "trained": bin_trained,
            "scheme": config.env.task_sample_scheme,
            "exp_name": run["exp_name"],
        }
    return results


# ---------------------------------------------------------------------------
# On-disk cache: one .npz per run, content-addressed like the PID eval cache
# ---------------------------------------------------------------------------

_STEP_DIFF_FIELDS = ("diff", "rel_diff", "err", "pid_err", "pid_steps", "bin_idx")


def _run_data_fingerprint(run: dict, args, test_points: list[float]) -> str:
    payload = {
        "kind": "execute_comparison_run_data",
        "checkpoint_dir": run["checkpoint_dir"],
        "config_path": run["config_path"],
        "test_points": test_points,
        "num_step_diff_episodes": args.num_step_diff_episodes,
        "seed": args.seed,
        "ref_tol_factor": args.ref_tol_factor,
        "num_bin_episodes": args.num_bin_episodes,
    }
    return fingerprint(payload)


def _save_run_data(
    path: str, latent_by_task: dict, step_diff: dict, bin_result: dict | None
) -> None:
    arrays = {"latent_task_indices": np.array(sorted(latent_by_task.keys()), dtype=np.int32)}
    for task_idx, t in latent_by_task.items():
        arrays[f"latent_{task_idx}_mu"] = np.asarray(t["mu"])
        arrays[f"latent_{task_idx}_var"] = np.asarray(t["var"])

    arrays["stepdiff_splits"] = np.array(list(step_diff.keys()))
    for split, d in step_diff.items():
        for key in _STEP_DIFF_FIELDS:
            arrays[f"stepdiff_{split}_{key}"] = np.asarray(d[key])

    if bin_result is not None:
        bins = np.asarray(bin_result["bins"], dtype=np.float64)
        arrays["bin_lo"] = bins[:, 0]
        arrays["bin_hi"] = bins[:, 1]
        arrays["bin_improvement"] = bin_result["improvement"]
        arrays["bin_trained"] = np.asarray(bin_result["trained"], dtype=bool)
        arrays["bin_scheme"] = np.array(bin_result["scheme"])

    atomic_savez(path, **arrays)


def _load_run_data(path: str) -> tuple[dict, dict, dict | None]:
    data = np.load(path, allow_pickle=False)

    latent_by_task = {}
    for task_idx in data["latent_task_indices"].tolist():
        latent_by_task[int(task_idx)] = {
            "mu": data[f"latent_{task_idx}_mu"],
            "var": data[f"latent_{task_idx}_var"],
            "latent_dim": data[f"latent_{task_idx}_mu"].shape[-1],
        }

    step_diff = {}
    for split in data["stepdiff_splits"].tolist():
        step_diff[split] = {key: data[f"stepdiff_{split}_{key}"] for key in _STEP_DIFF_FIELDS}

    bin_result = None
    if "bin_improvement" in data.files:
        bin_result = {
            "bins": list(zip(data["bin_lo"].tolist(), data["bin_hi"].tolist())),
            "improvement": data["bin_improvement"],
            "trained": data["bin_trained"].tolist(),
            "scheme": str(data["bin_scheme"]),
        }
    return latent_by_task, step_diff, bin_result


def run_and_save(runs: list[dict], args, out_dir: str) -> dict:
    """Run all compute stages, or load their cached results, and return them.

    The cache is all-or-nothing: if any run lacks a matching file, every run is
    recomputed. Returns {"test_points", "param_label", "latent_traces_by_task",
    "step_diff_traces", "bin_results", "ref_config"}.
    """
    ref_config = load_config_from_yaml(TrainConfig, runs[0]["config_path"])
    lo, hi = task_bounds(ref_config.env)
    param_label = task_label(ref_config.env.system)
    test_points = [float(v) for v in np.geomspace(lo, hi, args.num_task_points)]
    print(
        f"[*] System: {ref_config.env.system}  |  full range {param_label} in [{lo:.2f}, {hi:.2f}]"
    )
    print(f"[*] Task points: {[f'{v:.2f}' for v in test_points]}")

    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    paths = {
        run["run_uid"]: os.path.join(
            data_dir, f"{run['run_uid']}_{_run_data_fingerprint(run, args, test_points)}.npz"
        )
        for run in runs
    }

    if all(os.path.isfile(p) for p in paths.values()):
        print(f"[*] Reusing cached comparison data for all {len(runs)} run(s) in {data_dir}")
    else:
        print(f"[*] Computing comparison data for {len(runs)} run(s) ...")
        latent_by_task_by_run = {run["run_uid"]: {} for run in runs}
        for i, task_value in enumerate(test_points, start=1):
            print(f"[*] ({i}/{len(test_points)}) Rolling out at {param_label}={task_value:.2f} ...")
            traces = collect_latent_traces_for_task(
                runs, task_value, 1024, args.seed
            )  # num_envs hardcoded, see below
            for run in runs:
                latent_by_task_by_run[run["run_uid"]][i] = traces[run["run_uid"]]

        available_splits = [
            s for s in ("train", "val", "test") if getattr(ref_config.env, f"{s}_bins")
        ]
        print(
            f"[*] Collecting step-count and solution-error distributions "
            f"({args.num_step_diff_episodes} episodes/split/run, splits={available_splits}) ..."
        )
        step_diff_traces = collect_step_diff_traces(
            runs,
            args.num_step_diff_episodes,
            args.seed,
            ref_tol_factor=args.ref_tol_factor,
        )

        bin_results = {}
        if ref_config.env.test_bins:
            print(
                f"[*] Collecting trained-vs-held-out bin comparison ({args.num_bin_episodes} episodes/bin) ..."
            )
            bin_results = collect_bin_comparison(
                runs,
                args.num_bin_episodes,
                args.seed,
                step_diff_traces=step_diff_traces,
                step_diff_num_episodes=args.num_step_diff_episodes,
            )
        else:
            print("[*] Skipping trained-vs-held-out bin comparison (config.env.test_bins is empty)")

        for run in runs:
            uid = run["run_uid"]
            _save_run_data(
                paths[uid],
                latent_by_task_by_run.get(uid, {}),
                step_diff_traces.get(uid, {}),
                bin_results.get(uid),
            )
        print(f"[*] Saved raw comparison data -> {data_dir}")

    latent_traces_by_task: dict[int, dict] = {i: {} for i in range(1, len(test_points) + 1)}
    step_diff_traces = {}
    bin_results = {}
    for run in runs:
        uid = run["run_uid"]
        latent_by_task, step_diff, bin_result = _load_run_data(paths[uid])
        for i, t in latent_by_task.items():
            latent_traces_by_task[i][uid] = t
        step_diff_traces[uid] = step_diff
        if bin_result is not None:
            bin_result = dict(bin_result, exp_name=run["exp_name"])
            bin_results[uid] = bin_result

    return {
        "test_points": test_points,
        "param_label": param_label,
        "latent_traces_by_task": latent_traces_by_task,
        "step_diff_traces": step_diff_traces,
        "bin_results": bin_results,
        "ref_config": ref_config,
    }
