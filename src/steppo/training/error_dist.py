"""PID eval-dataset caching and error-distribution helpers."""

import dataclasses
import json
import os
import time

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from steppo.configs.base_config import PRECISION_DTYPES
from steppo.envs.ode.systems import get_rhs, get_system
from steppo.envs.ode_env import ODEEnv, ODEParams, _make_solver
from steppo.training.pid_solve import env_pid_controller
from steppo.training.pid_solve import solve_pid_batch as _pid_solve_batch
from steppo.training.pid_solve import solve_reference_batch as _solve_reference_batch
from steppo.training.trajectory_compare import (
    final_state_from_trajectory,
    local_error,
    relative_l2_error,
    to_log10_clipped,
)

_BATCH_SIZE = 64

DEFAULT_PID_CACHE_DIR = "data"

PID_EVAL_REF_TOL_FACTOR = 1e-2

_PID_EVAL_SPLIT_SEED_OFFSETS = {
    "train": 0,
    "val": 1_000_003,
    "test": 2_000_033,
}


def split_sampling_seed(seed: int, split: str | None) -> int:
    """Get RNG seed for a PID evaluation split."""
    if split is None:
        return int(seed)
    try:
        offset = _PID_EVAL_SPLIT_SEED_OFFSETS[split]
    except KeyError as exc:
        raise ValueError(f"Unknown PID evaluation split: {split!r}") from exc
    return (int(seed) + offset) % (2**32)


def sample_mu_from_bins(bins, scheme: str, num_episodes: int, seed: int) -> np.ndarray:
    """Sample from binned distribution; scheme: 'log-binned' or 'binned'."""
    rng = np.random.default_rng(seed)
    bins = np.asarray(bins, dtype=np.float64)
    idx = rng.integers(0, len(bins), size=num_episodes)
    lo, hi = bins[idx, 0], bins[idx, 1]
    if scheme == "log-binned":
        return np.exp(rng.uniform(np.log(lo), np.log(hi))).astype(np.float32)
    return rng.uniform(lo, hi).astype(np.float32)


def _oracle_try_dt(env, dt_log_gain, dummy_key, state, params, dt_candidate):
    """Try one explicit target dt via env.step."""
    action = jnp.log(dt_candidate / state.dt) / dt_log_gain
    _, new_state, _, done, info = env.step(dummy_key, state, action, params)
    return info["keep_step"], new_state, done


def _oracle_hill_climb(
    env,
    dt_log_gain,
    dummy_key,
    state,
    params,
    max_iters=6,
    pid_q=4,
    pid_safety=0.9,
    pid_min_factor=0.2,
    pid_max_factor=10.0,
    pid_kp=0.0,
    pid_ki=1.0,
    pid_kd=0.0,
):
    """Improve one PID step by probing the environment.

    From an accepted PID step, try larger steps with relative gains 25 %, 12.5 %,
    6.25 %, ...; from a rejected one, shrink by 25 % per iteration until a step is
    accepted or `max_iters` runs out. The PID step is the fallback.
    """
    from steppo.training.pid_controller import (
        compute_step_context_offset,
        pid_action_from_obs,
    )

    step_context_offset = compute_step_context_offset(
        env._config.obs_features, env._spec.feature_dims
    )

    def try_dt(dt):
        return _oracle_try_dt(env, dt_log_gain, dummy_key, state, params, dt)

    # This is the same closed-form PID expert used by the live PID rollout.
    pid_obs = env._spec.obs(env._obs_state(state), env._config)
    pid_action = pid_action_from_obs(
        pid_obs,
        step_context_offset,
        q=pid_q,
        safety=pid_safety,
        min_factor=pid_min_factor,
        max_factor=pid_max_factor,
        dt_log_gain=dt_log_gain,
        kp=pid_kp,
        ki=pid_ki,
        kd=pid_kd,
    )
    pid_dt = jnp.clip(
        state.dt * jnp.exp(pid_action.squeeze() * dt_log_gain),
        env.dt_min,
        env.dt_max,
    )
    pid_ok, pid_state, pid_done = try_dt(pid_dt)

    def cond_fn(carry):
        i, *_ = carry
        return i < max_iters

    def body_fn(carry):
        i, anchor, anchor_ok, best_state, best_done = carry
        relative_gain = jnp.float32(0.25) * jnp.power(jnp.float32(0.5), i)
        up = anchor * (1.0 + relative_gain)
        down = anchor * jnp.float32(0.75)

        def no_probe():
            return jnp.bool_(False), best_state, best_done

        up_ok, up_state, up_done = jax.lax.cond(
            anchor_ok,
            lambda: try_dt(up),
            no_probe,
        )
        down_ok, down_state, down_done = jax.lax.cond(
            jnp.logical_not(anchor_ok),
            lambda: try_dt(down),
            no_probe,
        )

        # An upward candidate must be accepted and beat the current best; a rejected
        # anchor keeps shrinking.
        accept_up = jnp.logical_and(
            anchor_ok,
            jnp.logical_and(up_ok, up_state.t > best_state.t),
        )
        accept_down = jnp.logical_and(
            jnp.logical_not(anchor_ok),
            jnp.logical_and(down_ok, down_state.t > state.t),
        )

        next_anchor = jnp.where(
            anchor_ok,
            jnp.where(accept_up, up_state.dt, anchor),
            down_state.dt,
        )
        next_anchor_ok = jnp.where(anchor_ok, True, down_ok)
        next_best_state = jax.tree.map(
            lambda up_leaf, down_leaf, best_leaf: jnp.where(
                accept_up,
                up_leaf,
                jnp.where(accept_down, down_leaf, best_leaf),
            ),
            up_state,
            down_state,
            best_state,
        )
        next_best_done = jnp.where(
            accept_up,
            up_done,
            jnp.where(accept_down, down_done, best_done),
        )
        return (
            i + 1,
            next_anchor,
            next_anchor_ok,
            next_best_state,
            next_best_done,
        )

    init = (jnp.int32(0), pid_state.dt, pid_ok, pid_state, pid_done)
    _, _, _, best_state, best_done = jax.lax.while_loop(cond_fn, body_fn, init)
    return best_state, best_done


def oracle_action(env, dt_log_gain, key, state, params, max_iters: int) -> Array:
    """Expert action from the oracle hill-climb search (needs the live env state)."""
    searched_state, _ = _oracle_hill_climb(env, dt_log_gain, key, state, params, max_iters)
    action = jnp.clip(jnp.log(searched_state.dt / state.dt) / dt_log_gain, -1.0, 1.0)
    return jnp.array([action], dtype=jnp.float32)


def solve_oracle_batch(env_config, task_params, keys, max_steps: int, max_iters: int = 20) -> dict:
    """Solve tasks with the oracle controller (hill-climb search at every step).

    Uses solve_pid_batch's key convention, so episodes match the PID solves.
    Requires `env_config.immediate_dt_action`. Returns {"steps", "final_y",
    "completed", "ts", "ys"}; `steps` counts accepted steps only and is
    meaningful only where `completed`. `ts`/`ys` hold the accepted steps,
    inf-padded to `max_steps` like solve_pid_batch's saved steps.
    """
    assert env_config.immediate_dt_action, (
        "solve_oracle_batch requires immediate_dt_action=True — the search "
        "candidate action must affect the step it's testing, not the next one."
    )
    env = ODEEnv(env_config, max_steps)
    dt_log_gain = jnp.asarray(env_config.dt_log_gain, dtype=jnp.float32)
    dummy_key = jax.random.PRNGKey(0)

    def one_task(mu, key):
        key_y0, key_pulse = jax.random.split(key)
        pulse_phase = jax.random.uniform(key_pulse, shape=(), dtype=jnp.float32)
        params = ODEParams(lam=mu, pulse_phase=pulse_phase, max_steps=max_steps)
        _, state0 = env.reset(key_y0, params)

        def scan_step(carry, i):
            state, done = carry
            hill_state, step_done = _oracle_hill_climb(
                env, dt_log_gain, dummy_key, state, params, max_iters
            )
            new_state = jax.tree.map(lambda n, o: jnp.where(done, o, n), hill_state, state)
            progressed = jnp.logical_and(jnp.logical_not(done), new_state.t > state.t)
            new_done = jnp.logical_or(done, step_done)
            return (new_state, new_done), (progressed, new_state.t, new_state.y)

        (final_state, _), (progress_trace, ts, ys) = jax.lax.scan(
            scan_step, (state0, jnp.bool_(False)), jnp.arange(max_steps)
        )
        steps = jnp.sum(progress_trace.astype(jnp.int32))
        completed = final_state.t >= jnp.asarray(env_config.t_end, final_state.t.dtype) * (
            1.0 - 1e-9
        )
        ts = jnp.where(progress_trace, ts, jnp.inf)
        ys = jnp.where(progress_trace[:, None], ys, jnp.inf)
        return steps, final_state.y, completed, ts, ys

    @jax.jit
    def solve_chunk(mus, ks):
        return jax.vmap(one_task)(mus, ks)

    task_params = jnp.asarray(task_params, dtype=jnp.float32)
    keys = jnp.asarray(keys)
    all_steps, all_y, all_done, all_ts, all_ys = [], [], [], [], []
    for start in range(0, len(task_params), _BATCH_SIZE):
        s, y, done, ts, ys = solve_chunk(
            task_params[start : start + _BATCH_SIZE], keys[start : start + _BATCH_SIZE]
        )
        all_steps.append(np.asarray(s))
        all_y.append(np.asarray(y))
        all_done.append(np.asarray(done))
        all_ts.append(np.asarray(ts))
        all_ys.append(np.asarray(ys))
    return {
        "steps": np.concatenate(all_steps),
        "final_y": np.concatenate(all_y),
        "completed": np.concatenate(all_done),
        "ts": np.concatenate(all_ts),
        "ys": np.concatenate(all_ys),
    }


def solve_pid_baseline_batch(env_config, task_params, keys, max_steps: int) -> dict:
    """Run the exact PID fallback used by the oracle, without search probes
    (an ODE-env rollout driven by pid_action_from_obs, not diffrax's own
    PIDController). max_iters=0 makes the oracle sweep's search degenerate
    to exactly this baseline."""
    return solve_oracle_batch(
        env_config,
        task_params,
        keys,
        max_steps,
        max_iters=0,
    )


def log10_rel_error(y: np.ndarray, y_ref: np.ndarray, atol: float) -> np.ndarray:
    """log10 relative L2 error of `y` vs. reference `y_ref`, for callers
    holding pre-solved final states. See trajectory_compare for the
    trajectory-level counterparts this generalizes."""
    return to_log10_clipped(relative_l2_error(y, y_ref, atol))


def collect_controller_steps_and_errors(
    env_config, stepsize_controller, probe: dict, refs: dict, max_steps: int
) -> dict:
    """{split: {"steps": (N,), "err": (N,) log10 relative L2 error}} for the
    given controller from one dense solve — `solve_pid_batch` returns both
    `final_y` and `steps` in one pass, avoiding a duplicate diffeqsolve."""
    out = {}
    for split, (mu, keys) in probe.items():
        result = _pid_solve_batch(
            env_config, stepsize_controller, mu, keys, max_steps, save_steps=False
        )
        out[split] = {
            "steps": result["steps"],
            "err": log10_rel_error(result["final_y"], refs[split], env_config.atol),
        }
    return out


# ─── canonical task sampling for the PID eval-dataset ──────────────────────


def sample_eval_tasks(env_config, num_envs: int, seed: int, bins, split: str | None = None) -> dict:
    """Sample `num_envs` tasks per bin for a PID evaluation dataset.

    Returns {"task_params", "episode_keys", "split" (bin index)}. Each named
    split uses its own RNG stream, so identical bins give different tasks.
    """
    sampling_seed = split_sampling_seed(seed, split)
    chunks, splits = [], []
    for i, (lo, hi) in enumerate(bins):
        vals = sample_mu_from_bins(
            [(lo, hi)], env_config.task_sample_scheme, num_envs, sampling_seed + i
        )
        chunks.append(vals)
        splits.append(np.full(num_envs, i, dtype=np.int8))
    task_params = np.concatenate(chunks)
    split = np.concatenate(splits)
    episode_keys = np.asarray(jax.random.split(jax.random.PRNGKey(sampling_seed), len(task_params)))
    return {"task_params": task_params, "episode_keys": episode_keys, "split": split}


PID_UNLIMITED_BUDGET_MULTIPLIER = 5
PID_EVAL_REF_TOL_FACTOR = 1e-2


def pid_eval_fingerprint(
    env_config,
    num_envs: int,
    seed: int,
    max_steps: int,
    split: str,
    bins,
    ref_tol_factor: float = PID_EVAL_REF_TOL_FACTOR,
    timing: dict | None = None,
    oracle: dict | None = None,
    budget_multiplier: int = PID_UNLIMITED_BUDGET_MULTIPLIER,
) -> str:
    """Hash of the full env config and every sampling/solve option; new fields invalidate caches."""
    from steppo.training.pid_solve import fingerprint as _fingerprint

    payload = {
        "kind": "pid_eval_dataset",
        "split": split,
        "env_config": dataclasses.asdict(env_config),
        "num_envs": int(num_envs),
        "seed": int(seed),
        "sampling_seed": split_sampling_seed(seed, split),
        "max_steps": int(max_steps),
        "bins": [[float(lo), float(hi)] for lo, hi in bins],
        "ref_tol_factor": float(ref_tol_factor),
        "timing": timing,
        "oracle": oracle,
        "budget_multiplier": int(budget_multiplier),
    }
    return _fingerprint(payload)


# ─── wall-clock timing benchmark (moved from research/ode/post_run_analysis/compare.py) ──


def _run_timed(run_fn, keys, batch_size, repeats, progress_label=None):
    """Run `run_fn` over batched keys, repeat `repeats` times.
    Returns (list_of_per_repeat_results, list_of_timings). `progress_label`,
    if set, prints progress after each repeat since unbudgeted repeats can
    take minutes each."""
    timings = []
    repeat_results = []
    for i in range(repeats):
        t0 = time.perf_counter()
        batch_results = []
        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start : start + batch_size]
            batch_res = run_fn(batch_keys)
            batch_results.append(jax.tree.map(np.asarray, batch_res))
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        repeat_results.append(jax.tree.map(lambda *arrs: np.concatenate(arrs), *batch_results))
        if progress_label:
            print(
                f"\n    [{progress_label}] repeat {i + 1}/{repeats} ({elapsed:.1f}s)",
                end="",
                flush=True,
            )
    return repeat_results, timings


def _aggregate(per_repeat: list[dict]) -> dict:
    """Mean and std of each metric across repeats."""
    out = {}
    for key in per_repeat[0]:
        vals = [m[key] for m in per_repeat]
        out[key] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def sample_task_with_pulse(spec, key, env_config):
    """Appends a pulse-phase seed to spec.sample_task's hidden param(s), same
    as ODEEnv.sample_task, so pulse-forced systems see a task[1] consistent
    with training."""
    key_task, key_pulse = jax.random.split(key)
    task = spec.sample_task(key_task, env_config)
    pulse_phase = jax.random.uniform(key_pulse, shape=(), dtype=jnp.float32)
    return jnp.concatenate([task, pulse_phase[None]])


def _time_pid_solve(
    env_config, num_episodes: int, max_steps: int, seed: int, repeats: int, progress_label=None
) -> dict:
    """Wall-clock benchmark of the diffrax-native PID solve. Samples its own
    tasks from the training distribution, independent of the frozen
    task_params/episode_keys `load_pid_batch` stores — this measures pure
    wall-clock cost, not accuracy on a fixed benchmark set."""
    spec = get_system(env_config.system)
    rhs = get_rhs(env_config)
    solver = _make_solver(env_config.rtol, env_config.atol)
    if env_config.precision == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = PRECISION_DTYPES[env_config.precision]

    sc = env_pid_controller(env_config)

    @jax.jit
    def solve_one(key):
        key_task, key_ep = jax.random.split(key)
        key_reset, _ = jax.random.split(key_ep)
        task = sample_task_with_pulse(spec, key_task, env_config).astype(dtype)
        y0 = spec.y0(task, key_reset, env_config).astype(dtype)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            solver,
            t0=jnp.asarray(0.0, dtype=dtype),
            t1=jnp.asarray(env_config.t_end, dtype=dtype),
            dt0=jnp.asarray(env_config.dt0, dtype=dtype),
            y0=y0,
            args=task,
            stepsize_controller=sc,
            max_steps=max_steps,
            throw=False,
        )
        return {
            "t_reached": sol.ts[-1],
            "num_steps": sol.stats["num_steps"],
            "accepted": sol.stats["num_accepted_steps"],
            "rejected": sol.stats["num_rejected_steps"],
        }

    @jax.jit
    def solve_batch(keys):
        return jax.vmap(solve_one)(keys)

    rng = jax.random.PRNGKey(seed)
    keys = jax.vmap(lambda i: jax.random.fold_in(rng, i))(jnp.arange(num_episodes))
    batch_size = min(num_episodes, 32)

    # Warmup: trigger JIT compilation, don't count in results
    warmup_keys = jax.random.split(jax.random.fold_in(rng, 2**31 - 1), batch_size)
    t0 = time.perf_counter()
    jax.tree.map(lambda x: x.block_until_ready(), solve_batch(warmup_keys))
    t_compile = time.perf_counter() - t0

    repeat_results, timings = _run_timed(solve_batch, keys, batch_size, repeats, progress_label)

    per_repeat = []
    for i, results in enumerate(repeat_results):
        t_reached = results["t_reached"]
        accepted = results["accepted"].astype(float)
        rejected = results["rejected"].astype(float)
        total = accepted + rejected
        per_repeat.append(
            {
                "success_rate": float(np.mean(t_reached >= env_config.t_end * 0.999)),
                "mean_steps": float(np.mean(results["num_steps"])),
                "mean_step_count": float(np.mean(total)),
                "mean_accept_ema": float(np.mean(accepted / np.maximum(total, 1))),
                "mean_t_reached": float(np.mean(t_reached)),
                "ms_per_ep": timings[i] / max(num_episodes, 1) * 1000,
            }
        )

    row = _aggregate(per_repeat)
    row["t_compile"] = t_compile
    row["t_runtime"] = float(np.median(timings))
    return row


# ─── the cache-aware entry point ────────────────────────────────────────────


def _cache_path(cache_root: str, system: str, split: str, fingerprint: str) -> str:
    from steppo.training.pid_solve import dataset_cache_dir

    return os.path.join(dataset_cache_dir(split, system, root=cache_root), f"{fingerprint}.npz")


def load_pid_batch(
    env_config,
    num_envs: int,
    seed: int,
    max_steps: int,
    bins,
    split: str,
    ref_tol_factor: float = PID_EVAL_REF_TOL_FACTOR,
    # with_timing: bool = False,
    timing_n: int = 128,
    timing_repeats: int = 8,
    timing_unlimited_repeats: int | None = None,
    timing_unlimited_max_steps: int = 8192,
    # with_oracle: bool = False,
    oracle_max_iters: int = 6,
    budget_multiplier: int = PID_UNLIMITED_BUDGET_MULTIPLIER,
    cache_dir: str = DEFAULT_PID_CACHE_DIR,
    force_regenerate: bool = False,
) -> dict:
    """Load or build the PID evaluation dataset for these parameters.

    Samples tasks, solves them with PID and with a tight-tolerance reference,
    optionally times the solves and runs the oracle, and caches the result
    under `<cache_dir>/<system>/evaluation/<fingerprint>.npz`.

    Returns {"task_params", "episode_keys", "split", "pid_steps", "pid_final_y",
    "ref_final_y", "ref_ts", "ref_ys", "pid_ts", "pid_local_err", "fingerprint",
    "timing", "oracle_steps", "oracle_final_y", "pid_steps_unlimited",
    "pid_final_y_unlimited", "oracle_ts", "oracle_local_err"}. `*_unlimited`
    re-solves PID with `max_steps * budget_multiplier`; `pid_local_err[i, t]`
    (`oracle_local_err[i, t]`) is the relative L2 error at `pid_ts[i, t]`
    (`oracle_ts[i, t]`) against the interpolated reference. Datasets written
    before the Oracle trajectory was stored return None for the oracle_* pair.
    """
    from steppo.training.pid_solve import atomic_savez

    timing_key = {
        "n": int(timing_n),
        "repeats": int(timing_repeats),
        "unlimited_repeats": int(timing_unlimited_repeats or min(timing_repeats, 4)),
        "unlimited_max_steps": int(timing_unlimited_max_steps),
    }
    oracle_key = {"max_iters": int(oracle_max_iters)}

    fingerprint = pid_eval_fingerprint(
        env_config,
        num_envs,
        seed,
        max_steps,
        split=split,
        bins=bins,
        ref_tol_factor=ref_tol_factor,
        timing=timing_key,
        oracle=oracle_key,
        budget_multiplier=budget_multiplier,
    )
    path = _cache_path(cache_dir, env_config.system, split, fingerprint)

    if not force_regenerate and os.path.isfile(path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta_json"]))
        return {
            "task_params": data["task_params"],
            "episode_keys": data["episode_keys"],
            "split": data["split"],
            "pid_steps": data["pid_steps"],
            "pid_final_y": data["pid_final_y"],
            "ref_final_y": data["ref_final_y"],
            "ref_ts": data["ref_ts"],
            "ref_ys": data["ref_ys"],
            "pid_ts": data["pid_ts"],
            "pid_local_err": data["pid_local_err"],
            "fingerprint": fingerprint,
            "timing": meta.get("timing"),
            "oracle_steps": data["oracle_steps"],
            "oracle_final_y": data["oracle_final_y"],
            "oracle_ts": data["oracle_ts"] if "oracle_ts" in data.files else None,
            "oracle_local_err": data["oracle_local_err"]
            if "oracle_local_err" in data.files
            else None,
            "pid_steps_unlimited": data["pid_steps_unlimited"]
            if "pid_steps_unlimited" in data.files
            else None,
            "pid_final_y_unlimited": data["pid_final_y_unlimited"]
            if "pid_final_y_unlimited" in data.files
            else None,
            "pid_completed": data["pid_completed"],
            "ref_completed": data["ref_completed"],
            "oracle_completed": data["oracle_completed"],
            "pid_completed_unlimited": data["pid_completed_unlimited"],
        }

    tasks = sample_eval_tasks(env_config, num_envs, seed, bins=bins, split=split)
    task_params = jnp.asarray(tasks["task_params"])
    episode_keys = jnp.asarray(tasks["episode_keys"])

    sc = env_pid_controller(env_config)
    pid_out = _pid_solve_batch(
        env_config,
        sc,
        task_params,
        episode_keys,
        max_steps,
        save_steps=True,
        save_ys=True,
    )
    pid_steps = pid_out["accepted"] + pid_out["rejected"]
    pid_final_y = final_state_from_trajectory(pid_out["ts"], pid_out["ys"])

    # Only the step-size controller is tightened by ref_tol_factor — see
    # pid_solve.solve_reference_batch for why the root finder must not be.
    ref_out = _solve_reference_batch(
        env_config,
        task_params,
        episode_keys,
        max_steps * 10,
        ref_tol_factor,
        save_steps=True,
        save_ys=True,
    )
    ref_final_y = final_state_from_trajectory(ref_out["ts"], ref_out["ys"])

    pid_local_err = local_error(
        ref_out["ts"], ref_out["ys"], pid_out["ts"], pid_out["ys"], env_config.atol
    )

    pid_unlimited_out = _pid_solve_batch(
        env_config,
        sc,
        task_params,
        episode_keys,
        max_steps * budget_multiplier,
    )

    budget_row = _time_pid_solve(env_config, timing_n, max_steps, seed, timing_repeats)
    unlimited_row = _time_pid_solve(
        env_config,
        timing_n,
        timing_unlimited_max_steps,
        seed,
        timing_key["unlimited_repeats"],
    )
    timing = {"budget": budget_row, "unlimited": unlimited_row}

    oracle_steps = oracle_final_y = None
    oracle_out = solve_oracle_batch(
        env_config,
        task_params,
        episode_keys,
        max_steps,
        max_iters=oracle_max_iters,
    )
    oracle_steps, oracle_final_y = oracle_out["steps"], oracle_out["final_y"]
    oracle_completed = oracle_out["completed"]
    oracle_local_err = local_error(
        ref_out["ts"], ref_out["ys"], oracle_out["ts"], oracle_out["ys"], env_config.atol
    )

    meta_json = json.dumps(
        {
            "fingerprint": fingerprint,
            "timing": timing,
        }
    )
    save_kwargs = dict(
        task_params=tasks["task_params"],
        episode_keys=tasks["episode_keys"],
        split=tasks["split"],
        pid_steps=pid_steps,
        pid_final_y=pid_final_y,
        ref_final_y=ref_final_y,
        ref_ts=ref_out["ts"],
        ref_ys=ref_out["ys"],
        pid_ts=pid_out["ts"],
        pid_local_err=pid_local_err,
        meta_json=np.array(meta_json),
        oracle_steps=oracle_steps,
        oracle_final_y=oracle_final_y,
        oracle_ts=oracle_out["ts"],
        oracle_local_err=oracle_local_err,
        pid_steps_unlimited=pid_unlimited_out["steps"],
        pid_final_y_unlimited=pid_unlimited_out["final_y"],
        pid_completed=pid_out["completed"],
        ref_completed=ref_out["completed"],
        oracle_completed=oracle_completed,
        pid_completed_unlimited=pid_unlimited_out["completed"],
    )
    atomic_savez(path, **save_kwargs)

    return {
        "task_params": tasks["task_params"],
        "episode_keys": tasks["episode_keys"],
        "split": tasks["split"],
        "pid_steps": pid_steps,
        "pid_final_y": pid_final_y,
        "ref_final_y": ref_final_y,
        "ref_ts": ref_out["ts"],
        "ref_ys": ref_out["ys"],
        "pid_ts": pid_out["ts"],
        "pid_local_err": pid_local_err,
        "fingerprint": fingerprint,
        "timing": timing,
        "oracle_steps": oracle_steps,
        "oracle_final_y": oracle_final_y,
        "oracle_ts": oracle_out["ts"],
        "oracle_local_err": oracle_local_err,
        "pid_steps_unlimited": pid_unlimited_out["steps"],
        "pid_final_y_unlimited": pid_unlimited_out["final_y"],
        "pid_completed": pid_out["completed"],
        "ref_completed": ref_out["completed"],
        "oracle_completed": oracle_completed,
        "pid_completed_unlimited": pid_unlimited_out["completed"],
    }


PID_EVAL_NUM_ENVS = 1024
PID_EVAL_SEED = 0
PID_EVAL_ORACLE_MAX_ITERS = 6

# Episodes per test_bin for compare_experiment.py's bin comparison.
PID_BIN_EVAL_NUM_ENVS = 1024

# num_envs/seed for compare.py's PID/RL/oracle comparison table (-n/--seed defaults).
PID_COMPARE_NUM_EPISODES = 128
PID_COMPARE_SEED = 42


def _pid_eval_cache_keys(
    env_config,
    num_envs,
    seed,
    max_steps,
    split,
    bins,
    ref_tol_factor,
    timing_n,
    timing_repeats,
    timing_unlimited_repeats,
    timing_unlimited_max_steps,
    oracle_max_iters,
    budget_multiplier=PID_UNLIMITED_BUDGET_MULTIPLIER,
):
    """Return the exact cache identity components shared by load/generation."""
    timing_key = {
        "n": int(timing_n),
        "repeats": int(timing_repeats),
        "unlimited_repeats": int(timing_unlimited_repeats or min(timing_repeats, 4)),
        "unlimited_max_steps": int(timing_unlimited_max_steps),
    }
    oracle_key = {"max_iters": int(oracle_max_iters)}
    fingerprint = pid_eval_fingerprint(
        env_config,
        num_envs,
        seed,
        max_steps,
        split=split,
        bins=bins,
        ref_tol_factor=ref_tol_factor,
        timing=timing_key,
        oracle=oracle_key,
        budget_multiplier=budget_multiplier,
    )
    return fingerprint, timing_key, oracle_key


def _read_pid_eval_cache(path, fingerprint):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
    return {
        "task_params": data["task_params"],
        "episode_keys": data["episode_keys"],
        "split": data["split"],
        "pid_steps": data["pid_steps"],
        "pid_final_y": data["pid_final_y"],
        "ref_final_y": data["ref_final_y"],
        "ref_ts": data["ref_ts"],
        "ref_ys": data["ref_ys"],
        "pid_ts": data["pid_ts"],
        "pid_local_err": data["pid_local_err"],
        "fingerprint": fingerprint,
        "timing": meta.get("timing"),
        "oracle_steps": data["oracle_steps"] if "oracle_steps" in data.files else None,
        "oracle_final_y": data["oracle_final_y"] if "oracle_final_y" in data.files else None,
        "oracle_ts": data["oracle_ts"] if "oracle_ts" in data.files else None,
        "oracle_local_err": data["oracle_local_err"] if "oracle_local_err" in data.files else None,
        "pid_steps_unlimited": data["pid_steps_unlimited"]
        if "pid_steps_unlimited" in data.files
        else None,
        "pid_final_y_unlimited": data["pid_final_y_unlimited"]
        if "pid_final_y_unlimited" in data.files
        else None,
        "pid_completed": data["pid_completed"],
        "ref_completed": data["ref_completed"],
        "oracle_completed": data["oracle_completed"],
        "pid_completed_unlimited": data["pid_completed_unlimited"],
    }


def load_cached_pid_batch(
    env_config,
    *,
    max_steps: int,
    bins,
    split: str,
    num_envs: int = PID_EVAL_NUM_ENVS,
    seed: int = PID_EVAL_SEED,
    ref_tol_factor: float = PID_EVAL_REF_TOL_FACTOR,
    timing_n: int = 128,
    timing_repeats: int = 8,
    timing_unlimited_repeats: int | None = None,
    timing_unlimited_max_steps: int = 8192,
    oracle_max_iters: int = PID_EVAL_ORACLE_MAX_ITERS,
    budget_multiplier: int = PID_UNLIMITED_BUDGET_MULTIPLIER,
    cache_dir: str = DEFAULT_PID_CACHE_DIR,
) -> dict:
    """Load a pre-generated evaluation dataset; never sample or solve on a miss."""
    fingerprint, _, _ = _pid_eval_cache_keys(
        env_config,
        num_envs,
        seed,
        max_steps,
        split,
        bins,
        ref_tol_factor,
        timing_n,
        timing_repeats,
        timing_unlimited_repeats,
        timing_unlimited_max_steps,
        oracle_max_iters,
        budget_multiplier,
    )
    path = _cache_path(cache_dir, env_config.system, split, fingerprint)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing pre-generated PID evaluation cache: {path}. "
            "Run bash scripts/generate-pid-datasets-all-ode.sh "
            f"{env_config.system} --force before training or analysis."
        )
    return _read_pid_eval_cache(path, fingerprint)
