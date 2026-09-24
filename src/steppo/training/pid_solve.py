"""Batched diffrax solves of (mu, key) tasks under a step-size controller, plus the
on-disk cache. Training data lives under data/<system>/training/ and evaluation
data under data/<system>/evaluation/<split>set/, so training never sees the
evaluation tasks.
"""

import dataclasses
import functools
import hashlib
import json
import os
import tempfile

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from steppo.configs.base_config import PRECISION_DTYPES
from steppo.envs.ode import _make_solver
from steppo.envs.ode.systems import get_rhs, get_system

_BATCH_SIZE = 1024
DATA_ROOT = "data"
_CACHE_FORMAT_VERSION = 6  # v5: float64 eval solves; v6: per-series completion masks + reference solve keeps the env root-finder tolerance

PID_BASELINE_SEED = 42
PID_WARP_SEED = 43

_PID_ENV_FIELDS = (
    "system",
    "precision",
    "t_end",
    "dt0",
    "rtol",
    "atol",
    "dt_min",
    "dt_max",
    "sample_y0",
    "y0_x",
    "y0_y",
    "cc_pulse_enabled",
    "cc_pulse_period",
    "cc_pulse_width",
    "cc_pulse_amplitude",
    "cc_pulse_dim",
    "cc_pulse_random",
    "fhn_pulse_enabled",
    "fhn_pulse_period",
    "fhn_pulse_width",
    "fhn_pulse_amplitude",
    "fhn_pulse_dim",
    "fhn_pulse_random",
    "sd_pulse_enabled",
    "sd_pulse_period",
    "sd_pulse_width",
    "sd_pulse_amplitude",
    "sd_pulse_dim",
    "sd_pulse_random",
)


def pid_env_payload(env_config) -> dict[str, object]:
    """Return only ODE-config fields that can affect a PID reference solve."""
    values = dataclasses.asdict(env_config)
    return {field: values[field] for field in _PID_ENV_FIELDS}


@functools.lru_cache(maxsize=None)
def _build_solve_batch(
    env_config,
    max_steps: int,
    save_steps: bool,
    save_ys: bool = False,
    solver_tols: tuple[float, float] | None = None,
):
    """Build the jitted batch solve, cached per (env_config, max_steps, save options).

    The controller is a call argument, so callers share one compiled program.
    `solver_tols` sets the root-finder tolerances independently of the
    controller (the reference solve tightens only step acceptance).
    `eqx.filter_jit` is needed because a LearnedController has static module
    fields; a different controller object recompiles once.
    """
    spec = get_system(env_config.system)
    rhs = get_rhs(env_config)
    solver_rtol, solver_atol = solver_tols or (env_config.rtol, env_config.atol)
    solver = _make_solver(solver_rtol, solver_atol)
    saveat = diffrax.SaveAt(steps=True) if save_steps else diffrax.SaveAt(t1=True)
    dtype = PRECISION_DTYPES[env_config.precision]

    @eqx.filter_jit
    def solve_batch(controller, mus, ks):
        def solve_one(mu, key):
            key_y0, key_pulse = jax.random.split(key)
            pulse_phase = jax.random.uniform(key_pulse, shape=(), dtype=dtype)
            task = jnp.array([mu, pulse_phase], dtype=dtype)
            y0 = spec.y0(task, key_y0, env_config).astype(dtype)
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(rhs),
                solver,
                t0=jnp.asarray(0.0, dtype=dtype),
                t1=jnp.asarray(env_config.t_end, dtype=dtype),
                dt0=jnp.asarray(env_config.dt0, dtype=dtype),
                y0=y0,
                args=task,
                stepsize_controller=controller,
                max_steps=max_steps,
                saveat=saveat,
                throw=False,
            )
            completed = sol.result == diffrax.RESULTS.successful
            if save_steps:
                ts = sol.ts
                t_reached = jnp.max(jnp.where(jnp.isfinite(ts), ts, 0.0))
                if save_ys:
                    return (
                        ts,
                        sol.ys,
                        sol.stats["num_accepted_steps"],
                        sol.stats["num_rejected_steps"],
                        t_reached,
                        completed,
                    )
                return (
                    ts,
                    sol.stats["num_accepted_steps"],
                    sol.stats["num_rejected_steps"],
                    t_reached,
                    completed,
                )
            steps = sol.stats["num_accepted_steps"] + sol.stats["num_rejected_steps"]
            return sol.ys[-1], steps, sol.ts[-1], completed

        return jax.vmap(solve_one)(mus, ks)

    return solve_batch


def env_pid_controller(env_config, tol_factor: float = 1.0, **coeffs) -> diffrax.PIDController:
    """diffrax PIDController at the env tolerances (scaled by `tol_factor`) and step bounds.

    `coeffs` are passed through (`pcoeff`, `icoeff`, `dcoeff`); the default is diffrax's I controller.
    """
    return diffrax.PIDController(
        rtol=env_config.rtol * tol_factor,
        atol=env_config.atol * tol_factor,
        dtmin=env_config.dt_min,
        dtmax=env_config.dt_max,
        force_dtmin=False,
        **coeffs,
    )


def solve_pid_batch(
    env_config,
    stepsize_controller,
    mu_values,
    keys,
    max_steps: int,
    save_steps: bool = False,
    save_ys: bool = False,
    batch_size: int | None = None,
    solver_tols: tuple[float, float] | None = None,
) -> dict:
    """Solve a batch of (mu, key) tasks under `stepsize_controller`.

    Returns {"final_y", "steps" (accepted + rejected), "t_reached", "completed"},
    or with `save_steps` {"ts", "accepted", "rejected", "t_reached",
    "completed"} plus "ys" when `save_ys` (inf-padded). Failed solves return a
    small step count, so mask aggregates on `completed`. `batch_size` chunks
    the vmap to bound memory.
    """
    if env_config.precision == "float64":
        jax.config.update("jax_enable_x64", True)
    solve_batch = _build_solve_batch(
        env_config,
        max_steps,
        save_steps,
        save_ys,
        tuple(solver_tols) if solver_tols is not None else None,
    )

    mu_values = jnp.asarray(mu_values, dtype=jnp.float32)
    n = len(mu_values)
    bs = batch_size or _BATCH_SIZE

    if save_steps and save_ys:
        all_ts, all_ys, all_acc, all_rej, all_tr, all_done = [], [], [], [], [], []
        for start in range(0, n, bs):
            ts, ys, acc, rej, tr, done = solve_batch(
                stepsize_controller, mu_values[start : start + bs], keys[start : start + bs]
            )
            all_ts.append(np.asarray(ts))
            all_ys.append(np.asarray(ys))
            all_acc.append(np.asarray(acc))
            all_rej.append(np.asarray(rej))
            all_tr.append(np.asarray(tr))
            all_done.append(np.asarray(done))
        return {
            "ts": np.concatenate(all_ts),
            "ys": np.concatenate(all_ys),
            "accepted": np.concatenate(all_acc),
            "rejected": np.concatenate(all_rej),
            "t_reached": np.concatenate(all_tr),
            "completed": np.concatenate(all_done),
        }

    if save_steps:
        all_ts, all_acc, all_rej, all_tr, all_done = [], [], [], [], []
        for start in range(0, n, bs):
            ts, acc, rej, tr, done = solve_batch(
                stepsize_controller, mu_values[start : start + bs], keys[start : start + bs]
            )
            all_ts.append(np.asarray(ts))
            all_acc.append(np.asarray(acc))
            all_rej.append(np.asarray(rej))
            all_tr.append(np.asarray(tr))
            all_done.append(np.asarray(done))
        return {
            "ts": np.concatenate(all_ts),
            "accepted": np.concatenate(all_acc),
            "rejected": np.concatenate(all_rej),
            "t_reached": np.concatenate(all_tr),
            "completed": np.concatenate(all_done),
        }

    all_y, all_steps, all_tr, all_done = [], [], [], []
    for start in range(0, n, bs):
        y, steps, tr, done = solve_batch(
            stepsize_controller, mu_values[start : start + bs], keys[start : start + bs]
        )
        all_y.append(np.asarray(y))
        all_steps.append(np.asarray(steps))
        all_tr.append(np.asarray(tr))
        all_done.append(np.asarray(done))

    return {
        "final_y": np.concatenate(all_y),
        "steps": np.concatenate(all_steps),
        "t_reached": np.concatenate(all_tr),
        "completed": np.concatenate(all_done),
    }


def solve_reference_batch(
    env_config,
    mu_values,
    keys,
    max_steps: int,
    ref_tol_factor: float,
    save_steps: bool = False,
    save_ys: bool = False,
    batch_size: int | None = None,
) -> dict:
    """Solve the reference trajectory that errors are measured against.

    Only step acceptance is tightened by `ref_tol_factor`; the root finder keeps
    the env tolerances (tightening both stalls nonsmooth systems at dt_min).
    """
    controller = env_pid_controller(env_config, tol_factor=ref_tol_factor)
    return solve_pid_batch(
        env_config,
        controller,
        mu_values,
        keys,
        max_steps,
        save_steps=save_steps,
        save_ys=save_ys,
        batch_size=batch_size,
        solver_tols=(env_config.rtol, env_config.atol),
    )


def solve_trajectory(env_config, stepsize_controller, xi: float, max_steps: int, key=None) -> dict:
    """Solve one task ξ; returns the accepted {"ts", "ys"} (no padding), "steps" and "completed".

    `key` (default PRNGKey(0)) seeds the initial state and pulse phase as in solve_pid_batch.
    """
    key = jax.random.PRNGKey(0) if key is None else key
    out = solve_pid_batch(
        env_config,
        stepsize_controller,
        np.asarray([xi]),
        key[None],
        max_steps,
        save_steps=True,
        save_ys=True,
    )
    valid = np.isfinite(out["ts"][0])
    return {
        "ts": out["ts"][0][valid],
        "ys": out["ys"][0][valid],
        "steps": int(out["accepted"][0] + out["rejected"][0]),
        "completed": bool(out["completed"][0]),
    }


def solve_reference_trajectory(
    env_config, xi: float, max_steps: int, ref_tol_factor: float, key=None
) -> dict:
    """Tight-tolerance reference for one task ξ (see solve_reference_batch), as solve_trajectory returns it."""
    key = jax.random.PRNGKey(0) if key is None else key
    out = solve_reference_batch(
        env_config,
        np.asarray([xi]),
        key[None],
        max_steps,
        ref_tol_factor,
        save_steps=True,
        save_ys=True,
    )
    valid = np.isfinite(out["ts"][0])
    return {
        "ts": out["ts"][0][valid],
        "ys": out["ys"][0][valid],
        "steps": int(out["accepted"][0] + out["rejected"][0]),
        "completed": bool(out["completed"][0]),
    }


def cache_dir(kind: str, system: str, root: str = DATA_ROOT) -> str:
    """`<root>/<system>/<kind>/` — kind is "training" or "evaluation". Two
    separate subtrees so a training cache entry and an evaluation cache entry
    are never the same file, even if their fingerprints coincide."""
    assert kind in ("training", "evaluation"), kind
    return os.path.join(root, system, kind)


DATASET_SPLITS = ("train", "val", "test")


def dataset_cache_dir(split: str, system: str, root: str = DATA_ROOT) -> str:
    """Return the evaluation cache directory for one named dataset split."""
    if split not in DATASET_SPLITS:
        raise ValueError(f"unknown dataset split {split!r}; expected one of {DATASET_SPLITS}")
    return os.path.join(root, system, "evaluation", f"{split}set")


def fingerprint(payload: dict) -> str:
    """Stable hash of a JSON-serialisable payload holding everything that affects a cached result."""
    blob = json.dumps(
        {"format_version": _CACHE_FORMAT_VERSION, **payload}, sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def atomic_savez(path: str, **arrays) -> None:
    """Write a .npz atomically: concurrent cache-load calls racing on the same
    fingerprint (e.g. parallel analysis jobs or sweep workers) never see a
    partial file — worst case is duplicate compute, never a corrupt entry."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
