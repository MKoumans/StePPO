"""Per-task reward normalisation by PID step counts.

Rewards are scaled by PID_steps(μ) / C, with C the geometric mean over a μ grid,
so easy tasks (few steps, large per-step rewards) do not dominate PPO. Solves
that ran out of budget are extrapolated to t_end, and the scale is clipped to
[SCALE_MIN, SCALE_MAX].
"""

import jax
import jax.numpy as jnp
import numpy as np

from steppo.training.eval import compute_pid_baseline_steps

SCALE_MIN: float = 0.25
SCALE_MAX: float = 4.0


def build_pid_scale_grid(
    env_config,
    mu_lo: float,
    mu_hi: float,
    grid_points: int = 24,
    num_repeats: int = 8,
    max_steps: int = 1000,
    force_regenerate: bool = False,
    require_cache: bool = False,
    verbose: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Log-log grid of PID steps over [mu_lo, mu_hi]: (log_mu_grid, log_pid_steps, log_c).

    log_c is the mean of log_pid_steps. Cached under data/<system>/training/.
    """
    mu_grid = np.geomspace(float(mu_lo), float(mu_hi), grid_points)
    baseline = compute_pid_baseline_steps(
        env_config,
        mus=list(mu_grid),
        num_repeats=num_repeats,
        max_steps=max_steps,
        force_regenerate=force_regenerate,
        require_cache=require_cache,
        verbose=verbose,
    )
    # Extrapolate solves that ran out of budget to the full interval.
    t_end = float(env_config.t_end)
    pid_steps = np.array(
        [
            max(baseline[mu]["steps"], 1.0) * t_end / max(baseline[mu]["t_reached"], 0.01 * t_end)
            for mu in mu_grid
        ]
    )
    log_mu_grid = jnp.asarray(np.log(mu_grid), dtype=jnp.float32)
    log_pid_steps = jnp.asarray(np.log(pid_steps), dtype=jnp.float32)
    log_c = jnp.mean(log_pid_steps)
    return log_mu_grid, log_pid_steps, log_c


def build_pid_warp_grid(
    env_config,
    mu_lo: float,
    mu_hi: float,
    grid_points: int = 24,
    num_repeats: int = 8,
    max_steps: int = 1000,
    num_knots: int = 64,
    use_cache: bool = True,
    force_regenerate: bool = False,
    require_cache: bool = False,
    verbose: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """PID effort curve Φ(μ, t) for time-warped progress rewards.

    For each μ on a log grid, records the times at which fractions k/K of the
    PID solver steps have been spent, so every unit of solver work earns the
    same reward. Returns float32 (log_mu_grid (G,), t_knots (G, K), fracs (K,)),
    cached under data/<system>/training/.
    """
    import os

    from steppo.training.pid_solve import (
        PID_WARP_SEED,
        atomic_savez,
        cache_dir,
        env_pid_controller,
        fingerprint,
        pid_env_payload,
        solve_pid_batch,
    )

    path = None
    if require_cache and not use_cache:
        raise ValueError("require_cache=True requires use_cache=True")
    if use_cache:
        payload = {
            "kind": "progress_warp_grid",
            "env_config": pid_env_payload(env_config),
            "pid_seed": PID_WARP_SEED,
            "mu_lo": float(mu_lo),
            "mu_hi": float(mu_hi),
            "grid_points": int(grid_points),
            "num_repeats": int(num_repeats),
            "max_steps": int(max_steps),
            "num_knots": int(num_knots),
        }
        fp = fingerprint(payload)
        path = os.path.join(cache_dir("training", env_config.system), f"warp_grid_{fp}.npz")
        if not force_regenerate and os.path.isfile(path):
            if verbose > 0:
                print(f"[+] Loaded PID progress-warp grid cache: {path}")
            data = np.load(path)
            return (
                jnp.asarray(data["log_mu_grid"]),
                jnp.asarray(data["t_knots"]),
                jnp.asarray(data["fracs"]),
            )
        if require_cache:
            raise FileNotFoundError(
                f"Missing pre-generated PID training cache: {path}. Run bash scripts/generate-pid-datasets-all-ode.sh "
                f"{env_config.system} --force before training."
            )

    sc = env_pid_controller(env_config)
    t_end = float(env_config.t_end)
    fracs = (np.arange(num_knots, dtype=np.float64) + 1.0) / num_knots

    mu_grid = np.geomspace(float(mu_lo), float(mu_hi), grid_points)
    rng = jax.random.PRNGKey(PID_WARP_SEED)
    rows = []
    for i, mu in enumerate(mu_grid):
        keys = jax.random.split(rng, num_repeats)
        rng = jax.random.fold_in(rng, i)
        out = solve_pid_batch(
            env_config,
            sc,
            np.full(num_repeats, mu, dtype=np.float32),
            keys,
            max_steps,
            save_steps=True,
        )
        ts_all, n_acc_all, t_reached_all = out["ts"], out["accepted"], out["t_reached"]
        log_knots = []
        for r in range(num_repeats):
            n = int(n_acc_all[r])
            if n < 2:  # PID died immediately — fall back to a linear-in-t warp
                t_k = fracs * t_end
            else:
                tr = max(float(t_reached_all[r]), 0.01 * t_end)
                t_acc = np.asarray(ts_all[r][:n], dtype=np.float64)
                n_ext = n * t_end / tr  # extrapolated total cost (= n if complete)
                s_k = fracs * n_ext
                t_k = np.where(
                    s_k <= n,
                    np.interp(s_k, np.arange(1, n + 1, dtype=np.float64), t_acc),
                    tr + (s_k - n) * (t_end - tr) / max(n_ext - n, 1e-9),
                )
            log_knots.append(np.log(np.maximum(t_k, 1e-30)))
        rows.append(np.exp(np.mean(log_knots, axis=0)))  # geometric mean over repeats
    t_knots = np.maximum.accumulate(np.stack(rows), axis=1)  # enforce monotone rows
    log_mu_grid = np.log(mu_grid).astype(np.float32)
    t_knots = t_knots.astype(np.float32)
    fracs32 = fracs.astype(np.float32)

    if path is not None:
        atomic_savez(path, log_mu_grid=log_mu_grid, t_knots=t_knots, fracs=fracs32)

    return jnp.asarray(log_mu_grid), jnp.asarray(t_knots), jnp.asarray(fracs32)


def reward_scale_for_mu(
    mu_raw: jnp.ndarray,
    log_mu_grid: jnp.ndarray,
    log_pid_steps: jnp.ndarray,
    log_c: jnp.ndarray,
) -> jnp.ndarray:
    """PID_steps(μ) / C, interpolated in log-log space and clipped to [SCALE_MIN, SCALE_MAX]."""
    log_steps = jnp.interp(jnp.log(mu_raw), log_mu_grid, log_pid_steps)
    return jnp.clip(jnp.exp(log_steps - log_c), SCALE_MIN, SCALE_MAX)
