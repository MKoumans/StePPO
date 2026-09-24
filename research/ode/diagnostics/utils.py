"""Shared output/timing helpers for the diagnostic scripts.

wallclock_vs_mu.py, wallclock_batch_size_sweep.py, and model_sweep.py write an
NPZ result alongside a human-readable summary (a PNG plot, plus a text or
markdown report); this module holds that shared plotting/report and
wallclock-timing logic so no script re-implements matplotlib setup or the
warmup/repeat timing loop on its own.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np


def time_controller(
    env_config,
    controller,
    mus,
    max_steps,
    mode="single",
    repeats=20,
    verbose=False,
    progress_label=None,
    report_batch_time=False,
    batch_size=None,
    return_samples=False,
):
    """Measure wallclock for the supplied task values under one controller.

    Thin wrapper around utils.time_solve_fn binding it to solve_pid_batch for
    this (env_config, controller) — see that function for the mode/
    report_batch_time semantics.
    """
    from steppo.training.pid_solve import solve_pid_batch

    def solve_fn(mu_batch, keys):
        return solve_pid_batch(
            env_config, controller, mu_batch, keys, max_steps, batch_size=len(mu_batch)
        )

    return time_solve_fn(
        solve_fn,
        mus,
        mode=mode,
        repeats=repeats,
        verbose=verbose,
        progress_label=progress_label,
        report_batch_time=report_batch_time,
        batch_size=batch_size,
        return_samples=return_samples,
    )


def time_solve_fn(
    solve_fn,
    mus,
    mode="single",
    repeats=20,
    verbose=False,
    progress_label=None,
    report_batch_time=False,
    batch_size=None,
    return_samples=False,
):
    """Wall-clock time of `solve_fn(mu_batch, keys)` per task value in `mus`.

    "single" solves each task value alone; "batched" solves all at once and reports
    ms per task value. With report_batch_time, each task value is repeated
    batch_size times and the whole batch time is reported.
    """
    key0 = jax.random.PRNGKey(0)

    if mode == "single":

        def solve_one(mu):
            out = solve_fn(np.asarray([mu]), key0[None])
            jax.block_until_ready(out)
            return out

        for _ in range(3):
            solve_one(mus[0])

        means = []
        all_times = []
        for mu in mus:
            times = []
            for r in range(repeats):
                t0 = time.perf_counter()
                solve_one(mu)
                dt = (time.perf_counter() - t0) * 1000.0
                times.append(dt)
                if verbose:
                    print(f"    mu={mu:8.2f} repeat={r:2d} time={dt:8.2f} ms")
            means.append(np.mean(times))
            all_times.append(times)
        means_arr = np.asarray(means)
        return (means_arr, np.asarray(all_times)) if return_samples else means_arr

    if mode == "batched":
        if report_batch_time:
            batch_n = int(batch_size or 1)
            keys = jax.vmap(lambda i: jax.random.fold_in(key0, i))(jnp.arange(batch_n))

            def solve_one_batch(mu):
                mu_batch = jnp.full((batch_n,), mu)
                out = solve_fn(mu_batch, keys)
                jax.block_until_ready(out)
                return out

            t0 = time.perf_counter()
            for _ in range(3):
                solve_one_batch(mus[0])
            if progress_label:
                print(
                    f"    [{progress_label}] warmup done ({(time.perf_counter() - t0):.1f}s)",
                    flush=True,
                )

            means = []
            all_times = []
            for mu in mus:
                times = []
                for r in range(repeats):
                    t0 = time.perf_counter()
                    solve_one_batch(mu)
                    elapsed = time.perf_counter() - t0
                    batch_ms = elapsed * 1000.0
                    times.append(batch_ms)
                    if verbose:
                        print(
                            f"    mu={mu:8.2f} batch(n={batch_n}) repeat={r:2d} "
                            f"batch_ms={batch_ms:8.2f}"
                        )
                    if progress_label:
                        print(
                            f"    [{progress_label}] mu={mu:.4g} repeat {r + 1}/{repeats} "
                            f"({elapsed:.1f}s)",
                            flush=True,
                        )
                means.append(np.mean(times))
                all_times.append(times)
            means_arr = np.asarray(means)
            return (means_arr, np.asarray(all_times)) if return_samples else means_arr

        keys = jax.vmap(lambda i: jax.random.fold_in(key0, i))(jnp.arange(len(mus)))

        def solve_all():
            out = solve_fn(mus, keys)
            jax.block_until_ready(out)
            return out

        t0 = time.perf_counter()
        for _ in range(3):
            solve_all()
        if progress_label:
            print(
                f"    [{progress_label}] warmup done ({(time.perf_counter() - t0):.1f}s)",
                flush=True,
            )

        times = []
        for r in range(repeats):
            t0 = time.perf_counter()
            solve_all()
            elapsed = time.perf_counter() - t0
            batch_ms = elapsed * 1000.0
            dt = batch_ms if report_batch_time else batch_ms / len(mus)
            times.append(dt)
            label = "batch_ms" if report_batch_time else "ms/mu"
            print_value = dt
            if verbose:
                print(f"    batch(n={len(mus)}) repeat={r:2d} {label}={print_value:8.2f}")
            if progress_label:
                print(
                    f"    [{progress_label}] repeat {r + 1}/{repeats} ({elapsed:.1f}s)", flush=True
                )
        return np.full(len(mus), np.mean(times))

    raise ValueError(f"unknown mode {mode!r}")


def write_simple_outputs(
    path, mus, pid_ms=None, rl_ms=None, metadata=None, report_batch_time=False
):
    """Write a compact plot and text report next to an NPZ result."""
    stem, _ = os.path.splitext(path)
    metadata = metadata or {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ylabel = "wall-clock per batch (ms)" if report_batch_time else "amortized wall-clock (ms/mu)"
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if pid_ms is not None:
        ax.plot(mus, pid_ms, "o-", label="PID")
    if rl_ms is not None:
        ax.plot(mus, rl_ms, "o-", label="RL")
    ax.set_xscale("log")
    ax.set_xlabel("mu")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Wall-clock vs mu ({metadata.get('system', 'ODE')})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    png_path = f"{stem}.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    lines = [
        "Wall-clock vs mu",
        f"system: {metadata.get('system', 'unknown')}",
        f"device: {metadata.get('device', 'unknown')}",
        f"mode: {metadata.get('mode', 'unknown')}",
        f"batch_size: {metadata.get('batch_size', 'unknown')}",
        f"max_steps: {metadata.get('max_steps', 'unknown')}",
        f"repeats: {metadata.get('repeats', 'unknown')}",
        "",
    ]
    if pid_ms is not None:
        lines.append(f"mean PID: {np.mean(pid_ms):.3f} ms")
    if rl_ms is not None:
        lines.append(f"mean RL: {np.mean(rl_ms):.3f} ms")
    if pid_ms is not None and rl_ms is not None:
        lines.extend(
            [
                f"mean RL - PID: {np.mean(rl_ms - pid_ms):.3f} ms",
                f"mean speedup (PID / RL): {np.mean(pid_ms) / np.mean(rl_ms):.3f}x",
                "",
            ]
        )
    lines.append("mu\tPID_ms\tRL_ms")
    for i, mu in enumerate(mus):
        pid_value = "" if pid_ms is None else f"{pid_ms[i]:.6f}"
        rl_value = "" if rl_ms is None else f"{rl_ms[i]:.6f}"
        lines.append(f"{mu:.9g}\t{pid_value}\t{rl_value}")
    txt_path = f"{stem}.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved: {png_path}")
    print(f"[+] Saved: {txt_path}")


def plot_batch_size_auc(system_arrays, out_path, sizes):
    """Plot saved wall-clock area and relative reduction by batch size.

    `system_arrays` maps system name -> the (n_sizes, 7) array `aggregate()`
    returns; `sizes` is the tuple of batch sizes used for the x-axis ticks.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for system, arr in system_arrays.items():
        x = arr[:, 0]
        axes[0].semilogx(x, arr[:, 3], "o-", label=system)
        axes[1].semilogx(x, arr[:, 4], "o-", label=system)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Saved AUC: PID − RL")
    axes[0].set_xlabel("batch size")
    axes[0].set_ylabel("saved area (ms·μ)")
    axes[1].set_title("Relative area reduction")
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel("(AUC PID − AUC RL) / AUC PID (%)")
    for ax in axes:
        ax.set_xticks(sizes)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.savefig(out_path, dpi=180)
    return out_path
