"""Sanity check: solve any registered ODE system with Kvaerno5 via diffrax, no step limit.

Generates per-(system, task-value) GIFs with two subplots:
  Top:    phase-space trajectory (first two state dims), drawn progressively
          (falls back to y-vs-t for 1-D systems like scalar_decay)
  Bottom: step size (dt) over time, coloring accepted steps

Usage:
  python sanity_check_vdp.py                              # sweep all registered systems
  python sanity_check_vdp.py --systems van_der_pol robertson
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx
from diffrax._root_finder._verychord import VeryChord
from equinox import EquinoxRuntimeError
from lineax import AutoLinearSolver
from matplotlib.animation import FuncAnimation

from steppo.envs.ode.systems import get_system

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path("outputs/gifs/sanity_check")


@dataclass
class SanitySpec:
    """Sweep + solver config for sanity-checking one ODE system.

    plot_kind: "phase" plots y[track_dims[1]] vs y[track_dims[0]] (conjugate
    variables, e.g. Van der Pol); "timeseries" plots y[d] vs t per track_dims
    with track_labels as legend (scalar/non-phase-plottable systems).
    """

    task_name: str  # label for the hidden task parameter, e.g. "mu"
    task_values: list  # values to sweep over
    y0: tuple  # fixed initial condition, len == system.y_dim
    t_end: float
    dt0: float = 1e-4
    rtol: float = 1e-4
    atol: float = 1e-7
    max_steps: int = 2**13
    plot_kind: str = "phase"
    track_dims: tuple = (0, 1)
    track_labels: tuple = None
    timeseries_yscale: str = "linear"  # "linear" or "symlog" (for widely separated magnitudes)

    # Optional periodic forcing, added to rhs(t, y, task)[forcing_dim].
    # f(t) -> scalar; sanity-check-only, does not touch the registered system.
    forcing: Optional[Callable[[float], float]] = None
    forcing_dim: int = 0

    # Set when the SANITY_SPECS key isn't itself a registered system name
    # (e.g. "chemical_cascade_forced" reuses the "chemical_cascade" rhs).
    system_override: Optional[str] = None


def gaussian_pulse_train(period: float, width: float, amplitude: float) -> Callable[[float], float]:
    """Narrow Gaussian bumps every `period` (width `width`) — smooth stand-in for a
    Dirac impulse train, so a stiff implicit solver needn't do exact-jump event handling."""

    def forcing(t):
        phase = jnp.mod(t + period / 2.0, period) - period / 2.0
        return amplitude * jnp.exp(-0.5 * (phase / width) ** 2)

    return forcing


# Defaults mirror configs/envs/ode/*/*.yaml, tuned so each system produces an
# interesting stiff/non-stiff trajectory within a reasonable step budget.
SANITY_SPECS = {
    "scalar_decay": SanitySpec(
        task_name="lambda",
        task_values=[1.0, 10.0, 50.0, 100.0],
        y0=(1.0,),
        t_end=10.0,
        dt0=0.01,
        rtol=1e-6,
        atol=1e-8,
        plot_kind="timeseries",
        track_dims=(0,),
        track_labels=("y",),
    ),
    "van_der_pol": SanitySpec(
        task_name="mu",
        task_values=[1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0],
        y0=(0.0, -2.0),
        t_end=50.0,
        dt0=1e-4,
        rtol=1e-4,
        atol=1e-7,
        plot_kind="phase",
    ),
    "fitzhugh_nagumo": SanitySpec(
        task_name="eps",
        task_values=[0.005, 0.01, 0.05, 0.2, 1.0],
        y0=(-1.0, 1.0),
        t_end=100.0,
        dt0=1e-4,
        rtol=1e-4,
        atol=1e-7,
        plot_kind="phase",
    ),
    "brusselator": SanitySpec(
        task_name="B",
        task_values=[2.0, 5.0, 10.0, 20.0, 50.0],
        y0=(1.0, 3.0),
        t_end=100.0,
        dt0=1e-4,
        rtol=1e-4,
        atol=1e-7,
        plot_kind="phase",
    ),
    "robertson": SanitySpec(
        task_name="k2",
        task_values=[1e5, 1e6, 1e7, 1e8, 1e9],
        y0=(1.0, 0.0, 0.0),
        t_end=1000.0,
        dt0=1e-6,
        rtol=1e-4,
        atol=1e-7,
        max_steps=2**15,
        plot_kind="timeseries",
        track_dims=(0, 1, 2),
        track_labels=("y1", "y2", "y3"),
    ),
    "chemical_cascade": SanitySpec(
        task_name="lambda",
        task_values=[0.01, 0.1, 1.0, 10.0],
        y0=tuple([1.0] * 5 + [0.0] * 45),
        # Transient finishes by t~10-50; t_end=1000 wastes the x-axis on a flat tail.
        t_end=60.0,
        dt0=1e-4,
        rtol=1e-4,
        atol=1e-7,
        # X5 (slow, 0.1 leak) is what propagates cascade delay across layers 1/5/10;
        # X1 reacts away too fast to spread. Layer i's X5 is at index i*5+4.
        plot_kind="timeseries",
        track_dims=(4, 24, 49),
        track_labels=("layer 1 (X5)", "layer 5 (X5)", "layer 10 (X5)"),
        timeseries_yscale="symlog",
    ),
    "fosm": SanitySpec(
        # D < k=2.0 required for finite-time convergence (Riley et al. Case 1).
        # 1.5 = paper's training value, 1.75 = paper's OOD eval value.
        # max_steps raised well above the other systems' default (2**13) to see
        # whether the solver ever converges, or genuinely never settles because
        # of sign(w) chattering at the sliding surface.
        task_name="D",
        task_values=[1.0, 1.5, 1.75, 1.9],
        y0=(0.1,),
        t_end=0.5,
        dt0=1e-5,
        rtol=1e-4,
        atol=1e-7,
        max_steps=2**18,
        plot_kind="timeseries",
        track_dims=(0,),
        track_labels=("w",),
    ),
    "chua": SanitySpec(
        # Raw sign(V1) -- expect the same chattering failure mode confirmed
        # for raw "fosm"; max_steps raised well above default to see whether
        # it ever converges or genuinely never settles.
        task_name="m",
        task_values=[0.8],
        y0=(0.15264, -0.02281, 0.38127),
        t_end=20.0,
        dt0=1e-5,
        rtol=1e-4,
        atol=1e-7,
        max_steps=2**18,
        plot_kind="timeseries",
        track_dims=(0, 1, 2),
        track_labels=("V1", "V2", "i"),
    ),
    "chua_smooth": SanitySpec(
        # Same task/IC as "chua", sign(V1) -> tanh(V1/eps) (see chua.py).
        # Default max_steps (2**13, same as every other system) -- the
        # question is whether smoothing alone gets it back under that budget.
        task_name="m",
        task_values=[0.8],
        y0=(0.15264, -0.02281, 0.38127),
        t_end=20.0,
        dt0=1e-5,
        rtol=1e-4,
        atol=1e-7,
        plot_kind="timeseries",
        track_dims=(0, 1, 2),
        track_labels=("V1", "V2", "i"),
    ),
    "fosm_smooth": SanitySpec(
        # Same task/IC as "fosm", but sign(w) -> tanh(w/eps) (see fosm.py).
        # Uses the *default* max_steps (2**13, same as every other system) —
        # the question is whether smoothing alone gets it back under that
        # budget, not whether a bigger budget eventually works.
        task_name="D",
        task_values=[1.0, 1.5, 1.75, 1.9],
        y0=(0.1,),
        t_end=0.5,
        dt0=1e-5,
        rtol=1e-4,
        atol=1e-7,
        plot_kind="timeseries",
        track_dims=(0,),
        track_labels=("w",),
    ),
    "chemical_cascade_forced": SanitySpec(
        task_name="lambda",
        task_values=[0.01, 0.1, 1.0, 10.0],
        y0=tuple([0.0] * 50),  # starts empty; the pulse train is the only input
        t_end=120.0,
        dt0=1e-4,
        rtol=1e-4,
        atol=1e-7,
        plot_kind="timeseries",
        track_dims=(4, 24, 49),
        track_labels=("layer 1 (X5)", "layer 5 (X5)", "layer 10 (X5)"),
        timeseries_yscale="symlog",
        # Impulse train into layer 1's X1: narrow enough to look impulsive, wide
        # enough for the adaptive controller to resolve without excess rejections.
        forcing=gaussian_pulse_train(period=15.0, width=0.3, amplitude=2.0),
        forcing_dim=0,
        system_override="chemical_cascade",
    ),
}


def solve(rhs, task_value: float, spec: SanitySpec):
    """Solve one registered ODE system with a high-budget diffrax solver."""
    task = jnp.array([task_value])
    rf = VeryChord(
        rtol=spec.rtol,
        atol=spec.atol,
        norm=optx.rms_norm,
        linear_solver=AutoLinearSolver(well_posed=None),
    )
    solver = diffrax.Kvaerno5(root_finder=rf)

    if spec.forcing is not None:

        def forced_rhs(t, y, args):
            return rhs(t, y, args).at[spec.forcing_dim].add(spec.forcing(t))

        term = diffrax.ODETerm(forced_rhs)
    else:
        term = diffrax.ODETerm(lambda t, y, args: rhs(t, y, args))
    stepsize_controller = diffrax.PIDController(rtol=spec.rtol, atol=spec.atol)
    saveat = diffrax.SaveAt(steps=True, t1=True)

    try:
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=spec.t_end,
            dt0=spec.dt0,
            y0=jnp.array(spec.y0),
            args=task,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=spec.max_steps,
            throw=True,
        )
    except EquinoxRuntimeError as e:
        print(f"  Error solving: {e}")
        return None

    return sol


def make_gif(
    system_name: str,
    spec: SanitySpec,
    task_value: float,
    sol,
    out_path: Path,
    n_frames: int = 80,
    fps: int = 10,
):
    """Render the sampled ODE trajectory as an animated diagnostic GIF."""
    n_accepted = int(sol.stats["num_accepted_steps"])
    n_rejected = int(sol.stats["num_rejected_steps"])

    ts = np.array(sol.ts[:n_accepted])
    ys = np.array(sol.ys[:n_accepted])
    dt = np.diff(ts)

    frame_indices = np.linspace(0, n_accepted - 1, min(n_frames, n_accepted), dtype=int)

    fig, (ax_traj, ax_dt) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.suptitle(
        f"{system_name}  {spec.task_name}={task_value:g}  "
        f"(accepted={n_accepted}, rejected={n_rejected})",
        fontsize=13,
        fontweight="bold",
    )

    # --- trajectory axis ---
    if spec.plot_kind == "phase":
        d0, d1 = spec.track_dims[0], spec.track_dims[1]
        y0, y1 = ys[:, d0], ys[:, d1]
        ax_traj.set_xlim(y0.min() - 0.2, y0.max() + 0.2)
        ax_traj.set_ylim(y1.min() - 0.2, y1.max() + 0.2)
        ax_traj.set_xlabel(f"y{d0}")
        ax_traj.set_ylabel(f"y{d1}")
        ax_traj.set_title("Phase-space trajectory")
        (line_traj,) = ax_traj.plot([], [], "b-", lw=0.6, alpha=0.8)
        (dot_traj,) = ax_traj.plot([], [], "ro", ms=4)

        def update_traj(idx):
            line_traj.set_data(y0[: idx + 1], y1[: idx + 1])
            dot_traj.set_data([y0[idx]], [y1[idx]])
            return line_traj, dot_traj

    elif spec.plot_kind == "timeseries":
        tracked = ys[:, spec.track_dims]  # (n_accepted, len(track_dims))
        labels = spec.track_labels or [f"y{d}" for d in spec.track_dims]
        ax_traj.set_xlim(ts[0], ts[-1])
        ax_traj.set_xlabel("t")
        ax_traj.set_ylabel("y")
        ax_traj.set_title("State trajectory")
        if spec.timeseries_yscale == "symlog":
            # Widely separated magnitudes (e.g. chemical_cascade layers):
            # linthresh keeps near-zero values linear, log-scales the rest so
            # faint downstream signals stay visible next to the main pulse.
            nonzero_abs = np.abs(tracked[tracked != 0])
            linthresh = max(float(nonzero_abs.min()), 1e-10) if nonzero_abs.size else 1e-6
            ax_traj.set_yscale("symlog", linthresh=linthresh)
        else:
            ax_traj.set_ylim(tracked.min() - 0.2, tracked.max() + 0.2)
        lines = [ax_traj.plot([], [], lw=1.2, label=label)[0] for label in labels]
        dots = [ax_traj.plot([], [], "o", ms=4, color=line.get_color())[0] for line in lines]
        ax_traj.legend(loc="upper right", fontsize=8)

        def update_traj(idx):
            for j, (line, dot) in enumerate(zip(lines, dots)):
                line.set_data(ts[: idx + 1], tracked[: idx + 1, j])
                dot.set_data([ts[idx]], [tracked[idx, j]])
            return (*lines, *dots)

    else:
        raise ValueError(f"Unknown plot_kind: {spec.plot_kind!r}")

    # --- step-size axis ---
    ax_dt.set_xlim(ts[0], ts[-1])
    dt_min = dt.min()
    dt_max = dt.max()
    margin = 0.1 * (np.log10(dt_max) - np.log10(dt_min) + 1e-8)
    ax_dt.set_ylim(dt_min * 10 ** (-margin), dt_max * 10**margin)
    ax_dt.set_yscale("log")
    ax_dt.set_xlabel("t")
    ax_dt.set_ylabel("Δt (step size)")
    ax_dt.set_title("Accepted step sizes")
    (line_dt,) = ax_dt.plot([], [], "b-", lw=0.5, alpha=0.7)
    scatter_dt = ax_dt.scatter([], [], s=4, c=[], cmap="coolwarm", vmin=dt_min, vmax=dt_max)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(frame):
        idx = frame_indices[frame]
        artists = update_traj(idx)
        if idx > 0:
            line_dt.set_data(ts[1 : idx + 1], dt[:idx])
            offsets = np.column_stack([ts[1 : idx + 1], dt[:idx]])
            scatter_dt.set_offsets(offsets)
            scatter_dt.set_array(dt[:idx])
        return (*artists, line_dt, scatter_dt)

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 // fps, blit=True)
    anim.save(str(out_path), writer="pillow", fps=fps, dpi=80)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    """Run sanity solves for selected systems and task values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(SANITY_SPECS),
        choices=list(SANITY_SPECS),
        help="ODE systems to sanity-check (default: all registered systems)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument(
        "--fps", type=int, default=10, help="Playback speed of the saved gif (lower = slower)"
    )
    args = parser.parse_args()

    for system_name in args.systems:
        spec = SANITY_SPECS[system_name]
        system = get_system(spec.system_override or system_name)
        out_dir = args.out_dir / system_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {system_name} (sweeping {spec.task_name}) ===")
        print(f"{spec.task_name:>10s}  {'accepted':>10s}  {'rejected':>10s}  {'total':>10s}")
        print("-" * 50)

        for value in spec.task_values:
            sol = solve(system.rhs, value, spec)
            if sol is None:
                print(f"{value:10.4g}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>10s}")
                continue

            acc = int(sol.stats["num_accepted_steps"])
            rej = int(sol.stats["num_rejected_steps"])
            print(f"{value:10.4g}  {acc:10d}  {rej:10d}  {acc + rej:10d}")

            gif_path = out_dir / f"{system_name}_{spec.task_name}_{value:g}.gif"
            print(f"  Generating GIF for {spec.task_name}={value:g}...")
            make_gif(system_name, spec, value, sol, gif_path, n_frames=args.n_frames, fps=args.fps)

    print(f"\nAll GIFs saved under {args.out_dir}")


if __name__ == "__main__":
    main()
