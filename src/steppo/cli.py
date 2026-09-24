"""`steppo` command line: solve one task with a released StePPO model and compare it with PID.

    steppo solve van_der_pol --xi 50                   # outputs/example/van_der_pol_50.{png,gif}
    steppo solve brusselator --xi 5 --plot traj.png    # traj.png and traj.gif
    steppo solve van_der_pol --xi 50 --gif y           # GIF shows only the trajectory
    steppo solve scalar_decay --xi 20 --controller steppo --gpus 0

From Python, `solve(system, xi)` returns the same trajectories. Models are
downloaded from Hugging Face (PAPER_MODELS) on first use.
"""

import argparse
import os
import sys
import time
from pathlib import Path

PAPER_MODELS = {
    "scalar_decay": "MKoumans/sd",
    "van_der_pol": "MKoumans/vdp",
    "brusselator": "MKoumans/bru",
}
# Step budget of the solves: BUDGET_MULTIPLIER × the model's rollout_steps, so PID reaches t_end.
BUDGET_MULTIPLIER = 10
CONTROLLERS = ("steppo", "pid")
COLORS = {"reference": "k", "steppo": "#2CA02C", "pid": "#4472C4"}
PLOT_DIR = "outputs/example"
# GIF panels: the state components, the accepted step sizes, the log10 error against the reference.
GIF_PANELS = ("y", "dt", "err")
JAX_CACHE_DIR = "~/.cache/steppo/jax"
# Dashed line in the log10 error panels of the plot and the GIF: a relative error of 100%.
ERROR_THRESHOLD = 0.0


def load_paper_model(system: str, revision: str = "main", refresh: bool = False):
    """The released StePPO model artifact for `system` (a LoadedModelArtifact).

    Uses the locally cached copy when there is one (no network access); the Hub
    is contacted only on first use or with `refresh`.
    """
    from huggingface_hub.errors import LocalEntryNotFoundError

    import steppo.envs.ode  # noqa: F401 — import order avoids a circular import
    from steppo.models.huggingface.hub import download_model

    if not refresh:
        try:
            return download_model(PAPER_MODELS[system], revision=revision, local_files_only=True)
        except LocalEntryNotFoundError:
            pass
    return download_model(PAPER_MODELS[system], revision=revision)


def out_of_distribution_note(xi: float, lo: float, hi: float) -> str | None:
    """A heads-up when ξ lies outside the trained range [lo, hi], else None."""
    if lo <= xi <= hi:
        return None
    return f"ξ = {xi:g} is outside the trained range [{lo:g}, {hi:g}] (out of distribution)"


def solve(system: str, xi: float, controllers=CONTROLLERS, artifact=None, on_solved=None) -> dict:
    """Solve task ξ of `system` with each controller and a tight-tolerance reference.

    Returns {"reference": ..., "steppo": ..., "pid": ...}, each {"ts", "ys",
    "steps", "completed"}; controller entries also carry "log10_e", the log10
    relative error against the reference at each accepted step.
    `artifact` reuses an already loaded model (see load_paper_model).
    `on_solved(name, result, seconds)` is called after each solve.
    """
    artifact = artifact or load_paper_model(system)
    from steppo.training.error_dist import PID_EVAL_REF_TOL_FACTOR
    from steppo.training.pid_solve import (
        env_pid_controller,
        solve_reference_trajectory,
        solve_trajectory,
    )
    from steppo.training.trajectory_compare import log10_trajectory_error

    on_solved = on_solved or (lambda name, result, seconds: None)
    env, budget = artifact.config.env, artifact.config.rollout_steps * BUDGET_MULTIPLIER
    start = time.perf_counter()
    reference = solve_reference_trajectory(env, xi, budget, PID_EVAL_REF_TOL_FACTOR)
    on_solved("reference", reference, time.perf_counter() - start)
    available = {"steppo": lambda: artifact.controller, "pid": lambda: env_pid_controller(env)}
    results = {"reference": reference}
    for name in controllers:
        start = time.perf_counter()
        result = solve_trajectory(env, available[name](), xi, budget)
        result["log10_e"] = log10_trajectory_error(reference, result["ts"], result["ys"], env.atol)
        on_solved(name, result, time.perf_counter() - start)
        results[name] = result
    return results


def plot(results: dict, title: str = "", path: str | None = None):
    """Plot every state component and the log10 error of `solve`'s results; save to `path` if given."""
    import matplotlib.pyplot as plt

    n_states = results["reference"]["ys"].shape[1]
    fig, axes = plt.subplots(n_states + 1, 1, figsize=(7, 2.2 * (n_states + 1)), sharex=True)
    for name, r in results.items():
        label = name if name == "reference" else f"{name} ({r['steps']} steps)"
        for i in range(n_states):
            axes[i].plot(
                r["ts"],
                r["ys"][:, i],
                color=COLORS[name],
                lw=0.8 if name == "reference" else 1,
                label=label,
            )
        if "log10_e" in r:
            axes[-1].plot(r["ts"], r["log10_e"], color=COLORS[name], lw=1)
    axes[-1].axhline(ERROR_THRESHOLD, color="k", lw=0.8, ls="--", alpha=0.3)
    low, high = axes[-1].get_ylim()  # keep the threshold off the top border
    axes[-1].set_ylim(low, max(high, ERROR_THRESHOLD + 0.05 * (high - low)))
    for i in range(n_states):
        axes[i].set_ylabel(f"$y_{i + 1}$")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    axes[-1].set_ylabel("$\\log_{10} e$")
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def animate(
    results: dict,
    path: str,
    title: str = "",
    panels=("y",),
    frames: int = 60,
    fps: int = 12,
    dpi: int = 72,
):
    """Save a GIF of `solve`'s results in which every controller takes one solver step per tick.

    `panels` picks rows from GIF_PANELS. Only accepted steps are stored, so a
    controller's rejections are spread evenly between them; ticks are grouped
    into at most `frames` frames.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    reference = results["reference"]
    controllers = {name: r for name, r in results.items() if name != "reference"}
    n_states = reference["ys"].shape[1]
    # One (panel, label, series(r) -> values) entry per row; ts are step ends, so dt_0 = ts[0] - 0.
    rows = []
    if "y" in panels:
        rows += [("y", f"$y_{i + 1}$", lambda r, i=i: r["ys"][:, i]) for i in range(n_states)]
    if "dt" in panels:
        rows.append(("dt", "$\\Delta t$", lambda r: np.diff(r["ts"], prepend=0.0)))
    if "err" in panels:
        rows.append(("err", "$\\log_{10} e$", lambda r: r["log10_e"]))

    fig, axes = plt.subplots(
        len(rows), 1, figsize=(7, 2.2 * len(rows) + 0.6), sharex=True, squeeze=False
    )
    axes = axes[:, 0]
    for ax, (panel, label, series) in zip(axes, rows):
        ax.set_ylabel(label)
        values = np.concatenate([series(r) for r in controllers.values()])
        values = values[np.isfinite(values)]
        if panel == "y":
            ax.plot(reference["ts"], series(reference), color="k", lw=0.8, alpha=0.3)
        elif panel == "dt":
            ax.set_yscale("log")
            ax.set_ylim(values.min() / 1.5, values.max() * 1.5)
        elif values.size:  # err: ignore the rare near-exact steps (e.g. at t = 0) when scaling
            low, high = np.percentile(values, 1), values.max()
            pad = 0.05 * max(high - low, 1.0)
            ax.set_ylim(low - pad, max(high, ERROR_THRESHOLD) + pad)
            ax.axhline(ERROR_THRESHOLD, color="k", lw=0.8, ls="--", alpha=0.3)
    axes[-1].set_xlim(0.0, reference["ts"][-1])
    axes[-1].set_xlabel("t")
    lines = {
        name: [ax.plot([], [], "o-", color=COLORS[name], lw=1, ms=2.5)[0] for ax in axes]
        for name in controllers
    }
    legend = axes[0].legend([lines[name][0] for name in controllers], list(controllers), fontsize=8)
    axes[0].set_title(title)
    fig.tight_layout()

    n_ticks = max(r["steps"] for r in controllers.values())
    stride = max(1, -(-n_ticks // frames))
    ticks = list(range(stride, n_ticks + stride, stride))
    ticks += [ticks[-1]] * fps  # hold the final frame for a second

    def draw(k):
        for (name, r), text in zip(controllers.items(), legend.get_texts()):
            steps = min(k, r["steps"])
            shown = len(r["ts"]) * steps // r["steps"]
            for line, (_, _, series) in zip(lines[name], rows):
                line.set_data(r["ts"][:shown], series(r)[:shown])
            done = " — reached t_end" if steps == r["steps"] and r["completed"] else ""
            text.set_text(f"{name}: {steps} steps{done}")
        return [line for ls in lines.values() for line in ls]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, draw, frames=np.asarray(ticks), blit=False).save(
        path, writer=PillowWriter(fps=fps), dpi=dpi
    )
    plt.close(fig)


def _gif_panels(values):
    """`--gif` values as a tuple of GIF_PANELS, accepting both `y dt` and `y,dt`."""
    panels = [p.strip() for v in values for p in v.split(",") if p.strip()]
    unknown = sorted(set(panels) - set(GIF_PANELS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown --gif panel(s) {', '.join(unknown)}; choose from {', '.join(GIF_PANELS)}"
        )
    return tuple(p for p in GIF_PANELS if p in panels)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="steppo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    solve_cmd = commands.add_parser(
        "solve",
        help="solve one task with StePPO and/or PID and save a plot and a GIF",
        description="Solve task ξ of SYSTEM with the released StePPO model and/or a PID "
        "controller, compare both with a tight-tolerance reference, and save a plot "
        "of the complete solve (PNG) and an animation of the solve progress (GIF).",
        epilog="examples:\n"
        "  steppo solve van_der_pol\n"
        "  steppo solve scalar_decay --xi 50 --gif y dt err --plot traj.png\n"
        "  steppo solve brusselator --xi 20 --controller steppo --gpus 0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    solve_cmd.add_argument("system", choices=list(PAPER_MODELS), help="ODE system to solve")
    solve_cmd.add_argument(
        "--xi", type=float, default=10.0, help="task parameter (λ, μ or B, depending on SYSTEM)"
    )
    solve_cmd.add_argument(
        "--controller",
        choices=["both", *CONTROLLERS],
        default="both",
        help="step-size controller(s) to run (default: both)",
    )
    solve_cmd.add_argument(
        "--plot",
        metavar="PATH",
        help=f"PNG path (default: {PLOT_DIR}/<system>_<xi>.png); the GIF goes next to it",
    )
    solve_cmd.add_argument(
        "--gif",
        nargs="+",
        metavar="PANEL",
        default=["y,dt,err"],
        help="GIF rows, any of: y (trajectory), dt (step sizes), err (log10 error against "
        "the reference). Default: all. E.g. `--gif y dt` or `--gif y,dt`",
    )
    solve_cmd.add_argument(
        "--refresh",
        action="store_true",
        help="download the model from Hugging Face again instead of using the cached copy",
    )
    solve_cmd.add_argument(
        "--gpus",
        metavar="IDS",
        help="GPU ids to use, e.g. 0 or 0,1, or `cpu` (default: GPUS, else cpu; a single "
        "solve is faster on the CPU)",
    )
    return parser


def _describe_device(devices) -> str:
    """One line naming the device JAX computes on and the memory it may use."""
    if not devices:
        return "JAX could not start; see the warning above"
    device = devices[0]
    if device.platform == "cpu":
        return "CPU (no GPU in use)"
    stats = device.memory_stats() or {}
    memory = (
        f", up to {stats['bytes_limit'] / 2**30:.1f} GiB, allocated as needed"
        if "bytes_limit" in stats
        else ""
    )
    others = f" (+{len(devices) - 1} more)" if len(devices) > 1 else ""
    return f"GPU {device.id}: {device.device_kind}{memory}{others}"


def _enable_compilation_cache():
    """Keep compiled XLA programs on disk, so rebuilding the model in later runs skips compiling.

    Location: JAX_COMPILATION_CACHE_DIR, else JAX_CACHE_DIR (a volume in docker-compose).
    """
    import jax

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.expanduser(JAX_CACHE_DIR)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    # Model construction compiles ~100 tiny programs; cache them all, not only slow ones.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        panels = _gif_panels(args.gif)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    # The output contains ξ and ∈, which a Windows pipe (e.g. `!steppo` in a
    # notebook) cannot encode in its default code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # Before JAX / huggingface_hub are imported: allocate GPU memory on demand
    # (pre-allocation fails when the display holds part of the GPU, e.g. WSL)
    # and silence the Hub's progress bars and notices.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # A single solve is faster on the CPU (per-step kernel launches), so the GPU is opt-in.
    if args.gpus is not None:
        os.environ["GPUS"] = args.gpus
    else:
        os.environ.setdefault("GPUS", "cpu")
    from steppo.utils.device import setup_devices
    from steppo.utils.task_params import task_bounds

    print(f"[Device] {_describe_device(setup_devices(verbose=False))}")
    _enable_compilation_cache()

    artifact = load_paper_model(args.system, refresh=args.refresh)
    env = artifact.config.env
    repo_id = PAPER_MODELS[args.system]
    print(f"[Model] StePPO model for {args.system} ready (https://huggingface.co/{repo_id})")

    controllers = CONTROLLERS if args.controller == "both" else (args.controller,)
    budget = artifact.config.rollout_steps * BUDGET_MULTIPLIER
    print(
        f"[Solve] {args.system}, ξ = {args.xi:g}: t ∈ [0, {env.t_end:g}], "
        f"rtol {env.rtol:g}, atol {env.atol:g}, at most {budget} steps"
    )
    ood_note = out_of_distribution_note(args.xi, *task_bounds(env))
    if ood_note:
        print(f"[Solve] Heads-up: {ood_note}")

    def report(name, r, seconds):
        status = "reached t_end" if r["completed"] else "did NOT reach t_end"
        errors = (
            f", mean log10 error {r['log10_e'].mean():.2f}, final {r['log10_e'][-1]:.2f}"
            if "log10_e" in r
            else ""
        )
        print(f"[Solve] {name:9s} {r['steps']:6d} steps, {status}{errors} ({seconds:.1f} s)")

    results = solve(args.system, args.xi, controllers, artifact=artifact, on_solved=report)

    title = f"{args.system}, ξ = {args.xi:g}"
    plot_path = Path(args.plot or f"{PLOT_DIR}/{args.system}_{args.xi:g}.png").resolve()
    plot(results, title, plot_path)
    print(f'[Output] Saved complete solve to "{plot_path}"')
    gif_path = plot_path.with_suffix(".gif")
    animate(results, gif_path, title, panels)
    print(f'[Output] Saved solve progress ({", ".join(panels)}) to "{gif_path}"')


if __name__ == "__main__":
    main()
