"""Fig. 2: one trajectory per system and its local error, for PID, StePPO and DeepONet.

Each system is solved at one task value ξ (default 5). The reference is a
tight-tolerance solve computed here, so no dataset is needed. The local error
is the log10 relative L2 error against the interpolated reference at each
method's own time points: accepted solver steps for PID and StePPO, the
200-point training grid for DeepONet. PID and StePPO both get BUDGET_MULTIPLIER
× rollout_steps steps (steppo.cli) so that each trajectory reaches
t_end; the policy does not observe the budget, so this does not change its
actions. A StePPO solve longer than its training budget
(rollout_steps) is reported.

    PYTHONPATH=. python paper/fig2_trajectory.py [--xi 5] [--gpus 0]
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse

import matplotlib

import steppo.envs.ode  # noqa: F401 — import order avoids a circular import

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper.utils import (
    COLORS,
    SYSTEMS,
    add_mode_argument,
    deeponet_grid,
    load_config,
    load_deeponet,
    load_steppo,
    out_path,
    save_csv,
)


def trajectories(system: str, xi: float, mode: str) -> tuple[dict, dict]:
    """(reference, {method: {"ts", "ys", "log10_e"[, "steps"]}}) for one system at task ξ."""
    from steppo.cli import BUDGET_MULTIPLIER
    from steppo.training.error_dist import PID_EVAL_REF_TOL_FACTOR
    from steppo.training.pid_solve import (
        env_pid_controller,
        solve_reference_trajectory,
        solve_trajectory,
    )
    from steppo.training.trajectory_compare import log10_trajectory_error

    config = load_config(system)
    env, budget = config.env, config.rollout_steps
    long_budget = budget * BUDGET_MULTIPLIER
    reference = solve_reference_trajectory(env, xi, long_budget, PID_EVAL_REF_TOL_FACTOR)
    methods = {
        "PID": solve_trajectory(env, env_pid_controller(env), xi, long_budget),
        "StePPO": solve_trajectory(env, load_steppo(system, config, mode), xi, long_budget),
    }
    if methods["StePPO"]["steps"] > budget:
        print(
            f"[!] {system}: StePPO used {methods['StePPO']['steps']} steps, "
            f"more than its training budget of {budget}"
        )
    t = deeponet_grid(env.t_end)
    methods["DeepONet"] = {"ts": t, "ys": load_deeponet(system, "complete", mode)([xi], t)[0]}
    for name, m in methods.items():
        m["log10_e"] = log10_trajectory_error(reference, m["ts"], m["ys"], env.atol)
        if "completed" in m and not m["completed"]:
            print(f"[!] {system}: {name} did not reach t_end within its step budget")
    return reference, methods


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--xi", type=float, default=5.0, help="task value ξ for every system")
    parser.add_argument("--systems", nargs="+", choices=list(SYSTEMS), default=list(SYSTEMS))
    add_mode_argument(parser)
    args = parser.parse_args()

    results = {system: trajectories(system, args.xi, args.mode) for system in args.systems}
    n_states = max(reference["ys"].shape[1] for reference, _ in results.values())
    fig, axes = plt.subplots(
        n_states + 1,
        len(args.systems),
        figsize=(4 * len(args.systems), 2.2 * (n_states + 1)),
        sharex="col",
        squeeze=False,
    )
    for col, (system, (reference, methods)) in enumerate(results.items()):
        dim = reference["ys"].shape[1]
        for i in range(dim):
            ax = axes[i, col]
            ax.plot(
                reference["ts"],
                reference["ys"][:, i],
                color=COLORS["Reference"],
                lw=0.8,
                label="Reference",
            )
            for name, m in methods.items():
                label = f"{name} ({m['steps']} steps)" if "steps" in m else name
                ax.plot(m["ts"], m["ys"][:, i], color=COLORS[name], lw=1, label=label)
            ax.set_ylabel(f"$y_{i + 1}$")
        for i in range(dim, n_states):
            axes[i, col].set_visible(False)
        axes[0, col].set_title(f"{system}, {SYSTEMS[system]['label']} = {args.xi:g}")
        axes[0, col].legend(fontsize=7)
        ax_e = axes[n_states, col]
        for name, m in methods.items():
            ax_e.plot(m["ts"], m["log10_e"], color=COLORS[name], lw=1)
        ax_e.set_ylabel("$\\log_{10} e$")
        ax_e.set_xlabel("t")
        save_csv(
            out_path(f"fig2_{system}.csv"),
            {
                "Reference": {
                    "t": reference["ts"],
                    **{f"y{i + 1}": reference["ys"][:, i] for i in range(reference["ys"].shape[1])},
                },
                **{
                    name: {
                        "t": m["ts"],
                        **{f"y{i + 1}": m["ys"][:, i] for i in range(m["ys"].shape[1])},
                        "log10_e": m["log10_e"],
                    }
                    for name, m in methods.items()
                },
            },
        )
    fig.tight_layout()
    fig.savefig(out_path("fig2_trajectory.pdf"))
    print(f"[+] {out_path('fig2_trajectory.pdf')}")


if __name__ == "__main__":
    main()
