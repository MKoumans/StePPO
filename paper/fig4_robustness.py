"""Fig. 4: robustness on Van der Pol under full-range and binned training.

(a) Time-integrated error e_int vs μ on the test split, for PID, StePPO
    (full range) and DeepONet trained on the full range and on the binned
    support; the shaded regions are the binned training intervals. PID and
    StePPO are computed as in Fig. 3; DeepONet is evaluated on its 200-point
    training grid against the dataset reference.
(b), (c) Trajectories at μ = 7 (inside the binned support) and μ = 2 (outside),
    for PID and both DeepONets against an on-the-fly reference, shown for
    t <= 25 with every 5th point marked.

    PYTHONPATH=. python paper/fig4_robustness.py [--bins 16] [--gpus 0]
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse

import matplotlib

import steppo.envs.ode  # noqa: F401 — import order avoids a circular import

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper.utils import (
    COLORS,
    add_mode_argument,
    binned_mean,
    deeponet_grid,
    deeponet_series,
    load_config,
    load_deeponet,
    load_test_set,
    out_path,
    save_csv,
)

SYSTEM = "van_der_pol"
BINNED_TRAIN_BINS = [(5.0, 15.0), (35.0, 45.0)]
EXAMPLE_MUS = (7.0, 2.0)
T_SHOW = 25.0
MARK_EVERY = 5


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bins", type=int, default=16, help="number of log-spaced μ bins")
    add_mode_argument(parser)
    args = parser.parse_args()

    from steppo.cli import BUDGET_MULTIPLIER
    from steppo.training.error_dist import PID_EVAL_REF_TOL_FACTOR
    from steppo.training.pid_solve import (
        env_pid_controller,
        solve_reference_trajectory,
        solve_trajectory,
    )
    from steppo.utils.figures.ode import make_bin_edges

    config = load_config(SYSTEM)
    env = config.env
    deeponets = {
        "DeepONet": load_deeponet(SYSTEM, "complete", args.mode),
        "DeepONet (binned)": load_deeponet(SYSTEM, "binned", args.mode),
    }
    data = load_test_set(SYSTEM, config, args.mode)
    for name, model in deeponets.items():
        data[name] = deeponet_series(model, data["cache"], env.t_end, env.atol)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    lo, hi = min(b[0] for b in env.test_bins), max(b[1] for b in env.test_bins)
    edges = make_bin_edges(lo, hi, None, args.bins, log_bins=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    rows = {}
    for name in ("PID", "StePPO", *deeponets):
        rows[name] = {"mu": centers, "e_int": binned_mean(data[name], "err_integrated", edges)}
        axes[0].plot(centers, rows[name]["e_int"], "o-", ms=3, color=COLORS[name], label=name)
    for b_lo, b_hi in BINNED_TRAIN_BINS:
        axes[0].axvspan(b_lo, b_hi, color="0.85", zorder=0)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("μ")
    axes[0].set_ylabel("$e_{int}$ ($\\log_{10}$)")
    axes[0].set_title("(a)")
    axes[0].legend(fontsize=7)
    save_csv(out_path("fig4a_van_der_pol.csv"), rows)

    long_budget = config.rollout_steps * BUDGET_MULTIPLIER
    t = deeponet_grid(env.t_end)
    for ax, mu, panel in zip(axes[1:], EXAMPLE_MUS, ("(b)", "(c)")):
        reference = solve_reference_trajectory(env, mu, long_budget, PID_EVAL_REF_TOL_FACTOR)
        pid = solve_trajectory(env, env_pid_controller(env), mu, long_budget)
        series = {
            "Reference": reference,
            "PID": pid,
            **{name: {"ts": t, "ys": model([mu], t)[0]} for name, model in deeponets.items()},
        }
        for name, s in series.items():
            shown = s["ts"] <= T_SHOW
            style = (
                dict(lw=0.8)
                if name == "Reference"
                else dict(marker="o", ms=2.5, markevery=MARK_EVERY, lw=1)
            )
            ax.plot(s["ts"][shown], s["ys"][shown, 0], color=COLORS[name], label=name, **style)
        ax.set_xlabel("t")
        ax.set_ylabel("$y_1$")
        ax.set_title(f"{panel} μ = {mu:g}")
        save_csv(
            out_path(f"fig4{panel[1]}_van_der_pol.csv"),
            {
                name: {"t": s["ts"], "y1": s["ys"][:, 0], "y2": s["ys"][:, 1]}
                for name, s in series.items()
            },
        )
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path("fig4_robustness.pdf"))
    print(f"[+] {out_path('fig4_robustness.pdf')}")


if __name__ == "__main__":
    main()
