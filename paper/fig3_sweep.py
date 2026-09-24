"""Fig. 3: solver steps and time-integrated error vs task parameter ξ, for Oracle, PID, StePPO and DeepONet.

Tasks, the reference and the Oracle come from each system's test-split
evaluation dataset (paper/download_data.py); PID and StePPO are solved on
the same episodes (see paper.utils.load_test_set). Values are means over
completed solves in log-spaced ξ bins. The time-integrated error e_int is the
time average of the log10 local error against the reference. DeepONet (trained
on the full range) is evaluated on its 200-point training grid and has no step
count, so it appears in the error panel only.

    PYTHONPATH=. python paper/fig3_sweep.py [--bins 16] [--systems ...] [--gpus 0]
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
    SYSTEMS,
    add_mode_argument,
    binned_mean,
    deeponet_series,
    load_config,
    load_deeponet,
    load_test_set,
    out_path,
    save_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bins", type=int, default=16, help="number of log-spaced ξ bins")
    parser.add_argument("--systems", nargs="+", choices=list(SYSTEMS), default=list(SYSTEMS))
    add_mode_argument(parser)
    args = parser.parse_args()

    from steppo.utils.figures.ode import make_bin_edges

    fig, axes = plt.subplots(
        2, len(args.systems), figsize=(4 * len(args.systems), 5), sharex="col", squeeze=False
    )
    for col, system in enumerate(args.systems):
        config = load_config(system)
        data = load_test_set(system, config, args.mode)
        lo, hi = min(b[0] for b in config.env.test_bins), max(b[1] for b in config.env.test_bins)
        edges = make_bin_edges(lo, hi, None, args.bins, log_bins=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        data["DeepONet"] = deeponet_series(
            load_deeponet(system, "complete", args.mode),
            data["cache"],
            config.env.t_end,
            config.env.atol,
        )
        rows = {}
        for name in ("Oracle", "PID", "StePPO", "DeepONet"):
            rows[name] = {"xi": centers}
            if "steps" in data[name]:
                rows[name]["steps"] = binned_mean(data[name], "steps", edges)
                axes[0, col].plot(
                    centers, rows[name]["steps"], "o-", ms=3, color=COLORS[name], label=name
                )
            rows[name]["e_int"] = binned_mean(data[name], "err_integrated", edges)
            axes[1, col].plot(
                centers, rows[name]["e_int"], "o-", ms=3, color=COLORS[name], label=name
            )
        axes[0, col].set_title(system)
        axes[0, col].set_ylabel("steps")
        axes[0, col].set_yscale("log")
        axes[1, col].set_ylabel("$e_{int}$ ($\\log_{10}$)")
        axes[1, col].set_xlabel(SYSTEMS[system]["label"])
        axes[1, col].set_xscale("log")
        axes[1, col].legend(fontsize=7)
        save_csv(out_path(f"fig3_{system}.csv"), rows)
    fig.tight_layout()
    fig.savefig(out_path("fig3_sweep.pdf"))
    print(f"[+] {out_path('fig3_sweep.pdf')}")


if __name__ == "__main__":
    main()
