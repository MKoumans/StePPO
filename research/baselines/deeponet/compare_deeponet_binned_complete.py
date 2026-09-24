"""Overlay the 'binned' vs 'complete' DeepONet retrain runs (bin-mean log10
relative L2 error vs. task parameter) for one system, and write both a CSV of
the underlying bin table and a PNG of the comparison plot.

Reads the ``*_eval_table_relerr.txt`` bin-summary tables written by
``deeponet_relerr_eval.py`` (bin_center, err_tend_log10rel_mean/std,
err_integ_log10rel_mean/std, count) for the ``retrain`` (complete-domain) and
``retrain_binned`` (binned-domain) labels of a system.

Example::

    python compare_deeponet_binned_complete.py --system van_der_pol
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from overleaf_tables import load_eval_table

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..", "..")
SYSTEMS = ("scalar_decay", "van_der_pol", "brusselator")
PARAM_LABELS = {
    "scalar_decay": r"$\lambda$",
    "van_der_pol": r"$\mu$",
    "brusselator": r"$B$",
}
COLORS = {"complete": "#4472C4", "binned": "#C00000"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--label-complete", default="retrain")
    parser.add_argument("--label-binned", default="retrain_binned")
    parser.add_argument(
        "--out-dir", default=os.path.join(REPO_ROOT, "outputs", "baselines", "20260915")
    )
    args = parser.parse_args()

    results_dir = os.path.join(HERE, "output-deeponet", args.system, "results")
    table_complete_path = os.path.join(
        results_dir, f"deeponet_{args.system}_{args.label_complete}_eval_table_relerr.txt"
    )
    table_binned_path = os.path.join(
        results_dir, f"deeponet_{args.system}_{args.label_binned}_eval_table_relerr.txt"
    )
    for path in (table_complete_path, table_binned_path):
        if not os.path.exists(path):
            parser.error(f"eval table not found: {path}")

    table_complete = load_eval_table(table_complete_path)
    table_binned = load_eval_table(table_binned_path)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- CSV: bin_center and complete/binned tend/integ mean, std, count ---
    csv_path = os.path.join(args.out_dir, f"{args.system}_binned_vs_complete_relerr.csv")
    fields = [
        "err_tend_log10rel_mean",
        "err_tend_log10rel_std",
        "err_integ_log10rel_mean",
        "err_integ_log10rel_std",
        "count",
    ]
    all_centers = np.union1d(table_complete["bin_center"], table_binned["bin_center"])
    header = ["bin_center"] + [f"complete_{f}" for f in fields] + [f"binned_{f}" for f in fields]

    def lookup(table, center):
        idx = np.where(np.isclose(table["bin_center"], center))[0]
        if len(idx) == 0:
            return [np.nan] * len(fields)
        i = idx[0]
        return [table[f][i] for f in fields]

    rows = []
    for c in all_centers:
        rows.append([c] + lookup(table_complete, c) + lookup(table_binned, c))
    np.savetxt(
        csv_path, np.array(rows), delimiter=",", header=",".join(header), comments="", fmt="%.10g"
    )
    print(f"[+] Saved: {csv_path}")

    # --- PNG: endpoint and time-integrated error, complete vs binned ---
    series = [
        ("err_tend_log10rel_mean", "Endpoint error"),
        ("err_integ_log10rel_mean", "Time-integrated error"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for ax, (field, title) in zip(axes, series):
        for key, table, label in (
            ("complete", table_complete, "complete"),
            ("binned", table_binned, "binned"),
        ):
            valid = np.isfinite(table[field])
            ax.plot(
                table["bin_center"][valid],
                table[field][valid],
                color=COLORS[key],
                linewidth=1.8,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(PARAM_LABELS[args.system])
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel(r"$\log_{10}$ relative $L^2$ error (bin mean)")
    secondary = axes[1].secondary_yaxis(
        "right",
        functions=(lambda value: np.power(10.0, value), lambda value: np.log10(value)),
    )
    secondary.set_yscale("log")
    secondary.set_ylabel(r"$10^y$: geometric-mean relative $L^2$ error")
    fig.suptitle(f"DeepONet {args.system}: binned vs. complete")
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, f"{args.system}_binned_vs_complete_relerr.png")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {png_path}")


if __name__ == "__main__":
    main()
