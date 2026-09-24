"""Overlay PID, StePPO and DeepONet (complete and binned) log10 relative
error vs task parameter for one system, from the *_eval_table_relerr.txt tables
written by pid_steppo_relerr_eval.py and deeponet_relerr_eval.py (same bins).

    python plot_pid_steppo_deeponet_sweep.py --system van_der_pol
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
TRAIN_BINS = {
    "van_der_pol": [(5.0, 15.0), (35.0, 45.0), (65.0, 80.0)],
}
PARAM_LABELS = {"scalar_decay": r"$\lambda$", "van_der_pol": r"$\mu$", "brusselator": r"$B$"}
COLORS = {
    "pid": "#7F7F7F",
    "steppo": "#2CA02C",
    "deeponet_complete": "#4472C4",
    "deeponet_binned": "#C00000",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=list(PARAM_LABELS))
    parser.add_argument("--label-steppo", default="steppo")
    parser.add_argument(
        "--out-dir", default=os.path.join(REPO_ROOT, "outputs", "baselines", "20260915")
    )
    args = parser.parse_args()

    results_dir = os.path.join(HERE, "output-deeponet", args.system, "results")
    tables = {
        "pid": load_eval_table(
            os.path.join(results_dir, "pid_eval_table_relerr.txt"), missing_ok=True
        ),
        "steppo": load_eval_table(
            os.path.join(results_dir, f"{args.label_steppo}_eval_table_relerr.txt"), missing_ok=True
        ),
        "deeponet_complete": load_eval_table(
            os.path.join(results_dir, f"deeponet_{args.system}_retrain_eval_table_relerr.txt"),
            missing_ok=True,
        ),
        "deeponet_binned": load_eval_table(
            os.path.join(
                results_dir, f"deeponet_{args.system}_retrain_binned_eval_table_relerr.txt"
            ),
            missing_ok=True,
        ),
    }
    missing = [k for k, v in tables.items() if v is None]
    if missing:
        parser.error(
            f"missing eval tables for: {missing} (run pid_steppo_relerr_eval.py / deeponet_relerr_eval.py first)"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, (field, title) in zip(
        axes,
        [
            ("err_tend_log10rel_mean", "Endpoint error"),
            ("err_integ_log10rel_mean", "Time-integrated error"),
        ],
    ):
        for lo, hi in TRAIN_BINS.get(args.system, []):
            ax.axvspan(lo, hi, color="orange", alpha=0.15)
        for key, table in tables.items():
            label = {
                "pid": "PID",
                "steppo": "StePPO",
                "deeponet_complete": "DeepONet (complete)",
                "deeponet_binned": "DeepONet (binned)",
            }[key]
            valid = np.isfinite(table[field])
            ax.plot(
                table["bin_center"][valid],
                table[field][valid],
                color=COLORS[key],
                lw=1.8,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(PARAM_LABELS[args.system])
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(r"$\log_{10}$ relative $L^2$ error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05)
    )
    fig.suptitle(
        f"{args.system}: PID / StePPO(binned) / DeepONet(complete+binned) vs. {PARAM_LABELS[args.system]}"
    )
    fig.tight_layout()

    png_path = os.path.join(args.out_dir, f"{args.system}_pid_steppo_deeponet_relerr.png")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    print(f"[+] Saved: {png_path}")

    csv_path = os.path.join(args.out_dir, f"{args.system}_pid_steppo_deeponet_relerr.csv")
    bin_centers = tables["deeponet_complete"]["bin_center"]
    header = ["bin_center"]
    columns = [bin_centers]
    for key, table in tables.items():
        for field, suffix in [
            ("err_tend_log10rel_mean", "endpoint"),
            ("err_integ_log10rel_mean", "integ"),
        ]:
            header.append(f"{key}_{suffix}")
            # join on bin_center: all tables share the same log-spaced edges
            lookup = dict(zip(table["bin_center"], table[field]))
            columns.append(np.array([lookup.get(c, np.nan) for c in bin_centers]))
    np.savetxt(
        csv_path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.10g",
    )
    print(f"[+] Saved: {csv_path}")


if __name__ == "__main__":
    main()
