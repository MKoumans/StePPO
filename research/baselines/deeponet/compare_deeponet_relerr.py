"""Overlay two DeepONet relerr-vs-parameter runs on the same axes.

Reads the compact ``*_eval_table_relerr.txt`` bin-summary tables written by
``deeponet_relerr_eval.py`` (bin_center, err_tend_log10rel_mean/std,
err_integ_log10rel_mean/std, count) for two runs of the same system and
overlays their bin-mean curves, so the two can be compared directly instead
of only side by side as separate PNGs.

Example::

    python compare_deeponet_relerr.py --system van_der_pol \
        --label-a retrain --label-b retrain_binned
"""

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from overleaf_tables import load_eval_table

HERE = os.path.dirname(__file__)
SYSTEMS = ("scalar_decay", "van_der_pol", "brusselator")
PARAM_LABELS = {
    "scalar_decay": r"$\lambda$",
    "van_der_pol": r"$\mu$",
    "brusselator": r"$B$",
}
COLORS = {"a": "#4472C4", "b": "#C00000"}


def load_train_bins(training_config_path: str):
    if not os.path.exists(training_config_path):
        return None
    with open(training_config_path, encoding="utf-8") as handle:
        return json.load(handle).get("train_bins")


def find_training_config(out_dir: str, system: str, label: str):
    """Find the label's training config, also trying the label without a trailing _<digits> suffix."""
    candidates = [label]
    stripped = re.sub(r"_\d+$", "", label)
    if stripped != label:
        candidates.append(stripped)
    for candidate in candidates:
        path = os.path.join(out_dir, f"deeponet_{system}_{candidate}_training_config.json")
        if os.path.exists(path):
            return path
    return os.path.join(out_dir, f"deeponet_{system}_{label}_training_config.json")


def plot_comparison(
    system: str, label_a: str, label_b: str, table_a, table_b, train_bins, output_path: str
):
    series = [
        ("err_tend_log10rel_mean", "Endpoint error"),
        ("err_integ_log10rel_mean", "Time-integrated error"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for ax, (field, title) in zip(axes, series):
        if train_bins:
            for lo, hi in train_bins:
                ax.axvspan(float(lo), float(hi), color="#F4A261", alpha=0.20, zorder=0)
        for key, table, label in (("a", table_a, label_a), ("b", table_b, label_b)):
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
        ax.set_xlabel(PARAM_LABELS[system])
        ax.grid(True, which="both", alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if train_bins:
            handles.append(Patch(facecolor="#F4A261", edgecolor="none", alpha=0.20))
            labels.append(f"{label_b} training domain")
        ax.legend(handles, labels, frameon=False, fontsize=9)

    axes[0].set_ylabel(r"$\log_{10}$ relative $L^2$ error (bin mean)")
    secondary = axes[1].secondary_yaxis(
        "right",
        functions=(lambda value: np.power(10.0, value), lambda value: np.log10(value)),
    )
    secondary.set_yscale("log")
    secondary.set_ylabel(r"$10^y$: geometric-mean relative $L^2$ error")
    fig.suptitle(f"DeepONet {system}: {label_a} vs {label_b}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--label-a", default="retrain")
    parser.add_argument("--label-b", default="retrain_binned")
    parser.add_argument("--table-a", default=None)
    parser.add_argument("--table-b", default=None)
    parser.add_argument(
        "--training-config",
        default=None,
        help="training JSON containing train_bins for label-b, for shading",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    out_dir = os.path.join(HERE, "output-deeponet", args.system, "results")
    table_a_path = args.table_a or os.path.join(
        out_dir, f"deeponet_{args.system}_{args.label_a}_eval_table_relerr.txt"
    )
    table_b_path = args.table_b or os.path.join(
        out_dir, f"deeponet_{args.system}_{args.label_b}_eval_table_relerr.txt"
    )
    training_config = args.training_config or find_training_config(
        out_dir, args.system, args.label_b
    )
    output = args.output or os.path.join(
        HERE,
        "output-deeponet",
        args.system,
        "plots",
        f"deeponet_{args.system}_{args.label_a}_vs_{args.label_b}_relerr_vs_parameter.png",
    )

    for path in (table_a_path, table_b_path):
        if not os.path.exists(path):
            parser.error(f"eval table not found: {path}")

    table_a = load_eval_table(table_a_path)
    table_b = load_eval_table(table_b_path)
    train_bins = load_train_bins(training_config)
    if train_bins:
        print(f"[*] Training bins loaded from: {training_config}")
    else:
        print(f"[!] Training config not found; no training-domain shading: {training_config}")

    plot_comparison(args.system, args.label_a, args.label_b, table_a, table_b, train_bins, output)


if __name__ == "__main__":
    main()
