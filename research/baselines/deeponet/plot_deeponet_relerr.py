"""Create a lightweight parameter-error plot from DeepONet evaluation output.

The evaluator writes one compact ``.npz`` summary containing the task
parameter and two errors per test sample.  This script reads only that summary
and does not load the large trajectory caches or rerun model inference.

Examples::

    python plot_deeponet_relerr.py --system scalar_decay --profile retrain
    python plot_deeponet_relerr.py --system van_der_pol --profile retrain
    python plot_deeponet_relerr.py --system brusselator --profile retrain
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(__file__)
SYSTEMS = ("scalar_decay", "van_der_pol", "brusselator")
PARAM_LABELS = {
    "scalar_decay": r"$\lambda$",
    "van_der_pol": r"$\mu$",
    "brusselator": r"$B$",
}


def _binned_mean(x: np.ndarray, y: np.ndarray, bins: int):
    edges = np.geomspace(np.min(x), np.max(x), bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    index = np.digitize(x, edges) - 1
    means = np.full(bins, np.nan)
    for i in range(bins):
        values = y[index == i]
        if values.size:
            means[i] = np.nanmean(values)
    valid = np.isfinite(means)
    return centers[valid], means[valid]


def plot_summary(
    system: str, profile: str, summary_path: str, output_path: str, bins: int = 64, train_bins=None
):
    data = np.load(summary_path, allow_pickle=False)
    task_param = np.asarray(data["task_test"], dtype=float)
    err_tend = np.asarray(data["err_tend_log10rel"], dtype=float)
    err_integrated = np.asarray(data["err_integrated_log10rel"], dtype=float)

    finite_x = np.isfinite(task_param) & (task_param > 0)
    if not np.any(finite_x):
        raise ValueError(f"No positive finite task parameters found in {summary_path}")

    x = task_param[finite_x]
    test_x_min, test_x_max = float(np.min(x)), float(np.max(x))
    series = [
        (err_tend[finite_x], "Endpoint error"),
        (err_integrated[finite_x], "Time-integrated error"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for ax, (values, title) in zip(axes, series):
        if train_bins:
            for lo, hi in train_bins:
                ax.axvspan(float(lo), float(hi), color="#F4A261", alpha=0.20, zorder=0)
        valid = np.isfinite(values)
        ax.scatter(
            x[valid],
            values[valid],
            s=9,
            alpha=0.18,
            color="#4472C4",
            edgecolors="none",
            label="test samples",
        )
        if np.count_nonzero(valid) >= 2:
            bx, by = _binned_mean(x[valid], values[valid], bins)
            ax.plot(bx, by, color="#C00000", linewidth=1.8, label="bin mean")
        ax.set_xscale("log")
        # The x range follows the test samples; training-bin shading must not widen it.
        ax.set_xlim(test_x_min, test_x_max)
        ax.set_title(title)
        ax.set_xlabel(PARAM_LABELS[system])
        ax.grid(True, which="both", alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if train_bins:
            handles.append(
                Patch(facecolor="#F4A261", edgecolor="none", alpha=0.20, label="training domain")
            )
            labels.append("training domain")
        ax.legend(handles, labels, frameon=False, fontsize=9)

    axes[0].set_ylabel(r"$\log_{10}$ relative $L^2$ error (linear axis)")
    # Right axis: 10**y, the geometric-mean relative error (the metric is averaged in log space).
    secondary = axes[1].secondary_yaxis(
        "right",
        functions=(lambda value: np.power(10.0, value), lambda value: np.log10(value)),
    )
    secondary.set_yscale("log")
    secondary.set_ylabel(r"$10^y$: geometric-mean relative $L^2$ error")
    fig.suptitle(f"DeepONet {system} ({profile})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--profile", choices=("retrain", "legacy"), default="retrain")
    parser.add_argument(
        "--label", default=None, help="summary/output label; useful for complete versus binned runs"
    )
    parser.add_argument("--summary", default=None, help="override the evaluator summary .npz path")
    parser.add_argument(
        "--training-config", default=None, help="training JSON containing the actual train_bins"
    )
    parser.add_argument("--output", default=None, help="override the output .png path")
    parser.add_argument("--bins", type=int, default=64)
    args = parser.parse_args()
    result_label = args.label or args.profile

    out_dir = os.path.join(HERE, "output-deeponet", args.system, "results")
    summary = args.summary or os.path.join(
        out_dir, f"deeponet_{args.system}_{result_label}_relerr_summary.npz"
    )
    training_config = args.training_config or os.path.join(
        out_dir, f"deeponet_{args.system}_{result_label}_training_config.json"
    )
    output = args.output or os.path.join(
        HERE,
        "output-deeponet",
        args.system,
        "plots",
        f"deeponet_{args.system}_{result_label}_relerr_vs_parameter.png",
    )
    if args.bins < 1:
        parser.error("--bins must be positive")
    if not os.path.exists(summary):
        parser.error(f"summary file not found: {summary}")
    train_bins = None
    if os.path.exists(training_config):
        with open(training_config, encoding="utf-8") as handle:
            train_bins = json.load(handle).get("train_bins")
        print(f"[*] Training bins loaded from: {training_config}")
    else:
        print(f"[!] Training config not found; no training-domain shading: {training_config}")
    plot_summary(args.system, result_label, summary, output, bins=args.bins, train_bins=train_bins)


if __name__ == "__main__":
    main()
