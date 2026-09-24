"""Plot last-step and time-integrated log10 relative error vs task parameter for
PID, RL and oracle (error_vs_mu.txt / error_integrated_vs_mu.txt, test split)
together with DeepONet's *_eval_table_relerr.txt from deeponet_relerr_eval.py.

    python plot_error_vs_param_with_deeponet.py --system van_der_pol
    python plot_error_vs_param_with_deeponet.py --all
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from overleaf_tables import (
    SYSTEMS,
    add_train_bins_arg,
    deeponet_dir,
    overleaf_outputs_dir,
    parse_flat_table,
    parse_split_table,
    parse_train_bins,
)


def plot_one(system: str, train_bins=None):
    spec = SYSTEMS[system]
    param_label = spec["param_label"]
    outputs_dir = overleaf_outputs_dir(system)
    dn_dir = deeponet_dir(system)
    dn_results = os.path.join(dn_dir, "results")
    dn_plots = os.path.join(dn_dir, "plots")
    os.makedirs(dn_plots, exist_ok=True)

    tend = parse_split_table(os.path.join(outputs_dir, "error_vs_mu.txt"), split="test")
    integ = parse_split_table(os.path.join(outputs_dir, "error_integrated_vs_mu.txt"), split="test")
    dn = parse_flat_table(os.path.join(dn_results, f"deeponet_{system}_eval_table_relerr.txt"))

    def filtered(d, count_key="pid_count"):
        mask = d[count_key] > 0
        return {k: v[mask] for k, v in d.items()}

    tend = filtered(tend)
    integ = filtered(integ)
    dn_mask = dn["count"] > 0

    colors = {"pid": "#4472C4", "policy": "#ED7D31", "oracle": "#70AD47", "deeponet": "#C00000"}

    def shade_train_bins(ax):
        if not train_bins:
            return
        for i, (lo, hi) in enumerate(train_bins):
            ax.axvspan(
                lo, hi, color="#2aa876", alpha=0.15, label="Training bins" if i == 0 else None
            )

    # --- last-step error ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(tend["bin_center"], tend["pid_mean"], "-", color=colors["pid"], label="PID")
    ax.fill_between(
        tend["bin_center"],
        tend["pid_mean"] - tend["pid_std"],
        tend["pid_mean"] + tend["pid_std"],
        color=colors["pid"],
        alpha=0.15,
    )
    ax.plot(tend["bin_center"], tend["policy_mean"], "-", color=colors["policy"], label="RL")
    ax.fill_between(
        tend["bin_center"],
        tend["policy_mean"] - tend["policy_std"],
        tend["policy_mean"] + tend["policy_std"],
        color=colors["policy"],
        alpha=0.15,
    )
    if "oracle_mean" in tend:
        ax.plot(
            tend["bin_center"], tend["oracle_mean"], "-", color=colors["oracle"], label="Oracle"
        )
        ax.fill_between(
            tend["bin_center"],
            tend["oracle_mean"] - tend["oracle_std"],
            tend["oracle_mean"] + tend["oracle_std"],
            color=colors["oracle"],
            alpha=0.15,
        )
    ax.plot(
        dn["bin_center"][dn_mask],
        dn["err_tend_log10rel_mean"][dn_mask],
        "--",
        color=colors["deeponet"],
        label="DeepONet",
    )
    ax.fill_between(
        dn["bin_center"][dn_mask],
        dn["err_tend_log10rel_mean"][dn_mask] - dn["err_tend_log10rel_std"][dn_mask],
        dn["err_tend_log10rel_mean"][dn_mask] + dn["err_tend_log10rel_std"][dn_mask],
        color=colors["deeponet"],
        alpha=0.12,
    )
    shade_train_bins(ax)
    ax.set_xscale("log")
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel("log10 relative L2 error (last step)", fontsize=12)
    ax.set_title(f"{system}: error vs {param_label} (test split), with DeepONet")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out1 = os.path.join(dn_plots, "error_vs_param_with_deeponet.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}")

    # --- time-integrated error (no oracle column) ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(integ["bin_center"], integ["pid_mean"], "-", color=colors["pid"], label="PID")
    ax.fill_between(
        integ["bin_center"],
        integ["pid_mean"] - integ["pid_std"],
        integ["pid_mean"] + integ["pid_std"],
        color=colors["pid"],
        alpha=0.15,
    )
    ax.plot(integ["bin_center"], integ["policy_mean"], "-", color=colors["policy"], label="RL")
    ax.fill_between(
        integ["bin_center"],
        integ["policy_mean"] - integ["policy_std"],
        integ["policy_mean"] + integ["policy_std"],
        color=colors["policy"],
        alpha=0.15,
    )
    ax.plot(
        dn["bin_center"][dn_mask],
        dn["err_integ_log10rel_mean"][dn_mask],
        "--",
        color=colors["deeponet"],
        label="DeepONet",
    )
    ax.fill_between(
        dn["bin_center"][dn_mask],
        dn["err_integ_log10rel_mean"][dn_mask] - dn["err_integ_log10rel_std"][dn_mask],
        dn["err_integ_log10rel_mean"][dn_mask] + dn["err_integ_log10rel_std"][dn_mask],
        color=colors["deeponet"],
        alpha=0.12,
    )
    shade_train_bins(ax)
    ax.set_xscale("log")
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel("log10 relative L2 error (time-integrated)", fontsize=12)
    ax.set_title(f"{system}: time-integrated error vs {param_label} (test split), with DeepONet")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out2 = os.path.join(dn_plots, "error_integrated_vs_param_with_deeponet.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")

    return out1, out2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=list(SYSTEMS))
    parser.add_argument("--all", action="store_true")
    add_train_bins_arg(parser)
    args = parser.parse_args()
    systems = list(SYSTEMS) if args.all else [args.system]
    if not systems or systems == [None]:
        parser.error("geef --system of --all")
    train_bins = parse_train_bins(args.train_bins)
    for s in systems:
        plot_one(s, train_bins=train_bins)


if __name__ == "__main__":
    main()
