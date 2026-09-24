"""Loss curves and error vs task parameter for the DeepONet baselines of all three
systems, from each system's saved losshistory.npz and eval_table.txt.

    python plot_all_systems.py
    python plot_all_systems.py --train-bins van_der_pol:5,15 --train-bins van_der_pol:25,35
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from overleaf_tables import (
    add_train_bins_by_system_arg,
    load_eval_table,
    parse_train_bins_by_system,
)

HERE = os.path.dirname(__file__)
BASE = os.path.join(HERE, "output-deeponet")

SYSTEMS = {
    "scalar_decay": dict(
        losshistory="deeponet_scalar_decay_paper_losshistory.npz",
        eval_npz="deeponet_scalar_decay_paper_eval.npz",
        param_label="lambda",
    ),
    "van_der_pol": dict(
        losshistory="deeponet_van_der_pol_losshistory.npz",
        eval_table="deeponet_van_der_pol_eval_table.txt",
        param_label="mu",
    ),
    "brusselator": dict(
        losshistory="deeponet_brusselator_losshistory.npz",
        eval_table="deeponet_brusselator_eval_table.txt",
        param_label="B",
    ),
}


def plot_system(name, cfg, train_bins=None):
    results_dir = os.path.join(BASE, name, "results")
    plots_dir = os.path.join(BASE, name, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"{name} — DeepONet")

    # --- Loss curve ---
    loss_path = None
    if cfg.get("losshistory"):
        candidate = os.path.join(results_dir, cfg["losshistory"])
        if os.path.exists(candidate):
            loss_path = candidate
    if loss_path is None:
        # fall back to any *_losshistory.npz in results_dir
        import glob

        candidates = glob.glob(os.path.join(results_dir, "*losshistory*.npz"))
        loss_path = candidates[0] if candidates else None

    if loss_path:
        lh = np.load(loss_path)
        loss_train = np.asarray(lh["loss_train"])
        loss_test = np.asarray(lh["loss_test"])
        # losses may be (N,) or (N, 1)
        if loss_train.ndim > 1:
            loss_train = loss_train[:, 0]
        if loss_test.ndim > 1:
            loss_test = loss_test[:, 0]
        axs[0].plot(lh["steps"], loss_train, label="train loss")
        axs[0].plot(lh["steps"], loss_test, label="test loss")
        axs[0].set_yscale("log")
        axs[0].set_xlabel("iteratie")
        axs[0].set_ylabel("loss")
        axs[0].set_title("Loss over training")
        axs[0].legend()
    else:
        axs[0].text(0.5, 0.5, "no loss history found", ha="center", va="center")

    # --- Error vs task parameter ---
    eval_path = None
    if cfg.get("eval_table"):
        candidate = os.path.join(results_dir, cfg["eval_table"])
        if os.path.exists(candidate):
            eval_path = candidate
    elif cfg.get("eval_npz"):
        candidate = os.path.join(results_dir, cfg["eval_npz"])
        if os.path.exists(candidate):
            eval_path = candidate

    if eval_path and eval_path.endswith(".txt"):
        data = load_eval_table(eval_path)
        bin_center = data["bin_center"]
        errf = data["error_full_l2_mean"]
        errt = data["error_tend_l2_mean"]
        axs[1].plot(bin_center, errf, marker="o", markersize=3, label="error_full_l2")
        axs[1].plot(bin_center, errt, marker="o", markersize=3, label="error_tend_l2")
        if train_bins:
            for i, (lo, hi) in enumerate(train_bins):
                axs[1].axvspan(
                    lo, hi, color="green", alpha=0.15, label="train-bins" if i == 0 else None
                )
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].set_xlabel(cfg["param_label"])
        axs[1].set_ylabel("mean L2 error vs oracle")
        axs[1].set_title(f"accuracy_vs_{cfg['param_label']}")
        axs[1].legend(fontsize=8)
    elif eval_path and eval_path.endswith(".npz"):
        data = np.load(eval_path)
        lam_test = data["lam_test"]
        per_sample = data.get("per_sample_rms", data.get("per_sample_l2"))
        order = np.argsort(lam_test)
        axs[1].plot(lam_test[order], per_sample[order], ".", alpha=0.4, markersize=4)
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].set_xlabel(cfg["param_label"])
        axs[1].set_ylabel("RMS error vs oracle (per sample)")
        axs[1].set_title(f"accuracy_vs_{cfg['param_label']}")
    else:
        axs[1].text(0.5, 0.5, "no evaluation data found", ha="center", va="center")

    fig.tight_layout()
    out_path = os.path.join(plots_dir, f"deeponet_{name}_summary.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_train_bins_by_system_arg(parser)
    args = parser.parse_args()
    train_bins_by_system = parse_train_bins_by_system(args.train_bins)

    paths = []
    for name, cfg in SYSTEMS.items():
        paths.append(plot_system(name, cfg, train_bins=train_bins_by_system.get(name)))
    print("\n".join(paths))
