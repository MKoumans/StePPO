"""Plot PID, RL and DeepONet wall-clock time vs task parameter per system.

PID comes from wallclock_vs_mu_{mode}.npz, RL from wallclock_rl_{system}_<tag>_{mode}.npz
(the newest, or --rl-tag), DeepONet from the files in _DEEPONET_WALLCLOCK.

    python plot_wallclock_vs_param.py --system van_der_pol
    python plot_wallclock_vs_param.py --all
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from overleaf_tables import (
    SYSTEMS,
    add_rl_tag_arg,
    add_train_bins_arg,
    deeponet_dir,
    find_rl_wallclock,
    parse_train_bins,
)

# (kind, filename template, x key). Only van_der_pol has per-mode DeepONet timings.
_DEEPONET_WALLCLOCK = {
    "scalar_decay": ("npz", "wallclock_deeponet.npz", "Bs"),
    "van_der_pol": ("npz", "wallclock_deeponet_vdp_{mode}.npz", "mus"),
    "brusselator": ("npz", "wallclock_deeponet_brusselator.npz", "Bs"),
}


def load_deeponet_wallclock(system: str, mode: str = "single"):
    """Return (x, ms, mode, device); mode and device are None for sources that do not record them."""
    kind, fname_tpl, xkey = _DEEPONET_WALLCLOCK[system]
    fname = fname_tpl.format(mode=mode)
    path = os.path.join(deeponet_dir(system), "results", fname)
    if not os.path.exists(path):
        return None, None, None, None
    if kind == "npz":
        d = np.load(path)
        mode = str(d["mode"]) if "mode" in d.files else None
        device = str(d["device"]) if "device" in d.files else None
        return d[xkey], d["mean_times_ms"], mode, device
    with open(path) as f:
        lines = [line for line in f if line.strip()]
    header = lines[0].split()
    rows = np.asarray([[float(x) for x in line.split()] for line in lines[1:]])
    cols = {name: rows[:, i] for i, name in enumerate(header)}
    return cols["bin_center"], cols["deeponet_ms_mean"], None, None


def plot_one(system: str, mode: str = "single", train_bins=None, rl_tag: str = None):
    """Plot one system for `mode` ("single" or "batched")."""
    spec = SYSTEMS[system]
    param_label = spec["param_label"]
    dn_dir = deeponet_dir(system)
    results_dir = os.path.join(dn_dir, "results")
    plots_dir = os.path.join(dn_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    wc_path = os.path.join(results_dir, f"wallclock_vs_mu_{mode}.npz")
    if not os.path.exists(wc_path):
        print(f"[!] {wc_path} not found; run wallclock_vs_mu.py --mode {mode} for {system} first")
        return None
    d = np.load(wc_path)
    mus, pid_ms = d["mus"], d["pid_ms"]
    pid_mode = str(d["mode"]) if "mode" in d.files else "?"
    pid_device = str(d["device"]) if "device" in d.files else "?"
    max_steps = int(d["max_steps"]) if "max_steps" in d.files else None

    rl_mus, rl_ms, found_rl_tag, rl_mode, rl_device = find_rl_wallclock(
        results_dir, system, mode, tag=rl_tag
    )
    dn_x, dn_ms, dn_mode, dn_device = load_deeponet_wallclock(system, mode=mode)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        mus, pid_ms, "-", color="#4472C4", label=f"PID ({pid_mode}/{pid_device})", linewidth=1.8
    )
    if rl_ms is not None:
        rl_label = f"RL-{found_rl_tag} ({rl_mode}/{rl_device})" if rl_mode else f"RL-{found_rl_tag}"
        ax.plot(rl_mus, rl_ms, "-", color="#ED7D31", label=rl_label, linewidth=1.8)
    if dn_x is not None:
        dn_label = f"DeepONet ({dn_mode}/{dn_device})" if dn_mode else "DeepONet"
        ax.plot(dn_x, dn_ms, "--", color="#C00000", label=dn_label, linewidth=1.8)

    if train_bins:
        for i, (lo, hi) in enumerate(train_bins):
            ax.axvspan(
                lo, hi, color="#2aa876", alpha=0.15, label="Training bins" if i == 0 else None
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel("wallclock (ms), warm", fontsize=12)
    budget_note = f"  (max_steps={max_steps})" if max_steps is not None else ""
    ax.set_title(f"{system}: wallclock vs {param_label} — PID/RL/DeepONet{budget_note}")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_path = os.path.join(plots_dir, f"wallclock_vs_param_{mode}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=list(SYSTEMS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=["single", "batched"], default="single")
    add_train_bins_arg(parser)
    add_rl_tag_arg(parser)
    args = parser.parse_args()
    systems = list(SYSTEMS) if args.all else [args.system]
    if not systems or systems == [None]:
        parser.error("geef --system of --all")
    train_bins = parse_train_bins(args.train_bins)
    for s in systems:
        plot_one(s, mode=args.mode, train_bins=train_bins, rl_tag=args.rl_tag)


if __name__ == "__main__":
    main()
