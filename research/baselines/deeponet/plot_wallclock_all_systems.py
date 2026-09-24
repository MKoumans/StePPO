"""Wall-clock vs task parameter, one plot per system x domain x mode, with PID,
that domain's DeepONet, and every RL variant tagged "{domain}-<variant>".

Reads wallclock_vs_mu_{mode}.npz (PID), wallclock_rl_{system}_{tag}_{mode}.npz
(RL) and wallclock_deeponet_{system}_{domain}_{mode}.npz (DeepONet).

    python plot_wallclock_all_systems.py --system van_der_pol --domain binned --mode single
    python plot_wallclock_all_systems.py --all
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from overleaf_tables import SYSTEMS, add_train_bins_arg, deeponet_dir, parse_train_bins

MODES = ("single", "batched")
DOMAINS = ("complete", "binned")

_RL_COLORS = plt.cm.tab10.colors  # cycled for any number of RL variants


def _find_rl_variants(results_dir: str, system: str, domain: str, mode: str):
    """RL timing files for tags "{domain}-<variant>" and `mode`."""
    pattern = os.path.join(results_dir, f"wallclock_rl_{system}_{domain}-*_{mode}.npz")
    paths = sorted(glob.glob(pattern))
    out = []
    for path in paths:
        d = np.load(path)
        tag = str(d["tag"]) if "tag" in d.files else os.path.basename(path)
        variant = tag[len(domain) + 1 :] if tag.startswith(domain + "-") else tag
        out.append((variant, d))
    return out


def plot_one(system: str, domain: str, mode: str, train_bins=None):
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
    pid_device = str(d["device"]) if "device" in d.files else "?"

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(
        mus,
        pid_ms,
        "-",
        color="#4472C4",
        label=f"PID ({mode}/{pid_device})",
        linewidth=2.0,
        zorder=5,
    )

    dn_path = os.path.join(results_dir, f"wallclock_deeponet_{system}_{domain}_{mode}.npz")
    if os.path.exists(dn_path):
        dd = np.load(dn_path)
        dn_device = str(dd["device"]) if "device" in dd.files else "?"
        ax.plot(
            dd["mus"],
            dd["mean_times_ms"],
            "--",
            color="#C00000",
            label=f"DeepONet-{domain} ({mode}/{dn_device})",
            linewidth=2.0,
            zorder=4,
        )
    else:
        print(f"[!] {dn_path} not found; skipping DeepONet")

    rl_variants = _find_rl_variants(results_dir, system, domain, mode)
    for i, (variant, dd) in enumerate(rl_variants):
        rl_device = str(dd["device"]) if "device" in dd.files else "?"
        ax.plot(
            dd["mus"],
            dd["rl_ms"],
            color=_RL_COLORS[i % len(_RL_COLORS)],
            label=f"RL-{variant} ({mode}/{rl_device})",
            linewidth=1.5,
            alpha=0.9,
        )
    if not rl_variants:
        print(
            f"[!] No RL variants found for {system}/{domain}/{mode} "
            f"(wallclock_rl_{system}_{domain}-*_{mode}.npz)"
        )

    if train_bins and domain == "binned":
        for i, (lo, hi) in enumerate(train_bins):
            ax.axvspan(
                lo, hi, color="#2aa876", alpha=0.15, label="Training bins" if i == 0 else None
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel("wallclock (ms), warm", fontsize=12)
    ax.set_title(f"{system}/{domain}: wallclock vs {param_label} — mode={mode}")
    ax.legend(fontsize=8, ncol=2 if len(rl_variants) > 4 else 1)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_path = os.path.join(plots_dir, f"wallclock_all_{domain}_{mode}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}  ({len(rl_variants)} RL variants)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=list(SYSTEMS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--domain", choices=list(DOMAINS) + ["both"], default="both")
    parser.add_argument("--mode", choices=list(MODES) + ["both"], default="both")
    add_train_bins_arg(parser)
    args = parser.parse_args()
    systems = list(SYSTEMS) if args.all else [args.system]
    if not systems or systems == [None]:
        parser.error("geef --system of --all")
    domains = list(DOMAINS) if args.domain == "both" else [args.domain]
    modes = list(MODES) if args.mode == "both" else [args.mode]
    train_bins = parse_train_bins(args.train_bins)
    for s in systems:
        for dm in domains:
            for m in modes:
                plot_one(s, dm, m, train_bins=train_bins)


if __name__ == "__main__":
    main()
