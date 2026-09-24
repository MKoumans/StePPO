"""Merge DeepONet, PID and RL wall-clock timings into one wallclock_vs_mu.npz/.txt.

PID comes from wallclock_vs_mu_{mode}.npz and RL from wallclock_rl_{system}_<tag>_{mode}.npz;
DeepONet and RL timings are interpolated (log-linear) onto the PID task grid.

    python merge_deeponet_wallclock.py --system van_der_pol
    python merge_deeponet_wallclock.py --all
"""

import argparse
import os

import numpy as np
from overleaf_tables import SYSTEMS, add_rl_tag_arg, deeponet_dir, find_rl_wallclock
from plot_wallclock_vs_param import load_deeponet_wallclock


def merge_one(system: str, mode: str = "single", rl_tag: str = None):
    results_dir = os.path.join(deeponet_dir(system), "results")
    wc_path = os.path.join(results_dir, f"wallclock_vs_mu_{mode}.npz")
    if not os.path.exists(wc_path):
        print(f"[!] {wc_path} not found; run wallclock_vs_mu.py --mode {mode} for {system} first")
        return
    d = np.load(wc_path)
    mus = d["mus"]
    pid_ms = d["pid_ms"]
    param_label = (
        str(d["param_label"]) if "param_label" in d.files else SYSTEMS[system]["param_label"]
    )

    dn_x, dn_ms, dn_mode, dn_device = load_deeponet_wallclock(system, mode=mode)
    if dn_x is None:
        print(f"[!] No DeepONet wall-clock data for {system}; skipping.")
        return
    deeponet_ms = np.interp(np.log(mus), np.log(dn_x), dn_ms)

    rl_mus, rl_ms_raw, found_rl_tag, rl_mode, rl_device = find_rl_wallclock(
        results_dir, system, mode, tag=rl_tag
    )
    rl_ms = None
    if rl_ms_raw is not None:
        rl_ms = np.interp(np.log(mus), np.log(rl_mus), rl_ms_raw)
        print(f"[*] RL-variant: {found_rl_tag}")
    elif rl_tag:
        print(f"[!] No wallclock_rl_{system}_{rl_tag}_{mode}.npz found.")

    save_kwargs = dict(mus=mus, pid_ms=pid_ms, deeponet_ms=deeponet_ms, param_label=param_label)
    if dn_mode is not None:
        save_kwargs["deeponet_mode"] = dn_mode
    if dn_device is not None:
        save_kwargs["deeponet_device"] = dn_device
    if rl_ms is not None:
        save_kwargs["rl_ms"] = rl_ms
        save_kwargs["rl_tag"] = found_rl_tag
        if rl_mode is not None:
            save_kwargs["rl_mode"] = rl_mode
        if rl_device is not None:
            save_kwargs["rl_device"] = rl_device
    np.savez(wc_path, **save_kwargs)
    print(f"[+] Bijgewerkt: {wc_path}")

    txt_path = os.path.splitext(wc_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        header = f"{param_label:>12} {'pid_ms':>12}"
        if rl_ms is not None:
            header += f" {'rl_ms':>12}"
        header += f" {'deeponet_ms':>14}"
        f.write(header + "\n")
        for i in range(len(mus)):
            row = f"{mus[i]:12.4f} {pid_ms[i]:12.4f}"
            if rl_ms is not None:
                row += f" {rl_ms[i]:12.4f}"
            row += f" {deeponet_ms[i]:14.4f}"
            f.write(row + "\n")
    print(f"[+] Bijgewerkt: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=list(SYSTEMS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=["single", "batched"], default="single")
    add_rl_tag_arg(parser)
    args = parser.parse_args()
    systems = list(SYSTEMS) if args.all else [args.system]
    if not systems or systems == [None]:
        parser.error("geef --system of --all")
    for s in systems:
        merge_one(s, mode=args.mode, rl_tag=args.rl_tag)


if __name__ == "__main__":
    main()
