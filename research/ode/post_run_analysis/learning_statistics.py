"""Aggregate ODE learning-statistics artifacts."""

import argparse
import glob
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

from steppo.utils.plotting import plot_l2_error_history


def _parse_table(lines: list[str]) -> tuple[list[str], list[list[float]]]:
    header = None
    rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if header is None:
            header = parts
            continue
        rows.append([float(v) for v in parts])
    return header or [], rows


def _read_blocks(path: str) -> dict[str, tuple[list[str], list[list[float]]]]:
    """Read named tables from a text file."""
    with open(path) as f:
        lines = f.read().splitlines()

    blocks: dict[str, list[str]] = {}
    current = None
    body: list[str] = []

    def flush():
        if current is not None:
            blocks[current] = body[:]

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and ":" in stripped:
            flush()
            current = stripped.lstrip("#").split(":", 1)[1].strip()
            body = []
            continue
        if stripped.startswith("#") or not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) == 1 and not any(c.isdigit() for c in tokens[0]):
            flush()
            current = tokens[0]
            body = []
            continue
        body.append(line)
    flush()

    return {name: _parse_table(body) for name, body in blocks.items()}


def _load_mu_returns(run_dir: str):
    path = os.path.join(run_dir, "mu_returns.txt")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        header, rows = _parse_table(f.readlines())
    if not rows:
        return None
    mus = header[1:]
    data = np.array(rows)  # (n_iters, 1 + n_mus)
    return {"mus": mus, "iterations": data[:, 0], "values": data[:, 1:]}


def aggregate_mu_returns(run_dirs: list[str], out_dir: str):
    """Aggregate per-task return histories across repeated runs."""
    per_seed = [d for r in run_dirs if (d := _load_mu_returns(r)) is not None]
    if not per_seed:
        print("  [!] No mu_returns.txt found in any run, skipping")
        return
    if len(per_seed) < len(run_dirs):
        print(f"  [!] mu_returns.txt found in {len(per_seed)}/{len(run_dirs)} runs")

    mus = per_seed[0]["mus"]
    if any(d["mus"] != mus for d in per_seed):
        print("  [!] mu bins differ across seeds; using the intersection order from the first run")

    min_len = min(len(d["iterations"]) for d in per_seed)
    iterations = per_seed[0]["iterations"][:min_len]
    stacked = np.stack([d["values"][:min_len] for d in per_seed])  # (n_seeds, n_iters, n_mus)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)

    _save_mu_returns_txt(
        iterations, mus, mean, std, len(per_seed), os.path.join(out_dir, "mu_returns_avg.txt")
    )
    _plot_mu_returns(
        iterations, mus, mean, std, len(per_seed), os.path.join(out_dir, "mu_returns_avg.png")
    )


def _save_mu_returns_txt(iterations, mus, mean, std, n_seeds, save_path):
    header = ["iteration"] + [f"{m}_mean" for m in mus] + [f"{m}_std" for m in mus]
    with open(save_path, "w") as f:
        f.write(f"# Per-mu return over training, averaged across {n_seeds} seeds\n")
        f.write("  ".join(f"{h:>14s}" for h in header) + "\n")
        for i, it in enumerate(iterations):
            row = [it] + list(mean[i]) + list(std[i])
            f.write(f"{row[0]:>14.0f}" + "".join(f"  {v:>14.4f}" for v in row[1:]) + "\n")
    print(f"  saved -> {save_path}")


def _plot_mu_returns(iterations, mus, mean, std, n_seeds, save_path):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(mus), 2)))[: len(mus)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (mu, color) in enumerate(zip(mus, colors)):
        ax.plot(
            iterations,
            mean[:, i],
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=mu.replace("mu=", "μ="),
        )
        ax.fill_between(
            iterations,
            mean[:, i] - std[:, i],
            mean[:, i] + std[:, i],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean episode return")
    ax.set_title(f"Per-μ return over training (mean ± std, n={n_seeds} seeds)")
    ax.legend(fontsize=8, ncol=2, title="μ bin")
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {save_path}")


def _load_efficiency(run_dir: str):
    path = os.path.join(run_dir, "efficiency.txt")
    if not os.path.isfile(path):
        return None
    blocks = _read_blocks(path)
    out = {}
    for subset, (header, rows) in blocks.items():
        if not rows:
            continue
        out[subset] = {"header": header, "data": np.array(rows)}
    return out or None


def aggregate_efficiency(run_dirs: list[str], out_dir: str):
    """Aggregate efficiency histories across repeated runs."""
    per_seed = [d for r in run_dirs if (d := _load_efficiency(r)) is not None]
    if not per_seed:
        print("  [!] No efficiency.txt found in any run, skipping")
        return
    if len(per_seed) < len(run_dirs):
        print(f"  [!] efficiency.txt found in {len(per_seed)}/{len(run_dirs)} runs")

    subsets: list[str] = []
    for d in per_seed:
        for s in d:
            if s not in subsets:
                subsets.append(s)

    agg = {}
    for subset in subsets:
        entries = [d[subset] for d in per_seed if subset in d]
        if not entries:
            continue
        min_len = min(len(e["data"]) for e in entries)
        header = entries[0]["header"]
        stacked = np.stack([e["data"][:min_len] for e in entries])  # (n_seeds, n_iters, n_cols)
        agg[subset] = {
            "header": header,
            "iterations": stacked[0, :, 0],
            "mean": stacked[:, :, 1:].mean(axis=0),
            "std": stacked[:, :, 1:].std(axis=0),
            "n_seeds": len(entries),
        }

    _save_efficiency_txt(agg, os.path.join(out_dir, "efficiency_avg.txt"))
    _plot_efficiency(agg, os.path.join(out_dir, "efficiency_avg.png"))


def _save_efficiency_txt(agg: dict, save_path: str):
    with open(save_path, "w") as f:
        for subset, d in agg.items():
            cols = d["header"][1:]
            header = ["iteration"] + [f"{c}_mean" for c in cols] + [f"{c}_std" for c in cols]
            f.write(f"# subset: {subset} (n={d['n_seeds']} seeds)\n")
            f.write("  ".join(f"{h:>18s}" for h in header) + "\n")
            for i, it in enumerate(d["iterations"]):
                row = [it] + list(d["mean"][i]) + list(d["std"][i])
                f.write(f"{row[0]:>18.0f}" + "".join(f"  {v:>18.4f}" for v in row[1:]) + "\n")
            f.write("\n")
    print(f"  saved -> {save_path}")


def _plot_efficiency(agg: dict, save_path: str):
    colors = {"train": "#2a78d6", "val": "#f4873c", "test": "#548235", "noise": "#9e4d9e"}
    labels = {
        "train": "Training distribution",
        "val": "Validation distribution",
        "test": "Test distribution",
        "noise": "Test (noise)",
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))
    ax1.set_title("Mean of per-task ratios")
    ax1.set_ylabel("PID-relative step improvement  (pid-policy)/pid")
    ax2.set_title("Aggregate step counts")
    ax2.set_ylabel("Mean steps per episode")
    ax3.set_title("Median of per-task ratios")
    ax3.set_ylabel("PID-relative step improvement  (pid-policy)/pid")
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Iteration")
        ax.grid(True, linewidth=0.4)

    n_seeds = None
    for subset, d in agg.items():
        cols = d["header"][1:]
        idx = {c: i for i, c in enumerate(cols)}
        color = colors.get(subset, "#898781")
        label = labels.get(subset, subset)
        it = d["iterations"]
        n_seeds = d["n_seeds"]

        mean_col = idx["mean"]
        m, s = d["mean"][:, mean_col], d["std"][:, mean_col]
        ax1.plot(it, m, color=color, linewidth=1.8, marker="o", markersize=7, label=label)
        ax1.fill_between(it, m - s, m + s, color=color, alpha=0.15, linewidth=0)

        pid_col, pol_col = idx["mean_pid_steps"], idx["mean_policy_steps"]
        pid_m, pid_s = d["mean"][:, pid_col], d["std"][:, pid_col]
        pol_m, pol_s = d["mean"][:, pol_col], d["std"][:, pol_col]
        ax2.plot(
            it,
            pid_m,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=color,
        )
        ax2.fill_between(it, pid_m - pid_s, pid_m + pid_s, color=color, alpha=0.15, linewidth=0)
        ax2.plot(
            it,
            pol_m,
            color=color,
            linewidth=1.8,
            marker="s",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
        )
        ax2.fill_between(it, pol_m - pol_s, pol_m + pol_s, color=color, alpha=0.15, linewidth=0)

        med_col = idx["median"]
        ax3.plot(
            it,
            d["mean"][:, med_col],
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=7,
            label=label,
        )

    ax1.axhline(0.0, color="#898781", linewidth=0.8, linestyle="--")
    ax3.axhline(0.0, color="#898781", linewidth=0.8, linestyle="--")
    ax1.legend(fontsize=8)
    ax3.legend(fontsize=8)

    # Split the legend by subset and controller.
    subset_handles = [
        Line2D([0], [0], color=colors.get(s, "#898781"), linewidth=1.8, label=labels.get(s, s))
        for s in agg
    ]
    shape_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=1.8,
            marker="o",
            markersize=8,
            markerfacecolor="#555555",
            markeredgecolor="#555555",
            label="PID (classical controller)",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=1.8,
            marker="s",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor="#555555",
            markeredgewidth=1.5,
            label="Policy (learned agent)",
        ),
    ]
    legend1 = ax2.legend(handles=subset_handles, fontsize=7, loc="upper left", title="Distribution")
    ax2.add_artist(legend1)
    ax2.legend(handles=shape_handles, fontsize=7, loc="upper right", title="Controller")
    fig.suptitle(
        f"PID-relative step efficiency over training (mean ± std across seeds, n={n_seeds})"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {save_path}")


def _load_steps_vs_mu(run_dir: str, filename: str = "steps_vs_mu.txt"):
    path = os.path.join(run_dir, filename)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        lines = f.readlines()

    blocks = _read_blocks(path)
    split_data = {
        split: {"header": header, "data": np.array(rows)}
        for split, (header, rows) in blocks.items()
        if rows and split in {"train", "val", "test"}
    }
    if split_data:
        return {"splits": split_data}

    header, rows = _parse_table(lines)
    if not rows:
        return None
    # Support legacy test-only tables.
    return {"splits": {"test": {"header": header, "data": np.array(rows)}}}


def aggregate_steps_vs_mu(
    run_dirs: list[str],
    out_dir: str,
    filename: str = "steps_vs_mu.txt",
    out_prefix: str = "steps_vs_mu",
    y_label: str = "Solver steps",
):
    """Aggregate solver-step curves by task parameter across repeated runs."""
    per_seed = [d for r in run_dirs if (d := _load_steps_vs_mu(r, filename)) is not None]
    if not per_seed:
        print(f"  [!] No {filename} found in any run, skipping")
        return
    if len(per_seed) < len(run_dirs):
        print(f"  [!] {filename} found in {len(per_seed)}/{len(run_dirs)} runs")

    split_names = []
    for data in per_seed:
        for split in data["splits"]:
            if split not in split_names:
                split_names.append(split)
    split_names.sort(key=lambda split: ({"train": 0, "val": 1, "test": 2}.get(split, 3), split))

    aggregate = {}
    for split in split_names:
        entries = [data["splits"][split] for data in per_seed if split in data["splits"]]
        if not entries:
            continue
        min_len = min(len(entry["data"]) for entry in entries)
        header = entries[0]["header"]
        stacked = np.stack([entry["data"][:min_len] for entry in entries])
        result = {
            "bin_center": stacked[0, :, 0],
        }
        for controller in ("pid", "policy", "oracle"):
            column = f"{controller}_mean"
            legacy_column = "rl_mean" if controller == "policy" else None
            if column in header:
                index = header.index(column)
            elif legacy_column is not None and legacy_column in header:
                index = header.index(legacy_column)
            else:
                continue
            # nan-tolerant: a bin is nan in a seed's table when nothing in
            # it solved (see figures/ode.py::bin_stats), which is missing
            # data for that seed, not a zero to average in or a value that
            # should erase the bin for every other seed.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-nan bin
                result[column] = np.nanmean(stacked[:, :, index], axis=0)
                result[f"{controller}_std"] = np.nanstd(stacked[:, :, index], axis=0)
        aggregate[split] = result

    if not aggregate:
        print("  [!] No usable split tables found, skipping")
        return
    os.makedirs(out_dir, exist_ok=True)
    _save_steps_vs_mu_txt(aggregate, len(per_seed), os.path.join(out_dir, f"{out_prefix}_avg.txt"))
    _plot_steps_vs_mu(
        aggregate, len(per_seed), os.path.join(out_dir, f"{out_prefix}_avg.png"), y_label=y_label
    )


def _save_steps_vs_mu_txt(aggregate: dict, n_seeds: int, save_path: str):
    with open(save_path, "w") as f:
        f.write(
            f"# Solver steps vs mu, averaged across {n_seeds} seeds "
            f"(std = seed-to-seed variability of the per-bin mean)\n"
        )
        for split, result in aggregate.items():
            columns = list(result)
            f.write(f"# split: {split}\n")
            f.write("  ".join(f"{column:>12s}" for column in columns) + "\n")
            for row in zip(*(result[column] for column in columns)):
                f.write("  ".join(f"{value:>12.4f}" for value in row) + "\n")
            f.write("\n")
    print(f"  saved -> {save_path}")


def _plot_steps_vs_mu(aggregate: dict, n_seeds: int, save_path: str, y_label: str = "Solver steps"):
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"train": "#4472C4", "val": "#ED7D31", "test": "#548235"}
    for split, result in aggregate.items():
        if "policy_mean" not in result:
            continue
        color = colors.get(split, "#898781")
        x = result["bin_center"]
        ax.plot(x, result["policy_mean"], color=color, linewidth=1.8, label=f"RL ({split})")
        ax.fill_between(
            x,
            result["policy_mean"] - result["policy_std"],
            result["policy_mean"] + result["policy_std"],
            color=color,
            alpha=0.16,
        )

    baseline_split = "test" if "test" in aggregate else next(iter(aggregate))
    baseline = aggregate[baseline_split]
    x = baseline["bin_center"]
    if "pid_mean" in baseline:
        ax.plot(
            x, baseline["pid_mean"], color="#222222", linewidth=1.8, label=f"PID ({baseline_split})"
        )
        ax.fill_between(
            x,
            baseline["pid_mean"] - baseline["pid_std"],
            baseline["pid_mean"] + baseline["pid_std"],
            color="#222222",
            alpha=0.10,
        )
    if "oracle_mean" in baseline:
        ax.plot(
            x,
            baseline["oracle_mean"],
            color="#70AD47",
            linewidth=1.8,
            linestyle="--",
            label=f"Oracle ({baseline_split})",
        )
        ax.fill_between(
            x,
            baseline["oracle_mean"] - baseline["oracle_std"],
            baseline["oracle_mean"] + baseline["oracle_std"],
            color="#70AD47",
            alpha=0.12,
        )

    ax.set_xscale("log")
    ax.set_xlabel("μ")
    ax.set_ylabel(y_label)
    ax.set_title(
        f"{y_label} vs μ (RL train/val/test, PID, oracle; mean ± seed-std, n={n_seeds} seeds)"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {save_path}")


def _load_z_sensitivity(run_dir: str):
    candidates = (
        os.path.join(run_dir, "z_sensitivity", "z_sensitivity.txt"),
        os.path.join(run_dir, "z_sensitivity.txt"),
    )
    path = next((candidate for candidate in candidates if os.path.isfile(candidate)), None)
    if path is None:
        return None
    blocks = _read_blocks(path)
    out = {}
    for sweep, (header, rows) in blocks.items():
        if not rows:
            continue
        out[sweep] = {"header": header, "data": np.array(rows)}
    return out or None


def aggregate_z_sensitivity(run_dirs: list[str], out_dir: str):
    """Aggregate latent-sensitivity results across repeated runs."""
    per_seed = [d for r in run_dirs if (d := _load_z_sensitivity(r)) is not None]
    if not per_seed:
        print("  [!] No z_sensitivity.txt found in any run, skipping")
        return
    if len(per_seed) < len(run_dirs):
        print(f"  [!] z_sensitivity.txt found in {len(per_seed)}/{len(run_dirs)} runs")

    sweeps: list[str] = []
    for d in per_seed:
        for s in d:
            if s not in sweeps:
                sweeps.append(s)

    agg = {}
    for sweep in sweeps:
        entries = [d[sweep] for d in per_seed if sweep in d]
        if not entries:
            continue
        min_len = min(len(e["data"]) for e in entries)
        header = entries[0]["header"]
        stacked = np.stack([e["data"][:min_len] for e in entries])
        agg[sweep] = {
            "header": header,
            "alpha": stacked[0, :, 0],
            "mean": stacked[:, :, 1:].mean(axis=0),
            "std": stacked[:, :, 1:].std(axis=0),
            "n_seeds": len(entries),
        }

    _save_z_sensitivity_txt(agg, os.path.join(out_dir, "z_sensitivity_avg.txt"))
    _plot_z_sensitivity(agg, os.path.join(out_dir, "z_sensitivity_avg.png"))


def _save_z_sensitivity_txt(agg: dict, save_path: str):
    with open(save_path, "w") as f:
        for sweep, d in agg.items():
            cols = d["header"][1:]
            header = ["alpha"] + [f"{c}_mean" for c in cols] + [f"{c}_std" for c in cols]
            f.write(f"# {sweep} (n={d['n_seeds']} seeds)\n")
            f.write("  ".join(f"{h:>14s}" for h in header) + "\n")
            for i, a in enumerate(d["alpha"]):
                row = [a] + list(d["mean"][i]) + list(d["std"][i])
                f.write(f"{row[0]:>14.2f}" + "".join(f"  {v:>14.4f}" for v in row[1:]) + "\n")
            f.write("\n")
    print(f"  saved -> {save_path}")


def _plot_z_sensitivity(agg: dict, save_path: str):
    blue, orange = "#2176AE", "#F4873C"
    sweep_colors = {"mean_noise": blue, "logvar_noise": orange}
    sweep_labels = {"mean_noise": "mean noise (μ)", "logvar_noise": "variance noise (logvar)"}

    any_header = next(iter(agg.values()))["header"]
    metrics = [("return", "Mean return", "Return vs. corruption")]
    if "length" in any_header:
        metrics.append(("length", "Mean episode length (steps)", "Steps vs. corruption"))
    if "success" in any_header:
        metrics.append(("success", "Success rate", "Success vs. corruption"))

    n_seeds = next(iter(agg.values()))["n_seeds"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.2))
    axes = np.atleast_1d(axes)
    fig.suptitle(
        f"Policy sensitivity to belief (z) corruption (mean ± std, n={n_seeds} seeds)", fontsize=13
    )

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for sweep, d in agg.items():
            cols = d["header"][1:]
            if key not in cols:
                continue
            i = cols.index(key)
            color = sweep_colors.get(sweep, "#898781")
            m, s = d["mean"][:, i], d["std"][:, i]
            ax.plot(d["alpha"], m, "-o", color=color, label=sweep_labels.get(sweep, sweep))
            ax.fill_between(d["alpha"], m - s, m + s, color=color, alpha=0.15)
        ax.set_xlabel("Mix fraction α  (0=true belief, 1=pure noise)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, linewidth=0.4)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {save_path}")


def aggregate_system_meta(run_dirs: list[str], out_dir: str):
    """Aggregate system-level metadata for a group of runs."""
    from research.ode.post_run_analysis.compare import (
        aggregate_bin_rows,
        read_meta_txt,
        write_meta_latex,
        write_meta_txt,
    )

    per_seed = []
    for run_dir in run_dirs:
        paths = [
            os.path.join(run_dir, "compare", "system_meta.txt"),
            os.path.join(run_dir, "system_meta.txt"),  # legacy standalone output
        ]
        path = next((candidate for candidate in paths if os.path.isfile(candidate)), None)
        if path is not None:
            rows = read_meta_txt(path)
            if rows:
                per_seed.append(rows)
    if not per_seed:
        print("  [!] No system_meta.txt found in any run, skipping")
        return
    if len(per_seed) < len(run_dirs):
        print(f"  [!] system_meta.txt found in {len(per_seed)}/{len(run_dirs)} runs")
    rows = aggregate_bin_rows(per_seed)
    write_meta_txt(
        rows, os.path.join(out_dir, "system_meta.txt"), system="seed_average", n_seeds=len(per_seed)
    )
    write_meta_latex(
        rows, os.path.join(out_dir, "system_meta.tex"), system="seed_average", n_seeds=len(per_seed)
    )


_GENERIC_EID_RE = re.compile(r"^(.+)_e(\d+)(?:_r(\d+))?$")


def _discover_repeat_groups(outputs_root: str, prefix: str | None) -> dict[int, list[str]]:
    """Discover repeat groups by experiment id."""
    pattern = (
        re.compile(rf"^{re.escape(prefix)}_e(\d+)(?:_r(\d+))?$") if prefix else _GENERIC_EID_RE
    )

    groups: dict[int, list[str]] = {}
    for metrics_path in sorted(
        glob.glob(os.path.join(outputs_root, "**", "*_metrics.json"), recursive=True)
    ):
        exp_name = os.path.basename(metrics_path)[: -len("_metrics.json")]
        match = pattern.match(exp_name)
        if not match:
            continue
        eid = int(match.group(2) if prefix is None else match.group(1))
        groups.setdefault(eid, []).append(os.path.dirname(metrics_path))
    return {eid: dirs for eid, dirs in groups.items() if len(dirs) > 1}


DEFAULT_METRICS = [
    "eval/mean_return",
    "eval/efficiency_train",
    "eval/efficiency_val",
    "eval/efficiency_test",
    "eval/efficiency_noise",
    "eval/task_embedding_mae",
    "eval/sigma_reduction_ratio",
    "eval/success_rate",
    "eval/belief_final_variance",
]

_STRIP_KEYS = {"run_uid", "seed", "repeat_batch_uid"}


def _config_signature(config: dict) -> str:
    """Hash config without run-identity fields."""
    pruned = {k: v for k, v in config.items() if k not in _STRIP_KEYS}
    blob = yaml.safe_dump(pruned, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def discover_runs(root: str, system: str, date: Optional[str]) -> list[dict]:
    """Discover completed runs for one system and optional date."""
    date_glob = date if date else "*"
    pattern = os.path.join(root, date_glob, "ode", system, "*", "checkpoints", "config.yaml")
    runs = []
    for config_path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(os.path.dirname(config_path))
        with open(config_path) as f:
            config = yaml.safe_load(f)
        runs.append(
            {
                "run_dir": run_dir,
                "run_uid": os.path.basename(run_dir),
                "seed": config.get("seed"),
                "repeat_batch_uid": config.get("repeat_batch_uid") or "",
                "exp_name": config.get("exp_name", "experiment"),
                "signature": _config_signature(config),
                "mtime": os.path.getmtime(run_dir),
            }
        )
    return runs


def group_into_batches(runs: list[dict]) -> dict[str, list[dict]]:
    """Group tagged or legacy runs into repeat batches."""
    tagged = defaultdict(list)
    legacy = defaultdict(list)
    for run in runs:
        if run["repeat_batch_uid"]:
            tagged[run["repeat_batch_uid"]].append(run)
        else:
            legacy[run["signature"]].append(run)

    batches = {}
    for batch_uid, batch_runs in tagged.items():
        batches[f"tagged_{batch_uid}"] = sorted(batch_runs, key=lambda r: (r["seed"], r["mtime"]))

    for sig, sig_runs in legacy.items():
        by_seed = defaultdict(list)
        for run in sig_runs:
            by_seed[run["seed"]].append(run)
        for seed_runs in by_seed.values():
            seed_runs.sort(key=lambda r: r["mtime"])

        max_occurrences = max((len(v) for v in by_seed.values()), default=0)
        for occurrence_idx in range(max_occurrences):
            sub_batch = [
                runs_for_seed[occurrence_idx]
                for runs_for_seed in by_seed.values()
                if occurrence_idx < len(runs_for_seed)
            ]
            sub_batch.sort(key=lambda r: (r["seed"], r["mtime"]))
            batches[f"legacy_{sig[:8]}_b{occurrence_idx}"] = sub_batch

    return batches


def print_manifest(batches: dict[str, list[dict]], min_seeds: int):
    """Print repeat groups and indicate which meet the seed threshold."""
    print(
        f"[*] Discovered {sum(len(v) for v in batches.values())} runs in {len(batches)} candidate batch(es):\n"
    )
    for label, runs in sorted(batches.items()):
        flag = "" if label.startswith("tagged_") else "  (heuristic grouping - verify below)"
        skip = " -> SKIPPED (< min-seeds)" if len(runs) < min_seeds else ""
        print(f"  {label}{flag}{skip}")
        for run in runs:
            print(f"      seed={run['seed']!s:<3} uid={run['run_uid']}  dir={run['run_dir']}")
        print()


def load_metrics(run: dict) -> Optional[list]:
    """Load the metrics artifact associated with one discovered run."""
    path = os.path.join(run["run_dir"], "outputs", f"{run['exp_name']}_metrics.json")
    if not os.path.isfile(path):
        print(f"[!] Skipping {run['run_uid']}: no metrics file at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def aggregate_batch(label: str, runs: list[dict], metric_keys: list[str], out_root: str):
    """Aggregate selected metrics for one repeat batch and write outputs."""
    import matplotlib.pyplot as plt

    per_run_metrics = []
    for run in runs:
        metrics = load_metrics(run)
        if metrics is not None:
            per_run_metrics.append((run, metrics))

    if len(per_run_metrics) < 2:
        print(f"[!] {label}: fewer than 2 runs with usable metrics, skipping")
        return

    out_dir = os.path.join(out_root, label)
    os.makedirs(out_dir, exist_ok=True)

    summary_lines = [f"Batch: {label}", f"Seeds: {[r['seed'] for r, _ in per_run_metrics]}", ""]

    for key in metric_keys:
        indices_per_run = []
        values_per_run = []
        for run, metrics in per_run_metrics:
            idx = [i for i, it in enumerate(metrics) if key in it]
            vals = [it[key] for it in metrics if key in it]
            indices_per_run.append(idx)
            values_per_run.append(vals)

        if not indices_per_run or not indices_per_run[0]:
            print(f"[!] {label}: metric '{key}' not found in any run, skipping")
            continue

        reference_idx = indices_per_run[0]
        if any(idx != reference_idx for idx in indices_per_run[1:]):
            print(
                f"[!] {label}: metric '{key}' logged at different iterations across seeds "
                f"(mismatched configs?), skipping"
            )
            continue

        values = np.array(values_per_run, dtype=float)  # (num_seeds, num_points)
        mean = values.mean(axis=0)
        std = values.std(axis=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(reference_idx, mean, color="C0", linewidth=1.8)
        ax.fill_between(reference_idx, mean - std, mean + std, color="C0", alpha=0.2)
        ax.set_xlabel("iteration")
        ax.set_ylabel(key)
        ax.set_title(f"{label}: {key} (n={len(per_run_metrics)} seeds)")
        fig.tight_layout()
        fig_path = os.path.join(out_dir, f"{key.replace('/', '_')}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        summary_lines.append(f"{key}: final={mean[-1]:.4f} +/- {std[-1]:.4f}")

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"[+] {label}: aggregated {len(per_run_metrics)} seeds -> {out_dir}")


_L2_PAIRS = (
    ("policy_err_mean", "policy_err_std"),
    ("policy_err_integrated_mean", "policy_err_integrated_std"),
    ("pid_err_mean", "pid_err_std"),
    ("pid_err_integrated_mean", "pid_err_integrated_std"),
)


def _load_l2_error(run_dir: str):
    path = os.path.join(run_dir, "l2_error.txt")
    if not os.path.isfile(path):
        return None
    blocks = _read_blocks(path)
    result = {}
    for subset, (header, rows) in blocks.items():
        if rows:
            result[subset] = {"header": header, "data": np.array(rows)}
    return result or None


def aggregate_l2_error(run_dirs: list[str], out_dir: str):
    """Aggregate last-step and integrated L2 error histories."""
    per_run = [data for run in run_dirs if (data := _load_l2_error(run)) is not None]
    if not per_run:
        print("  [!] No l2_error.txt found in any run, skipping")
        return
    if len(per_run) < len(run_dirs):
        print(f"  [!] l2_error.txt found in {len(per_run)}/{len(run_dirs)} runs")

    subsets = []
    for data in per_run:
        for subset in data:
            if subset not in subsets:
                subsets.append(subset)

    aggregate = {}
    for subset in subsets:
        entries = [data[subset] for data in per_run if subset in data]
        min_len = min(len(entry["data"]) for entry in entries)
        header = entries[0]["header"]
        stacked = np.stack([entry["data"][:min_len] for entry in entries])
        result = {"iterations": stacked[0, :, 0], "n_runs": len(entries)}
        for mean_name, std_name in _L2_PAIRS:
            if mean_name not in header:
                continue
            index = header.index(mean_name)
            result[mean_name] = stacked[:, :, index].mean(axis=0)
            result[std_name] = stacked[:, :, index].std(axis=0)
        if len(result) > 2:
            aggregate[subset] = result

    if not aggregate:
        print("  [!] No usable L2-error blocks found, skipping")
        return
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "l2_error_avg.txt")
    png_path = os.path.join(out_dir, "l2_error_avg.png")
    _save_l2_error_txt(aggregate, txt_path)
    history = []
    for subset, data in aggregate.items():
        columns = [name for pair in _L2_PAIRS for name in pair if name in data]
        for index, iteration in enumerate(data["iterations"]):
            history.append(
                (
                    int(iteration),
                    {
                        subset: {column: float(data[column][index]) for column in columns},
                    },
                )
            )
    plot_l2_error_history(history, png_path)
    print(f"  saved -> {png_path}")


def _save_l2_error_txt(aggregate: dict, save_path: str):
    with open(save_path, "w") as f:
        for subset, data in aggregate.items():
            columns = [name for pair in _L2_PAIRS for name in pair if name in data]
            f.write(f"# subset: {subset}\n")
            f.write("  ".join(f"{name:>18s}" for name in ["iteration"] + columns) + "\n")
            for index, iteration in enumerate(data["iterations"]):
                values = [data[column][index] for column in columns]
                f.write(
                    f"{iteration:>18.0f}" + "".join(f"  {value:>18.4f}" for value in values) + "\n"
                )
            f.write("\n")
    print(f"  saved -> {save_path}")


def run_averaging(run_dirs: list[str], out_dir: str):
    """Run every enabled learning-statistics aggregation."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] Averaging {len(run_dirs)} seed(s) -> {out_dir}")
    print("[*] Per-mu return over training ...")
    aggregate_mu_returns(run_dirs, out_dir)
    print("[*] Efficiency ...")
    aggregate_efficiency(run_dirs, out_dir)
    print("[*] Steps vs mu ...")
    aggregate_steps_vs_mu(run_dirs, out_dir)
    print("[*] Error (last-step) vs mu ...")
    aggregate_steps_vs_mu(
        run_dirs,
        out_dir,
        filename="error_vs_mu.txt",
        out_prefix="error_vs_mu",
        y_label="log10 relative L2 error (last step)",
    )
    print("[*] Error (time-integrated) vs mu ...")
    aggregate_steps_vs_mu(
        run_dirs,
        out_dir,
        filename="error_integrated_vs_mu.txt",
        out_prefix="error_integrated_vs_mu",
        y_label="log10 relative L2 error (time-integrated)",
    )
    print("[*] Z-sensitivity ...")
    aggregate_z_sensitivity(run_dirs, out_dir)
    print("[*] L2 error ...")
    aggregate_l2_error(run_dirs, out_dir)
    print("[*] System metadata ...")
    aggregate_system_meta(run_dirs, out_dir)


def _build_parser():
    parser = argparse.ArgumentParser(description="Aggregate ODE learning-statistics artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    average = subparsers.add_parser("average-seeds", help="Average per-run analysis artifacts")
    average.add_argument("--runs", nargs="+", help="Run output directories")
    average.add_argument(
        "--discover-experiment",
        metavar="OUTPUTS_ROOT",
        help="Discover repeat groups under an experiment outputs root",
    )
    average.add_argument("--prefix", help="Optional experiment-name prefix for discovery")
    average.add_argument(
        "--out", required=True, help="Output directory or discovery-mode parent directory"
    )

    repeat = subparsers.add_parser("aggregate-repeats", help="Aggregate training metrics by repeat")
    repeat.add_argument("--system", required=True, help="ODE system name")
    repeat.add_argument("--root", default="outputs/runs", help="Root runs directory")
    repeat.add_argument("--date", default=None, help="Restrict to one run date")
    repeat.add_argument("--out", default=None, help="Output directory")
    repeat.add_argument(
        "--metrics", default=",".join(DEFAULT_METRICS), help="Comma-separated metric keys"
    )
    repeat.add_argument(
        "--min-seeds", type=int, default=2, help="Skip batches with fewer usable seeds"
    )
    repeat.add_argument("--dry-run", action="store_true", help="Print grouping without aggregating")
    return parser


def _run_average_seeds(args):
    if args.runs and args.discover_experiment:
        raise SystemExit("--runs and --discover-experiment are mutually exclusive")
    if args.discover_experiment:
        groups = _discover_repeat_groups(args.discover_experiment, args.prefix)
        if not groups:
            print(
                f"[*] No repeat groups (2+ seeds) found under {args.discover_experiment}, nothing to average"
            )
            return
        for eid in sorted(groups):
            run_averaging(groups[eid], os.path.join(args.out, f"seed_average_e{eid}"))
        return
    if not args.runs:
        raise SystemExit("--runs is required (or use --discover-experiment)")
    missing = [run for run in args.runs if not os.path.isdir(run)]
    if missing:
        raise SystemExit(f"Run dir(s) not found: {missing}")
    run_averaging(args.runs, args.out)


def _run_aggregate_repeats(args):
    metric_keys = [key.strip() for key in args.metrics.split(",") if key.strip()]
    runs = discover_runs(args.root, args.system, args.date)
    if not runs:
        print(f"[!] No runs found under {args.root}/*/ode/{args.system}/*/checkpoints/config.yaml")
        return
    batches = group_into_batches(runs)
    print_manifest(batches, args.min_seeds)
    if args.dry_run:
        return
    out_root = args.out or os.path.join("outputs", "repeat_analysis", args.system)
    os.makedirs(out_root, exist_ok=True)
    for label, batch_runs in sorted(batches.items()):
        if len(batch_runs) >= args.min_seeds:
            aggregate_batch(label, batch_runs, metric_keys, out_root)


def main(argv=None):
    """Parse options and aggregate learning statistics for discovered runs."""
    args = _build_parser().parse_args(argv)
    if args.command == "average-seeds":
        _run_average_seeds(args)
        print(f"\n[+] All seed-averaged outputs saved under: {args.out}")
    elif args.command == "aggregate-repeats":
        _run_aggregate_repeats(args)


if __name__ == "__main__":
    main()
