"""Reusable figures and plotting helpers for ODE experiments."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from steppo.utils.task_params import task_bounds, task_label


def get_param_range(config):
    """Return ``(min_value, max_value, label)`` for the active ODE parameter."""
    return (*task_bounds(config.env), task_label(config.env.system))


def make_bin_edges(mu_lo, mu_hi, bin_width, num_bins, log_bins):
    """Return linear or logarithmic bin edges; a zero-width range becomes one real bin."""
    if mu_lo == mu_hi:
        pad = mu_lo * 0.05 if log_bins else bin_width / 2
        pad = pad if pad > 0 else 1e-6
        return np.array([mu_lo - pad, mu_hi + pad])
    if log_bins:
        return np.geomspace(mu_lo, mu_hi, num_bins + 1)
    return np.arange(mu_lo, mu_hi + bin_width, bin_width)


def _completion_mask(series, length):
    """Boolean mask of the episodes in `series` whose solve reached t_end,
    all-true for a series that carries no mask (see `bin_stats`)."""
    completed = series.get("completed")
    if completed is None:
        return np.ones(length, dtype=bool)
    return np.asarray(completed, dtype=bool)


def bin_stats(data, bin_edges, value_key="steps"):
    """Per-bin mean, std, count and completion fraction of `data[value_key]`.

    Only solves that reached t_end are averaged: an aborted solve has a small
    step count that would read as a cheap success. Empty bins are NaN.
    """
    mu, steps = data["mu"], data[value_key].astype(float)
    completed = _completion_mask(data, len(mu))
    bin_idx = np.digitize(mu, bin_edges) - 1
    means, stds, counts, completion = [], [], [], []
    for i in range(len(bin_edges) - 1):
        in_bin = bin_idx == i
        usable = in_bin & completed
        completion.append(float(completed[in_bin].mean()) if in_bin.any() else np.nan)
        if usable.any():
            means.append(np.mean(steps[usable]))
            stds.append(np.std(steps[usable]))
            counts.append(int(usable.sum()))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            counts.append(0)
    return np.array(means), np.array(stds), counts, np.array(completion)


def incomplete_spans(bin_centers, completion):
    """Merged [(lo, hi), ...] ranges where some solve did not reach t_end (NaN bins are not shaded)."""
    completion = np.asarray(completion, dtype=float)
    failing = np.isfinite(completion) & (completion < 1.0)
    spans, start = [], None
    for i, is_failing in enumerate(failing):
        if is_failing and start is None:
            start = i
        elif not is_failing and start is not None:
            spans.append((float(bin_centers[start]), float(bin_centers[i - 1])))
            start = None
    if start is not None:
        spans.append((float(bin_centers[start]), float(bin_centers[-1])))
    return spans


def plot_steps_vs_mu(
    pid_data,
    rl_data,
    mu_min,
    mu_max,
    mu_lo,
    mu_hi,
    bin_width,
    num_bins,
    log_bins,
    out_path,
    param_label="μ",
    train_bins=None,
    comparison_label="RL (diffeqsolve)",
    oracle_data=None,
):
    """Plot PID and optional RL solver-step distributions against the ODE parameter."""
    bin_edges = make_bin_edges(mu_lo, mu_hi, bin_width, num_bins, log_bins)
    bin_centers = (
        np.sqrt(bin_edges[:-1] * bin_edges[1:])
        if log_bins
        else (bin_edges[:-1] + bin_edges[1:]) / 2
    )
    bar_widths = (bin_edges[1:] - bin_edges[:-1]) * 0.9

    pid_means, pid_stds, _, _ = bin_stats(pid_data, bin_edges)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        bin_centers,
        pid_means,
        bar_widths,
        yerr=pid_stds,
        capsize=2,
        label="PID (budget)",
        color="#4472C4",
        alpha=0.55,
    )

    if rl_data is not None:
        rl_means, rl_stds, _, _ = bin_stats(rl_data, bin_edges)
        ax.bar(
            bin_centers,
            rl_means,
            bar_widths,
            yerr=rl_stds,
            capsize=2,
            label=comparison_label,
            color="#ED7D31",
            alpha=0.55,
        )

    if oracle_data is not None:
        oracle_means, oracle_stds, _, _ = bin_stats(oracle_data, bin_edges)
        ax.bar(
            bin_centers,
            oracle_means,
            bar_widths,
            yerr=oracle_stds,
            capsize=2,
            label="Oracle",
            color="#70AD47",
            alpha=0.55,
        )

    if train_bins:
        for i, (lo, hi) in enumerate(train_bins):
            ax.axvspan(
                lo,
                hi,
                color="#2aa876",
                alpha=0.15,
                label="Trained bins" if i == 0 else None,
            )
            ax.axvline(lo, color="#2aa876", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axvline(hi, color="#2aa876", linestyle="--", linewidth=1.0, alpha=0.7)
    else:
        ax.axvline(
            mu_min,
            color="k",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Train dist [{mu_min:.0f}, {mu_max:.0f}]",
        )
        ax.axvline(mu_max, color="k", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvspan(mu_lo, mu_min, color="grey", alpha=0.08)
        ax.axvspan(mu_max, mu_hi, color="grey", alpha=0.08)

    if log_bins:
        ax.set_xscale("log")
    ax.set_xlabel(param_label, fontsize=13)
    ax.set_ylabel("Solver steps", fontsize=13)
    ax.set_title(f"Solver steps vs {param_label} (stiffness parameter)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    if not log_bins:
        tick_positions = bin_centers[:: max(1, len(bin_centers) // 20)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{value:.0f}" for value in tick_positions])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def save_steps_vs_mu_splits_txt(split_data, save_path, mu_min=None, mu_max=None, value_key="steps"):
    """Write per-split PID/RL/oracle bin tables of `value_key` ("steps", "err" or "err_integrated")."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        all_data = list(split_data.values())
        all_values = []
        for data in all_data:
            all_values.append(np.asarray(data["pid"]["mu"]))
            for key in ("policy", "oracle"):
                if data.get(key) is not None:
                    all_values.append(np.asarray(data[key]["mu"]))
        mu_lo = float(min(np.min(values) for values in all_values))
        mu_hi = float(max(np.max(values) for values in all_values))
        if mu_min is not None:
            mu_lo = min(float(mu_min), mu_lo)
        if mu_max is not None:
            mu_hi = max(float(mu_max), mu_hi)
        log_bins = mu_lo > 0.0 and all(np.all(values > 0.0) for values in all_values)
        bin_edges = make_bin_edges(mu_lo, mu_hi, 5.0, 64, log_bins)
        bin_centers = (
            np.sqrt(bin_edges[:-1] * bin_edges[1:])
            if log_bins
            else (bin_edges[:-1] + bin_edges[1:]) / 2
        )
        for split, data in split_data.items():
            header = ["bin_center"]
            cols = [bin_centers]
            for name, key in (
                ("pid", "pid"),
                ("policy", "policy"),
                ("oracle", "oracle"),
                ("pid_unlimited", "pid_unlimited"),
            ):
                if data.get(key) is None:
                    continue
                means, stds, counts, completion = bin_stats(data[key], bin_edges, value_key)
                header += [f"{name}_mean", f"{name}_std", f"{name}_count", f"{name}_completion"]
                cols += [means, stds, counts, completion]

            f.write(f"# split: {split}" + chr(10))
            f.write("  ".join(header) + chr(10))
            for row in zip(*cols):
                f.write(
                    "  ".join(
                        f"{value:.4f}"
                        if isinstance(value, (float, np.floating))
                        else str(int(value))
                        for value in row
                    )
                    + chr(10)
                )
            f.write(chr(10))


def save_convergence_table(split_data, save_path, param_label="μ") -> None:
    """Write an unbinned per-split, per-method table for a single-task configuration.

    Rows give mean ± std of solver steps (and errors, when present) over solves
    that reached t_end, plus the completed fraction.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    methods = [
        ("PID", "pid", "pid_error"),
        ("PID (unlimited budget)", "pid_unlimited", None),
        ("RL", "policy", "policy_error"),
        ("Oracle", "oracle", "oracle_error"),
    ]
    with open(save_path, "w") as f:
        for split, data in split_data.items():
            pid_mu = np.asarray(data["pid"]["mu"], dtype=float)
            mu_value = float(pid_mu[0]) if len(pid_mu) else float("nan")
            f.write(f"# split: {split}  ({param_label} = {mu_value:.6g})\n")
            f.write(
                f"{'method':<24}{'steps (mean±std, n)':<26}"
                f"{'err (mean±std)':<20}{'err_integrated (mean±std)':<26}"
                f"{'completion':<12}\n"
            )
            for label, steps_key, err_key in methods:
                steps_series = data.get(steps_key)
                if steps_series is None:
                    continue
                steps = np.asarray(steps_series["steps"], dtype=float)
                done = _completion_mask(steps_series, len(steps))
                completion_str = f"{done.mean():.2f}" if len(done) else "n/a"
                steps = steps[done]
                steps_str = (
                    f"{steps.mean():.2f}±{steps.std():.2f} (n={len(steps)})"
                    if len(steps)
                    else "n/a"
                )
                err_str, err_int_str = "n/a", "n/a"
                err_series = data.get(err_key) if err_key else None
                if err_series is not None:
                    err_done = _completion_mask(err_series, len(err_series.get("err", [])))
                    err = np.asarray(err_series.get("err", []), dtype=float)[err_done]
                    if len(err):
                        err_str = f"{err.mean():.3f}±{err.std():.3f}"
                    err_int = np.asarray(err_series.get("err_integrated", []), dtype=float)
                    if len(err_int):
                        err_int = err_int[_completion_mask(err_series, len(err_int))]
                    if len(err_int):
                        err_int_str = f"{err_int.mean():.3f}±{err_int.std():.3f}"
                f.write(
                    f"{label:<24}{steps_str:<26}{err_str:<20}"
                    f"{err_int_str:<26}{completion_str:<12}\n"
                )
            f.write("\n")
    print(f"  → {save_path}")


def plot_steps_vs_mu_splits(
    split_data,
    mu_min,
    mu_max,
    out_path,
    param_label="μ",
    train_bins=None,
    value_key="steps",
    y_label="Solver steps",
):
    """Plot binned `value_key` for RL per split and for PID/oracle on the test split."""
    if not split_data:
        return

    all_mu = [np.asarray(data["pid"]["mu"]) for data in split_data.values()]
    mu_lo = float(min(np.min(values) for values in all_mu))
    mu_hi = float(max(np.max(values) for values in all_mu))
    if mu_min is not None:
        mu_lo = min(float(mu_min), mu_lo)
    if mu_max is not None:
        mu_hi = max(float(mu_max), mu_hi)
    log_bins = mu_lo > 0.0 and all(np.all(values > 0.0) for values in all_mu)
    bin_edges = make_bin_edges(mu_lo, mu_hi, 5.0, 64, log_bins)
    bin_centers = (
        np.sqrt(bin_edges[:-1] * bin_edges[1:])
        if log_bins
        else (bin_edges[:-1] + bin_edges[1:]) / 2
    )

    colors = {"train": "#4472C4", "val": "#ED7D31", "test": "#548235"}
    fig, ax = plt.subplots(figsize=(14, 6))

    for split, data in split_data.items():
        policy = data.get("policy")
        if policy is None:
            continue
        means, stds, _, _ = bin_stats(policy, bin_edges, value_key)
        color = colors.get(split, "#898781")
        ax.plot(bin_centers, means, color=color, linewidth=1.8, label=f"RL ({split})")
        ax.fill_between(bin_centers, means - stds, means + stds, color=color, alpha=0.16)

    baseline_split = "test" if "test" in split_data else next(iter(split_data))
    baseline = split_data[baseline_split]
    pid_means, pid_stds, _, pid_completion = bin_stats(baseline["pid"], bin_edges, value_key)
    ax.plot(
        bin_centers,
        pid_means,
        color="#222222",
        linewidth=1.8,
        linestyle="-",
        label=f"PID ({baseline_split})",
    )
    ax.fill_between(
        bin_centers, pid_means - pid_stds, pid_means + pid_stds, color="#222222", alpha=0.10
    )
    if baseline.get("oracle") is not None:
        oracle_means, oracle_stds, _, _ = bin_stats(baseline["oracle"], bin_edges, value_key)
        ax.plot(
            bin_centers,
            oracle_means,
            color="#70AD47",
            linewidth=1.8,
            linestyle="--",
            label=f"Oracle ({baseline_split})",
        )
        ax.fill_between(
            bin_centers,
            oracle_means - oracle_stds,
            oracle_means + oracle_stds,
            color="#70AD47",
            alpha=0.12,
        )
    if baseline.get("pid_unlimited") is not None:
        pid_u_means, pid_u_stds, _, _ = bin_stats(baseline["pid_unlimited"], bin_edges, value_key)
        ax.plot(
            bin_centers,
            pid_u_means,
            color="#222222",
            linewidth=1.4,
            linestyle=":",
            label=f"PID, unlimited budget ({baseline_split})",
        )
        ax.fill_between(
            bin_centers,
            pid_u_means - pid_u_stds,
            pid_u_means + pid_u_stds,
            color="#222222",
            alpha=0.06,
        )

    # Shade ranges where the baseline did not always solve.
    for i, (lo, hi) in enumerate(incomplete_spans(bin_centers, pid_completion)):
        ax.axvspan(
            lo,
            hi,
            color="#C00000",
            alpha=0.10,
            zorder=0,
            label=f"PID incomplete ({baseline_split})" if i == 0 else None,
        )
    if train_bins:
        for i, (lo, hi) in enumerate(train_bins):
            ax.axvspan(
                lo, hi, color="#2aa876", alpha=0.12, label="Training bins" if i == 0 else None
            )
    if log_bins:
        ax.set_xscale("log")
    ax.set_xlabel(param_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(f"{y_label} vs {param_label} (train/val/test RL, PID, oracle)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def error_output_path(steps_out_path: str, metric: str) -> str:
    """Map '.../steps_vs_mu.png' to '.../<metric>_vs_mu.png'; other names get a '_<metric>' suffix."""
    dirname, base = os.path.split(steps_out_path)
    name, ext = os.path.splitext(base)
    new_name = f"{metric}_vs_mu" if name == "steps_vs_mu" else f"{name}_{metric}"
    return os.path.join(dirname, new_name + ext)


def build_error_split_data(split_data: dict, value_key: str) -> dict | None:
    """Reshape per-split *_error series into the pid/policy/oracle layout used for steps.

    Series without `value_key` are omitted; returns None when there is no error data.
    """
    result = {}
    for split, series in split_data.items():
        pid_error = series.get("pid_error")
        if pid_error is None or value_key not in pid_error:
            continue
        policy_error = series.get("policy_error")
        oracle_error = series.get("oracle_error")
        result[split] = {
            "pid": pid_error,
            "policy": policy_error
            if policy_error is not None and value_key in policy_error
            else None,
            "oracle": oracle_error
            if oracle_error is not None and value_key in oracle_error
            else None,
        }
    return result or None


def write_steps_vs_mu_error_outputs(
    split_data,
    mu_min,
    mu_max,
    steps_out_path,
    param_label="μ",
    train_bins=None,
) -> list[str]:
    """Write error_vs_mu and error_integrated_vs_mu plots and tables beside `steps_out_path`.

    Metrics without data are skipped. Returns the written .txt paths.
    """
    written = []
    for metric, value_key, y_label in [
        ("error", "err", "log10 relative L2 error (last step)"),
        ("error_integrated", "err_integrated", "log10 relative L2 error (time-integrated)"),
    ]:
        error_split_data = build_error_split_data(split_data, value_key)
        if error_split_data is None:
            continue
        err_out_path = error_output_path(steps_out_path, metric)
        plot_steps_vs_mu_splits(
            error_split_data,
            mu_min,
            mu_max,
            err_out_path,
            param_label=param_label,
            train_bins=train_bins,
            value_key=value_key,
            y_label=y_label,
        )
        err_txt_path = os.path.splitext(err_out_path)[0] + ".txt"
        save_steps_vs_mu_splits_txt(
            error_split_data, err_txt_path, mu_min, mu_max, value_key=value_key
        )
        written.append(err_txt_path)
    return written
