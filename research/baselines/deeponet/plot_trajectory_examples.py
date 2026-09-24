"""Plot trajectories (top) and log relative error (bottom) of the oracle, RL and
DeepONet against the reference, per system.

Reads generate_trajectory_examples.py's output-deeponet/<system>/results/trajectory_examples.npz,
queries the DeepONet on the reference time points, and writes a PNG and CSV per example.

Run in the DeepONet container:
    python plot_trajectory_examples.py --system van_der_pol
    python plot_trajectory_examples.py --all
"""

import argparse
import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from deeponet_relerr_eval import _SYSTEMS, load_model_for_system, predict_at
from overleaf_tables import align_to_reference, trajectory_column_names
from relerr_metrics import relative_l2_error, to_log10_clipped

HERE = os.path.dirname(__file__)

_COLORS = plt.cm.viridis(np.linspace(0.15, 0.9, 4))
_STYLES = {"oracle": "-", "rl": "-", "deeponet": "--"}
_LABELS = {"oracle": "Oracle", "rl": "RL", "deeponet": "DeepONet"}


def state_dim0(y: np.ndarray) -> np.ndarray:
    """First state dimension, shown in the top panel."""
    return y[..., 0]


def plot_one(system: str, profile: str = "retrain", checkpoint: str | None = None):
    spec = _SYSTEMS[system]
    param_label = spec["param_label"]
    dn_dir = os.path.join(HERE, "output-deeponet", system)
    results_dir = os.path.join(dn_dir, "results")
    plots_dir = os.path.join(dn_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    npz_path = os.path.join(results_dir, "trajectory_examples.npz")
    d = np.load(npz_path, allow_pickle=True)
    task_params = d["task_params"]
    ref_ts, ref_ys = d["ref_ts"], d["ref_ys"]
    if ref_ys.ndim == 2:
        ref_ys = ref_ys[:, :, None]
    oracle_ts, oracle_ys = d["oracle_ts"], d["oracle_ys"]
    has_rl = "rl_ts" in d.files
    rl_ts, rl_ys = (d["rl_ts"], d["rl_ys"]) if has_rl else (None, None)
    atol = float(d["atol"])

    model, _, _ = load_model_for_system(
        system,
        ref_ys_ndim3=(ref_ys.shape[-1] > 1),
        profile=profile,
        checkpoint=checkpoint,
    )

    n = len(task_params)
    output_paths = []
    for i in range(n):
        color = _COLORS[i % len(_COLORS)]
        fig, axs = plt.subplots(
            2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]}
        )
        valid_ref = np.isfinite(ref_ts[i])
        t_ref = ref_ts[i][valid_ref].astype(np.float64)
        y_ref = ref_ys[i][valid_ref]
        label_param = f"{param_label}={task_params[i]:.3g}"

        axs[0].plot(
            t_ref,
            state_dim0(y_ref),
            color=color,
            linewidth=1.0,
            alpha=0.5,
            label=f"reference ({label_param})",
        )

        # --- Oracle ---
        valid_o = np.isfinite(oracle_ts[i])
        t_o = oracle_ts[i][valid_o].astype(np.float64)
        y_o = oracle_ys[i][valid_o]
        if len(t_o) >= 2:
            axs[0].plot(
                t_o,
                state_dim0(y_o),
                _STYLES["oracle"],
                color=color,
                linewidth=1.6,
                marker="o",
                markersize=3,
                markevery=max(1, len(t_o) // 12),
            )
            y_o_at_ref = np.stack(
                [np.interp(t_ref, t_o, y_o[:, k]) for k in range(y_o.shape[-1])], axis=-1
            )
            e_o = to_log10_clipped(relative_l2_error(y_o_at_ref, y_ref, atol))
            axs[1].plot(
                t_ref,
                e_o,
                _STYLES["oracle"],
                color=color,
                linewidth=1.6,
                marker="o",
                markersize=3,
                markevery=max(1, len(t_ref) // 12),
            )

        # --- RL ---
        if has_rl:
            valid_r = np.isfinite(rl_ts[i])
            t_r = rl_ts[i][valid_r].astype(np.float64)
            y_r = rl_ys[i][valid_r]
            if len(t_r) >= 2:
                axs[0].plot(
                    t_r,
                    state_dim0(y_r),
                    _STYLES["rl"],
                    color=color,
                    linewidth=1.6,
                    marker="s",
                    markersize=3,
                    markevery=max(1, len(t_r) // 12),
                )
                y_r_at_ref = np.stack(
                    [np.interp(t_ref, t_r, y_r[:, k]) for k in range(y_r.shape[-1])], axis=-1
                )
                e_r = to_log10_clipped(relative_l2_error(y_r_at_ref, y_ref, atol))
                axs[1].plot(
                    t_ref,
                    e_r,
                    _STYLES["rl"],
                    color=color,
                    linewidth=1.6,
                    marker="s",
                    markersize=3,
                    markevery=max(1, len(t_ref) // 12),
                )

        # --- DeepONet, queried on the reference time points ---
        y_dn = predict_at(model, system, task_params[i], t_ref)
        axs[0].plot(
            t_ref,
            state_dim0(y_dn),
            _STYLES["deeponet"],
            color=color,
            linewidth=1.6,
            marker="^",
            markersize=3,
            markevery=max(1, len(t_ref) // 12),
        )
        e_dn = to_log10_clipped(relative_l2_error(y_dn, y_ref, atol))
        axs[1].plot(
            t_ref,
            e_dn,
            _STYLES["deeponet"],
            color=color,
            linewidth=1.6,
            marker="^",
            markersize=3,
            markevery=max(1, len(t_ref) // 12),
        )

        # Export all plotted trajectories on the same reference time grid.
        csv_columns = ["t"]
        csv_values = [t_ref]

        def add_csv_series(prefix: str, ts: np.ndarray, ys: np.ndarray):
            aligned = align_to_reference(t_ref, ts, ys)
            csv_columns.extend(trajectory_column_names(prefix, aligned.shape[-1]))
            csv_values.extend(aligned[:, dim] for dim in range(aligned.shape[-1]))

        add_csv_series("reference", t_ref, y_ref)
        add_csv_series("oracle", t_o, y_o)
        if has_rl:
            add_csv_series("rl", t_r, y_r)
        add_csv_series("deeponet", t_ref, y_dn)

        axs[0].set_ylabel("y (first state dimension)")
        axs[0].set_title(
            f"{system}: trajectory {i + 1}/{n} ({label_param}) — "
            "Oracle (o) / RL (s) / DeepONet (^) vs reference"
        )
        axs[1].set_yscale("log")
        axs[1].set_xlabel("t")
        axs[1].set_ylabel("relative L2 error vs reference")

        method_handles = [
            plt.Line2D(
                [0],
                [0],
                color=color,
                lw=1.6,
                linestyle=_STYLES["oracle"],
                marker="o",
                markersize=4,
                label="Oracle",
            )
        ]
        if has_rl:
            method_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=1.6,
                    linestyle=_STYLES["rl"],
                    marker="s",
                    markersize=4,
                    label="RL",
                )
            )
        method_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                lw=1.6,
                linestyle=_STYLES["deeponet"],
                marker="^",
                markersize=4,
                label="DeepONet",
            )
        )
        method_handles.append(
            plt.Line2D([0], [0], color=color, lw=1.0, alpha=0.5, label="reference")
        )
        axs[0].legend(handles=method_handles, fontsize=8, loc="lower left")

        fig.tight_layout()
        out_path = os.path.join(plots_dir, f"trajectory_example_{i + 1:02d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        output_paths.append(out_path)
        print(f"Saved: {out_path}")

        csv_path = os.path.join(plots_dir, f"trajectory_example_{i + 1:02d}.csv")
        np.savetxt(
            csv_path,
            np.column_stack(csv_values),
            delimiter=",",
            header=",".join(csv_columns),
            comments="",
            fmt="%.10g",
        )
        print(f"Saved: {csv_path}")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=list(_SYSTEMS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--profile", choices=["retrain", "legacy"], default="retrain")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="optional explicit checkpoint path relative to repository root",
    )
    args = parser.parse_args()
    systems = list(_SYSTEMS) if args.all else [args.system]
    if not systems or systems == [None]:
        parser.error("geef --system of --all")
    if args.all and args.checkpoint:
        parser.error("--checkpoint can only be used with one --system")
    for s in systems:
        plot_one(s, profile=args.profile, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
