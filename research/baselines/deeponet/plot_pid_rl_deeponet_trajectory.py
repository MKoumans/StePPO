"""Trajectory and local-error comparison of PID, PID (unlimited budget), RL and
DeepONet against the reference, for one task per system (Fig. 2).

Reads generate_pid_rl_trajectory_example.py's
output-deeponet/<system>/results/pid_rl_vs_reference_trajectory.npz, queries the
system's DeepONet on the reference time grid, and writes one CSV and one PNG per
system with columns t, PID_y*, PID_err, PIDu_y*, PIDu_err, RL_y*, RL_err, D_y*, D_err.
The error is log10 of diffrax's PID error norm against the reference:

    err(t) = ||y(t) - y_ref(t)|| / (atol + rtol * max(|y_ref(t)|))

Run in the DeepONet container:
    python plot_pid_rl_deeponet_trajectory.py [--system SYSTEM ...]
"""

import argparse
import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from overleaf_tables import align_to_reference, trajectory_column_names
from relerr_metrics import pid_scaled_error, to_log10_clipped
from train_deeponet_unified import _build_msffn_trunk

HERE = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(HERE, "..", "..", "..", "outputs", "baselines", "20260915")

SYSTEMS = {
    "scalar_decay": dict(state_dim=1, branch_layers=[1, 128, 128], trunk_layers=[1, 128, 128, 128]),
    "van_der_pol": dict(
        state_dim=2,
        branch_layers=[1, 256, 256, 256, 256, 256],
        trunk_layers=[1, 256, 256, 256, 256, 256],
    ),
    "brusselator": dict(
        state_dim=2,
        branch_layers=[1, 256, 256, 256, 256, 256],
        trunk_layers=[1, 256, 256, 256, 256, 256],
    ),
}
MSFFN_SIGMAS = [
    1.0,
    10.0,
]  # trunk scales of the retrain_complete checkpoints (see their training_config.json)

_STYLES = {"PID": "-", "PIDu": ":", "RL": "-", "D": "--"}
_MARKERS = {"PID": "o", "PIDu": "d", "RL": "s", "D": "^"}
_COLORS = {"PID": "#4472C4", "PIDu": "#4472C4", "RL": "#2CA02C", "D": "#C00000"}


def load_deeponet_msffn(
    checkpoint: str, t_end: float, branch_layers: list, trunk_layers: list, num_outputs: int
):
    """Rebuild the MsFFN-trunk DeepONet of train_deeponet_unified.py and restore its checkpoint."""
    import deepxde as dde
    import torch

    t_grid_col = np.linspace(0.0, t_end, 200)[:, None].astype(np.float32)
    dummy_branch = np.zeros((1, 1), dtype=np.float32)
    if num_outputs > 1:
        dummy_y = np.zeros((1, len(t_grid_col), num_outputs), dtype=np.float32)
    else:
        dummy_y = np.zeros((1, len(t_grid_col)), dtype=np.float32)
    data = dde.data.TripleCartesianProd(
        X_train=(dummy_branch, t_grid_col),
        y_train=dummy_y,
        X_test=(dummy_branch, t_grid_col),
        y_test=dummy_y,
    )
    net_kwargs = (
        dict(num_outputs=num_outputs, multi_output_strategy="independent")
        if num_outputs > 1
        else {}
    )
    net = dde.nn.DeepONetCartesianProd(
        branch_layers, trunk_layers, "relu", "Glorot normal", **net_kwargs
    )
    activation_fn = dde.nn.activations.get("relu")
    if num_outputs > 1:
        net.trunk = torch.nn.ModuleList(
            _build_msffn_trunk(trunk_layers, activation_fn, MSFFN_SIGMAS)
            for _ in range(num_outputs)
        )
    else:
        net.trunk = _build_msffn_trunk(trunk_layers, activation_fn, MSFFN_SIGMAS)
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.restore(checkpoint, verbose=1)
    print(f"[*] Checkpoint restored: {checkpoint}")
    return model


def plot_one(system: str):
    spec = SYSTEMS[system]
    state_dim = spec["state_dim"]
    out_dir = os.path.join(HERE, "output-deeponet", system)
    npz_path = os.path.join(out_dir, "results", "pid_rl_vs_reference_trajectory.npz")
    ckpt_path = os.path.join(
        out_dir, "models", f"deeponet_{system}_retrain_complete_final-30000.pt"
    )

    d = np.load(npz_path, allow_pickle=True)
    mu = float(d["task_param"])
    atol = float(d["atol"])
    rtol = float(d["rtol"])

    valid_ref = np.isfinite(d["ref_ts"])
    t_ref = d["ref_ts"][valid_ref].astype(np.float64)
    y_ref = d["ref_ys"][valid_ref]
    if y_ref.ndim == 1:
        y_ref = y_ref[:, None]

    def as2d(ys):
        return ys[:, None] if ys.ndim == 1 else ys

    y_pid = align_to_reference(t_ref, d["pid_ts"], as2d(d["pid_ys"]), mask_beyond_end=True)
    y_pidu = align_to_reference(
        t_ref, d["pid_unlimited_ts"], as2d(d["pid_unlimited_ys"]), mask_beyond_end=True
    )
    y_rl = align_to_reference(t_ref, d["rl_ts"], as2d(d["rl_ys"]), mask_beyond_end=True)

    t_end = float(t_ref[-1])
    model = load_deeponet_msffn(
        ckpt_path,
        t_end,
        spec["branch_layers"],
        spec["trunk_layers"],
        num_outputs=state_dim,
    )
    y_pred = np.asarray(
        model.predict((np.array([[mu]], dtype=np.float32), t_ref.astype(np.float32)[:, None]))
    )
    y_dn = y_pred[0, :][:, None] if y_pred.ndim == 2 else y_pred[0]

    def scaled_error(y):
        """log10 scaled error, NaN where the trajectory is NaN (method stopped early)."""
        err = to_log10_clipped(pid_scaled_error(y, y_ref, atol, rtol))
        err[np.any(np.isnan(y), axis=-1)] = np.nan
        return err

    err_pid = scaled_error(y_pid)
    err_pidu = scaled_error(y_pidu)
    err_rl = scaled_error(y_rl)
    err_dn = scaled_error(y_dn)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- CSV ---
    columns = ["t"]
    values = [t_ref]
    for prefix, y, err in (
        ("PID", y_pid, err_pid),
        ("PIDu", y_pidu, err_pidu),
        ("RL", y_rl, err_rl),
        ("D", y_dn, err_dn),
    ):
        columns.extend(trajectory_column_names(prefix, state_dim))
        columns.append(f"{prefix}_err")
        values.extend(y[:, dim] for dim in range(state_dim))
        values.append(err)
    csv_path = os.path.join(PLOTS_DIR, f"{system}_pid_rl_deeponet_trajectory.csv")
    np.savetxt(
        csv_path,
        np.column_stack(values),
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt="%.10g",
    )
    print(f"[+] Saved: {csv_path}")

    # --- PNG: trajectories (one panel per state dim) and scaled local error ---
    fig, axs = plt.subplots(
        state_dim + 1,
        1,
        figsize=(9, 3.2 * (state_dim + 1)),
        sharex=True,
        gridspec_kw={"height_ratios": [1] * state_dim + [1.1]},
    )

    every = max(1, len(t_ref) // 15)

    def draw(ax, y_series_by_method, ylabel, legend=False):
        ax.plot(
            t_ref,
            y_series_by_method["reference"],
            color="black",
            lw=1.0,
            alpha=0.4,
            label="reference",
        )
        for name in ("PID", "PIDu", "RL", "D"):
            label = {"PID": "PID", "PIDu": "PID unlimited", "RL": "RL", "D": "DeepONet"}[name]
            alpha = 0.7 if name == "PIDu" else 1.0
            lw = 1.2 if name == "PIDu" else 1.6
            ax.plot(
                t_ref,
                y_series_by_method[name],
                _STYLES[name],
                color=_COLORS[name],
                lw=lw,
                alpha=alpha,
                marker=_MARKERS[name],
                markersize=3,
                markevery=every,
                label=label if legend else None,
            )
        ax.set_ylabel(ylabel)

    for dim in range(state_dim):
        draw(
            axs[dim],
            {
                "reference": y_ref[:, dim],
                "PID": y_pid[:, dim],
                "PIDu": y_pidu[:, dim],
                "RL": y_rl[:, dim],
                "D": y_dn[:, dim],
            },
            ylabel="y" if state_dim == 1 else f"y{dim}",
            legend=(dim == 0),
        )
    axs[0].legend(fontsize=8, loc="best")
    axs[0].set_title(f"{system} (task_param={mu:.3g}): PID / RL / DeepONet vs. reference")

    err_ax = axs[state_dim]
    for name in ("PID", "PIDu", "RL", "D"):
        err = {"PID": err_pid, "PIDu": err_pidu, "RL": err_rl, "D": err_dn}[name]
        alpha = 0.7 if name == "PIDu" else 1.0
        lw = 1.2 if name == "PIDu" else 1.6
        err_ax.plot(
            t_ref,
            err,
            _STYLES[name],
            color=_COLORS[name],
            lw=lw,
            alpha=alpha,
            marker=_MARKERS[name],
            markersize=3,
            markevery=every,
        )
    err_ax.set_xlabel("t")
    err_ax.set_ylabel(r"$\log_{10}$ PID-scaled error vs. reference")

    fig.tight_layout()
    png_path = os.path.join(PLOTS_DIR, f"{system}_pid_rl_deeponet_trajectory.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system", action="append", choices=list(SYSTEMS), help="repeatable; default = all systems"
    )
    args = parser.parse_args()
    for system in args.system or list(SYSTEMS):
        plot_one(system)


if __name__ == "__main__":
    main()
