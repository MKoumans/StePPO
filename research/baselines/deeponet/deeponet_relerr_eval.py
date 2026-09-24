"""Re-evaluate a trained DeepONet checkpoint in the same error metric as the
PID/RL/Oracle pipeline: log10 of the atol-clipped relative L2 error
(relerr_metrics.py), on the same cached test set.

Run in the DeepONet container:
    python deeponet_relerr_eval.py --system van_der_pol --profile retrain
"""

import argparse
import glob
import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np
import yaml
from overleaf_tables import bin_stats, load_oracle_testset
from relerr_metrics import log10_rel_error, time_integrated_log10_rel_error

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..", "..")

_SYSTEMS = {
    "scalar_decay": dict(
        config="configs/envs/ode/scalar_decay/scalar_decay_default.yaml",
        data_dir="data/scalar_decay/evaluation",
        out_dir="output-deeponet/scalar_decay",
        legacy_ckpt_glob="output-deeponet/scalar_decay/models/deeponet_scalar_decay_paper_final-*.pt",
        retrain_ckpt_glob="output-deeponet/scalar_decay/models/deeponet_scalar_decay_retrain_complete_final-*.pt",
        param_label="lambda",
        net=dict(branch=[1, 40, 40], trunk=[1, 40, 40, 40], multi_output=False),
        retrain_net=dict(
            branch=[1, 128, 128], trunk=[1, 128, 128, 128], multi_output=False, msffn_sigmas=2
        ),
    ),
    "van_der_pol": dict(
        config="configs/envs/ode/van_der_pol/van_der_pol_default.yaml",
        data_dir="data/van_der_pol/evaluation",
        out_dir="output-deeponet/van_der_pol",
        legacy_ckpt_glob="output-deeponet/van_der_pol/models/deeponet_van_der_pol_complete_final-*.pt",
        retrain_ckpt_glob="output-deeponet/van_der_pol/models/deeponet_van_der_pol_retrain_complete_final-*.pt",
        param_label="mu",
        net=dict(
            branch=[1, 256, 256, 256, 256, 256],
            trunk=[1, 256, 256, 256, 256, 256],
            multi_output=True,
        ),
        retrain_net=dict(
            branch=[1, 256, 256, 256, 256, 256],
            trunk=[1, 256, 256, 256, 256, 256],
            multi_output=True,
            msffn_sigmas=2,
        ),
    ),
    "brusselator": dict(
        config="configs/envs/ode/brusselator/brusselator_default.yaml",
        data_dir="data/brusselator/evaluation",
        out_dir="output-deeponet/brusselator",
        legacy_ckpt_glob="output-deeponet/brusselator/models/deeponet_brusselator_eval_final-*.pt",
        retrain_ckpt_glob="output-deeponet/brusselator/models/deeponet_brusselator_retrain_complete_final-*.pt",
        param_label="B",
        net=dict(branch=[1, 128, 128, 128, 128], trunk=[1, 128, 128, 128, 128], multi_output=True),
        retrain_net=dict(
            branch=[1, 256, 256, 256, 256, 256],
            trunk=[1, 256, 256, 256, 256, 256],
            multi_output=True,
            msffn_sigmas=2,
        ),
    ),
}

N_BINS = 64


def load_model_for_system(
    system: str, ref_ys_ndim3: bool = True, profile: str = "retrain", checkpoint: str | None = None
):
    """Rebuild the DeepONet for `system` and restore its final checkpoint; returns (model, atol, t_end)."""
    import deepxde as dde

    spec = _SYSTEMS[system]
    with open(os.path.join(REPO_ROOT, spec["config"])) as f:
        cfg = yaml.safe_load(f)["env"]
    atol = float(cfg["atol"])
    t_end = float(cfg["t_end"])

    branch_dim = 1
    if profile not in ("legacy", "retrain"):
        raise ValueError(f"Unknown profile: {profile}")
    net_spec = spec["retrain_net"] if profile == "retrain" else spec["net"]
    t_grid_col = np.linspace(0.0, t_end, 200)[:, None].astype(np.float32)
    dummy_branch = np.zeros((1, branch_dim), dtype=np.float32)
    # DeepXDE expects y as (n_branch, n_t) for single-output nets and
    # (n_branch, n_t, dim) for multi-output ones; restore validates the shape.
    if net_spec["multi_output"]:
        dummy_y = np.zeros((1, len(t_grid_col), 2 if ref_ys_ndim3 else 1), dtype=np.float32)
    else:
        dummy_y = np.zeros((1, len(t_grid_col)), dtype=np.float32)
    data = dde.data.TripleCartesianProd(
        X_train=(dummy_branch, t_grid_col),
        y_train=dummy_y,
        X_test=(dummy_branch, t_grid_col),
        y_test=dummy_y,
    )
    if net_spec["multi_output"]:
        net = dde.nn.DeepONetCartesianProd(
            net_spec["branch"],
            net_spec["trunk"],
            "relu",
            "Glorot normal",
            num_outputs=2,
            multi_output_strategy="independent",
        )
    else:
        net = dde.nn.DeepONetCartesianProd(
            net_spec["branch"], net_spec["trunk"], "relu", "Glorot normal"
        )
    if net_spec.get("msffn_sigmas"):
        # Retrain checkpoints use the MsFFN trunk, so swap it in before restoring.
        import torch
        from train_deeponet_unified import _build_msffn_trunk

        activation_fn = dde.nn.activations.get("relu")
        n_sigmas = net_spec["msffn_sigmas"]
        # Sigma values only initialise buffers that restore() overwrites.
        placeholder_sigmas = [1.0] * n_sigmas
        if net_spec["multi_output"]:
            net.trunk = torch.nn.ModuleList(
                _build_msffn_trunk(net_spec["trunk"], activation_fn, placeholder_sigmas)
                for _ in range(2)
            )
        else:
            net.trunk = _build_msffn_trunk(net_spec["trunk"], activation_fn, placeholder_sigmas)
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)

    if checkpoint is not None:
        ckpt = checkpoint if os.path.isabs(checkpoint) else os.path.join(REPO_ROOT, checkpoint)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt}")
    else:
        ckpt_glob = spec["retrain_ckpt_glob"] if profile == "retrain" else spec["legacy_ckpt_glob"]
        # ckpt_glob is relative to this directory (output-deeponet/).
        ckpt_candidates = glob.glob(os.path.join(HERE, ckpt_glob))
        if not ckpt_candidates:
            raise FileNotFoundError(f"No final {profile} checkpoint found for {ckpt_glob}")
        # Use the most recently written checkpoint, not the last filename.
        ckpt = max(ckpt_candidates, key=os.path.getmtime)
    model.restore(ckpt, verbose=1)
    print(f"[*] Checkpoint hersteld: {ckpt}")
    return model, atol, t_end


def predict_at(model, system: str, task_param: float, t_valid: np.ndarray) -> np.ndarray:
    """Query the DeepONet at arbitrary times for one task parameter; returns (T, dim)."""
    y_pred = model.predict(
        (np.array([[task_param]], dtype=np.float32), t_valid.astype(np.float32)[:, None])
    )
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2:
        return y_pred[0, :][:, None]
    return y_pred[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=list(_SYSTEMS))
    parser.add_argument(
        "--profile",
        choices=["retrain", "legacy"],
        default="retrain",
        help="checkpoint architecture and default filename pattern",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="optional explicit checkpoint path relative to the repository root",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="result filename label; useful for separating complete and binned runs",
    )
    args = parser.parse_args()
    spec = _SYSTEMS[args.system]
    result_label = args.label or args.profile

    with open(os.path.join(REPO_ROOT, spec["config"])) as f:
        cfg = yaml.safe_load(f)["env"]
    atol = float(cfg["atol"])
    print(f"[*] System={args.system}  atol={atol:.1e}")

    d = load_oracle_testset(os.path.join(REPO_ROOT, spec["data_dir"]))
    task_test = d["task_params"].astype(np.float64)
    ref_ts = d["ref_ts"]
    ref_ys = d["ref_ys"]
    if ref_ys.ndim == 2:
        ref_ys = ref_ys[:, :, None]  # scalar_decay: (N, T) -> (N, T, 1)

    model, _, _ = load_model_for_system(
        args.system, profile=args.profile, checkpoint=args.checkpoint
    )

    # --- Per-sample log10 relative error on the reference time points ---
    n = len(task_test)
    err_tend = np.full(n, np.nan)
    err_integrated = np.full(n, np.nan)
    for i in range(n):
        valid = np.isfinite(ref_ts[i])
        if valid.sum() < 2:
            continue
        t_valid = ref_ts[i][valid].astype(np.float32)
        y_true = ref_ys[i][valid]
        y_pred = model.predict((np.array([[task_test[i]]], dtype=np.float32), t_valid[:, None]))
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 2:
            y_pred = y_pred[0, :][:, None]  # single-output: (1, T) -> (T, 1)
        else:
            y_pred = y_pred[0]  # multi-output: (1, T, dim) -> (T, dim)
        err_tend[i] = log10_rel_error(y_pred[-1:], y_true[-1:], atol)[0]
        err_integrated[i] = time_integrated_log10_rel_error(
            t_valid.astype(np.float64), y_pred, y_true, atol
        )

    print(
        f"[+] log10 relative error (last step): mean={np.nanmean(err_tend):.3f}, "
        f"sum={np.nansum(err_tend):.3f}, n={np.isfinite(err_tend).sum()}"
    )
    print(
        f"[+] log10 relative error (time-integrated): mean={np.nanmean(err_integrated):.3f}, "
        f"sum={np.nansum(err_integrated):.3f}"
    )

    # --- Log-spaced bins for the comparison plot ---
    bin_edges = np.logspace(np.log10(task_test.min()), np.log10(task_test.max()), N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    tend_mean, tend_std, counts = bin_stats(task_test, err_tend, bin_edges)
    integ_mean, integ_std, _ = bin_stats(task_test, err_integrated, bin_edges)

    out_dir = os.path.join(HERE, spec["out_dir"], "results")
    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(
        out_dir, f"deeponet_{args.system}_{result_label}_eval_table_relerr.txt"
    )
    with open(table_path, "w") as f:
        f.write(
            f"{'bin_center':>12} {'err_tend_log10rel_mean':>24} {'err_tend_log10rel_std':>23} "
            f"{'err_integ_log10rel_mean':>25} {'err_integ_log10rel_std':>24} {'count':>7}\n"
        )
        for i in range(N_BINS):
            if counts[i] == 0:
                continue
            f.write(
                f"{bin_centers[i]:12.4f} {tend_mean[i]:24.4f} {tend_std[i]:23.4f} "
                f"{integ_mean[i]:25.4f} {integ_std[i]:24.4f} {counts[i]:7d}\n"
            )
    print(f"[+] Saved: {table_path}")

    summary_path = os.path.join(
        out_dir, f"deeponet_{args.system}_{result_label}_relerr_summary.npz"
    )
    np.savez(
        summary_path,
        task_test=task_test,
        err_tend_log10rel=err_tend,
        err_integrated_log10rel=err_integrated,
        atol=atol,
    )
    print(f"[+] Saved: {summary_path}")


if __name__ == "__main__":
    main()
