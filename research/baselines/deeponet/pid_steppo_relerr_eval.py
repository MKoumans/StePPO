"""PID and StePPO error tables vs task parameter, on the same test set, bins
(N_BINS log-spaced) and metric as deeponet_relerr_eval.py, so all methods can
be overlaid.

Needs the JAX environment (solve_pid_batch, LearnedController, Hugging Face download):
    uv run python research/baselines/deeponet/pid_steppo_relerr_eval.py --system van_der_pol
    uv run python research/baselines/deeponet/pid_steppo_relerr_eval.py --system van_der_pol --skip-rl

Writes pid_eval_table_relerr.txt and <label>_eval_table_relerr.txt to
output-deeponet/<system>/results/, in the column layout of the DeepONet tables.
"""

import argparse
import os

import jax.numpy as jnp
import numpy as np
from overleaf_tables import align_to_reference, bin_stats, load_oracle_testset
from relerr_metrics import relative_l2_error, time_integrated_log10_rel_error, to_log10_clipped

from steppo.configs.base_config import TrainConfig, apply_precision, load_config_from_yaml
from steppo.envs.ode import ODEEnv  # noqa: F401 — import order avoids a circular import
from steppo.training.pid_solve import env_pid_controller, solve_pid_batch

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..", "..")
N_BINS = 64  # must match deeponet_relerr_eval.py

_SYSTEMS = {
    "scalar_decay": dict(
        config="configs/envs/ode/scalar_decay/scalar_decay_default.yaml",
        data_dir="data/scalar_decay/evaluation",
        out_dir="output-deeponet/scalar_decay",
        hf_repo="MKoumans/sd",
        hf_revision=None,
    ),
    "van_der_pol": dict(
        config="configs/envs/ode/van_der_pol/van_der_pol_default.yaml",
        data_dir="data/van_der_pol/evaluation",
        out_dir="output-deeponet/van_der_pol",
        hf_repo="MKoumans/vdp",
        hf_revision=None,
    ),
    "brusselator": dict(
        config="configs/envs/ode/brusselator/brusselator_default.yaml",
        data_dir="data/brusselator/evaluation",
        out_dir="output-deeponet/brusselator",
        hf_repo="MKoumans/bru",
        hf_revision=None,
    ),
}


def compute_errors(ts_all, ys_all, ref_ts, ref_ys, atol):
    """Per-sample (err_tend, err_integrated), as in deeponet_relerr_eval.py."""
    n = len(ref_ts)
    err_tend = np.full(n, np.nan)
    err_integrated = np.full(n, np.nan)
    for i in range(n):
        valid_ref = np.isfinite(ref_ts[i])
        if valid_ref.sum() < 2:
            continue
        t_ref = ref_ts[i][valid_ref].astype(np.float64)
        y_true = ref_ys[i][valid_ref]
        y_pred = align_to_reference(t_ref, ts_all[i], ys_all[i])
        if np.isnan(y_pred).any():
            continue
        err_tend[i] = to_log10_clipped(relative_l2_error(y_pred[-1:], y_true[-1:], atol))[0]
        err_integrated[i] = time_integrated_log10_rel_error(t_ref, y_pred, y_true, atol)
    return err_tend, err_integrated


def write_table(path, bin_centers, tend_mean, tend_std, integ_mean, integ_std, counts):
    with open(path, "w") as f:
        f.write(
            f"{'bin_center':>12} {'err_tend_log10rel_mean':>24} {'err_tend_log10rel_std':>23} "
            f"{'err_integ_log10rel_mean':>25} {'err_integ_log10rel_std':>24} {'count':>7}\n"
        )
        for i in range(len(bin_centers)):
            if counts[i] == 0:
                continue
            f.write(
                f"{bin_centers[i]:12.4f} {tend_mean[i]:24.4f} {tend_std[i]:23.4f} "
                f"{integ_mean[i]:25.4f} {integ_std[i]:24.4f} {counts[i]:7d}\n"
            )
    print(f"[+] Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=list(_SYSTEMS))
    parser.add_argument("--hf-repo", default=None, help="override the RL policy's HF repo id")
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--skip-rl", action="store_true", help="only compute the PID table")
    parser.add_argument(
        "--label",
        default="steppo",
        help="result filename label for the RL table",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    spec = _SYSTEMS[args.system]
    hf_repo = args.hf_repo or spec["hf_repo"]
    hf_revision = args.hf_revision if args.hf_revision is not None else spec["hf_revision"]

    config = load_config_from_yaml(TrainConfig, os.path.join(REPO_ROOT, spec["config"]))
    apply_precision(config)
    env_config = config.env
    atol = float(env_config.atol)
    max_steps = config.rollout_steps
    print(f"[*] system={args.system} atol={atol:.1e} max_steps={max_steps}")

    d = load_oracle_testset(os.path.join(REPO_ROOT, spec["data_dir"]))
    task_test = d["task_params"].astype(np.float64)
    episode_keys = jnp.asarray(d["episode_keys"])
    ref_ts, ref_ys = d["ref_ts"], d["ref_ys"]
    if ref_ys.ndim == 2:
        ref_ys = ref_ys[:, :, None]

    bin_edges = np.logspace(np.log10(task_test.min()), np.log10(task_test.max()), N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    out_dir = os.path.join(HERE, spec["out_dir"], "results")
    os.makedirs(out_dir, exist_ok=True)

    mu_batch = jnp.asarray(task_test, dtype=jnp.float32)

    # --- PID: diffrax PIDController with the training step budget ---
    pid_sc = env_pid_controller(env_config)
    print("[*] Solving PID over the full testset ...")
    pid_out = solve_pid_batch(
        env_config,
        pid_sc,
        mu_batch,
        episode_keys,
        max_steps,
        save_steps=True,
        save_ys=True,
        batch_size=args.batch_size,
    )
    pid_err_tend, pid_err_integ = compute_errors(pid_out["ts"], pid_out["ys"], ref_ts, ref_ys, atol)
    tend_mean, tend_std, counts = bin_stats(task_test, pid_err_tend, bin_edges)
    integ_mean, integ_std, _ = bin_stats(task_test, pid_err_integ, bin_edges)
    write_table(
        os.path.join(out_dir, "pid_eval_table_relerr.txt"),
        bin_centers,
        tend_mean,
        tend_std,
        integ_mean,
        integ_std,
        counts,
    )

    if args.skip_rl:
        return

    # --- StePPO: LearnedController from the Hugging Face artifact ---
    from steppo.models.huggingface.hub import download_model

    print(f"[*] Downloading RL policy: {hf_repo}{'@' + hf_revision if hf_revision else ''} ...")
    download_kwargs = {"revision": hf_revision} if hf_revision else {}
    artifact = download_model(hf_repo, **download_kwargs)
    assert artifact.config.env.system == args.system, (
        f"HF repo {hf_repo} is for system '{artifact.config.env.system}', not '{args.system}'"
    )
    print("[*] Solving StePPO over the full testset ...")
    rl_out = solve_pid_batch(
        env_config,
        artifact.controller,
        mu_batch,
        episode_keys,
        max_steps,
        save_steps=True,
        save_ys=True,
        batch_size=args.batch_size,
    )
    rl_err_tend, rl_err_integ = compute_errors(rl_out["ts"], rl_out["ys"], ref_ts, ref_ys, atol)
    tend_mean, tend_std, counts = bin_stats(task_test, rl_err_tend, bin_edges)
    integ_mean, integ_std, _ = bin_stats(task_test, rl_err_integ, bin_edges)
    write_table(
        os.path.join(out_dir, f"{args.label}_eval_table_relerr.txt"),
        bin_centers,
        tend_mean,
        tend_std,
        integ_mean,
        integ_std,
        counts,
    )


if __name__ == "__main__":
    main()
