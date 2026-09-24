"""Latent (z) sensitivity analysis: sweeps mix fraction alpha mixing belief_mu/
belief_logvar towards standard-normal noise (mean and variance swept
independently), and plots return/episode-length/success-rate vs. alpha.

Usage:
    python research/ode/post_run_analysis/z_sensitivity.py \\
        --config configs/ode_default.yaml \\
        --checkpoint outputs/checkpoints/checkpoint_1000 \\
        --num_envs 128 --out outputs/analysis
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import os

import flax.nnx as nnx
import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import dataclasses

import matplotlib.pyplot as plt

from research.ode.post_run_analysis.analysis_common import try_load_checkpoint
from research.ode.post_run_analysis.rollout_cache import load_or_compute
from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv
from steppo.training.eval import eval_episode
from steppo.utils.checkpoint import build_models, resolve_checkpoint

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse checkpoint, latent-mixing, and output options."""
    p = argparse.ArgumentParser(description="ODE latent (z) sensitivity analysis")
    p.add_argument(
        "--config", type=str, default=None, help="YAML config file (e.g. configs/ode_default.yaml)"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Orbax checkpoint directory to load weights from",
    )
    p.add_argument(
        "--system",
        type=str,
        default=None,
        help="Override env.system (e.g. scalar_decay, van_der_pol)",
    )
    p.add_argument(
        "-n",
        "--num_envs",
        type=int,
        default=128,
        help="Tasks sampled per alpha value (averaged in plots)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override rollout_steps from config (eval max_steps)",
    )
    p.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=None,
        help="Mix-fraction grid to sweep, e.g. 0 0.1 0.25 0.5 0.75 1.0 "
        "(default: 11-point grid from 0 to 1)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", type=str, default="outputs/analysis", help="Output directory for PNG/txt files"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sweep rollout
# ---------------------------------------------------------------------------


def run_alpha_sweep(
    vae,
    policy,
    env,
    rng_key,
    num_envs: int,
    max_steps: int,
    alphas: list[float],
):
    """Runs both the mean-noise sweep (alpha_mu=alpha) and variance-noise sweep
    (alpha_logvar=alpha); returns a dict of per-sweep lists of per-alpha results."""
    vae_graphdef, vae_params = nnx.split(vae)
    policy_graphdef, policy_params = nnx.split(policy)
    has_t = hasattr(env, "t_end")

    @jax.jit
    def run_batch(vae_params, policy_params, keys, alpha_mu, alpha_logvar):
        vae_ = nnx.merge(vae_graphdef, vae_params)
        policy_ = nnx.merge(policy_graphdef, policy_params)

        def run_one(key):
            key_task, key_ep = jax.random.split(key)
            params = env.sample_task(key_task)
            return eval_episode(
                vae_,
                policy_,
                env,
                params,
                key_ep,
                max_steps=max_steps,
                episodes_per_trial=1,
                z_noise_alpha_mu=alpha_mu,
                z_noise_alpha_logvar=alpha_logvar,
            )

        return jax.vmap(run_one)(keys)

    def sweep_one(alpha_mu_fixed: bool):
        """True sweeps alpha_mu (alpha_logvar=0); False sweeps alpha_logvar."""
        rows = []
        for i, alpha in enumerate(alphas):
            keys = jax.random.split(jax.random.fold_in(rng_key, i), num_envs)
            alpha_mu, alpha_logvar = (alpha, 0.0) if alpha_mu_fixed else (0.0, alpha)
            batch = run_batch(vae_params, policy_params, keys, alpha_mu, alpha_logvar)
            batch = jax.tree.map(np.array, batch)

            row = {
                "alpha": alpha,
                "mean_return": float(batch["total_return"].mean()),
                "std_return": float(batch["total_return"].std()),
                "mean_length": float(batch["episode_length"].mean()),
            }
            if has_t:
                t_end = float(env.t_end)
                ep_t = batch["ep_t_reached"][:, 0]
                row["success_rate"] = float((ep_t >= t_end).mean())
            rows.append(row)
        return rows

    print("[*] Sweeping mean-noise (alpha_mu) ...")
    mean_sweep = sweep_one(alpha_mu_fixed=True)
    print("[*] Sweeping variance-noise (alpha_logvar) ...")
    logvar_sweep = sweep_one(alpha_mu_fixed=False)
    return {"mean_noise": mean_sweep, "logvar_noise": logvar_sweep}


# ---------------------------------------------------------------------------
# Plotting / reporting
# ---------------------------------------------------------------------------

_BLUE = "#2176AE"
_ORANGE = "#F4873C"


def plot_sensitivity(results: dict, out_dir: str, has_success: bool):
    """Plot return, episode length, and success versus latent mixing."""
    mean_rows = results["mean_noise"]
    logvar_rows = results["logvar_noise"]
    alphas = [r["alpha"] for r in mean_rows]

    n_panels = 3 if has_success else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.2))
    fig.suptitle("Policy sensitivity to belief (z) corruption", fontsize=13)

    def _plot(ax, key, ylabel, title):
        m = [r[key] for r in mean_rows]
        lv = [r[key] for r in logvar_rows]
        ax.plot(alphas, m, "-o", color=_BLUE, label="mean noise (μ)")
        ax.plot(alphas, lv, "-o", color=_ORANGE, label="variance noise (logvar)")
        ax.set_xlabel("Mix fraction α  (0=true belief, 1=pure noise)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, linewidth=0.4)
        ax.legend(fontsize=9)

    _plot(axes[0], "mean_return", "Mean return", "Return vs. corruption")
    _plot(axes[1], "mean_length", "Mean episode length (steps)", "Steps vs. corruption")
    if has_success:
        _plot(axes[2], "success_rate", "Success rate", "Success vs. corruption")

    fig.tight_layout()
    path = os.path.join(out_dir, "z_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def save_txt(results: dict, path: str):
    """Write latent-sensitivity results as a text table."""
    has_success = "success_rate" in results["mean_noise"][0]
    with open(path, "w") as f:
        for name, rows in results.items():
            f.write(f"{name}\n")
            hdr = f"{'alpha':>6s} {'return':>10s} {'ret_std':>10s} {'length':>10s}"
            if has_success:
                hdr += f" {'success':>8s}"
            f.write(hdr + "\n")
            for r in rows:
                line = (
                    f"{r['alpha']:6.2f} {r['mean_return']:10.3f} "
                    f"{r['std_return']:10.3f} {r['mean_length']:10.1f}"
                )
                if has_success:
                    line += f" {r['success_rate']:8.2f}"
                f.write(line + "\n")
            f.write("\n")
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run latent mean and variance sensitivity sweeps."""
    args = parse_args()

    if args.checkpoint is not None:
        args.checkpoint = resolve_checkpoint(args.checkpoint)

    if args.config is None and args.checkpoint is not None:
        run_dir = os.path.dirname(args.checkpoint)
        bundled = os.path.join(run_dir, "config.yaml")
        if os.path.isfile(bundled):
            args.config = bundled
            print(f"[*] Auto-detected config: {bundled}")
        else:
            raise SystemExit(
                "--config is required (no config.yaml found in checkpoint dir); "
                "pass --config explicitly for checkpoints from older runs."
            )

    if args.config is not None:
        config = load_config_from_yaml(TrainConfig, args.config, strict=False)
    else:
        config = TrainConfig()
    if args.system is not None:
        config.env.system = args.system

    max_steps = args.steps if args.steps is not None else config.rollout_steps
    alphas = args.alphas if args.alphas else list(np.linspace(0.0, 1.0, 11))
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] System : {config.env.system}")
    print(f"[*] Envs   : {args.num_envs}  Max steps: {max_steps}  Seed: {args.seed}")
    print(f"[*] Alphas : {[round(a, 2) for a in alphas]}")
    print(f"[*] Output : {out_dir}")

    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, args.seed)

    if args.checkpoint is not None:
        vae, policy = try_load_checkpoint(
            vae, policy, args.checkpoint, backbone=config.backbone, algo=config.algo
        )
    else:
        print("[!] No checkpoint supplied — using random weights.")

    rng = jax.random.PRNGKey(args.seed)
    cache_payload = {
        "checkpoint": args.checkpoint,
        "env_config": dataclasses.asdict(config.env),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "max_steps": max_steps,
        "alphas": [float(a) for a in alphas],
    }
    results = load_or_compute(
        "z_sensitivity",
        cache_payload,
        lambda: run_alpha_sweep(vae, policy, env, rng, args.num_envs, max_steps, alphas),
    )

    has_success = "success_rate" in results["mean_noise"][0]
    plot_sensitivity(results, out_dir, has_success)
    save_txt(results, os.path.join(out_dir, "z_sensitivity.txt"))

    print(f"\n[+] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
