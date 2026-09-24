"""Timing of the belief-variance collapse vs task parameter, in and out of distribution.

A task value is in distribution if it lies in one of config.env.train_bins. The
sweep draws log-uniformly over train_bins and test_bins. Writes to --out:
collapse_onset_vs_mu.png, sigma_id_vs_ood.png and task_vs_belief_id_ood.png.

Usage:
    python research/ode/post_run_analysis/collapse_vs_task.py \\
        --config <config.yaml> --checkpoint <run>/checkpoints \\
        -n 512 --log_bins --out outputs/analysis/collapse_vs_task
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import dataclasses
import os

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.ode.post_run_analysis.analysis_common import (
    _savefig,
    freeze_inactive,
    try_load_checkpoint,
)
from research.ode.post_run_analysis.rollout_cache import load_or_compute
from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv, ODEParams
from steppo.utils.checkpoint import build_models, resolve_checkpoint
from steppo.utils.task_params import task_label

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse checkpoint, task-range, and collapse-analysis options."""
    p = argparse.ArgumentParser(
        description="Belief variance collapse timing vs. task parameter (incl. OOD)"
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--system",
        type=str,
        default=None,
        help="Override env.system (e.g. scalar_decay, van_der_pol)",
    )
    p.add_argument(
        "-n",
        "--num_episodes",
        type=int,
        default=512,
        help="Total environments swept across the full mu range",
    )
    p.add_argument("--steps", type=int, default=None, help="Override rollout_steps from config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Collapse onset = first step where total belief variance "
        "drops to this fraction of its initial value",
    )
    p.add_argument("--bin_width", type=float, default=5.0, help="mu bin width (linear mode)")
    p.add_argument("--num_bins", type=int, default=32, help="Number of bins (log mode)")
    p.add_argument(
        "--log_bins",
        action="store_true",
        default=True,
        help="Use logarithmic bin spacing (default: on — mu is log-uniform sampled)",
    )
    p.add_argument(
        "--linear_bins",
        dest="log_bins",
        action="store_false",
        help="Use linear bin spacing instead",
    )
    p.add_argument("--out", type=str, default="outputs/analysis/collapse_vs_task")
    return p.parse_args()


def _param_label(system: str) -> str:
    label = task_label(system)
    if label == "param":
        raise ValueError(
            f"Unknown system '{system}' — add its display symbol to _PARAM_FIELDS "
            "in src/steppo/utils/figures/ode.py"
        )
    return label


def _is_in_bins(values: np.ndarray, bins) -> np.ndarray:
    """Set membership: whether each value falls in the union of closed [lo, hi]
    intervals in `bins` — pure set principle, no scalar min/max range check."""
    if not bins:
        return np.zeros_like(values, dtype=bool)
    lo, hi = np.asarray(bins, dtype=np.float64).T
    return ((values[:, None] >= lo[None, :]) & (values[:, None] <= hi[None, :])).any(axis=1)


def _bins_extent(*bin_groups) -> tuple[float, float]:
    """Overall [lo, hi] spanned by the union of every bin across the given groups."""
    all_bins = [b for group in bin_groups for b in group]
    if not all_bins:
        raise ValueError("no bins configured to sweep")
    los, his = zip(*all_bins)
    return float(min(los)), float(max(his))


def _complement_intervals(bins, lo: float, hi: float) -> list[tuple[float, float]]:
    """Gaps within [lo, hi] not covered by any interval in `bins` — the OOD region
    of the sweep, as a set of shaded intervals (there can be more than one when
    train_bins is disjoint)."""
    if not bins:
        return [(lo, hi)]
    gaps = []
    cursor = lo
    for b_lo, b_hi in sorted(bins):
        if b_lo > cursor:
            gaps.append((cursor, min(b_lo, hi)))
        cursor = max(cursor, b_hi)
    if cursor < hi:
        gaps.append((cursor, hi))
    return gaps


def _sample_mu_values(num_episodes, mu_lo, mu_hi, seed):
    """Sample mu values log-uniformly over [mu_lo, mu_hi]."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(mu_lo), np.log(mu_hi), size=num_episodes)).astype(np.float32)


# ---------------------------------------------------------------------------
# Rollout (belief-only — trimmed version of analyze_rollout.collect_analysis_rollout)
# ---------------------------------------------------------------------------


def collect_belief_rollout(vae, policy, env, env_params_batch, num_envs: int, T: int, rng_key):
    """Run T rollout steps, tracking only belief_mu/belief_logvar/active.

    Episodes are not reset on done — state and belief freeze so each trace
    covers a single contiguous ODE integration, matching analyze_rollout.py.
    """
    prior_mu, prior_logvar = vae.get_prior()

    keys_reset = jax.random.split(rng_key, num_envs)
    obs0, env_state0 = jax.vmap(env.reset)(keys_reset, env_params_batch)

    gru_hidden0 = vae.encoder.init_hidden((num_envs,))
    belief_mu0 = jnp.broadcast_to(prior_mu, (num_envs, vae.config.total_latent_dim))
    belief_logvar0 = jnp.broadcast_to(prior_logvar, (num_envs, vae.config.total_latent_dim))
    done0 = jnp.zeros(num_envs, dtype=jnp.bool_)

    action_space = policy.action_space
    action_dim = policy.action_dim

    _freeze = freeze_inactive

    def scan_step(carry, rng_key_i):
        obs, env_state, gru_hidden, belief_mu, belief_logvar, done_so_far = carry
        key_act, key_env = jax.random.split(rng_key_i)
        active = ~done_so_far

        z = jnp.concatenate([belief_mu, belief_logvar], axis=-1)
        key_acts = jax.random.split(key_act, num_envs)
        actions, _lp, _v = jax.vmap(lambda o, z_, k: policy.act(o, z_, k, deterministic=True))(
            obs, z, key_acts
        )

        key_envs = jax.random.split(key_env, num_envs)
        next_obs, next_env_state, rewards, dones, _ = jax.vmap(
            lambda k, s, a, p: env.step(k, s, a, p)
        )(key_envs, env_state, actions, env_params_batch)
        rewards = rewards.reshape(num_envs, 1)

        def freeze_tree(n, o):
            return jax.tree.map(lambda na, oa: _freeze(active, na, oa), n, o)

        final_obs = _freeze(active, next_obs, obs)
        final_env_state = freeze_tree(next_env_state, env_state)
        final_dones = jnp.logical_or(done_so_far, dones)

        if action_space == "discrete":
            actions_enc = jax.nn.one_hot(actions, action_dim).astype(jnp.float32)
        else:
            actions_enc = actions.astype(jnp.float32)

        new_mu, new_logvar, new_hidden = jax.vmap(
            lambda a, o, r, h: vae.encoder.encode_step(a, o, r, h)
        )(actions_enc, next_obs, rewards, gru_hidden)

        new_mu = _freeze(active, new_mu, belief_mu)
        new_logvar = _freeze(active, new_logvar, belief_logvar)
        new_hidden = jax.tree.map(lambda na, oa: _freeze(active, na, oa), new_hidden, gru_hidden)

        carry = (final_obs, final_env_state, new_hidden, new_mu, new_logvar, final_dones)
        ys = {"belief_mu": new_mu, "belief_logvar": new_logvar, "active": active}
        return carry, ys

    init_carry = (obs0, env_state0, gru_hidden0, belief_mu0, belief_logvar0, done0)
    keys_scan = jax.random.split(rng_key, T)
    _, traces = jax.lax.scan(scan_step, init_carry, keys_scan)
    return traces


# ---------------------------------------------------------------------------
# Collapse-onset metric
# ---------------------------------------------------------------------------


def compute_collapse_onset(belief_logvar: np.ndarray, threshold_frac: float):
    """First rollout step where total belief variance drops to `threshold_frac`
    of its initial value. belief_logvar: (T, E, latent_dim). Returns
    (onset_step (E,) int, var_sum (T, E)); envs that never cross the
    threshold get onset_step = T (sentinel).
    """
    var_sum = np.exp(np.clip(belief_logvar, -10, 10)).sum(axis=-1)  # (T, E)
    T = var_sum.shape[0]
    threshold = threshold_frac * var_sum[0]  # (E,) — prior is broadcast, so ~constant
    crossed = var_sum <= threshold[None, :]  # (T, E)
    has_crossed = crossed.any(axis=0)
    onset = np.where(has_crossed, crossed.argmax(axis=0), T)
    return onset, var_sum


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_ID_COLOR = "#2176AE"
_OOD_COLOR = "#D7263D"


def make_bin_edges(mu_lo, mu_hi, bin_width, num_bins, log_bins):
    if log_bins:
        return np.geomspace(mu_lo, mu_hi, num_bins + 1)
    return np.arange(mu_lo, mu_hi + bin_width, bin_width)


def plot_collapse_onset_vs_mu(
    mu_values,
    onset,
    train_bins,
    mu_lo,
    mu_hi,
    bin_width,
    num_bins,
    log_bins,
    param_label,
    out_dir,
):
    """Binned mean±std collapse-onset step vs. mu — same visual language as
    steps_vs_mu.py's histogram, but a single series (no PID/RL split).
    train_bins boundaries are dashed (one pair per interval, since train_bins
    can be disjoint); OOD gaps (_complement_intervals) are shaded."""
    bin_edges = make_bin_edges(mu_lo, mu_hi, bin_width, num_bins, log_bins)
    bin_centers = (
        np.sqrt(bin_edges[:-1] * bin_edges[1:])
        if log_bins
        else (bin_edges[:-1] + bin_edges[1:]) / 2
    )
    bar_widths = (bin_edges[1:] - bin_edges[:-1]) * 0.9

    bin_idx = np.digitize(mu_values, bin_edges) - 1
    means, stds = [], []
    for i in range(len(bin_edges) - 1):
        mask = bin_idx == i
        if mask.any():
            means.append(onset[mask].mean())
            stds.append(onset[mask].std())
        else:
            means.append(0.0)
            stds.append(0.0)
    means, stds = np.array(means), np.array(stds)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        bin_centers,
        means,
        bar_widths,
        yerr=stds,
        capsize=2,
        color="#6A4C93",
        alpha=0.65,
        label="Collapse onset",
    )

    train_label = "Train bins " + ", ".join(
        f"[{lo:.3g}, {hi:.3g}]" for lo, hi in sorted(train_bins)
    )
    for i, (lo, hi) in enumerate(sorted(train_bins)):
        ax.axvline(
            lo,
            color="k",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=train_label if i == 0 else None,
        )
        ax.axvline(hi, color="k", linestyle="--", linewidth=1.5, alpha=0.7)
    for gap_lo, gap_hi in _complement_intervals(train_bins, mu_lo, mu_hi):
        ax.axvspan(gap_lo, gap_hi, color="grey", alpha=0.08)

    if log_bins:
        ax.set_xscale("log")
    ax.set_xlabel(param_label, fontsize=13)
    ax.set_ylabel("Collapse onset (rollout step)", fontsize=13)
    ax.set_title(
        f"Belief variance collapse onset vs {param_label} (stiffness parameter)", fontsize=14
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    if not log_bins:
        tick_positions = bin_centers[:: max(1, len(bin_centers) // 20)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{v:.3g}" for v in tick_positions])

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "collapse_onset_vs_mu.png"))


def plot_sigma_id_vs_ood(var_sum, is_id, out_dir):
    """Plot belief-variance collapse for in-distribution and OOD tasks."""
    T = var_sum.shape[0]
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 4))
    for mask, color, label in (
        (is_id, _ID_COLOR, "In-distribution"),
        (~is_id, _OOD_COLOR, "Out-of-distribution"),
    ):
        if not mask.any():
            continue
        sub = var_sum[:, mask]
        mean = sub.mean(axis=1)
        std = sub.std(axis=1)
        ax.plot(steps, mean, color=color, linewidth=1.8, label=f"{label} (n={mask.sum()})")
        ax.fill_between(steps, np.maximum(mean - std, 0), mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Total belief variance (sum over latent dims)")
    ax.set_title("Belief variance over rollout — ID vs. OOD", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "sigma_id_vs_ood.png"))


def plot_task_vs_belief_id_ood(belief_mu, active, mu_values, is_id, out_dir, n_dims_max=3):
    """Plot task parameter against final belief means by distribution."""
    T, E, _ = belief_mu.shape
    last_step = np.maximum(active.sum(axis=0).astype(int) - 1, 0)
    final_mu = belief_mu[last_step, np.arange(E), :]
    n_dims = min(belief_mu.shape[-1], n_dims_max)

    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 4), squeeze=False)
    fig.suptitle("Belief μ (episode end) vs. true task param — ID vs. OOD", fontsize=13)

    for d in range(n_dims):
        ax = axes[0][d]
        ax.scatter(
            mu_values[is_id],
            final_mu[is_id, d],
            c=_ID_COLOR,
            s=35,
            alpha=0.75,
            edgecolors="none",
            label="In-distribution",
        )
        ax.scatter(
            mu_values[~is_id],
            final_mu[~is_id, d],
            c=_OOD_COLOR,
            s=40,
            alpha=0.8,
            marker="^",
            edgecolors="none",
            label="Out-of-distribution",
        )

        r_id = _safe_corr(mu_values[is_id], final_mu[is_id, d])
        r_ood = _safe_corr(mu_values[~is_id], final_mu[~is_id, d])
        ax.set_title(f"z[{d}]  (r_id={r_id:+.3f}, r_ood={r_ood:+.3f})", fontsize=10)
        ax.set_xlabel("True param")
        ax.set_ylabel(f"belief_mu[{d}]")
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "task_vs_belief_id_ood.png"))


def _safe_corr(x, y):
    if len(x) < 2 or x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run belief-collapse analysis across trained and OOD task values."""
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

    config = (
        load_config_from_yaml(TrainConfig, args.config, strict=False)
        if args.config
        else TrainConfig()
    )
    if args.system is not None:
        config.env.system = args.system
    if args.steps is not None:
        config.rollout_steps = args.steps

    T = config.rollout_steps
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    train_bins = list(config.env.train_bins)
    test_bins = list(config.env.test_bins)
    if not train_bins:
        raise SystemExit(
            f"config.env.train_bins is empty for system '{config.env.system}' — "
            "nothing to define in-distribution against."
        )
    param_label = _param_label(config.env.system)
    mu_lo, mu_hi = _bins_extent(train_bins, test_bins)

    print(f"[*] System      : {config.env.system}")
    print(f"[*] Episodes    : {args.num_episodes}  Steps: {T}  Seed: {args.seed}")
    print(f"[*] Train {param_label:1s} bins  : {train_bins}")
    print(
        f"[*] Sweep {param_label:1s} range : [{mu_lo:.4g}, {mu_hi:.4g}] (union of train_bins + test_bins)"
    )
    print(f"[*] Threshold   : {args.threshold} (fraction of initial variance)")
    print(f"[*] Output      : {out_dir}")

    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, args.seed)
    if args.checkpoint is not None:
        vae, policy = try_load_checkpoint(
            vae, policy, args.checkpoint, backbone=config.backbone, algo=config.algo
        )
    else:
        print("[!] No checkpoint supplied — using random weights.")

    mu_values = _sample_mu_values(args.num_episodes, mu_lo, mu_hi, args.seed)
    is_id = _is_in_bins(mu_values, train_bins)
    env_params_batch = ODEParams(
        lam=jnp.asarray(mu_values),
        pulse_phase=jnp.zeros_like(jnp.asarray(mu_values)),
        max_steps=T,
    )

    vae_graphdef, vae_params = nnx.split(vae)
    policy_graphdef, policy_params = nnx.split(policy)

    @jax.jit
    def run_rollout(vae_params, policy_params, env_params_batch, rng_key):
        vae_ = nnx.merge(vae_graphdef, vae_params)
        policy_ = nnx.merge(policy_graphdef, policy_params)
        return collect_belief_rollout(
            vae_, policy_, env, env_params_batch, args.num_episodes, T, rng_key
        )

    print("[*] Running rollout sweep …")
    rng = jax.random.PRNGKey(args.seed)
    cache_payload = {
        "checkpoint": args.checkpoint,
        "env_config": dataclasses.asdict(config.env),
        "num_episodes": args.num_episodes,
        "steps": T,
        "seed": args.seed,
    }
    traces = load_or_compute(
        "collapse_vs_task",
        cache_payload,
        lambda: jax.tree.map(
            np.array, run_rollout(vae_params, policy_params, env_params_batch, rng)
        ),
    )

    onset, var_sum = compute_collapse_onset(traces["belief_logvar"], args.threshold)

    n_never = int((onset == T).sum())
    n_id, n_ood = int(is_id.sum()), int((~is_id).sum())
    if n_id:
        print(
            f"[*] Collapse onset — ID  (n={n_id}):  mean={onset[is_id].mean():.1f}  "
            f"median={np.median(onset[is_id]):.1f}"
        )
    else:
        print("[*] Collapse onset — ID  (n=0): no swept points fell inside train_bins")
    if n_ood:
        print(
            f"[*] Collapse onset — OOD (n={n_ood}): mean={onset[~is_id].mean():.1f}  "
            f"median={np.median(onset[~is_id]):.1f}"
        )
    else:
        print("[*] Collapse onset — OOD (n=0): no swept points fell outside train_bins")
    print(f"[*] Never collapsed: {n_never}/{args.num_episodes}")

    print("[*] Generating plots …")
    plot_collapse_onset_vs_mu(
        mu_values,
        onset,
        train_bins,
        mu_lo,
        mu_hi,
        args.bin_width,
        args.num_bins,
        args.log_bins,
        param_label,
        out_dir,
    )
    plot_sigma_id_vs_ood(var_sum, is_id, out_dir)
    plot_task_vs_belief_id_ood(traces["belief_mu"], traces["active"], mu_values, is_id, out_dir)

    print(f"\n[+] All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
