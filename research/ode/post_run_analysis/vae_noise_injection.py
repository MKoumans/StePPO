"""Encoder sensitivity to a burst of observation noise.

For a few tasks, a clean rollout is re-encoded with noise spliced into the
observation at --spike_fracs of t_end (actions held fixed), and the belief
(mu, sigma) is compared with the clean one. Noise at ratio r is
obs + r * feature_std * eps, eps ~ N(0, 1).

Writes to --out: vae_noise_task{i}_<param><value>.png (belief traces),
vae_noise_dose_response.png (deviation vs ratio) and vae_noise_table.txt.

Usage:
    python research/ode/post_run_analysis/vae_noise_injection.py \\
        --config <config.yaml> --checkpoint <run>/checkpoints --num_samples 4 --out outputs/analysis
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import os

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv
from steppo.utils.checkpoint import build_models, resolve_checkpoint

try:
    from .analysis_common import (
        _BLUE,
        _ORANGE,
        _savefig,
        freeze_inactive,
        get_task_param,
        try_load_checkpoint,
    )
except ImportError:
    from analysis_common import (
        _BLUE,
        _ORANGE,
        _savefig,
        freeze_inactive,
        get_task_param,
        try_load_checkpoint,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse checkpoint, noise-spike, rollout, and output options."""
    p = argparse.ArgumentParser(description="ODE VAE input-noise injection analysis")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (e.g. configs/envs/ode/van_der_pol/van_der_pol_experiment.yml)",
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
        help="Override env.system (e.g. scalar_decay, van_der_pol, chemical_cascade)",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of distinct tasks to sample and plot individually",
    )
    p.add_argument("--steps", type=int, default=None, help="Override rollout_steps from config")
    p.add_argument(
        "--noise_ratios",
        nargs="*",
        type=float,
        default=None,
        help="White-noise mix ratios in [0,1] to sweep (default: 0.0 0.1 0.25 0.5 0.75 1.0)",
    )
    p.add_argument(
        "--spike_fracs",
        nargs="*",
        type=float,
        default=None,
        help="Fractions of t_end at which to inject a noise spike (default: 0.25 0.5 0.75)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", type=str, default="outputs/analysis", help="Output directory for PNG/txt files"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Clean rollout collection (records raw obs, unlike collect_analysis_rollout)
# ---------------------------------------------------------------------------


def collect_obs_rollout(vae, policy, env, env_params_batch, num_envs: int, T: int, rng_key):
    """Roll out T steps recording the raw observation; state freezes after done.

    Returns (T, num_envs, ...) arrays: obs, action, reward, state_t, active.
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
        ys = {
            "obs": final_obs,  # (E, obs_dim) — what the encoder saw
            "action": actions_enc,  # (E, action_dim)
            "reward": rewards.squeeze(-1),  # (E,)
            "state_t": final_env_state.t,  # (E,)
            "active": active,  # (E,) bool — True while step is "live"
        }
        return carry, ys

    init_carry = (obs0, env_state0, gru_hidden0, belief_mu0, belief_logvar0, done0)
    keys_scan = jax.random.split(rng_key, T)
    _, traces = jax.lax.scan(scan_step, init_carry, keys_scan)
    return traces


# ---------------------------------------------------------------------------
# Noise injection + re-encoding
# ---------------------------------------------------------------------------


def spike_step_indices(state_t: np.ndarray, t_end: float, spike_fracs: list[float]) -> list[int]:
    """First rollout-step index at which ODE time t crosses each frac*t_end."""
    idxs = []
    for frac in spike_fracs:
        target = frac * t_end
        hit = np.searchsorted(state_t, target, side="left")
        idxs.append(int(np.clip(hit, 0, len(state_t) - 1)))
    return idxs


def encode_clean_and_distorted(
    vae,
    obs: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    spike_idxs: list[int],
    noise_ratios: list[float],
    rng: np.random.Generator,
):
    """Re-encode a single task's trajectory under the normal obs and under
    obs distorted by noise spikes at spike_idxs, for each ratio. Returns
    (normal, distorted): each a dict (or {ratio: dict}) of "mu"/"sigma"
    arrays (T+1, latent_dim).
    """
    actions_j = jnp.asarray(action)
    rewards_j = jnp.asarray(reward)

    mu_n, logvar_n = vae.encoder.encode_trajectory(actions_j, jnp.asarray(obs), rewards_j)
    mu_n, logvar_n = np.array(mu_n), np.array(logvar_n)
    normal = {"mu": mu_n, "sigma": np.exp(0.5 * np.clip(logvar_n, -10, 10))}

    feature_std = obs.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1e-6, feature_std)

    # Fixed noise direction per spike so ratio purely scales magnitude.
    eps_per_spike = rng.standard_normal(size=(len(spike_idxs), obs.shape[-1]))

    distorted = {}
    for ratio in noise_ratios:
        if ratio == 0.0:
            distorted[ratio] = {"mu": mu_n.copy(), "sigma": normal["sigma"].copy()}
            continue
        obs_noisy = obs.copy()
        for spike_idx, eps in zip(spike_idxs, eps_per_spike):
            obs_noisy[spike_idx] = obs[spike_idx] + ratio * feature_std * eps
        mu_d, logvar_d = vae.encoder.encode_trajectory(actions_j, jnp.asarray(obs_noisy), rewards_j)
        mu_d, logvar_d = np.array(mu_d), np.array(logvar_d)
        distorted[ratio] = {"mu": mu_d, "sigma": np.exp(0.5 * np.clip(logvar_d, -10, 10))}

    return normal, distorted


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_task_trajectories(
    task_idx: int,
    param_label: str,
    param_value: float,
    normal: dict,
    distorted: dict,
    spike_idxs: list[int],
    noise_ratios: list[float],
    latent_dim: int,
    out_dir: str,
):
    """Plot clean and distorted belief trajectories for one task."""
    n_dims = min(latent_dim, 6)
    fig, axes = plt.subplots(n_dims, 2, figsize=(10, 2.6 * n_dims), squeeze=False)
    fig.suptitle(
        f"Task {task_idx}  ({param_label}={param_value:.3g}) — belief response to "
        f"VAE input-noise spikes",
        fontsize=13,
    )

    ratios_nonzero = [r for r in noise_ratios if r != 0.0]
    cmap = matplotlib.colormaps["plasma"]
    colors = {
        r: cmap(0.15 + 0.75 * i / max(len(ratios_nonzero) - 1, 1))
        for i, r in enumerate(ratios_nonzero)
    }

    steps = np.arange(normal["mu"].shape[0])  # 0 = prior, 1..T = posterior

    for d in range(n_dims):
        for col, key, ylabel, color in (
            (0, "mu", "μ value", _BLUE),
            (1, "sigma", "σ value", _ORANGE),
        ):
            ax = axes[d][col]
            ax.plot(
                steps, normal[key][:, d], "-", color=color, linewidth=2.0, label="normal", zorder=5
            )
            for ratio in ratios_nonzero:
                ax.plot(
                    steps,
                    distorted[ratio][key][:, d],
                    "--",
                    color=colors[ratio],
                    linewidth=1.3,
                    alpha=0.9,
                    label=f"r={ratio:g}",
                )
            for spike_idx in spike_idxs:
                ax.axvline(spike_idx + 1, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
            ax.set_title(f"z[{d}] — {key}", fontsize=10)
            ax.set_xlabel("Rollout step")
            ax.set_ylabel(ylabel)
            ax.grid(True, linewidth=0.4)
            if col == 1 and key == "sigma":
                ax.set_ylim(bottom=0)
            if d == 0 and col == 1:
                ax.legend(fontsize=7, loc="upper right", ncol=1)

    fig.tight_layout()
    fname = f"vae_noise_task{task_idx}_{param_label.replace(' ', '')}{param_value:.3g}.png"
    _savefig(fig, os.path.join(out_dir, fname))


def plot_dose_response(all_results: list[dict], noise_ratios: list[float], out_dir: str):
    """Plot belief deviation as a function of injected observation noise."""
    """Mean |Δmu| and mean |Δsigma| after the first spike, vs. ratio, per task."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("Belief deviation from normal vs. noise ratio", fontsize=13)

    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i / max(len(all_results) - 1, 1)) for i in range(len(all_results))]

    for res, color in zip(all_results, colors):
        label = f"task {res['task_idx']} ({res['param_label']}={res['param_value']:.3g})"
        axes[0].plot(res["ratios"], res["mu_dev"], "-o", color=color, label=label, markersize=4)
        axes[1].plot(res["ratios"], res["sigma_dev"], "-o", color=color, label=label, markersize=4)

    axes[0].set_title("Mean |Δμ| after spikes", fontsize=11)
    axes[1].set_title("Mean |Δσ| after spikes", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Noise ratio r")
        ax.grid(True, linewidth=0.4)
    axes[0].set_ylabel("|Δμ|")
    axes[1].set_ylabel("|Δσ|")
    axes[1].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "vae_noise_dose_response.png"))


def save_table(all_results: list[dict], spike_fracs: list[float], path: str):
    """Write VAE noise-injection summaries as a text table."""
    with open(path, "w") as f:
        for res in all_results:
            f.write(
                f"task {res['task_idx']}  {res['param_label']}={res['param_value']:.4g}"
                f"  spike_steps={res['spike_idxs']}  (fracs={spike_fracs})\n"
            )
            f.write(
                f"{'ratio':>6s} {'mu_dev':>10s} {'sigma_dev':>10s} "
                f"{'mu_dev_end':>11s} {'sigma_dev_end':>13s}\n"
            )
            for r, mu_dev, sig_dev, mu_end, sig_end in zip(
                res["ratios"],
                res["mu_dev"],
                res["sigma_dev"],
                res["mu_dev_end"],
                res["sigma_dev_end"],
            ):
                f.write(f"{r:6.2f} {mu_dev:10.4f} {sig_dev:10.4f} {mu_end:11.4f} {sig_end:13.4f}\n")
            f.write("\n")
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Reusable entry point (called standalone below, or from analyse_rollout.py)
# ---------------------------------------------------------------------------


def run_noise_injection_analysis(
    vae,
    policy,
    env,
    config,
    out_dir: str,
    num_samples: int = 4,
    noise_ratios: list[float] = None,
    spike_fracs: list[float] = None,
    seed: int = 0,
):
    """Sample `num_samples` tasks, roll out once, and produce the noise-spike
    trajectory figures + dose-response plot + table in `out_dir`. Reusable
    from both this script's __main__ and analyse_rollout.py's main().
    """
    T = config.rollout_steps
    noise_ratios = noise_ratios if noise_ratios else [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    spike_fracs = spike_fracs if spike_fracs else [0.25, 0.5, 0.75]
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[*] VAE noise injection: {num_samples} samples, ratios={noise_ratios}, "
        f"spike_fracs={spike_fracs} (of t_end={float(config.env.t_end)})"
    )

    rng = jax.random.PRNGKey(seed)
    rng, key_task, key_rollout = jax.random.split(rng, 3)
    task_keys = jax.random.split(key_task, num_samples)
    env_params_batch = jax.vmap(env.sample_task)(task_keys)
    true_params, param_label = get_task_param(env_params_batch, config.env.system)

    print("[*] Collecting clean rollout for all sampled tasks …")
    vae_graphdef, vae_params = nnx.split(vae)
    policy_graphdef, policy_params = nnx.split(policy)

    @jax.jit
    def run_rollout(vae_params, policy_params, env_params_batch, rng_key):
        vae_ = nnx.merge(vae_graphdef, vae_params)
        policy_ = nnx.merge(policy_graphdef, policy_params)
        return collect_obs_rollout(vae_, policy_, env, env_params_batch, num_samples, T, rng_key)

    traces = run_rollout(vae_params, policy_params, env_params_batch, key_rollout)
    traces = jax.tree.map(np.array, traces)
    print("[*] Rollout complete. Injecting noise and re-encoding per task …")

    t_end = float(config.env.t_end)
    latent_dim = vae.config.total_latent_dim
    np_rng = np.random.default_rng(seed)

    all_results = []
    for s in range(num_samples):
        active_s = traces["active"][:, s]
        valid_len = int(active_s.sum()) if active_s.any() else T
        valid_len = max(valid_len, 1)

        obs_s = traces["obs"][:valid_len, s]
        action_s = traces["action"][:valid_len, s]
        reward_s = traces["reward"][:valid_len, s]
        t_s = traces["state_t"][:valid_len, s]

        spike_idxs = spike_step_indices(t_s, t_end, spike_fracs)

        normal, distorted = encode_clean_and_distorted(
            vae,
            obs_s,
            action_s,
            reward_s,
            spike_idxs,
            noise_ratios,
            np_rng,
        )

        post_spike_start = min(spike_idxs) + 1  # +1: index 0 in mu/sigma arrays is the prior
        mu_dev, sigma_dev, mu_dev_end, sigma_dev_end = [], [], [], []
        for ratio in noise_ratios:
            d_mu = np.abs(
                distorted[ratio]["mu"][post_spike_start:] - normal["mu"][post_spike_start:]
            )
            d_sig = np.abs(
                distorted[ratio]["sigma"][post_spike_start:] - normal["sigma"][post_spike_start:]
            )
            mu_dev.append(float(d_mu.mean()))
            sigma_dev.append(float(d_sig.mean()))
            mu_dev_end.append(float(np.abs(distorted[ratio]["mu"][-1] - normal["mu"][-1]).mean()))
            sigma_dev_end.append(
                float(np.abs(distorted[ratio]["sigma"][-1] - normal["sigma"][-1]).mean())
            )

        param_value = float(true_params[s])
        plot_task_trajectories(
            s,
            param_label,
            param_value,
            normal,
            distorted,
            spike_idxs,
            noise_ratios,
            latent_dim,
            out_dir,
        )

        all_results.append(
            {
                "task_idx": s,
                "param_label": param_label,
                "param_value": param_value,
                "spike_idxs": spike_idxs,
                "ratios": noise_ratios,
                "mu_dev": mu_dev,
                "sigma_dev": sigma_dev,
                "mu_dev_end": mu_dev_end,
                "sigma_dev_end": sigma_dev_end,
            }
        )
        print(f"  task {s} ({param_label}={param_value:.3g}): spike steps {spike_idxs} done.")

    plot_dose_response(all_results, noise_ratios, out_dir)
    save_table(all_results, spike_fracs, os.path.join(out_dir, "vae_noise_table.txt"))


# ---------------------------------------------------------------------------
# Main (standalone CLI)
# ---------------------------------------------------------------------------


def main():
    """Run trajectory-level VAE observation-noise analysis."""
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
    if args.steps is not None:
        config.rollout_steps = args.steps

    seed = args.seed
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] System       : {config.env.system}")
    print(f"[*] Steps        : {config.rollout_steps}  Seed: {seed}")
    print(f"[*] Output       : {out_dir}")

    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, seed)

    if args.checkpoint is not None:
        vae, policy = try_load_checkpoint(
            vae, policy, args.checkpoint, backbone=config.backbone, algo=config.algo
        )
    else:
        print("[!] No checkpoint supplied — using random weights.")

    run_noise_injection_analysis(
        vae,
        policy,
        env,
        config,
        out_dir,
        num_samples=args.num_samples,
        noise_ratios=args.noise_ratios,
        spike_fracs=args.spike_fracs,
        seed=seed,
    )

    print(f"\n[+] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
