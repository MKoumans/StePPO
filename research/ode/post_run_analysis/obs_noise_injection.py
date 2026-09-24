"""Observation-noise sensitivity of the policy's action, compared across models.

Each model is rolled out once without noise on one task, recording the inputs
of every step. A second pass replays each step with noise added to that step's
observation only (re-running the encoder) and measures |Δaction| against the
action actually taken; the executed trajectory is never perturbed. Noise at
ratio r is obs + r * feature_std * eps, eps ~ N(0, 1), with feature_std from
the model's clean trajectory.

Writes to --out: obs_noise_comparison_<param><value>.png (|Δaction| per step
and the state trajectory, one column per model) and obs_noise_table.txt.

Usage:
    python research/ode/post_run_analysis/obs_noise_injection.py \\
        --model <config.yaml>:<checkpoint>:<label> [--model ...] --out outputs/analysis
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
    from .analysis_common import _savefig, get_task_param, try_load_checkpoint
except ImportError:
    from analysis_common import _savefig, get_task_param, try_load_checkpoint


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_model_spec(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"--model must be 'config.yaml:checkpoint_dir[:label]', got: {spec!r}"
        )
    config_path, checkpoint_dir = parts[0], parts[1]
    label = parts[2] if len(parts) > 2 else os.path.basename(checkpoint_dir.rstrip("/"))
    return {"config": config_path, "checkpoint": checkpoint_dir, "label": label}


def parse_args():
    """Parse model, noise, rollout, and output options."""
    p = argparse.ArgumentParser(
        description="ODE observation-noise local action sensitivity + trajectory comparison"
    )
    p.add_argument(
        "--model",
        action="append",
        type=_parse_model_spec,
        dest="models",
        required=True,
        help="config.yaml:checkpoint_dir[:label] — repeatable, one per model to compare",
    )
    p.add_argument(
        "--noise_ratios",
        nargs="*",
        type=float,
        default=None,
        help="White-noise mix ratios in [0,1] to sweep (default: 0.0 0.1 0.25 0.5 0.75 1.0)",
    )
    p.add_argument("--steps", type=int, default=None, help="Override rollout_steps from config")
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for task sampling and noise generation (shared across models)",
    )
    p.add_argument(
        "--out", type=str, default="outputs/analysis", help="Output directory for PNG/txt files"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Clean rollout, recording the inputs that produced each step's belief and action.
# ---------------------------------------------------------------------------


def collect_probe_rollout(vae, policy, env, env_params, T: int, rng_key):
    """Single-environment rollout; returns (T, ...) arrays, including the observation
    (`obs_pre`) and encoder state (`gru_hidden_pre`) that fed each step.
    """
    prior_mu, prior_logvar = vae.get_prior()

    obs0, env_state0 = env.reset(rng_key, env_params)
    gru_hidden0 = vae.encoder.init_hidden(())
    done0 = jnp.bool_(False)

    action_space = policy.action_space
    action_dim = policy.action_dim

    def scan_step(carry, rng_key_i):
        obs, env_state, gru_hidden, belief_mu, belief_logvar, done_so_far = carry
        key_act, key_env = jax.random.split(rng_key_i)
        active = ~done_so_far

        z = jnp.concatenate([belief_mu, belief_logvar], axis=-1)
        action, _lp, _v = policy.act(obs, z, key_act, deterministic=True)

        next_obs, next_env_state, reward, done, _ = env.step(key_env, env_state, action, env_params)

        final_obs = jnp.where(active, next_obs, obs)
        final_env_state = jax.tree.map(
            lambda na, oa: jnp.where(active, na, oa), next_env_state, env_state
        )
        final_done = jnp.logical_or(done_so_far, done)

        if action_space == "discrete":
            action_enc = jax.nn.one_hot(action, action_dim).astype(jnp.float32)
        else:
            action_enc = action.astype(jnp.float32)

        new_mu, new_logvar, new_hidden = vae.encoder.encode_step(
            action_enc, next_obs, reward, gru_hidden
        )
        new_mu = jnp.where(active, new_mu, belief_mu)
        new_logvar = jnp.where(active, new_logvar, belief_logvar)
        # gru_hidden may be a pytree, not a bare array.
        new_hidden = jax.tree.map(lambda na, oa: jnp.where(active, na, oa), new_hidden, gru_hidden)

        carry_out = (final_obs, final_env_state, new_hidden, new_mu, new_logvar, final_done)
        ys = {
            "obs_pre": obs,  # (obs_dim,) — carry-in obs, fed this step's belief+action
            "gru_hidden_pre": gru_hidden,  # (hidden,) — carry-in hidden, fed this step's own encode
            "actions_enc": action_enc,  # (action_dim,)
            "action": action,  # (action_dim,) raw action (pre one-hot)
            "reward": reward,  # scalar
            "belief_mu": belief_mu,  # (latent_dim,) — carry-in z used for this step's action
            "belief_logvar": belief_logvar,
            "state_t": final_env_state.t,
            "state_y": final_env_state.y,
            "keep_step": next_env_state.last_keep_step,
            "active": active,
        }
        return carry_out, ys

    init_carry = (obs0, env_state0, gru_hidden0, prior_mu, prior_logvar, done0)
    keys_scan = jax.random.split(rng_key, T)
    _, traces = jax.lax.scan(scan_step, init_carry, keys_scan)
    traces["prior_mu"] = prior_mu
    traces["prior_logvar"] = prior_logvar
    return traces


# ---------------------------------------------------------------------------
# Local action-sensitivity probe
# ---------------------------------------------------------------------------


def compute_action_deviation(
    vae, policy, traces: dict, noise_ratios: list[float], rng: np.random.Generator
):
    """Per noise ratio, the mean |Δaction| at every step when that step's observation is noisy."""
    obs_pre = np.asarray(traces["obs_pre"])  # (T, obs_dim)
    gru_hidden_pre = traces["gru_hidden_pre"]  # array, or pytree of arrays — leading axis T
    actions_enc = np.asarray(traces["actions_enc"])
    reward = np.asarray(traces["reward"])
    action_actual = np.asarray(traces["action"])  # (T, action_dim) or (T,) for discrete
    prior_mu = np.asarray(traces["prior_mu"])
    prior_logvar = np.asarray(traces["prior_logvar"])
    T = obs_pre.shape[0]

    if action_actual.ndim == 1:
        action_actual = action_actual[:, None]

    feature_std = obs_pre.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1e-6, feature_std)
    eps = rng.standard_normal(size=(T, obs_pre.shape[-1]))

    def act_deterministic(obs, z):
        a, _lp, _v = policy.act(
            jnp.asarray(obs), jnp.asarray(z), jax.random.PRNGKey(0), deterministic=True
        )
        a = np.asarray(a)
        return a[None] if a.ndim == 0 else a

    deviations = {}
    for ratio in noise_ratios:
        if ratio == 0.0:
            deviations[ratio] = np.zeros(T)
            continue

        obs_noisy = obs_pre + ratio * feature_std * eps  # (T, obs_dim)
        actions_noisy = np.zeros_like(action_actual)

        # Step 0: z is the fixed prior (independent of obs), no encode needed.
        z0 = np.concatenate([prior_mu, prior_logvar])
        actions_noisy[0] = act_deterministic(obs_noisy[0], z0)

        if T > 1:

            def probe_step(obs_i_noisy, prev_action_enc, prev_reward, prev_hidden):
                mu, logvar, _ = vae.encoder.encode_step(
                    prev_action_enc,
                    obs_i_noisy,
                    prev_reward,
                    prev_hidden,
                )
                z = jnp.concatenate([mu, logvar], axis=-1)
                a, _lp, _v = policy.act(obs_i_noisy, z, jax.random.PRNGKey(0), deterministic=True)
                return a

            gru_hidden_pre_head = jax.tree.map(lambda a: jnp.asarray(a)[:-1], gru_hidden_pre)
            actions_1_to_end = jax.vmap(probe_step)(
                jnp.asarray(obs_noisy[1:]),
                jnp.asarray(actions_enc[:-1]),
                jnp.asarray(reward[:-1]),
                gru_hidden_pre_head,
            )
            actions_1_to_end = np.asarray(actions_1_to_end)
            if actions_1_to_end.ndim == 1:
                actions_1_to_end = actions_1_to_end[:, None]
            actions_noisy[1:] = actions_1_to_end

        deviations[ratio] = np.abs(actions_noisy - action_actual).mean(axis=-1)

    return deviations


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison_figure(
    models_data: list[dict],
    noise_ratios: list[float],
    param_label: str,
    param_value: float,
    system_name: str,
    out_dir: str,
):
    """Plot action and belief changes caused by observation noise."""
    n_models = len(models_data)
    fig, axes = plt.subplots(
        2, n_models, figsize=(5.2 * n_models, 7.5), squeeze=False, sharey="row"
    )
    fig.suptitle(
        f"Observation-noise robustness — {system_name}  ({param_label}={param_value:.3g})",
        fontsize=13,
    )

    ratios_nonzero = [r for r in noise_ratios if r != 0.0]
    cmap = matplotlib.colormaps["plasma"]
    colors = {
        r: cmap(0.15 + 0.75 * i / max(len(ratios_nonzero) - 1, 1))
        for i, r in enumerate(ratios_nonzero)
    }
    y_cmap = matplotlib.colormaps["tab10"]

    for m, data in enumerate(models_data):
        ax_dev = axes[0][m]
        steps = np.arange(len(next(iter(data["deviations"].values()))))
        for ratio in ratios_nonzero:
            ax_dev.plot(
                steps,
                data["deviations"][ratio],
                "-",
                color=colors[ratio],
                linewidth=1.3,
                alpha=0.9,
                label=f"r={ratio:g}",
            )
        ax_dev.set_title(f"{data['label']} — local |Δaction|", fontsize=10)
        ax_dev.set_xlabel("Rollout step")
        ax_dev.set_ylabel("|Δaction|")
        ax_dev.grid(True, linewidth=0.4)
        ax_dev.set_ylim(bottom=0)
        if m == 0:
            ax_dev.legend(fontsize=7, loc="upper right")

        ax_traj = axes[1][m]
        state_t = np.asarray(data["traces"]["state_t"])
        state_y = np.asarray(data["traces"]["state_y"])
        keep_step = np.asarray(data["traces"]["keep_step"], dtype=bool)
        y_dim = state_y.shape[-1]
        for d in range(y_dim):
            ax_traj.plot(
                state_t,
                state_y[:, d],
                "-o",
                color=y_cmap(d % 10),
                markersize=2.5,
                linewidth=1.0,
                alpha=0.8,
                label=f"y[{d}]",
            )
        rejected = ~keep_step
        if rejected.any():
            for d in range(y_dim):
                ax_traj.plot(
                    state_t[rejected],
                    state_y[rejected, d],
                    "x",
                    color="red",
                    markersize=6,
                    markeredgewidth=1.5,
                    zorder=5,
                    label="rejected step" if d == 0 else None,
                )
        ax_traj.set_title(f"{data['label']} — trajectory", fontsize=10)
        ax_traj.set_xlabel("ODE time t")
        ax_traj.set_ylabel("state y")
        ax_traj.grid(True, linewidth=0.4)
        if m == 0:
            ax_traj.legend(fontsize=7, loc="best")

    fig.tight_layout()
    fname = f"obs_noise_comparison_{param_label.replace(' ', '')}{param_value:.3g}.png"
    _savefig(fig, os.path.join(out_dir, fname))


def save_table(models_data: list[dict], noise_ratios: list[float], path: str):
    """Write observation-noise sensitivity summaries as a text table."""
    with open(path, "w") as f:
        for data in models_data:
            keep = np.asarray(data["traces"]["keep_step"], dtype=bool)
            active = np.asarray(data["traces"]["active"], dtype=bool)
            n_accept = int((keep & active).sum())
            n_reject = int((~keep & active).sum())
            f.write(f"model {data['label']}  accepted={n_accept}  rejected={n_reject}\n")
            f.write(f"{'ratio':>6s} {'mean|Δaction|':>14s}\n")
            for ratio in noise_ratios:
                f.write(f"{ratio:6.2f} {float(data['deviations'][ratio].mean()):14.5f}\n")
            f.write("\n")
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _truncate_traces(traces: dict, valid_len: int) -> dict:
    """Trim every per-step field to the episode's valid length, dropping the
    frozen post-done tail (whose raw keep_step/state values are meaningless
    once the env stops actually integrating). Mirrors the valid_len trimming
    in vae_noise_injection.py."""
    out = {}
    for k, v in traces.items():
        if k in ("prior_mu", "prior_logvar"):
            out[k] = v
        else:
            out[k] = jax.tree.map(lambda a: np.asarray(a)[:valid_len], v)
    return out


def _load_model(spec: dict, steps_override: int, seed: int):
    checkpoint = resolve_checkpoint(spec["checkpoint"])
    config = load_config_from_yaml(TrainConfig, spec["config"], strict=False)
    if steps_override is not None:
        config.rollout_steps = steps_override

    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, seed)
    vae, policy = try_load_checkpoint(
        vae, policy, checkpoint, backbone=config.backbone, algo=config.algo
    )
    return config, env, vae, policy


def main():
    """Run observation-noise injection analysis for selected models."""
    args = parse_args()
    noise_ratios = args.noise_ratios if args.noise_ratios else [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Models: {[m['label'] for m in args.models]}")
    print(f"[*] Noise ratios: {noise_ratios}")

    loaded = [_load_model(spec, args.steps, args.seed) for spec in args.models]
    system_names = {config.env.system for config, _, _, _ in loaded}
    if len(system_names) > 1:
        raise SystemExit(
            f"--model configs must share the same env.system to be comparable on one task; "
            f"got: {system_names}"
        )
    system_name = next(iter(system_names))

    ref_config, ref_env, _, _ = loaded[0]
    rng = jax.random.PRNGKey(args.seed)
    rng, key_task, key_rollout = jax.random.split(rng, 3)
    env_params = ref_env.sample_task(key_task)
    true_param, param_label = get_task_param(
        jax.tree.map(lambda a: a[None], env_params),
        ref_config.env.system,
    )
    param_value = float(true_param[0])
    print(f"[*] Sampled task: {param_label}={param_value:.4g}")

    np_rng = np.random.default_rng(args.seed)
    models_data = []
    for spec, (config, env, vae, policy) in zip(args.models, loaded):
        print(f"[*] Rolling out '{spec['label']}' …")
        vae_graphdef, vae_params = nnx.split(vae)
        policy_graphdef, policy_params = nnx.split(policy)

        @jax.jit
        def run_rollout(vae_params, policy_params, rng_key):
            vae_ = nnx.merge(vae_graphdef, vae_params)
            policy_ = nnx.merge(policy_graphdef, policy_params)
            return collect_probe_rollout(
                vae_, policy_, env, env_params, config.rollout_steps, rng_key
            )

        traces = run_rollout(vae_params, policy_params, key_rollout)
        traces = jax.tree.map(np.array, traces)

        active = np.asarray(traces["active"])
        valid_len = int(active.sum()) if active.any() else config.rollout_steps
        valid_len = max(valid_len, 1)
        traces = _truncate_traces(traces, valid_len)

        print("    computing local action deviation …")
        deviations = compute_action_deviation(vae, policy, traces, noise_ratios, np_rng)

        models_data.append({"label": spec["label"], "traces": traces, "deviations": deviations})

    plot_comparison_figure(
        models_data, noise_ratios, param_label, param_value, system_name, out_dir
    )
    save_table(models_data, noise_ratios, os.path.join(out_dir, "obs_noise_table.txt"))

    print(f"\n[+] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
