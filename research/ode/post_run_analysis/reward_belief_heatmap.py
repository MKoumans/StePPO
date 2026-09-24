"""Belief-conditioned decoder landscapes over the step-size axis.

VariBAD's gridworld figure (Zintgraf et al., 2020, Fig. 3a) sweeps the reward
decoder over the state axis, since gridworld reward depends only on state.
Our ODE reward depends on (state, action, z) with action = step-size scale,
so the analogous sweep is over the ACTION axis: for each point along one
real rollout, hold the observed state and belief z fixed and evaluate a
decoder over a grid of candidate step sizes. The (t, candidate dt) heatmap
shows what the model expects at each step-size choice at that point in the
trajectory, with the policy's actually-chosen step size overlaid as a trace.

Three heatmaps are produced (whichever decoders the checkpoint's VAE has):
reward, state (reduced to ||predicted next state - current state||, since
state is typically multi-dimensional), and accept (reduced to
sigmoid(logit), predicted step-acceptance probability).

Usage:
    python research/ode/post_run_analysis/reward_belief_heatmap.py \\
        --config configs/envs/ode/van_der_pol/van_der_pol_default.yaml \\
        --checkpoint outputs/checkpoints/checkpoint_1000 \\
        --out outputs/analysis
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.ode.post_run_analysis.analysis_common import col_edges, try_load_checkpoint
from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv, ODEParams
from steppo.utils.checkpoint import build_models, resolve_checkpoint

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse checkpoint, task, sweep, and output options."""
    p = argparse.ArgumentParser(
        description="Belief-conditioned reward-vs-step-size heatmap for one ODE rollout"
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (e.g. configs/envs/ode/van_der_pol/van_der_pol_default.yaml)",
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
        "--lam",
        type=float,
        default=None,
        help="Fixed task parameter (ODEParams.lam) for the rollout; "
        "sampled via env.sample_task if omitted",
    )
    p.add_argument(
        "--action_grid_size",
        type=int,
        default=41,
        help="Number of candidate step-size actions swept per trajectory point",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override rollout_steps from config — raise this if the rollout "
        "hits the step budget before reaching t_end",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", type=str, default="outputs/analysis", help="Output directory for the PNG"
    )
    p.add_argument(
        "--max_legend_dims",
        type=int,
        default=10,
        help="Suppress the per-dimension state legend in the trajectory "
        "plot when state_dim exceeds this (high-dim systems like "
        "chemical_cascade otherwise produce an unreadable legend)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


def collect_rollout(vae, policy, env, params: ODEParams, rng_key, max_steps: int) -> dict:
    """Roll out one deterministic episode, recording per-step (obs, belief,
    dt-before, dt-after, t). Plain Python loop (not lax.scan) — a single
    trajectory for plotting, not a batched/jitted training rollout.
    """
    key_reset, key_run = jax.random.split(rng_key)
    obs, env_state = env.reset(key_reset, params)
    gru_hidden = vae.encoder.init_hidden()
    belief_mu, belief_logvar = vae.get_prior()

    obs_list, mu_list, dt_before_list, dt_after_list, t_list = [], [], [], [], []
    y_list, t_after_list, reward_list = [], [], []

    for i in range(max_steps):
        key_i = jax.random.fold_in(key_run, i)
        key_act, key_step = jax.random.split(key_i)

        z_policy = jnp.concatenate([belief_mu, belief_logvar], axis=-1)
        action, _, _ = policy.act(obs, z_policy, key_act, deterministic=True)

        obs_list.append(obs)
        mu_list.append(belief_mu)
        dt_before_list.append(env_state.dt)
        t_list.append(env_state.t)

        obs_next, env_state_next, reward, done, _ = env.step(key_step, env_state, action, params)
        dt_after_list.append(env_state_next.dt)
        y_list.append(env_state_next.y)
        t_after_list.append(env_state_next.t)
        reward_list.append(reward)

        action_enc = jnp.asarray(action, dtype=jnp.float32)
        reward_enc = jnp.reshape(reward, (1,)).astype(jnp.float32)
        belief_mu, belief_logvar, gru_hidden = vae.encoder.encode_step(
            action_enc, obs_next, reward_enc, gru_hidden
        )

        obs, env_state = obs_next, env_state_next
        if bool(done):
            break

    return {
        "obs": np.stack([np.asarray(o) for o in obs_list]),  # (T, obs_dim)
        "mu": np.stack([np.asarray(m) for m in mu_list]),  # (T, latent_dim)
        "dt_before": np.asarray(dt_before_list, dtype=np.float64),  # (T,)
        "dt_after": np.asarray(dt_after_list, dtype=np.float64),  # (T,)
        "t": np.asarray(t_list, dtype=np.float64),  # (T,)
        "y": np.stack([np.asarray(y) for y in y_list]),  # (T, y_dim)
        "t_after": np.asarray(t_after_list, dtype=np.float64),  # (T,)
        "reward": np.asarray(reward_list, dtype=np.float64),  # (T,)
    }


# ---------------------------------------------------------------------------
# Decoder-vs-step-size sweep
# ---------------------------------------------------------------------------


def compute_decoder_heatmap(
    decode_scalar_fn,
    rollout: dict,
    action_grid_size: int,
    dt_min: float,
    dt_max: float,
    dt_log_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """For every trajectory step, sweep candidate actions -> candidate step
    sizes (same formula as ODEEnv.step) and evaluate `decode_scalar_fn(zs,
    states, actions) -> (G,)`. Returns (predicted_value, dt_grid), both
    (T, action_grid_size); dt_grid varies per row since it scales that row's
    dt_before.
    """
    action_grid = jnp.linspace(-1.0, 1.0, action_grid_size)  # (G,)
    obs = jnp.asarray(rollout["obs"])  # (T, obs_dim)
    mu = jnp.asarray(rollout["mu"])  # (T, latent_dim)
    dt_before = jnp.asarray(rollout["dt_before"])  # (T,)

    def value_row(state_i, z_i, dt_before_i):
        dt_candidate = jnp.clip(
            dt_before_i * jnp.exp(action_grid * dt_log_gain), dt_min, dt_max
        )  # (G,)
        states = jnp.broadcast_to(state_i, (action_grid_size, state_i.shape[-1]))
        zs = jnp.broadcast_to(z_i, (action_grid_size, z_i.shape[-1]))
        actions = action_grid[:, None]  # (G, 1)
        value = decode_scalar_fn(zs, states, actions)
        return value, dt_candidate

    predicted_value, dt_grid = jax.vmap(value_row)(obs, mu, dt_before)
    return np.asarray(predicted_value), np.asarray(dt_grid)


def _reward_scalar_fn(vae):
    def fn(zs, states, actions):
        out = vae.reward_decoder(zs, states, actions)
        return out[0] if isinstance(out, tuple) else out

    return fn


def _state_scalar_fn(vae):
    """||predicted next state - current state||, since state is generally
    multi-dimensional and can't be shown as one heatmap per-dimension."""

    def fn(zs, states, actions):
        out = vae.state_decoder(zs, states, actions)
        pred_state = out[0] if isinstance(out, tuple) else out
        return jnp.linalg.norm(pred_state - states, axis=-1)

    return fn


def _accept_scalar_fn(vae):
    def fn(zs, states, actions):
        logit = vae.accept_decoder(zs, states, actions)
        return jax.nn.sigmoid(logit)

    return fn


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_heatmap(
    rollout: dict,
    predicted_value: np.ndarray,
    dt_grid: np.ndarray,
    system: str,
    lam,
    out_path: str,
    log_scale_y: bool,
    title: str,
    colorbar_label: str,
    cmap: str = "RdYlBu_r",
):
    """Plot decoder values over time and candidate step sizes."""
    t = rollout["t"]
    dt_actual = rollout["dt_after"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # dt_grid varies per row, so build the mesh from per-cell (t, dt) coords.
    T, G = predicted_value.shape
    t_edges = np.concatenate([t, [t[-1] + (t[-1] - t[-2] if T > 1 else 1.0)]])
    dt_edges = col_edges(dt_grid)  # (T, G+1)
    dt_edges = np.concatenate([dt_edges, dt_edges[-1:]], axis=0)  # (T+1, G+1)
    mesh = ax.pcolormesh(
        np.tile(t_edges[:, None], (1, G + 1)),
        dt_edges,
        predicted_value,
        shading="flat",
        cmap=cmap,
    )
    fig.colorbar(mesh, ax=ax, label=colorbar_label)

    ax.plot(t, dt_actual, color="black", linewidth=1.8, label="Step size actually chosen")
    if log_scale_y:
        ax.set_yscale("log")

    lam_str = f"{lam:.3g}" if lam is not None else "sampled"
    ax.set_xlabel("Trajectory time t")
    ax.set_ylabel("Candidate step size Δt")
    ax.set_title(f"{title} — system={system}, λ={lam_str}")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_trajectory(
    rollout: dict, system: str, lam, out_path: str, log_scale_dt: bool, max_legend_dims: int = 10
):
    """Three-panel view of the whole rollout: ODE state y(t), step size Δt(t),
    and reward(t) — the actual trajectory the reward heatmap is built from."""
    t = rollout["t_after"]
    y = rollout["y"]
    dt = rollout["dt_after"]
    reward = rollout["reward"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    if y.ndim == 1:
        axes[0].plot(t, y, linewidth=1.2)
    else:
        show_legend = y.shape[1] <= max_legend_dims
        for d in range(y.shape[1]):
            axes[0].plot(t, y[:, d], linewidth=1.0, label=f"y[{d}]" if show_legend else None)
        if show_legend:
            axes[0].legend(fontsize=8)
    axes[0].set_ylabel("State y")
    axes[0].set_title("ODE state trajectory")
    axes[0].grid(True, linewidth=0.4)

    axes[1].plot(t, dt, color="tab:orange", linewidth=1.2)
    if log_scale_dt:
        axes[1].set_yscale("log")
    axes[1].set_ylabel("Step size Δt")
    axes[1].set_title("Step size chosen by the policy")
    axes[1].grid(True, linewidth=0.4)

    axes[2].plot(t, reward, color="tab:green", linewidth=1.2)
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Trajectory time t")
    axes[2].set_title("Reward received")
    axes[2].grid(True, linewidth=0.4)

    lam_str = f"{lam:.3g}" if lam is not None else "sampled"
    fig.suptitle(f"Full rollout — system={system}, λ={lam_str}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Collect one rollout and render decoder heatmaps."""
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
        if args.config is not None
        else TrainConfig()
    )
    if args.system is not None:
        config.env.system = args.system
    if args.steps is not None:
        config.rollout_steps = args.steps

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] System : {config.env.system}")
    print(f"[*] Steps  : {config.rollout_steps}  Seed: {args.seed}")
    print(f"[*] Output : {out_dir}")

    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, args.seed)

    if args.checkpoint is not None:
        vae, policy = try_load_checkpoint(
            vae, policy, args.checkpoint, backbone=config.backbone, algo=config.algo
        )
    else:
        print("[!] No checkpoint supplied - using random weights.")

    rng = jax.random.PRNGKey(args.seed)
    key_task, key_roll = jax.random.split(rng)

    if args.lam is not None:
        pulse_phase = jax.random.uniform(key_task, shape=(), dtype=jnp.float32)
        params = ODEParams(lam=args.lam, pulse_phase=pulse_phase, max_steps=config.rollout_steps)
    else:
        params = env.sample_task(key_task)

    print("[*] Rolling out one episode ...")
    rollout = collect_rollout(vae, policy, env, params, key_roll, config.rollout_steps)
    print(f"[*] Rollout length: {len(rollout['t'])} steps")

    log_scale_y = (config.env.dt_max / max(config.env.dt_min, 1e-30)) > 10.0
    dt_kwargs = dict(
        dt_min=float(config.env.dt_min),
        dt_max=float(config.env.dt_max),
        dt_log_gain=float(config.env.dt_log_gain),
    )

    print("[*] Sweeping candidate step sizes — reward decoder ...")
    predicted_reward, dt_grid = compute_decoder_heatmap(
        _reward_scalar_fn(vae), rollout, args.action_grid_size, **dt_kwargs
    )
    plot_heatmap(
        rollout,
        predicted_reward,
        dt_grid,
        config.env.system,
        args.lam,
        os.path.join(out_dir, "reward_belief_heatmap.png"),
        log_scale_y,
        title="Belief-conditioned reward landscape",
        colorbar_label="Predicted reward (reward decoder)",
        cmap="RdYlBu_r",
    )

    if vae.state_decoder is not None:
        print("[*] Sweeping candidate step sizes — state decoder ...")
        predicted_state_delta, dt_grid = compute_decoder_heatmap(
            _state_scalar_fn(vae), rollout, args.action_grid_size, **dt_kwargs
        )
        plot_heatmap(
            rollout,
            predicted_state_delta,
            dt_grid,
            config.env.system,
            args.lam,
            os.path.join(out_dir, "state_belief_heatmap.png"),
            log_scale_y,
            title="Belief-conditioned state-change landscape",
            colorbar_label="||predicted next state - current state|| (state decoder)",
            cmap="viridis",
        )
    else:
        print("[*] No state decoder on this checkpoint (state_loss_coeff <= 0) — skipping.")

    if vae.accept_decoder is not None:
        print("[*] Sweeping candidate step sizes — accept decoder ...")
        predicted_accept_prob, dt_grid = compute_decoder_heatmap(
            _accept_scalar_fn(vae), rollout, args.action_grid_size, **dt_kwargs
        )
        plot_heatmap(
            rollout,
            predicted_accept_prob,
            dt_grid,
            config.env.system,
            args.lam,
            os.path.join(out_dir, "accept_belief_heatmap.png"),
            log_scale_y,
            title="Belief-conditioned step-acceptance landscape",
            colorbar_label="Predicted P(step accepted) (accept decoder)",
            cmap="viridis",
        )
    else:
        print("[*] No accept decoder on this checkpoint (accept_loss_coeff <= 0) — skipping.")

    traj_path = os.path.join(out_dir, "trajectory.png")
    plot_trajectory(
        rollout,
        config.env.system,
        args.lam,
        traj_path,
        log_scale_y,
        max_legend_dims=args.max_legend_dims,
    )

    print(f"\n[+] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
