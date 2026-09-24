"""Single-run rollout diagnostics for a trained ODE controller.

Writes PNG panels to --out; panels marked * overlay the PID baseline:
  belief_mu.png, belief_sigma.png   belief mean / std over rollout steps
  ode_trajectory.png *              y(t)
  task_vs_belief.png                final belief vs true task parameter
  rejection_rate.png *              fraction of rejected steps
  policy_action.png *               policy action vs PID action
  belief_by_task_bin.png            belief traces per task-parameter quantile
  acceptance_vs_task.png *          acceptance rate vs task parameter
  completion_vs_task.png *          steps to completion vs task parameter
  vae_noise_*                       VAE input-noise sensitivity (skip with --noise_samples 0)

Usage:
    python research/ode/post_run_analysis/analyse_rollout.py \\
        --checkpoint outputs/runs/<date>/ode/<system>/<run_uid>/checkpoints \\
        --num_envs 64 --out outputs/analysis

Without --checkpoint, randomly initialised weights are used (to test the plots).
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
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv, ODEParams
from steppo.models.backbones import get_backbone
from steppo.training.pid_controller import compute_step_context_offset, pid_action_from_obs
from steppo.utils.checkpoint import build_models, resolve_checkpoint

try:
    from .analysis_common import (
        _ALPHA,
        _BLUE,
        _GREEN,
        _ORANGE,
        _savefig,
        freeze_inactive,
        get_task_param,
        try_load_checkpoint,
    )
    from .rollout_cache import load_or_compute
    from .vae_noise_injection import run_noise_injection_analysis
except ImportError:
    from analysis_common import (
        _ALPHA,
        _BLUE,
        _GREEN,
        _ORANGE,
        _savefig,
        freeze_inactive,
        get_task_param,
        try_load_checkpoint,
    )
    from rollout_cache import load_or_compute
    from vae_noise_injection import run_noise_injection_analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse checkpoint, rollout, plotting, and output options."""
    p = argparse.ArgumentParser(description="ODE rollout analysis")
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
        help="Environments to roll out (averaged in plots)",
    )
    p.add_argument("--steps", type=int, default=None, help="Override rollout_steps from config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", type=str, default="outputs/analysis", help="Output directory for PNG files"
    )
    p.add_argument(
        "--mu_table",
        nargs="*",
        type=float,
        default=None,
        help="Print per-mu comparison table (policy vs PID). "
        "Pass mu values or omit args for default [1,5,10,20,50,100,200]",
    )
    p.add_argument(
        "--noise_samples",
        type=int,
        default=4,
        help="Tasks to sample for the VAE input-noise injection panel (0 disables it)",
    )
    p.add_argument(
        "--noise_ratios",
        nargs="*",
        type=float,
        default=None,
        help="White-noise mix ratios in [0,1] for the noise injection panel "
        "(default: 0.0 0.1 0.25 0.5 0.75 1.0)",
    )
    p.add_argument(
        "--noise_spike_fracs",
        nargs="*",
        type=float,
        default=None,
        help="Fractions of t_end at which to inject a noise spike (default: 0.25 0.5 0.75)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Analysis rollout
# ---------------------------------------------------------------------------


def collect_analysis_rollout(
    vae,
    policy,
    env,
    env_params_batch,
    num_envs: int,
    T: int,
    rng_key,
):
    """Run T rollout steps across num_envs environments.

    Episodes are NOT reset on done — the state is frozen so each trace covers
    a single contiguous ODE integration. Belief encoding also freezes after done.

    Returns a dict of (T, num_envs, ...) JAX arrays.
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
        active = ~done_so_far  # (num_envs,) bool

        # ── Build z and act (deterministic) ───────────────────────
        z = jnp.concatenate([belief_mu, belief_logvar], axis=-1)
        key_acts = jax.random.split(key_act, num_envs)
        actions, _lp, _v = jax.vmap(lambda o, z_, k: policy.act(o, z_, k, deterministic=True))(
            obs, z, key_acts
        )

        # ── Env step ───────────────────────────────────────────────
        key_envs = jax.random.split(key_env, num_envs)
        next_obs, next_env_state, rewards, dones, _ = jax.vmap(
            lambda k, s, a, p: env.step(k, s, a, p)
        )(key_envs, env_state, actions, env_params_batch)
        rewards = rewards.reshape(num_envs, 1)

        # ── Freeze state / obs after episode ends ─────────────────
        def freeze_tree(n, o):
            return jax.tree.map(lambda na, oa: _freeze(active, na, oa), n, o)

        final_obs = _freeze(active, next_obs, obs)
        final_env_state = freeze_tree(next_env_state, env_state)
        final_dones = jnp.logical_or(done_so_far, dones)

        # ── Encode (freeze belief after done) ─────────────────────
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

        # ── Record ────────────────────────────────────────────────
        # keep_step from the raw (pre-freeze) env step tells us acceptance
        ys = {
            "belief_mu": new_mu,  # (E, latent_dim)
            "belief_logvar": new_logvar,  # (E, latent_dim)
            "state_y": final_env_state.y,  # (E, y_dim)
            "state_t": final_env_state.t,  # (E,)
            "action": actions,  # (E, action_dim)
            "reward": rewards.squeeze(-1),  # (E,)
            "keep_step": next_env_state.last_keep_step,  # (E,) bool
            "active": active,  # (E,) bool
        }
        return carry, ys

    init_carry = (obs0, env_state0, gru_hidden0, belief_mu0, belief_logvar0, done0)
    keys_scan = jax.random.split(rng_key, T)

    _, traces = jax.lax.scan(scan_step, init_carry, keys_scan)
    return traces


def collect_pid_analysis_rollout(
    env,
    env_params_batch,
    num_envs: int,
    T: int,
    step_context_offset: int,
    rng_key,
):
    """Run T rollout steps using the PID controller (no learned policy).

    Same structure as collect_analysis_rollout but actions come from
    pid_action_from_obs.  No VAE encoding — belief fields are absent.
    """
    keys_reset = jax.random.split(rng_key, num_envs)
    obs0, env_state0 = jax.vmap(env.reset)(keys_reset, env_params_batch)
    done0 = jnp.zeros(num_envs, dtype=jnp.bool_)

    _freeze = freeze_inactive

    def scan_step(carry, rng_key_i):
        obs, env_state, done_so_far = carry
        active = ~done_so_far

        actions = jax.vmap(
            lambda o: pid_action_from_obs(o, step_context_offset, dt_log_gain=env.dt_log_gain)
        )(obs)

        key_envs = jax.random.split(rng_key_i, num_envs)
        next_obs, next_env_state, rewards, dones, _ = jax.vmap(
            lambda k, s, a, p: env.step(k, s, a, p)
        )(key_envs, env_state, actions, env_params_batch)
        rewards = rewards.reshape(num_envs, 1)

        def freeze_tree(n, o):
            return jax.tree.map(lambda na, oa: _freeze(active, na, oa), n, o)

        final_obs = _freeze(active, next_obs, obs)
        final_env_state = freeze_tree(next_env_state, env_state)
        final_dones = jnp.logical_or(done_so_far, dones)

        carry = (final_obs, final_env_state, final_dones)
        ys = {
            "state_y": final_env_state.y,
            "state_t": final_env_state.t,
            "action": actions,
            "reward": rewards.squeeze(-1),
            "keep_step": next_env_state.last_keep_step,
            "active": active,
        }
        return carry, ys

    init_carry = (obs0, env_state0, done0)
    keys_scan = jax.random.split(rng_key, T)
    _, traces = jax.lax.scan(scan_step, init_carry, keys_scan)
    return traces


def collect_diffeqsolve_analysis_rollout(
    sc,
    env,
    config,
    env_params_batch,
    num_envs: int,
    T: int,
    rng_key,
):
    """Run diffeqsolve with a LearnedController and return (traces, stats).

    traces: dict with state_y, state_t, active in (T, E, ...) format.
            Only accepted steps are stored; padding repeats the final state.
    stats:  dict with per-episode scalars (E,): acceptance_rate, total_steps, t_reached.
    """
    import diffrax

    from steppo.envs.ode import _make_solver
    from steppo.envs.ode.systems import get_rhs

    rhs = get_rhs(config.env)
    solver = _make_solver(config.env.rtol, config.env.atol)

    keys_reset = jax.random.split(rng_key, num_envs)
    _, env_states = jax.vmap(env.reset)(keys_reset, env_params_batch)
    y0s = env_states.y
    lams = env_params_batch.lam
    pulse_phases = env_params_batch.pulse_phase

    def solve_one(y0, lam, pulse_phase):
        task = jnp.array([lam, pulse_phase], dtype=jnp.float32)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            solver,
            t0=0.0,
            t1=config.env.t_end,
            dt0=config.env.dt0,
            y0=y0,
            args=task,
            stepsize_controller=sc,
            max_steps=T,
            saveat=diffrax.SaveAt(steps=True),
            throw=False,
        )
        num_acc = sol.stats["num_accepted_steps"]
        num_rej = sol.stats["num_rejected_steps"]
        total = num_acc + num_rej
        step_idx = jnp.arange(T)
        active = step_idx < num_acc
        t_reached = sol.ts[jnp.clip(num_acc - 1, 0, T - 1)]

        return {
            "state_y": sol.ys,
            "state_t": sol.ts,
            "active": active,
            "acceptance_rate": num_acc / jnp.maximum(total, 1),
            "total_steps": total,
            "t_reached": t_reached,
        }

    results = jax.vmap(solve_one)(y0s, lams, pulse_phases)

    traces = {
        "state_y": jnp.moveaxis(results["state_y"], 0, 1),
        "state_t": jnp.moveaxis(results["state_t"], 0, 1),
        "active": jnp.moveaxis(results["active"], 0, 1),
    }
    stats = {
        "acceptance_rate": results["acceptance_rate"],
        "total_steps": results["total_steps"],
        "t_reached": results["t_reached"],
    }
    return traces, stats


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_belief_mu(traces: dict, out_dir: str, latent_dim: int):
    """Mean ± std of belief_mu across environments over rollout steps."""
    mu = np.array(traces["belief_mu"])  # (T, E, latent_dim)
    T = mu.shape[0]
    steps = np.arange(T)

    n_dims = min(latent_dim, 6)
    cols = min(n_dims, 3)
    rows = (n_dims + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    fig.suptitle("Belief μ over rollout (mean ± 1σ across environments)", fontsize=13)

    for d in range(n_dims):
        ax = axes[d // cols][d % cols]
        m = mu[:, :, d]  # (T, E)
        mean = m.mean(axis=1)
        std = m.std(axis=1)
        ax.plot(steps, mean, color=_BLUE, linewidth=1.8)
        ax.fill_between(steps, mean - std, mean + std, alpha=_ALPHA, color=_BLUE)
        ax.set_title(f"z[{d}]", fontsize=10)
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("μ value")
        ax.grid(True, linewidth=0.4)

    for d in range(n_dims, rows * cols):
        axes[d // cols][d % cols].set_visible(False)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "belief_mu.png"))


def plot_belief_sigma(traces: dict, out_dir: str, latent_dim: int):
    """Mean ± std of belief_sigma = exp(0.5*logvar) across environments."""
    logvar = np.array(traces["belief_logvar"])  # (T, E, latent_dim)
    sigma = np.exp(0.5 * np.clip(logvar, -10, 10))
    T = sigma.shape[0]
    steps = np.arange(T)

    n_dims = min(latent_dim, 6)
    cols = min(n_dims, 3)
    rows = (n_dims + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    fig.suptitle("Belief σ over rollout (mean ± 1σ across environments)", fontsize=13)

    for d in range(n_dims):
        ax = axes[d // cols][d % cols]
        s = sigma[:, :, d]  # (T, E)
        mean = s.mean(axis=1)
        std = s.std(axis=1)
        ax.plot(steps, mean, color=_ORANGE, linewidth=1.8)
        ax.fill_between(steps, np.maximum(mean - std, 0), mean + std, alpha=_ALPHA, color=_ORANGE)
        ax.set_title(f"z[{d}]", fontsize=10)
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("σ value")
        ax.set_ylim(bottom=0)
        ax.grid(True, linewidth=0.4)

    for d in range(n_dims, rows * cols):
        axes[d // cols][d % cols].set_visible(False)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "belief_sigma.png"))


def plot_ode_trajectory(
    traces: dict,
    out_dir: str,
    system_name: str,
    pid_traces: dict = None,
    diffeqsolve_traces: dict = None,
):
    """y(t) vs ODE time t for each y-dimension. Connected dots per environment."""
    y_all = np.array(traces["state_y"])  # (T, E, y_dim)
    t_all = np.array(traces["state_t"])  # (T, E)
    T, E, y_dim = y_all.shape

    n_show = min(E, 16)
    colors = cm.tab20(np.linspace(0, 1, n_show))

    cols = min(y_dim, 3)
    rows = (y_dim + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    fig.suptitle(f"ODE state trajectory — {system_name}", fontsize=13)

    dim_labels = {0: "y₀", 1: "y₁", 2: "y₂"}

    for d in range(y_dim):
        ax = axes[d // cols][d % cols]
        for e in range(n_show):
            t_env = t_all[:, e]
            y_env = y_all[:, e, d]
            label = "Policy (env.step)" if e == 0 else None
            ax.plot(
                t_env,
                y_env,
                "-o",
                color=colors[e],
                markersize=2,
                linewidth=0.8,
                alpha=0.65,
                label=label,
            )
        if pid_traces is not None:
            pid_y = np.array(pid_traces["state_y"])
            pid_t = np.array(pid_traces["state_t"])
            for e in range(n_show):
                label = "PID" if e == 0 else None
                ax.plot(
                    pid_t[:, e],
                    pid_y[:, e, d],
                    "--",
                    color=colors[e],
                    markersize=0,
                    linewidth=0.6,
                    alpha=0.40,
                    label=label,
                )
        if diffeqsolve_traces is not None:
            ds_y = np.array(diffeqsolve_traces["state_y"])
            ds_t = np.array(diffeqsolve_traces["state_t"])
            ds_active = np.array(diffeqsolve_traces["active"])
            for e in range(min(n_show, ds_y.shape[1])):
                mask = ds_active[:, e].astype(bool)
                label = "RL (diffeqsolve)" if e == 0 else None
                ax.plot(
                    ds_t[mask, e],
                    ds_y[mask, e, d],
                    ":",
                    color=colors[e],
                    linewidth=0.9,
                    alpha=0.50,
                    label=label,
                )
        if pid_traces is not None or diffeqsolve_traces is not None:
            ax.legend(fontsize=7, loc="best")
        ax.set_xlabel("ODE time t")
        ax.set_ylabel(dim_labels.get(d, f"y[{d}]"))
        ax.set_title(f"State dimension {d}", fontsize=10)
        ax.grid(True, linewidth=0.4)

    if y_dim == 2 and rows == 1 and cols < 3:
        ax_phase = fig.add_axes([0.68, 0.15, 0.28, 0.70])
        for e in range(n_show):
            ax_phase.plot(
                y_all[:, e, 0], y_all[:, e, 1], "-", color=colors[e], linewidth=0.8, alpha=0.55
            )
        if pid_traces is not None:
            for e in range(n_show):
                ax_phase.plot(
                    pid_y[:, e, 0], pid_y[:, e, 1], "--", color=colors[e], linewidth=0.5, alpha=0.35
                )
        if diffeqsolve_traces is not None:
            for e in range(min(n_show, ds_y.shape[1])):
                mask = ds_active[:, e].astype(bool)
                ax_phase.plot(
                    ds_y[mask, e, 0],
                    ds_y[mask, e, 1],
                    ":",
                    color=colors[e],
                    linewidth=0.6,
                    alpha=0.40,
                )
        ax_phase.set_xlabel("y₀")
        ax_phase.set_ylabel("y₁")
        ax_phase.set_title("Phase portrait", fontsize=9)
        ax_phase.grid(True, linewidth=0.4)

    for d in range(y_dim, rows * cols):
        axes[d // cols][d % cols].set_visible(False)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "ode_trajectory.png"))


def plot_task_vs_belief(
    traces: dict,
    out_dir: str,
    true_params: np.ndarray,
    system_name: str,
):
    """Scatter: final belief_mu[dim] vs true task parameter, one subplot per dim.

    Uses the last active step per env so the belief has had maximum time to
    accumulate evidence.  Pearson r quantifies how linearly each latent dim
    encodes the task parameter.
    """
    mu = np.array(traces["belief_mu"])  # (T, E, latent_dim)
    active = np.array(traces["active"])  # (T, E) bool

    T, E, latent_dim = mu.shape

    # Final belief = belief at the last step the env was still active
    last_step = np.maximum(active.sum(axis=0).astype(int) - 1, 0)  # (E,)
    final_mu = mu[last_step, np.arange(E), :]  # (E, latent_dim)

    n_dims = min(latent_dim, 3)
    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 4), squeeze=False)
    fig.suptitle(
        f"Belief μ (at episode end) vs true task param  [{system_name}]",
        fontsize=13,
    )

    for d in range(n_dims):
        ax = axes[0][d]
        x = true_params  # (E,)
        y = final_mu[:, d]  # (E,)

        r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-8 and y.std() > 1e-8 else 0.0

        sc = ax.scatter(x, y, c=x, cmap="viridis", s=40, alpha=0.85, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="True param")

        # Linear trend line
        if x.std() > 1e-8:
            m, b = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, m * xs + b, color="red", linewidth=1.2, linestyle="--", alpha=0.7)

        ax.set_title(f"z[{d}]  (r = {r:+.3f})", fontsize=10)
        ax.set_xlabel("True param")
        ax.set_ylabel(f"belief_mu[{d}]")
        ax.grid(True, linewidth=0.4)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "task_vs_belief.png"))


def _rejection_rate_arrays(traces: dict):
    keep = np.array(traces["keep_step"], dtype=np.float32)
    active = np.array(traces["active"], dtype=np.float32)
    denom = np.maximum(active.sum(axis=1), 1.0)
    rej_mean = ((1.0 - keep) * active).sum(axis=1) / denom
    rej_all = 1.0 - keep
    rej_std = (rej_all * active).std(axis=1)
    return rej_mean, rej_std


def plot_rejection_rate(traces: dict, out_dir: str, pid_traces: dict = None):
    """Fraction of rejected ODE solver steps at each rollout timestep."""
    rej_mean, rej_std = _rejection_rate_arrays(traces)
    steps = np.arange(len(rej_mean))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_title("ODE solver step rejection rate over rollout", fontsize=12)
    ax.plot(steps, rej_mean, color=_GREEN, linewidth=1.8, label="Policy")
    ax.fill_between(
        steps,
        np.maximum(rej_mean - rej_std, 0),
        np.minimum(rej_mean + rej_std, 1),
        alpha=_ALPHA,
        color=_GREEN,
    )

    if pid_traces is not None:
        pid_mean, pid_std = _rejection_rate_arrays(pid_traces)
        pid_steps = np.arange(len(pid_mean))
        ax.plot(pid_steps, pid_mean, color="grey", linewidth=1.5, linestyle="--", label="PID")
        ax.fill_between(
            pid_steps,
            np.maximum(pid_mean - pid_std, 0),
            np.minimum(pid_mean + pid_std, 1),
            alpha=0.15,
            color="grey",
        )

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Fraction rejected")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "rejection_rate.png"))


def plot_policy_action(traces: dict, out_dir: str, pid_traces: dict = None):
    """Policy action (dt scale) mean ± std over environments vs. rollout step.

    A positive action expands the next step size; negative shrinks it.
    Overlaid with belief_sigma[0] on a secondary axis so you can see
    whether uncertainty narrows as the policy becomes more aggressive.
    """
    actions = np.array(traces["action"])  # (T, E, action_dim) or (T, E, 1)
    if actions.ndim == 3:
        actions = actions[:, :, 0]  # take first (only) dim → (T, E)
    sigma = np.exp(0.5 * np.clip(np.array(traces["belief_logvar"])[:, :, 0], -10, 10))  # (T, E)

    steps = np.arange(actions.shape[0])

    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    ax2 = ax1.twinx()

    a_mean = actions.mean(axis=1)
    a_std = actions.std(axis=1)
    s_mean = sigma.mean(axis=1)

    ax1.plot(steps, a_mean, color=_BLUE, linewidth=1.8, label="Policy action mean")
    ax1.fill_between(steps, a_mean - a_std, a_mean + a_std, alpha=_ALPHA, color=_BLUE)

    if pid_traces is not None:
        pid_act = np.array(pid_traces["action"])
        if pid_act.ndim == 3:
            pid_act = pid_act[:, :, 0]
        pid_steps = np.arange(pid_act.shape[0])
        pid_mean = pid_act.mean(axis=1)
        pid_std = pid_act.std(axis=1)
        ax1.plot(
            pid_steps,
            pid_mean,
            color="grey",
            linewidth=1.5,
            linestyle="--",
            label="PID action mean",
        )
        ax1.fill_between(
            pid_steps, pid_mean - pid_std, pid_mean + pid_std, alpha=0.15, color="grey"
        )

    ax1.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax1.set_xlabel("Rollout step")
    ax1.set_ylabel("Action (tanh-scaled)", color=_BLUE)
    ax1.tick_params(axis="y", labelcolor=_BLUE)

    ax2.plot(steps, s_mean, color=_ORANGE, linewidth=1.4, linestyle="--", label="belief σ[0] mean")
    ax2.set_ylabel("Belief σ[0]", color=_ORANGE)
    ax2.tick_params(axis="y", labelcolor=_ORANGE)
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    ax1.set_title("Policy action (dt scale) and belief σ[0] over rollout", fontsize=12)
    ax1.grid(True, linewidth=0.4)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "policy_action.png"))


def plot_belief_by_task_bin(
    traces: dict,
    out_dir: str,
    true_params: np.ndarray,
    param_label: str,
    latent_dim: int,
    n_bins: int = 4,
):
    """Belief_mu traces averaged within quantile bins of the task parameter.

    Reveals whether the encoder's posterior shifts systematically with task
    difficulty.  Each bin contains an equal number of environments.
    """
    if true_params.std() < 1e-6:
        print("  [skip] belief_by_task_bin — all envs share the same task param.")
        return

    mu = np.array(traces["belief_mu"])  # (T, E, latent_dim)
    T = mu.shape[0]
    steps = np.arange(T)

    # Quantile-based bins (equal population per bin)
    q = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(true_params, q)
    bin_idx = np.minimum(
        np.searchsorted(bin_edges[1:], true_params, side="left"),
        n_bins - 1,
    )  # (E,) values in [0, n_bins-1]

    colors = cm.viridis(np.linspace(0.1, 0.9, n_bins))

    n_dims = min(latent_dim, 3)
    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 4), squeeze=False)
    fig.suptitle(
        f"Belief μ grouped by {param_label} quantile bins  (n_bins={n_bins})",
        fontsize=13,
    )

    for d in range(n_dims):
        ax = axes[0][d]
        for i in range(n_bins):
            mask = bin_idx == i
            if mask.sum() == 0:
                continue
            lo, hi = bin_edges[i], bin_edges[i + 1]
            label = f"{lo:.1f}≤{param_label}<{hi:.1f} (n={mask.sum()})"
            group = mu[:, mask, d]  # (T, n_group)
            mean = group.mean(axis=1)
            ax.plot(steps, mean, color=colors[i], linewidth=1.8, label=label)
        ax.set_title(f"z[{d}]", fontsize=10)
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("μ value")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, linewidth=0.4)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "belief_by_task_bin.png"))


def _acceptance_rate_per_env(traces: dict):
    keep = np.array(traces["keep_step"], dtype=np.float32)
    active = np.array(traces["active"], dtype=np.float32)
    total_active = active.sum(axis=0)
    total_kept = (keep * active).sum(axis=0)
    return np.where(total_active > 0, total_kept / total_active, np.nan)


def plot_acceptance_vs_task(
    traces: dict,
    out_dir: str,
    true_params: np.ndarray,
    param_label: str,
    pid_traces: dict = None,
    diffeqsolve_stats: dict = None,
):
    """Scatter: per-env solver acceptance rate vs. ground-truth task parameter."""
    accept_rate = _acceptance_rate_per_env(traces)

    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(
        true_params,
        accept_rate,
        c=true_params,
        cmap="viridis",
        s=40,
        alpha=0.85,
        edgecolors="none",
        label="Policy (env.step)",
    )
    plt.colorbar(sc, ax=ax, label=f"True {param_label}")

    if pid_traces is not None:
        pid_accept = _acceptance_rate_per_env(pid_traces)
        ax.scatter(
            true_params,
            pid_accept,
            c="none",
            edgecolors="grey",
            linewidths=1.0,
            s=40,
            alpha=0.70,
            marker="D",
            label="PID",
        )

    if diffeqsolve_stats is not None:
        ds_accept = np.array(diffeqsolve_stats["acceptance_rate"])
        ax.scatter(
            true_params,
            ds_accept,
            c="none",
            edgecolors=_GREEN,
            linewidths=1.0,
            s=40,
            alpha=0.70,
            marker="s",
            label="RL (diffeqsolve)",
        )

    if pid_traces is not None or diffeqsolve_stats is not None:
        ax.legend(fontsize=9)

    ax.set_xlabel(f"True {param_label}")
    ax.set_ylabel("Acceptance rate")
    ax.set_title(f"Solver acceptance rate vs. {param_label}", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "acceptance_vs_task.png"))


def plot_completion_vs_task(
    traces: dict,
    out_dir: str,
    true_params: np.ndarray,
    param_label: str,
    T: int,
    pid_traces: dict = None,
    diffeqsolve_stats: dict = None,
    t_end: float = None,
):
    """Scatter: steps to episode completion vs. ground-truth task parameter."""
    active = np.array(traces["active"], dtype=np.float32)
    n_active = active.sum(axis=0)
    never_done = active[-1] > 0

    fig, ax = plt.subplots(figsize=(8, 4))

    sc = ax.scatter(
        true_params,
        n_active,
        c=true_params,
        cmap="viridis",
        s=40,
        alpha=0.85,
        edgecolors="none",
        zorder=2,
        label="Policy (env.step)",
    )
    plt.colorbar(sc, ax=ax, label=f"True {param_label}")

    if never_done.any():
        ax.scatter(
            true_params[never_done],
            n_active[never_done],
            c="red",
            marker="x",
            s=60,
            linewidths=1.5,
            zorder=3,
            label=f"Policy still active at T={T}",
        )

    if pid_traces is not None:
        pid_active = np.array(pid_traces["active"], dtype=np.float32)
        pid_n_active = pid_active.sum(axis=0)
        pid_never = pid_active[-1] > 0
        ax.scatter(
            true_params,
            pid_n_active,
            c="none",
            edgecolors="grey",
            linewidths=1.0,
            s=40,
            alpha=0.70,
            marker="D",
            zorder=2,
            label="PID",
        )
        if pid_never.any():
            ax.scatter(
                true_params[pid_never],
                pid_n_active[pid_never],
                c="orange",
                marker="x",
                s=60,
                linewidths=1.5,
                zorder=3,
                label=f"PID still active at T={T}",
            )

    if diffeqsolve_stats is not None:
        ds_steps = np.array(diffeqsolve_stats["total_steps"], dtype=np.float32)
        ds_t = np.array(diffeqsolve_stats["t_reached"])
        # diffrax clips the final step to land exactly on t_end, while env.step
        # overshoots — so completion must be judged against t_end itself.
        done_threshold = (
            float(t_end) if t_end is not None else float(np.nanmax(ds_t[np.isfinite(ds_t)]))
        )
        ds_done = ds_t >= done_threshold * 0.999
        ax.scatter(
            true_params,
            ds_steps,
            c="none",
            edgecolors=_GREEN,
            linewidths=1.0,
            s=40,
            alpha=0.70,
            marker="s",
            zorder=2,
            label="RL (diffeqsolve)",
        )
        ds_never = ~ds_done
        if ds_never.any():
            ax.scatter(
                true_params[ds_never],
                ds_steps[ds_never],
                c=_GREEN,
                marker="x",
                s=60,
                linewidths=1.5,
                zorder=3,
                label="diffeqsolve incomplete",
            )

    ax.legend(fontsize=9)
    ax.set_xlabel(f"True {param_label}")
    ax.set_ylabel("Steps to completion")
    ax.set_title(f"Episode completion steps vs. {param_label}", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "completion_vs_task.png"))


def make_trajectory_gifs(
    traces: dict,
    out_dir: str,
    system_name: str,
    pid_traces: dict = None,
    diffeqsolve_traces: dict = None,
    n_show: int = 16,
    n_frames: int = 80,
):
    """Animated GIFs of ODE trajectories drawn progressively.

    Produces trajectory_phase.gif (phase portrait, 2D+ systems only)
    and trajectory_yt.gif (state dimensions vs ODE time).
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D

    y_pol = np.array(traces["state_y"])
    t_pol = np.array(traces["state_t"])
    T, E, y_dim = y_pol.shape
    n_show = min(n_show, E)
    frame_idx = np.linspace(0, T - 1, n_frames, dtype=int)

    #                   (y, t, active, linestyle, color,    label)
    methods = [
        (y_pol, t_pol, None, "-", _BLUE, "Policy (env.step)"),
    ]
    if pid_traces is not None:
        methods.append(
            (
                np.array(pid_traces["state_y"]),
                np.array(pid_traces["state_t"]),
                None,
                "--",
                _ORANGE,
                "PID",
            )
        )
    if diffeqsolve_traces is not None:
        methods.append(
            (
                np.array(diffeqsolve_traces["state_y"]),
                np.array(diffeqsolve_traces["state_t"]),
                np.array(diffeqsolve_traces["active"]),
                ":",
                _GREEN,
                "RL (diffeqsolve)",
            )
        )

    def _safe_ravel(arr, active, max_e):
        sub = arr[:, :max_e]
        if active is None:
            return sub.ravel()
        return sub[active[:, :max_e].astype(bool)]

    legend_handles = [
        Line2D([0], [0], ls=ls, color=col, lw=2.0, label=lbl) for _, _, _, ls, col, lbl in methods
    ]

    # --- Phase portrait GIF (2D+ systems) ---
    if y_dim >= 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title(f"Phase portrait — {system_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("y₀")
        ax.set_ylabel("y₁")

        v0 = np.concatenate([_safe_ravel(m[0][..., 0], m[2], n_show) for m in methods])
        v1 = np.concatenate([_safe_ravel(m[0][..., 1], m[2], n_show) for m in methods])
        pad = 0.2
        ax.set_xlim(v0.min() - pad, v0.max() + pad)
        ax.set_ylim(v1.min() - pad, v1.max() + pad)
        ax.grid(True, linewidth=0.4)

        phase_artists = []
        for y_m, t_m, act_m, ls, col, label in methods:
            n_e = min(n_show, y_m.shape[1])
            lines, dots = [], []
            for e in range(n_e):
                (ln,) = ax.plot([], [], ls, color=col, lw=0.9, alpha=0.5)
                (dt,) = ax.plot([], [], "o", color=col, ms=3, alpha=0.7)
                lines.append(ln)
                dots.append(dt)
            phase_artists.append((lines, dots, y_m, act_m, n_e))

        ax.legend(handles=legend_handles, fontsize=8, loc="best")
        fig.tight_layout()

        def update_phase(frame):
            idx = frame_idx[frame]
            result = []
            for lines, dots, y_m, act_m, n_e in phase_artists:
                for e in range(n_e):
                    if act_m is not None:
                        mask = act_m[: idx + 1, e].astype(bool)
                        lines[e].set_data(y_m[: idx + 1, e, 0][mask], y_m[: idx + 1, e, 1][mask])
                        if mask.any():
                            last = np.where(mask)[0][-1]
                            dots[e].set_data([y_m[last, e, 0]], [y_m[last, e, 1]])
                        else:
                            dots[e].set_data([], [])
                    else:
                        lines[e].set_data(y_m[: idx + 1, e, 0], y_m[: idx + 1, e, 1])
                        dots[e].set_data([y_m[idx, e, 0]], [y_m[idx, e, 1]])
                    result.extend([lines[e], dots[e]])
            return result

        anim = FuncAnimation(fig, update_phase, frames=n_frames, interval=50, blit=True)
        path = os.path.join(out_dir, "trajectory_phase.gif")
        anim.save(path, writer="pillow", fps=16, dpi=100)
        plt.close(fig)
        print(f"  saved → {path}")

    # --- y vs t GIF ---
    n_dims = min(y_dim, 4)
    fig, axes = plt.subplots(n_dims, 1, figsize=(9, 3 * n_dims), squeeze=False)
    fig.suptitle(f"State vs time — {system_name}", fontsize=13, fontweight="bold")
    dim_labels = {0: "y₀", 1: "y₁", 2: "y₂", 3: "y₃"}

    yt_artists = []
    for d in range(n_dims):
        ax = axes[d][0]
        vals_y = np.concatenate([_safe_ravel(m[0][..., d], m[2], n_show) for m in methods])
        vals_t = np.concatenate([_safe_ravel(m[1], m[2], n_show) for m in methods])
        margin = 0.1 * (vals_y.max() - vals_y.min() + 1e-6)
        ax.set_xlim(vals_t.min(), vals_t.max())
        ax.set_ylim(vals_y.min() - margin, vals_y.max() + margin)
        ax.set_xlabel("ODE time t")
        ax.set_ylabel(dim_labels.get(d, f"y[{d}]"))
        ax.grid(True, linewidth=0.4)

        dim_art = []
        for y_m, t_m, act_m, ls, col, label in methods:
            n_e = min(n_show, y_m.shape[1])
            lines, dots = [], []
            for e in range(n_e):
                (ln,) = ax.plot([], [], ls, color=col, lw=0.9, alpha=0.5)
                (dt,) = ax.plot([], [], "o", color=col, ms=3, alpha=0.7)
                lines.append(ln)
                dots.append(dt)
            dim_art.append((lines, dots, y_m, t_m, act_m, n_e))
        yt_artists.append(dim_art)

    axes[0][0].legend(handles=legend_handles, fontsize=8, loc="best")
    fig.tight_layout()

    def update_yt(frame):
        idx = frame_idx[frame]
        result = []
        for d, dim_art in enumerate(yt_artists):
            for lines, dots, y_m, t_m, act_m, n_e in dim_art:
                for e in range(n_e):
                    if act_m is not None:
                        mask = act_m[: idx + 1, e].astype(bool)
                        lines[e].set_data(t_m[: idx + 1, e][mask], y_m[: idx + 1, e, d][mask])
                        if mask.any():
                            last = np.where(mask)[0][-1]
                            dots[e].set_data([t_m[last, e]], [y_m[last, e, d]])
                        else:
                            dots[e].set_data([], [])
                    else:
                        lines[e].set_data(t_m[: idx + 1, e], y_m[: idx + 1, e, d])
                        dots[e].set_data([t_m[idx, e]], [y_m[idx, e, d]])
                    result.extend([lines[e], dots[e]])
        return result

    anim = FuncAnimation(fig, update_yt, frames=n_frames, interval=50, blit=True)
    path = os.path.join(out_dir, "trajectory_yt.gif")
    anim.save(path, writer="pillow", fps=16, dpi=100)
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Per-mu evaluation table
# ---------------------------------------------------------------------------


def _steps_to_completion(traces: dict):
    """Return per-env steps to completion (active steps count)."""
    active = np.array(traces["active"], dtype=np.float32)  # (T, E)
    return active.sum(axis=0)  # (E,)


def _solver_stats(traces: dict):
    """Return per-env (accepted, rejected) counts."""
    keep = np.array(traces["keep_step"], dtype=np.float32)
    active = np.array(traces["active"], dtype=np.float32)
    accepted = (keep * active).sum(axis=0)
    rejected = ((1.0 - keep) * active).sum(axis=0)
    return accepted, rejected


def run_mu_table(
    vae,
    policy,
    env,
    config,
    mus: list[float],
    num_repeats: int,
    T: int,
    sco: int,
    rng_key,
    sc=None,
):
    """Evaluate policy, PID, and optionally diffeqsolve at each mu value."""
    vae_graphdef, vae_params = nnx.split(vae)
    policy_graphdef, policy_params = nnx.split(policy)

    @jax.jit
    def run_policy(vae_params, policy_params, env_params_batch, rng_key):
        vae_ = nnx.merge(vae_graphdef, vae_params)
        policy_ = nnx.merge(policy_graphdef, policy_params)
        return collect_analysis_rollout(
            vae_, policy_, env, env_params_batch, num_repeats, T, rng_key
        )

    @jax.jit
    def run_pid(env_params_batch, rng_key):
        return collect_pid_analysis_rollout(env, env_params_batch, num_repeats, T, sco, rng_key)

    run_diffeqsolve = None
    if sc is not None:

        @jax.jit
        def run_diffeqsolve(env_params_batch, rng_key):
            return collect_diffeqsolve_analysis_rollout(
                sc, env, config, env_params_batch, num_repeats, T, rng_key
            )

    results = []
    all_mu_traces = []
    for mu in mus:
        rng_key, k1, k2, k3 = jax.random.split(rng_key, 4)
        env_params_batch = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_repeats,) + x.shape) if hasattr(x, "shape") else x,
            ODEParams(lam=jnp.float32(mu), max_steps=T),
        )

        pol_traces = run_policy(vae_params, policy_params, env_params_batch, k1)
        pol_traces = jax.tree.map(np.array, pol_traces)

        pid_traces = run_pid(env_params_batch, k2)
        pid_traces = jax.tree.map(np.array, pid_traces)

        pol_steps = _steps_to_completion(pol_traces)
        pol_acc, pol_rej = _solver_stats(pol_traces)
        pol_t = np.array(pol_traces["state_t"][-1])

        pid_steps = _steps_to_completion(pid_traces)
        pid_acc, pid_rej = _solver_stats(pid_traces)
        pid_t = np.array(pid_traces["state_t"][-1])

        row = {
            "mu": mu,
            "pol_steps": float(pol_steps.mean()),
            "pol_acc": float(pol_acc.mean()),
            "pol_rej": float(pol_rej.mean()),
            "pol_t": float(pol_t.mean()),
            "pid_steps": float(pid_steps.mean()),
            "pid_acc": float(pid_acc.mean()),
            "pid_rej": float(pid_rej.mean()),
            "pid_t": float(pid_t.mean()),
        }

        trace_set = {"pol": pol_traces, "pid": pid_traces}

        if run_diffeqsolve is not None:
            ds_traces, ds_stats = run_diffeqsolve(env_params_batch, k3)
            ds_stats = jax.tree.map(np.array, ds_stats)
            row["ds_steps"] = float(ds_stats["total_steps"].mean())
            row["ds_acc"] = float(ds_stats["acceptance_rate"].mean())
            row["ds_t"] = float(ds_stats["t_reached"].mean())
            trace_set["ds"] = jax.tree.map(np.array, ds_traces)

        all_mu_traces.append(trace_set)
        results.append(row)
        print(f"  mu={mu:.0f} done.")

    return results, all_mu_traces


def _format_speedup(pol_steps, pid_steps):
    if pol_steps <= 0 or pid_steps <= 0:
        return "  N/A"
    ratio = pid_steps / pol_steps
    if ratio >= 1:
        return f"{ratio:5.2f}×"
    else:
        return f"{ratio:5.2f}×"


def _build_rows(results: list[dict]):
    has_ds = "ds_steps" in results[0] if results else False
    rows = []
    for r in results:
        pol_total = r["pol_acc"] + r["pol_rej"]
        pol_rate = r["pol_acc"] / max(pol_total, 1) * 100
        row = {
            "mu": r["mu"],
            "pol_steps": r["pol_steps"],
            "pol_acc": r["pol_acc"],
            "pol_rej": r["pol_rej"],
            "pol_rate": pol_rate,
            "pol_t": r["pol_t"],
            "pid_steps": r["pid_steps"],
            "pid_t": r["pid_t"],
            "speedup": _format_speedup(r["pol_steps"], r["pid_steps"]),
        }
        if has_ds:
            row["ds_steps"] = r["ds_steps"]
            row["ds_acc"] = r.get("ds_acc", 0) * 100
            row["ds_t"] = r["ds_t"]
            row["ds_speedup"] = _format_speedup(r["ds_steps"], r["pid_steps"])
        rows.append(row)
    return rows


def print_mu_table(results: list[dict], T: int, t_end: float):
    """Print per-task rollout metrics."""
    rows = _build_rows(results)
    has_ds = "ds_steps" in rows[0] if rows else False
    ds_block = "─" * 28 + "┼" if has_ds else ""
    sep = "─" * 6 + "──┼" + "─" * 36 + "┼" + ds_block + "─" * 22 + "┼"
    ds_hdr1 = f"{'RL (diffeqsolve)':^28s}│" if has_ds else ""
    hdr1 = f"{'':>6s}  │{'POLICY (env.step)':^36s}│{ds_hdr1}{'PID':^22s}│"
    ds_hdr2 = f"{'steps':>8s} {'acc%':>6s} {'t':>6s} {'Δ':>5s} │" if has_ds else ""
    hdr2 = (
        f"{'mu':>6s}  │"
        f"{'steps':>8s} {'acc':>7s} {'rej':>7s} {'rate':>6s} {'t':>6s}"
        f" │{ds_hdr2}{'steps':>8s} {'t':>6s} {'Δ':>5s} │"
    )
    print(f"\nPer-μ evaluation (budget={T}, t_end={t_end})")
    print(sep)
    print(hdr1)
    print(hdr2)
    print(sep)
    for r in rows:
        ds_cols = ""
        if has_ds:
            ds_cols = (
                f"{r['ds_steps']:8.0f} {r['ds_acc']:5.1f}% {r['ds_t']:6.1f} {r['ds_speedup']:>5s} │"
            )
        print(
            f"{r['mu']:6.0f}  │"
            f"{r['pol_steps']:8.0f} {r['pol_acc']:7.0f} {r['pol_rej']:7.0f} "
            f"{r['pol_rate']:5.1f}% {r['pol_t']:6.1f}"
            f" │{ds_cols}{r['pid_steps']:8.0f} {r['pid_t']:6.1f} {r['speedup']:>5s} │"
        )
    print(sep)


def save_mu_table_txt(results: list[dict], T: int, t_end: float, path: str):
    """Write per-task rollout metrics as a text table."""
    import io

    buf = io.StringIO()

    def _print(s=""):
        buf.write(s + "\n")

    rows = _build_rows(results)
    has_ds = "ds_steps" in rows[0] if rows else False
    _print(f"Per-μ evaluation (budget={T}, t_end={t_end})")
    _print()
    ds_hdr = f"  {'DS steps':>9s} {'DS acc%':>7s} {'DS t':>6s} {'DS Δ':>6s}" if has_ds else ""
    _print(
        f"{'mu':>6s}  {'steps':>8s} {'acc':>7s} {'rej':>7s} {'rate':>6s} {'t':>6s}"
        f"{ds_hdr}  {'PID steps':>10s} {'PID t':>6s} {'speedup':>7s}"
    )
    _print("-" * (80 + (35 if has_ds else 0)))
    for r in rows:
        ds_cols = ""
        if has_ds:
            ds_cols = (
                f"  {r['ds_steps']:9.0f} {r['ds_acc']:6.1f}% {r['ds_t']:6.1f} {r['ds_speedup']:>6s}"
            )
        _print(
            f"{r['mu']:6.0f}  {r['pol_steps']:8.0f} {r['pol_acc']:7.0f} {r['pol_rej']:7.0f} "
            f"{r['pol_rate']:5.1f}% {r['pol_t']:6.1f}"
            f"{ds_cols}  {r['pid_steps']:10.0f} {r['pid_t']:6.1f} {r['speedup']:>7s}"
        )
    _print("-" * (80 + (35 if has_ds else 0)))

    with open(path, "w") as f:
        f.write(buf.getvalue())
    print(f"  saved → {path}")


def save_mu_table_latex(results: list[dict], T: int, t_end: float, path: str):
    """Write per-task rollout metrics as a LaTeX table."""
    rows = _build_rows(results)
    has_ds = "ds_steps" in rows[0] if rows else False
    ds_cols = "rrr" if has_ds else ""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{Per-$\\mu$ evaluation (budget={T}, $t_{{\\mathrm{{end}}}}={t_end}$)}}",
        r"\label{tab:mu_eval}",
        f"\\begin{{tabular}}{{r rrrrc {ds_cols}rrr}}",
        r"\toprule",
    ]
    if has_ds:
        lines += [
            r" & \multicolumn{5}{c}{\textbf{Policy}} & \multicolumn{3}{c}{\textbf{RL (diffeqsolve)}} & \multicolumn{3}{c}{\textbf{PID}} \\",
            r"\cmidrule(lr){2-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}",
            r"$\mu$ & Steps & Acc & Rej & Rate & $t$ & Steps & Acc\% & $t$ & Steps & $t$ & Speedup \\",
        ]
    else:
        lines += [
            r" & \multicolumn{5}{c}{\textbf{Policy}} & \multicolumn{3}{c}{\textbf{PID}} \\",
            r"\cmidrule(lr){2-6} \cmidrule(lr){7-9}",
            r"$\mu$ & Steps & Acc & Rej & Rate & $t$ & Steps & $t$ & Speedup \\",
        ]
    lines.append(r"\midrule")
    for r in rows:
        ds_data = ""
        if has_ds:
            ds_data = f" & {r['ds_steps']:.0f} & {r['ds_acc']:.1f}\\% & {r['ds_t']:.1f}"
        lines.append(
            f"  {r['mu']:.0f} & {r['pol_steps']:.0f} & {r['pol_acc']:.0f} & {r['pol_rej']:.0f} & "
            f"{r['pol_rate']:.1f}\\% & {r['pol_t']:.1f}"
            f"{ds_data} & "
            f"{r['pid_steps']:.0f} & {r['pid_t']:.1f} & {r['speedup'].strip()} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run single-rollout diagnostics and save their figures and tables."""
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────
    if args.checkpoint is not None:
        args.checkpoint = resolve_checkpoint(args.checkpoint)

    # Auto-detect config.yaml bundled with checkpoint if --config not given.
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

    T = config.rollout_steps
    num_envs = args.num_envs
    seed = args.seed
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] System : {config.env.system}")
    print(f"[*] Envs   : {num_envs}  Steps: {T}  Seed: {seed}")
    print(f"[*] Output : {out_dir}")

    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

    # ── Build models ──────────────────────────────────────────────
    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, seed)

    if args.checkpoint is not None:
        vae, policy = try_load_checkpoint(
            vae, policy, args.checkpoint, backbone=config.backbone, algo=config.algo
        )
    else:
        print("[!] No checkpoint supplied — using random weights.")

    # ── Build learned controller (if checkpoint available) ──────
    sc = None
    if args.checkpoint is not None:
        from steppo.envs.ode.learned_controller import LearnedController

        sc = LearnedController.from_checkpoint(args.checkpoint, config, env)
        print("[*] Built LearnedController from checkpoint.")

    # ── Per-mu table mode ────────────────────────────────────────
    if args.mu_table is not None:
        mus = args.mu_table if args.mu_table else [1, 5, 10, 20, 50, 100, 200]
        rng = jax.random.PRNGKey(seed)
        feature_dims = env._spec.feature_dims
        sco = compute_step_context_offset(config.env.obs_features, feature_dims)
        print(f"[*] Running per-μ table: {mus}")
        results, all_mu_traces = run_mu_table(
            vae,
            policy,
            env,
            config,
            mus,
            num_repeats=args.num_envs,
            T=T,
            sco=sco,
            rng_key=rng,
            sc=sc,
        )
        t_end_val = float(config.env.t_end)
        print_mu_table(results, T, t_end_val)
        save_mu_table_txt(results, T, t_end_val, os.path.join(out_dir, "mu_table.txt"))
        save_mu_table_latex(results, T, t_end_val, os.path.join(out_dir, "mu_table.tex"))

        print("[*] Generating per-μ trajectory GIFs …")
        for mu, trace_set in zip(mus, all_mu_traces):
            mu_dir = os.path.join(out_dir, f"mu_{mu:.0f}")
            os.makedirs(mu_dir, exist_ok=True)
            make_trajectory_gifs(
                trace_set["pol"],
                mu_dir,
                config.env.system,
                pid_traces=trace_set["pid"],
                diffeqsolve_traces=trace_set.get("ds"),
            )
        print(f"\n[+] All outputs saved to: {out_dir}")
        return

    # ── Sample tasks + RL/PID/diffeqsolve rollouts (single cached stage) ──
    feature_dims = env._spec.feature_dims
    sco = compute_step_context_offset(config.env.obs_features, feature_dims)

    def _compute_rollouts():
        rng_ = jax.random.PRNGKey(seed)
        rng_, key_task, key_rollout = jax.random.split(rng_, 3)
        task_keys = jax.random.split(key_task, num_envs)
        env_params_batch_ = jax.vmap(env.sample_task)(task_keys)
        true_params_, param_label_ = get_task_param(env_params_batch_, config.env.system)

        print("[*] Compiling and running analysis rollout …")
        vae_graphdef, vae_params = nnx.split(vae)
        policy_graphdef, policy_params = nnx.split(policy)

        @jax.jit
        def run_rollout(vae_params, policy_params, env_params_batch, rng_key):
            vae_ = nnx.merge(vae_graphdef, vae_params)
            policy_ = nnx.merge(policy_graphdef, policy_params)
            return collect_analysis_rollout(
                vae_, policy_, env, env_params_batch, num_envs, T, rng_key
            )

        traces_ = run_rollout(vae_params, policy_params, env_params_batch_, key_rollout)
        traces_ = jax.tree.map(np.array, traces_)
        print("[*] Rollout complete.")

        print("[*] Running PID baseline rollout …")
        rng_, key_pid = jax.random.split(rng_)

        @jax.jit
        def run_pid_rollout(env_params_batch, rng_key):
            return collect_pid_analysis_rollout(env, env_params_batch, num_envs, T, sco, rng_key)

        pid_traces_ = run_pid_rollout(env_params_batch_, key_pid)
        pid_traces_ = jax.tree.map(np.array, pid_traces_)
        print("[*] PID rollout complete.")

        diffeqsolve_traces_ = None
        diffeqsolve_stats_ = None
        if sc is not None:
            print("[*] Running diffeqsolve (LearnedController) rollout …")
            rng_, key_ds = jax.random.split(rng_)

            @jax.jit
            def run_diffeqsolve_rollout(env_params_batch, rng_key):
                return collect_diffeqsolve_analysis_rollout(
                    sc, env, config, env_params_batch, num_envs, T, rng_key
                )

            diffeqsolve_traces_, diffeqsolve_stats_ = run_diffeqsolve_rollout(
                env_params_batch_, key_ds
            )
            diffeqsolve_traces_ = jax.tree.map(np.array, diffeqsolve_traces_)
            diffeqsolve_stats_ = jax.tree.map(np.array, diffeqsolve_stats_)
            print("[*] diffeqsolve rollout complete.")

        return (
            env_params_batch_,
            true_params_,
            param_label_,
            traces_,
            pid_traces_,
            diffeqsolve_traces_,
            diffeqsolve_stats_,
        )

    cache_payload = {
        "checkpoint": args.checkpoint,
        "env_config": dataclasses.asdict(config.env),
        "num_envs": num_envs,
        "steps": T,
        "seed": seed,
        "has_learned_controller": sc is not None,
    }
    (
        env_params_batch,
        true_params,
        param_label,
        traces,
        pid_traces,
        diffeqsolve_traces,
        diffeqsolve_stats,
    ) = load_or_compute(
        "analyse_rollout",
        cache_payload,
        _compute_rollouts,
    )
    print("[*] Generating plots …")

    model_config = getattr(config, get_backbone(config.backbone).config_attr)
    latent_dim = model_config.total_latent_dim
    system_name = config.env.system

    # ── Plots ─────────────────────────────────────────────────────
    plot_belief_mu(traces, out_dir, latent_dim)
    plot_belief_sigma(traces, out_dir, latent_dim)
    plot_ode_trajectory(
        traces, out_dir, system_name, pid_traces=pid_traces, diffeqsolve_traces=diffeqsolve_traces
    )
    plot_task_vs_belief(traces, out_dir, true_params, system_name)
    plot_rejection_rate(traces, out_dir, pid_traces=pid_traces)
    plot_policy_action(traces, out_dir, pid_traces=pid_traces)
    plot_belief_by_task_bin(traces, out_dir, true_params, param_label, latent_dim)
    plot_acceptance_vs_task(
        traces,
        out_dir,
        true_params,
        param_label,
        pid_traces=pid_traces,
        diffeqsolve_stats=diffeqsolve_stats,
    )
    plot_completion_vs_task(
        traces,
        out_dir,
        true_params,
        param_label,
        T,
        pid_traces=pid_traces,
        diffeqsolve_stats=diffeqsolve_stats,
        t_end=float(config.env.t_end),
    )

    print("[*] Generating trajectory GIFs …")
    make_trajectory_gifs(
        traces, out_dir, system_name, pid_traces=pid_traces, diffeqsolve_traces=diffeqsolve_traces
    )

    if args.noise_samples > 0:
        print("[*] Running VAE input-noise injection analysis …")
        run_noise_injection_analysis(
            vae,
            policy,
            env,
            config,
            out_dir,
            num_samples=args.noise_samples,
            noise_ratios=args.noise_ratios,
            spike_fracs=args.noise_spike_fracs,
            seed=seed,
        )

    print(f"\n[+] All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
