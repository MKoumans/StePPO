"""Ground-truth reward-vs-step-size sweep: no VAE decoder involved.

Unlike `reward_belief_heatmap.py` (which plots the VAE reward decoder's
predictions, a learned approximator that can extrapolate off-distribution),
this replays a real policy rollout and at every point re-runs the REAL
`env.step()` (deterministic given state/action/params) across a grid of
candidate actions, recording actual reward/accept-reject/t reached.

Outputs:
  1. (t, candidate dt) heatmap of real reward, with the policy's actual step
     size overlaid against the myopic-greedy (argmax real one-step reward)
     step size — the gap between the two is reward left on the table.
  2. A full myopic-greedy trajectory from t=0 vs. the policy's actual step
     count and the PID baseline for the same λ (PID is itself greedy/local,
     so this is an apples-to-apples comparison, not just "vs. PID").

Caveat: myopic-greedy is a one-step-lookahead heuristic, not a provable
optimum — treat it as a strong reference, not a formal upper bound.

Usage:
    python research/ode/post_run_analysis/ground_truth_action_sweep.py \\
        --config configs/envs/ode/scalar_decay/scalar_decay_default.yaml \\
        --checkpoint <run>/checkpoints \\
        --lam 5 \\
        --out outputs/analysis
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import os

import diffrax
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.ode.post_run_analysis.analysis_common import col_edges, try_load_checkpoint
from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv, ODEParams, _make_solver
from steppo.envs.ode.systems import get_rhs
from steppo.training.eval import compute_pid_baseline_steps
from steppo.training.pid_solve import env_pid_controller
from steppo.utils.checkpoint import build_models, resolve_checkpoint

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse system, task, action-grid, and output options."""
    p = argparse.ArgumentParser(
        description="Ground-truth (no-decoder) reward-vs-step-size sweep for one ODE rollout"
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (e.g. configs/envs/ode/scalar_decay/scalar_decay_default.yaml)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Orbax checkpoint directory to load policy weights from",
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
        help="Fixed task parameter (ODEParams.lam); sampled via env.sample_task if omitted",
    )
    p.add_argument(
        "--action_grid_size",
        type=int,
        default=41,
        help="Number of candidate step-size actions swept per trajectory point",
    )
    p.add_argument("--steps", type=int, default=None, help="Override rollout_steps from config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default="outputs/analysis",
        help="Output directory for the PNG and summary txt",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Policy rollout, keeping the full ODEState at every step for exact replay
# ---------------------------------------------------------------------------


def collect_policy_rollout(vae, policy, env, params: ODEParams, rng_key, max_steps: int) -> dict:
    """Roll out one deterministic episode, keeping the full pre-step `ODEState`
    (not just derived scalars) so each point can be replayed through the real env."""
    key_reset, key_run = jax.random.split(rng_key)
    obs, env_state = env.reset(key_reset, params)
    gru_hidden = vae.encoder.init_hidden()
    belief_mu, belief_logvar = vae.get_prior()

    states, t_list, dt_before_list, dt_after_list = [], [], [], []
    t_after_list, reward_list, action_list = [], [], []

    for i in range(max_steps):
        key_i = jax.random.fold_in(key_run, i)
        key_act, key_step = jax.random.split(key_i)

        z_policy = jnp.concatenate([belief_mu, belief_logvar], axis=-1)
        action, _, _ = policy.act(obs, z_policy, key_act, deterministic=True)

        states.append(env_state)
        t_list.append(env_state.t)
        dt_before_list.append(env_state.dt)
        action_list.append(jnp.squeeze(action))

        obs_next, env_state_next, reward, done, _ = env.step(key_step, env_state, action, params)
        dt_after_list.append(env_state_next.dt)
        t_after_list.append(env_state_next.t)  # cumulative time actually reached
        reward_list.append(reward)

        action_enc = jnp.asarray(action, dtype=jnp.float32)
        reward_enc = jnp.reshape(reward, (1,)).astype(jnp.float32)
        belief_mu, belief_logvar, gru_hidden = vae.encoder.encode_step(
            action_enc, obs_next, reward_enc, gru_hidden
        )

        obs, env_state = obs_next, env_state_next
        if bool(done):
            break

    stacked_states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
    return {
        "states": stacked_states,  # ODEState, leaves (T, ...)
        "t": np.asarray(t_list, dtype=np.float64),  # (T,)  pre-step time
        "dt_before": np.asarray(dt_before_list, dtype=np.float64),  # (T,)
        "dt_after": np.asarray(dt_after_list, dtype=np.float64),  # (T,)
        "t_after": np.asarray(t_after_list, dtype=np.float64),  # (T,)  cumulative t reached
        "action": np.asarray(action_list, dtype=np.float64),  # (T,)
        "reward": np.asarray(reward_list, dtype=np.float64),  # (T,)
    }


# ---------------------------------------------------------------------------
# Real-env sweep: candidate actions -> real reward/accept/t_reached
# ---------------------------------------------------------------------------


def sweep_real_actions(
    env,
    states,
    dt_before,
    params: ODEParams,
    action_grid_size: int,
    dt_min: float,
    dt_max: float,
    dt_log_gain: float,
):
    """For every trajectory step, sweep candidate actions through the REAL
    `env.step()` (no VAE) at that step's exact pre-step ODEState.
    Returns (reward, accepted, t_reached, dt_grid), all (T, action_grid_size)."""
    action_grid = jnp.linspace(-1.0, 1.0, action_grid_size)  # (G,)
    dummy_key = jax.random.PRNGKey(0)  # env.step doesn't consume it

    def row(state_i, dt_before_i):
        def step_one(a):
            dt_candidate = jnp.clip(dt_before_i * jnp.exp(a * dt_log_gain), dt_min, dt_max)
            _, _, reward, _, info = env.step(dummy_key, state_i, a, params)
            return reward, info["keep_step"], info["t_reached"], dt_candidate

        return jax.vmap(step_one)(action_grid)

    reward, accepted, t_reached, dt_grid = jax.vmap(row)(states, jnp.asarray(dt_before))
    return np.asarray(reward), np.asarray(accepted), np.asarray(t_reached), np.asarray(dt_grid)


def simulate_greedy_trajectory(
    env,
    params: ODEParams,
    action_grid_size: int,
    dt_min: float,
    dt_max: float,
    dt_log_gain: float,
    t_end: float,
    max_steps: int,
) -> dict:
    """Replay from t=0, always taking the action maximizing real one-step reward
    (myopic greedy, no VAE/policy). Reward is monotonic in step size up to the
    acceptance boundary (accepted steps give strictly more progress reward for
    larger dt; rejected steps give 0), so argmax lands on the largest-accepted step."""
    action_grid = jnp.linspace(-1.0, 1.0, action_grid_size)
    dummy_key = jax.random.PRNGKey(0)

    obs, state = env.reset(jax.random.PRNGKey(0), params)
    accepted_count, rejected_count = 0, 0
    step_count = 0
    t_after_list = []
    for _ in range(max_steps):

        def step_one(a):
            _, new_state, reward, done, info = env.step(dummy_key, state, a, params)
            return reward, new_state, done, info["keep_step"]

        rewards, new_states, dones, keeps = jax.vmap(step_one)(action_grid)
        best = int(jnp.argmax(rewards))
        state = jax.tree_util.tree_map(lambda x: x[best], new_states)
        accepted_count += int(keeps[best])
        rejected_count += int(not bool(keeps[best]))
        step_count += 1
        t_after_list.append(float(state.t))
        if bool(dones[best]):
            break
    return {
        "steps": step_count,
        "accepted": accepted_count,
        "rejected": rejected_count,
        "t_reached": float(state.t),
        "t_after": np.asarray(t_after_list, dtype=np.float64),  # cumulative t reached per step
    }


def run_pid_dt_trajectory(env_config, lam: float, pulse_phase: float, max_steps: int):
    """Run the real PID controller and record its accepted step sizes."""
    """Run diffrax's real PIDController once for this λ, recording its (pre-step t,
    dt taken) trace. SaveAt(steps=True) records time after each ACCEPTED step, so
    dt_i = t_i - t_{i-1} is only the accepted-step trace (rejected sizes aren't
    recoverable this way) — but that's what's comparable to the policy's dt(t)."""
    rhs = get_rhs(env_config)
    solver = _make_solver(env_config.rtol, env_config.atol)
    sc = env_pid_controller(env_config)
    task = jnp.array([lam, pulse_phase], dtype=jnp.float32)
    y0 = jnp.ones((1,), dtype=jnp.float32)  # scalar_decay's fixed y0; fine as a default
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        solver,
        t0=0.0,
        t1=float(env_config.t_end),
        dt0=float(env_config.dt0),
        y0=y0,
        args=task,
        stepsize_controller=sc,
        max_steps=max_steps,
        saveat=diffrax.SaveAt(steps=True),
        throw=False,
    )
    ts = np.asarray(sol.ts)
    ts = ts[np.isfinite(ts)]
    ts = np.concatenate([[0.0], ts])
    t_pre = ts[:-1]
    dt_taken = np.diff(ts)
    return t_pre, dt_taken


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ground_truth_heatmap(
    t,
    reward_grid,
    dt_grid,
    dt_actual,
    dt_greedy,
    system: str,
    lam,
    out_path: str,
    log_scale_y: bool,
):
    """Reward background + policy/greedy overlay only — PID is deliberately NOT
    drawn here: the heatmap sweeps actions at the POLICY's own visited states,
    so a PID trace (from its own, independently-simulated states) at a shared
    (t, dt) coordinate would misleadingly sit on a color reflecting the policy's
    state, not PID's. See plot_step_index_comparison for the honest comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    T, G = reward_grid.shape
    t_edges = np.concatenate([t, [t[-1] + (t[-1] - t[-2] if T > 1 else 1.0)]])
    dt_edges = col_edges(dt_grid)
    dt_edges = np.concatenate([dt_edges, dt_edges[-1:]], axis=0)
    mesh = ax.pcolormesh(
        np.tile(t_edges[:, None], (1, G + 1)),
        dt_edges,
        reward_grid,
        shading="flat",
        cmap="RdYlBu_r",
    )
    fig.colorbar(mesh, ax=ax, label="Real reward (actual env.step(), no decoder)")

    ax.plot(t, dt_actual, color="black", linewidth=1.8, label="Step size actually chosen (policy)")
    ax.plot(
        t,
        dt_greedy,
        color="lime",
        linewidth=1.4,
        linestyle="--",
        label="Myopic-greedy step size (argmax real reward)",
    )
    if log_scale_y:
        ax.set_yscale("log")

    lam_str = f"{lam:.3g}" if lam is not None else "sampled"
    ax.set_xlabel("Trajectory time t")
    ax.set_ylabel("Candidate step size Δt")
    ax.set_title(
        f"Ground-truth reward landscape (policy's own states) — system={system}, λ={lam_str}"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_step_index_comparison(
    policy_t_after, greedy_t_after, pid_t_after, system: str, lam, out_path: str, t_end: float
):
    """Compare cumulative time reached per trajectory step index."""
    fig, ax = plt.subplots(figsize=(7, 5))

    def steps_axis(t_after):
        return np.arange(1, len(t_after) + 1)

    ax.plot(
        steps_axis(policy_t_after),
        policy_t_after,
        color="black",
        marker="o",
        markersize=4,
        linewidth=1.8,
        label=f"Policy ({len(policy_t_after)} steps)",
    )
    ax.plot(
        steps_axis(greedy_t_after),
        greedy_t_after,
        color="lime",
        marker="^",
        markersize=4,
        linewidth=1.6,
        linestyle="--",
        label=f"Myopic-greedy ({len(greedy_t_after)} steps)",
    )
    ax.plot(
        steps_axis(pid_t_after),
        pid_t_after,
        color="deepskyblue",
        marker="s",
        markersize=4,
        linewidth=1.6,
        linestyle=":",
        label=f"PID controller ({len(pid_t_after)} steps)",
    )

    ax.axhline(t_end, color="gray", linewidth=1.0, linestyle="-.", alpha=0.6, label="t_end")
    lam_str = f"{lam:.3g}" if lam is not None else "sampled"
    ax.set_xlabel("Environment step index")
    ax.set_ylabel("Cumulative time reached")
    ax.set_title(f"Steps to completion — system={system}, λ={lam_str}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Sweep real one-step rewards and compare policy, greedy, and PID paths."""
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
        params = ODEParams(
            lam=jnp.float32(args.lam), pulse_phase=pulse_phase, max_steps=config.rollout_steps
        )
    else:
        params = env.sample_task(key_task)

    lam_value = float(args.lam) if args.lam is not None else float(params.lam)

    print("[*] Rolling out one policy episode ...")
    rollout = collect_policy_rollout(vae, policy, env, params, key_roll, config.rollout_steps)
    print(f"[*] Rollout length: {len(rollout['t'])} steps")

    log_scale_y = (config.env.dt_max / max(config.env.dt_min, 1e-30)) > 10.0
    dt_kwargs = dict(
        dt_min=float(config.env.dt_min),
        dt_max=float(config.env.dt_max),
        dt_log_gain=float(config.env.dt_log_gain),
    )

    print("[*] Sweeping real candidate actions through env.step() (no decoder) ...")
    reward_grid, accepted_grid, t_reached_grid, dt_grid = sweep_real_actions(
        env, rollout["states"], rollout["dt_before"], params, args.action_grid_size, **dt_kwargs
    )
    greedy_idx = np.argmax(reward_grid, axis=1)
    dt_greedy = dt_grid[np.arange(len(greedy_idx)), greedy_idx]
    reward_greedy = reward_grid[np.arange(len(greedy_idx)), greedy_idx]

    print("[*] Simulating a full myopic-greedy trajectory from t=0 (no VAE/policy) ...")
    greedy_traj = simulate_greedy_trajectory(
        env,
        params,
        args.action_grid_size,
        **dt_kwargs,
        t_end=float(config.env.t_end),
        max_steps=config.rollout_steps,
    )

    print("[*] Computing PID baseline steps for the same λ ...")
    pid_result = compute_pid_baseline_steps(
        config.env,
        mus=[lam_value],
        num_repeats=16,
        max_steps=config.rollout_steps * 10,
    )[lam_value]

    print("[*] Running the real PID controller once to trace its (t, dt) ...")
    t_pid, dt_pid = run_pid_dt_trajectory(
        config.env,
        lam_value,
        float(params.pulse_phase),
        max_steps=config.rollout_steps * 10,
    )
    pid_t_after = t_pid + dt_pid  # cumulative time reached after each PID step

    plot_ground_truth_heatmap(
        rollout["t"],
        reward_grid,
        dt_grid,
        rollout["dt_after"],
        dt_greedy,
        config.env.system,
        lam_value,
        os.path.join(out_dir, "ground_truth_action_heatmap.png"),
        log_scale_y,
    )

    plot_step_index_comparison(
        rollout["t_after"],
        greedy_traj["t_after"],
        pid_t_after,
        config.env.system,
        lam_value,
        os.path.join(out_dir, "ground_truth_step_index.png"),
        t_end=float(config.env.t_end),
    )

    policy_steps = len(rollout["t"])
    per_step_gap = reward_greedy - rollout["reward"]

    summary_lines = [
        f"system={config.env.system}  lambda={lam_value:.4g}  checkpoint={args.checkpoint}",
        "",
        f"{'':>18} {'steps':>8} {'accepted':>9} {'rejected':>9} {'total_reward':>13}",
        f"{'PID baseline':>18} {pid_result['steps']:8.2f} {pid_result['accepted']:9.2f} "
        f"{pid_result['rejected']:9.2f} {'--':>13}",
        f"{'Myopic-greedy':>18} {greedy_traj['steps']:8d} {greedy_traj['accepted']:9d} "
        f"{greedy_traj['rejected']:9d} {'--':>13}",
        f"{'Policy (actual)':>18} {policy_steps:8d} {'--':>9} {'--':>9} {rollout['reward'].sum():13.3f}",
        "",
        "Per-step reward left on the table (myopic-greedy argmax - policy's actual reward),",
        f"at the {len(rollout['t'])} states the policy actually visited:",
        f"  mean={per_step_gap.mean():.4f}  total={per_step_gap.sum():.4f}  max={per_step_gap.max():.4f}",
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    summary_path = os.path.join(out_dir, "ground_truth_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"\n  saved -> {summary_path}")

    print(f"\n[+] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
