"""Profile where the trained-controller per-attempt overhead comes from.

This is deliberately a microbenchmark for the exact checkpoint used in the
PID/RL comparison. It times compiled device-resident kernels, synchronously.
Run with JAX_PLATFORMS=cpu.
"""

import argparse
import json
import time
from pathlib import Path

import diffrax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv, ODEState
from steppo.envs.ode.learned_controller import LearnedController
from steppo.envs.ode.systems import get_system
from steppo.utils.checkpoint import build_models


def restore_on_cpu(checkpoint: Path, config: TrainConfig, env: ODEEnv):
    """Restore CUDA-sharded checkpoint leaves onto the local CPU."""
    checkpointer = ocp.StandardCheckpointer()
    metadata = checkpointer.metadata(checkpoint.resolve()).item_metadata.tree
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    target = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
        metadata,
    )
    raw = checkpointer.restore(checkpoint.resolve(), target=target)
    vae, policy = build_models(config, env, config.seed)
    models = []
    for model, key in ((vae, "vae_params"), (policy, "ppo_params")):
        graph, state = nnx.split(model)
        leaves, treedef = jax.tree.flatten(state)
        saved = jax.tree.leaves(raw[key])
        if len(leaves) != len(saved):
            raise ValueError(f"{key}: parameter structure mismatch")
        models.append(nnx.merge(graph, jax.tree.unflatten(treedef, saved)))
    return models[0], models[1], raw


def time_compiled(fn, args, repeats):
    """Compile a function, warm it up, and collect synchronized timings."""
    exe = jax.jit(fn).lower(*args).compile()
    for _ in range(10):
        jax.block_until_ready(exe(*args))
    samples = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        result = jax.block_until_ready(exe(*args))
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return exe, result, np.asarray(samples)


def main():
    """Profile learned-controller solve and bookkeeping overhead."""
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-zero-specialization", action="store_true")
    parser.add_argument("--hlo-output", type=Path)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise SystemExit("Run with JAX_PLATFORMS=cpu")
    config = load_config_from_yaml(
        TrainConfig, str(args.checkpoint.parent / "config.yaml"), strict=False
    )
    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy, raw = restore_on_cpu(args.checkpoint, config, env)
    controller = LearnedController.from_models(
        vae,
        policy,
        config,
        env,
        specialize_zero_inference=not args.disable_zero_specialization,
    )
    spec = get_system(config.env.system)
    task = jnp.array([args.mu, 0.0], dtype=env.t0.dtype)
    y0 = spec.y0(task, jax.random.PRNGKey(0), config.env).astype(env.t0.dtype)

    # Representative post-attempt values taken from a real controller state.
    dt = jnp.asarray(config.env.dt0, dtype=env.t0.dtype)
    y = y0
    obs = controller.obs_fn(
        ODEState(
            y=y,
            t=env.t0,
            dt=dt,
            solver_state_flag=jnp.bool_(True),
            solver_state_f=jnp.zeros_like(y),
            last_log_error=jnp.float32(0.0),
            last_keep_step=jnp.bool_(True),
            step_count=jnp.int32(0),
            budget_exhausted=jnp.float32(0.0),
            accept_ema=jnp.float32(1.0),
            log_error_ema=jnp.float32(0.0),
            log_error_delta=jnp.float32(0.0),
            reject_streak=jnp.int32(0),
        )
    )
    mu, logvar = controller._f32(controller.encoder.prior())
    z = controller._z(mu, logvar)
    hidden = controller._f32(controller.encoder.init_hidden())
    action = controller._f32(controller.policy.infer_action(obs, z, task=None))
    reward = jnp.float32(0.0)
    y1 = y + jnp.float64(1e-4)
    yerr = jnp.zeros_like(y)
    t0 = env.t0
    t1 = env.t0 + dt

    probes = {}
    # Standalone neural and bookkeeping kernels. Inputs and outputs stay on CPU.
    probes["policy"] = (lambda o, z: controller.policy.infer_action(o, z, task=None), (obs, z))
    probes["encoder"] = (
        lambda a, o, r, h: controller.encoder.encode_step(a, o, r, h, task=None),
        (action, obs, jnp.reshape(reward, (1,)), hidden),
    )
    probes["encoder_then_policy"] = (
        lambda a, o, r, h, z0: (
            lambda out: controller.policy.infer_action(o, controller._z(out[0], out[1]), task=None)
        )(controller.encoder.encode_step(a, o, r, h, task=None)),
        (action, obs, jnp.reshape(reward, (1,)), hidden, z),
    )

    def bookkeeping(y, t, dt, error, keep):
        state = ODEState(
            y=y,
            t=t,
            dt=dt,
            solver_state_flag=jnp.bool_(True),
            solver_state_f=jnp.zeros_like(y),
            last_log_error=jnp.log(error + 1e-10),
            last_keep_step=keep,
            step_count=jnp.int32(1),
            budget_exhausted=jnp.float32(1.0) / jnp.float32(config.rollout_steps),
            accept_ema=keep.astype(jnp.float32),
            log_error_ema=jnp.log(error + 1e-10),
            log_error_delta=jnp.log(error + 1e-10),
            reject_streak=jnp.int32(0),
        )
        o = controller.obs_fn(state)
        t_capped = jnp.minimum(t, jnp.float32(config.env.t_end))
        r = (
            (t_capped - t) / jnp.float32(config.env.t_end) * jnp.float32(config.env.reward_factor)
            + jnp.float32(-config.env.survival_penalty)
        ).astype(jnp.float32)
        return o, r

    probes["observation_reward"] = (bookkeeping, (y1, t1, dt, jnp.float32(0.8), jnp.bool_(True)))

    # Full controller adaptation on a synthetic attempt, including all state
    # updates but excluding the ODE solver. This is the cleanest estimate of
    # controller-only work per Diffrax attempt.
    controller_state = controller.init(
        env.terms,
        env.t0,
        env.t0 + env.dt0,
        y0,
        env.dt0,
        task,
        env.terms.vf,
        jnp.float32(5.0),
    )[1]

    probes["adapt_step_size"] = (
        lambda state: controller.adapt_step_size(
            t0, t1, y, y1, task, yerr, jnp.float32(5.0), state
        ),
        (controller_state,),
    )

    rows = []
    for name, (fn, fn_args) in probes.items():
        executable, result, samples = time_compiled(fn, fn_args, args.repeats)
        analysis = executable.cost_analysis()
        if name == "adapt_step_size" and args.hlo_output is not None:
            args.hlo_output.parent.mkdir(parents=True, exist_ok=True)
            args.hlo_output.write_text(executable.as_text())
        row = {
            "probe": name,
            "median_ms": float(np.median(samples)),
            "p10_ms": float(np.percentile(samples, 10)),
            "p90_ms": float(np.percentile(samples, 90)),
            "cost_analysis": analysis,
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "cost_analysis"}), flush=True)

    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "iteration": int(raw["iteration"]),
        "system": config.env.system,
        "mu": args.mu,
        "device": str(jax.devices()[0]),
        "jax": jax.__version__,
        "diffrax": diffrax.__version__,
        "encoder_type": config.vae.encoder.encoder_type,
        "zero_specialization": not args.disable_zero_specialization,
        "encoder_inputs": list(config.vae.encoder.encoder_inputs),
        "linear_hidden_dims": list(config.vae.encoder.linear_hidden_dims),
        "actor_hidden_dims": list(config.ppo.policy.hidden_dims),
        "repeats": args.repeats,
        "note": "Compiled CPU microbenchmarks. Loading, compilation, warmup and host conversion excluded. adapt_step_size measures LearnedController logic only and uses one synthetic attempted step; it excludes Diffrax solver work.",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
