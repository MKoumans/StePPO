"""CPU single-trajectory PID versus trained LearnedController latency.

JAX_PLATFORMS=cpu PYTHONPATH=. python research/ode/inference/benchmark_checkpoint.py CHECKPOINT --mu 10 --output results.json

``--mu`` is the first task parameter (``mu`` for Van der Pol, ``B`` for
Brusselator); the task name is retained for command-line compatibility.
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
from steppo.envs.ode import ODEEnv, _make_solver
from steppo.envs.ode.learned_controller import LearnedController
from steppo.envs.ode.systems import get_system
from steppo.training.pid_solve import env_pid_controller
from steppo.utils.checkpoint import build_models


def compile_solve(env, controller, y0, task, max_steps, solver=None):
    """Compile one diffrax solve for repeated controller timing."""

    def solve(y0, task):
        sol = diffrax.diffeqsolve(
            env.terms,
            env.solver if solver is None else solver,
            t0=env.t0,
            t1=env.t_end,
            dt0=env.dt0,
            y0=y0,
            args=task,
            stepsize_controller=controller,
            max_steps=max_steps,
            saveat=diffrax.SaveAt(t1=True),
            throw=False,
        )
        return (
            sol.ys[0],
            sol.result == diffrax.RESULTS.successful,
            sol.stats["num_steps"],
            sol.stats["num_accepted_steps"],
            sol.stats["num_rejected_steps"],
        )

    return jax.jit(solve).lower(y0, task).compile()


def main():
    """Benchmark a learned controller on one fixed task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-zero-specialization", action="store_true")
    cli = parser.parse_args()
    if cli.repeats < 1:
        parser.error("--repeats must be positive")
    if jax.default_backend() != "cpu":
        parser.error("Run with JAX_PLATFORMS=cpu")
    config_path = cli.checkpoint.parent / "config.yaml"
    config = load_config_from_yaml(TrainConfig, str(config_path), strict=False)
    env = ODEEnv(config.env, config.rollout_steps)
    # Explicit CPU targets override CUDA sharding stored by the training run.
    with ocp.StandardCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(cli.checkpoint.resolve()).item_metadata.tree
        sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
        target = jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
            metadata,
        )
        raw = checkpointer.restore(cli.checkpoint.resolve(), target=target)
    if config.backbone != "varibad" or config.algo != "ppo":
        parser.error("This benchmark currently restores VariBAD/PPO checkpoints")
    vae, policy = build_models(config, env, config.seed)
    restored_models = []
    changed = []
    for model, key in [(vae, "vae_params"), (policy, "ppo_params")]:
        graph, state = nnx.split(model)
        initial_leaves, tree = jax.tree.flatten(state)
        saved_leaves = jax.tree.leaves(raw[key])
        if len(initial_leaves) != len(saved_leaves):
            raise ValueError(f"{key}: checkpoint structure mismatch")
        for initial, saved in zip(initial_leaves, saved_leaves):
            if initial.shape != saved.shape or initial.dtype != saved.dtype:
                raise ValueError(f"{key}: checkpoint shape/dtype mismatch")
        changed.append(
            sum(
                not np.array_equal(np.asarray(a), np.asarray(b))
                for a, b in zip(initial_leaves, saved_leaves)
            )
        )
        restored_models.append(nnx.merge(graph, jax.tree.unflatten(tree, saved_leaves)))
    if not all(changed):
        raise ValueError("Expected trained parameters different from initialization")
    envelopes = {k: float(v) for k, v in raw.get("control_envelopes", {}).items()}
    vae, policy = restored_models
    learned = LearnedController.from_models(
        vae,
        policy,
        config,
        env,
        control_envelopes=envelopes if config.pid_fallback.enabled else None,
        specialize_zero_inference=not cli.disable_zero_specialization,
    )
    print(
        f"Restored iteration {int(raw['iteration'])} on CPU; changed parameter leaves {changed}",
        flush=True,
    )
    pid = env_pid_controller(config.env)
    task = jnp.array([cli.mu, 0.0], dtype=env.t0.dtype)
    y0 = (
        get_system(config.env.system)
        .y0(task, jax.random.PRNGKey(0), config.env)
        .astype(env.t0.dtype)
    )
    max_steps = cli.max_steps or config.eval_max_steps or config.rollout_steps
    cases = []
    for name, controller in [("pid", pid), ("trained_rl", learned)]:
        executable = compile_solve(env, controller, y0, task, max_steps)
        for _ in range(5):
            result = jax.block_until_ready(executable(y0, task))
        if not bool(result[1]):
            raise RuntimeError(f"{name} failed: {jax.device_get(result)}")
        cases.append(dict(name=name, executable=executable, samples=[]))
        print(f"Compiled/warmed {name}; {int(result[2])} attempts", flush=True)
    rng = np.random.default_rng(0)
    for _ in range(cli.repeats):
        for i in rng.permutation(len(cases)):
            case = cases[i]
            start = time.perf_counter_ns()
            result = jax.block_until_ready(case["executable"](y0, task))
            case["samples"].append((time.perf_counter_ns() - start) / 1e6)
            case["result"] = result
    # Untimed numerical reference, using a tighter implicit solve.
    reference_controller = diffrax.PIDController(rtol=1e-10, atol=1e-12, dtmax=config.env.dt_max)
    reference_executable = compile_solve(
        env, reference_controller, y0, task, 100000, solver=_make_solver(1e-10, 1e-12)
    )
    reference = jax.device_get(jax.block_until_ready(reference_executable(y0, task)))
    if not bool(reference[1]):
        raise RuntimeError("Reference solve failed")
    rows = []
    for case in cases:
        final, success, steps, accepted, rejected = jax.device_get(case["result"])
        samples = case["samples"]
        row = dict(
            controller=case["name"],
            median_ms=float(np.median(samples)),
            p10_ms=float(np.percentile(samples, 10)),
            p90_ms=float(np.percentile(samples, 90)),
            steps=int(steps),
            accepted=int(accepted),
            rejected=int(rejected),
            success=bool(success),
            final_y=final.tolist(),
            terminal_linf_error=float(np.max(np.abs(final - reference[0]))),
            us_per_attempt=float(np.median(samples) * 1000 / int(steps)),
            samples_ms=samples,
        )
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)
    output = dict(
        checkpoint=str(cli.checkpoint.resolve()),
        config=str(config_path.resolve()),
        checkpoint_iteration=int(raw["iteration"]),
        changed_parameter_leaves=changed,
        mu=cli.mu,
        y0=np.asarray(y0).tolist(),
        t_end=float(env.t_end),
        rtol=config.env.rtol,
        atol=config.env.atol,
        dt0=float(env.dt0),
        dtmin=config.env.dt_min,
        dtmax=config.env.dt_max,
        max_steps=max_steps,
        controller_budget=config.rollout_steps,
        encoder=config.vae.encoder.encoder_type,
        encoder_inputs=list(config.vae.encoder.encoder_inputs),
        encoder_hidden_dims=list(config.vae.encoder.linear_hidden_dims),
        actor_hidden_dims=list(config.ppo.policy.hidden_dims),
        progress_warp=learned.warp is not None,
        zero_specialization=not cli.disable_zero_specialization,
        solver=type(env.solver).__name__,
        adjoint="RecursiveCheckpointAdjoint",
        device=str(jax.devices()[0]),
        jax=jax.__version__,
        diffrax=diffrax.__version__,
        repeats=cli.repeats,
        reference_y=reference[0].tolist(),
        reference_steps=int(reference[2]),
        note="Trained weights restored from checkpoint. All cases precompiled, five warmups each, timed single-sample calls randomly interleaved and synchronized. Loading, compilation, warmup, host conversion and reference solve excluded.",
        results=rows,
    )
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
