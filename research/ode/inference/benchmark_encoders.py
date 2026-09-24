"""Initialized-model latency ablation; not a trained-controller quality comparison.

Run separately on each backend:
  JAX_PLATFORMS=cpu PYTHONPATH=. python research/ode/inference/benchmark_encoders.py --output /tmp/cpu.json
  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda PYTHONPATH=. python research/ode/inference/benchmark_encoders.py --output /tmp/gpu.json
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

# Avoid reserving most of the GPU for this small benchmark.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from steppo.configs.base_config import TrainConfig
from steppo.envs.ode import ODEEnv
from steppo.envs.ode.learned_controller import LearnedController
from steppo.training.pid_solve import env_pid_controller
from steppo.utils.checkpoint import build_models


def main():
    """Benchmark initialized encoder variants across input batch sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--include-pid", action="store_true")
    parser.add_argument("--adjoint", choices=("default", "forward", "both"), default="both")
    cli = parser.parse_args()
    if cli.repeats < 1:
        parser.error("--repeats must be positive")
    rows = []
    cases = []
    kinds = ("pid", "gru", "linear", "zero") if cli.include_pid else ("gru", "linear", "zero")
    for kind in kinds:
        config = TrainConfig()
        config.rollout_steps = 2048
        config.env.system = "scalar_decay"
        config.env.precision = "float64"
        config.env.t_end = 1.0
        config.env.dt0 = 0.01
        config.env.dt_min = 0.001
        config.env.dt_max = 0.05
        config.env.rtol = 1e-5
        config.env.atol = 1e-7
        config.env.progress_warp = False
        config.env.sd_pulse_enabled = False
        config.env.immediate_dt_action = True
        config.vae.latent_dim = 4
        config.vae.latent_dim_long = 0
        arch = config.vae.encoder
        arch.encoder_type = "zero" if kind == "pid" else kind
        arch.hidden_size = 32
        arch.linear_hidden_dims = (32, 32)
        arch.state_embed_dim = 8
        arch.action_embed_dim = 4
        arch.encoder_inputs = ("action", "state")
        arch.activation = "tanh"
        arch.normalize = "none"
        config.ppo.policy.hidden_dims = (64, 32, 32)
        config.ppo.policy.policy_inputs = ("state", "z")
        config.ppo.policy.activation = "tanh"
        config.ppo.policy.normalize = "none"
        env = ODEEnv(config.env, config.rollout_steps)
        if kind == "pid":
            controller = env_pid_controller(config.env)
        else:
            vae, policy = build_models(config, env, seed=0)
            controller = LearnedController.from_models(vae, policy, config, env)
        y0 = jnp.array([1.0], dtype=jnp.float64)
        task = jnp.array([1.0, 0.0], dtype=jnp.float64)
        for adjoint_name, adjoint in (
            ("default", diffrax.RecursiveCheckpointAdjoint()),
            ("forward", diffrax.ForwardMode()),
        ):
            if cli.adjoint != "both" and adjoint_name != cli.adjoint:
                continue

            def solve(y0, task):
                sol = diffrax.diffeqsolve(
                    env.terms,
                    env.solver,
                    t0=env.t0,
                    t1=env.t_end,
                    dt0=env.dt0,
                    y0=y0,
                    args=task,
                    stepsize_controller=controller,
                    max_steps=config.rollout_steps,
                    saveat=diffrax.SaveAt(t1=True),
                    adjoint=adjoint,
                    throw=False,
                )
                return (
                    sol.ys[0],
                    sol.result == diffrax.RESULTS.successful,
                    sol.stats["num_steps"],
                    sol.stats["num_accepted_steps"],
                    sol.stats["num_rejected_steps"],
                )

            executable = jax.jit(solve).lower(y0, task).compile()
            for _ in range(5):
                result = jax.block_until_ready(executable(y0, task))
            cases.append(
                dict(
                    encoder=kind,
                    adjoint=adjoint_name,
                    executable=executable,
                    inputs=(y0, task),
                    samples=[],
                )
            )
            print(f"Compiled and warmed {kind}/{adjoint_name}", flush=True)
    # Compile all cases before timing; shuffle call order within each repetition.
    order_rng = np.random.default_rng(0)
    for _ in range(cli.repeats):
        for i in order_rng.permutation(len(cases)):
            case = cases[i]
            start = time.perf_counter_ns()
            result = jax.block_until_ready(case["executable"](*case["inputs"]))
            case["samples"].append((time.perf_counter_ns() - start) / 1e6)
            case["result"] = result
    for case in cases:
        samples = case["samples"]
        final, success, steps, accepted, rejected = jax.device_get(case["result"])
        if not bool(success):
            raise RuntimeError(f"{case['encoder']}/{case['adjoint']} failed")
        error = float(abs(final[0] - np.exp(-1.0)))
        if error > 1e-5:
            raise RuntimeError(f"{case['encoder']}/{case['adjoint']}: error {error}")
        row = dict(
            encoder=case["encoder"],
            adjoint=case["adjoint"],
            median_ms=float(np.median(samples)),
            p10_ms=float(np.percentile(samples, 10)),
            p90_ms=float(np.percentile(samples, 90)),
            steps=int(steps),
            accepted=int(accepted),
            rejected=int(rejected),
            absolute_error=error,
            us_per_attempt=float(np.median(samples) * 1000 / int(steps)),
            samples_ms=samples,
        )
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)
    output = dict(
        jax=jax.__version__,
        diffrax=diffrax.__version__,
        device=str(jax.devices()[0]),
        device_kind=jax.devices()[0].device_kind,
        host=platform.processor(),
        repeats=cli.repeats,
        solver=type(env.solver).__name__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        timing_order="All cases precompiled; randomized interleaved calls",
        note="Initialized models, shared actor seed/dimensions. Reward excluded. Scalar decay lambda=1, y0=1, t_end=1. ODE float64; models float32. Compilation, five warmups, and host transfers excluded.",
        results=rows,
    )
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
