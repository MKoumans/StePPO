# Test runs

Install the coverage plugin with `python -m pip install -r tests/requirements.txt`.
The project already provides pytest and pytest-xdist in its development group.
`tests/conftest.py` puts this checkout's `src/` and repository root first on
`sys.path`, so an IDE's editable install cannot silently make a worktree test
another checkout's code.

Run coverage for the runtime and research modules with:

```bash
pytest \
  --cov=src/steppo \
  --cov=research/ode \
  --cov=research/ode/diagnostics \
  --cov=research/ode/inference \
  --cov=research/pid_tuning \
  --cov=research/baselines/deeponet \
  --cov-branch --cov-report=term-missing
```

JAX uses a writable cache under the system temporary directory by default. Set
`JAX_COMPILATION_CACHE_DIR` to choose another writable location.

On GPU hosts, JAX tests use the GPU. On hosts without GPU passthrough, numerical
solver, gradient, neural training, rollout, and end-to-end pipeline tests are
skipped, while CPU-safe tests still run; long neural workloads never fall back
to CPU silently.
Set `STEPPO_REQUIRE_GPU=1` in GPU CI to fail early if device access is missing,
or set `JAX_PLATFORMS=cpu STEPPO_REQUIRE_GPU=0` for a CPU-only run. Setting
`JAX_PLATFORMS` before pytest starts also avoids CUDA initialization attempts
from pytest plugins on hosts without a driver. For parallel GPU runs,
`TEST_GPU_POOL` assigns one listed physical GPU to each pytest-xdist worker in
round-robin order. Clear a preset `CUDA_VISIBLE_DEVICES` so the pool can assign
the per-worker device, and provide a pool of currently available GPUs. Example:

```bash
env -u CUDA_VISIBLE_DEVICES \
  TEST_GPU_POOL=0,2,3 \
  STEPPO_REQUIRE_GPU=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  pytest -n 3 --cov=src/steppo --cov=research/ode --cov-branch --cov-report=term-missing
```

The full PID solver sweep is opt-in because it solves the same trajectories
across multiple ODE systems. Run it with:

```bash
STEPPO_RUN_SLOW_SOLVER_TESTS=1 pytest tests/envs/test_pid_env_vs_diffrax.py
```

The full PID solver sweep is opt-in even on GPU hosts. Repository code cannot
grant device passthrough to a sandbox; CPU-only hosts skip GPU-dependent tests
automatically and continue with CPU-safe tests.

## Coverage priorities

Coverage is used to find untested behavior, not as a substitute for checking
what the behavior means. The highest-value tests assert deterministic contracts:
ODE equations and environment transitions, cache identities and checkpoint
compatibility, replay-buffer alignment, PPO/VAE loss math and masks, evaluation
accounting, and research metric aggregation. Tiny CPU tests cover those seams;
GPU runs cover real solver integration, neural updates, full rollouts, and
checkpoint round trips.

Research entry points that mostly launch long experiments or render figures
should be tested through their small deterministic pieces: argument parsing,
input selection, metric transforms, and exported table schemas. Their full CLI
and plotting lines are not a useful unit-coverage target when they require
large external artifacts or hardware. A zero-coverage entry point remains a
visible gap; it should get a small smoke test when its data or runtime can be
made deterministic without copying an entire experiment into the test suite.
