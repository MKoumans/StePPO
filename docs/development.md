# Development

## Pre-commit checks

Install `pre-commit` in your Python tooling environment and enable the Git hook
once per checkout:

```bash
uv run pre-commit install
```

The hook formats and lints staged Python files with Ruff, then checks common
staged-file issues in YAML/TOML and whitespace, merge markers, and private
keys. Ruff applies safe fixes automatically; review and stage those changes
before committing. Run `uv run pre-commit run --all-files` to apply/check
the hooks across the repository.

The hook stays fast and does not run the full test suite. Tests are run
separately as described below; the full suite includes slow JAX/GPU work.

## Tests

```bash
uv run pytest                              # all tests (benchmark/pipeline excluded by default)
uv run pytest tests/models/test_vae.py            # one module
uv run pytest tests/models/test_vae.py::test_name # one test
uv run pytest --cov=src/steppo --cov=research --cov-branch --cov-report=term-missing
uv run pytest -m benchmark                 # multi-GPU throughput benchmarks
uv run pytest -m pipeline                  # slow end-to-end CLI-script pipeline tests
```

`pyproject.toml` sets `addopts = "-m 'not benchmark and not pipeline'"`, so the
two slow marker groups are opt-in.

The coverage command reports line and branch coverage for both `src/steppo` and
the top-level `research` package. Treat missed lines as prompts for review, not
as a coverage target by themselves: exercised code can still have weak
assertions.

### Full-suite runs

```bash
TEST_GPU_POOL=0,1,2 uv run pytest -n 3     # ~650s -> ~300s on a 3-GPU box
```

Use this for a pre-push or CI-style check — **not** for iterating on one test.
xdist makes every worker independently import and collect the full test set
regardless of how few tests are selected, so on a single test it is pure
overhead. Plain `pytest path::test` is already fast.

## GPU / JAX gotchas

These are the constraints most likely to bite when adding test files or
entrypoints:

- `tests/conftest.py` sets `GPUS`/`CUDA_VISIBLE_DEVICES` and disables XLA memory
  preallocation **before JAX is imported**. This must stay the first thing that
  runs in the test session. `--config`-driven scripts get the same treatment from
  `setup_devices()` on the first two lines of `research/ode/run.py`.
- `jax_enable_x64` is flipped globally, with no reset, by `apply_precision()` —
  which any trainer built over a float64-precision ODE env calls — and by
  `ODEEnv.__init__`. This leaks across tests depending on run order;
  `conftest.py` restores it after every test. Keep that fixture if you add a new
  test file that builds a trainer.
- `conftest.py` sets a persistent XLA compilation cache dir (`.jax_cache/`,
  gitignored) so jitted functions reuse compiled executables across separate
  `pytest` invocations.
- **Never add a per-test `jax.clear_caches()` fixture** "to avoid OOM". One used
  to exist in `test_ode_comprehensive.py` and made every test re-trace from
  scratch (~140 tiny recompiles per test, ~7–9 s each) for no real memory
  benefit; removing it cut that file's runtime from ~480 s to ~190 s with no OOM
  at the existing `XLA_PYTHON_CLIENT_MEM_FRACTION`.

## Style

Ruff handles linting, import sorting, and formatting. Match the surrounding
code for module-level docstrings that explain *why*, dense inline comments on
non-obvious numerics, and dataclass fields documented in place.

## Extending

### A new ODE system

See [environments.md](environments.md#adding-a-system) — one file under
`src/steppo/envs/ode/systems/` plus a registry entry and task bounds.

### A new belief-model backbone

1. Implement a model class exposing the `VariBADVAE` interface
   (`get_prior`, `encode`, `decode`, `sample_z`, plus an `.encoder` with
   `prior`/`init_hidden`/`encode_step`/`encode_trajectory`/`sample`).
2. Implement a `Trainer` subclass with `_init_belief_model`,
   `_update_belief_model` and `_belief_checkpoint_fields`.
3. Register both in `_load_backbones()` in `src/steppo/models/backbones.py`, with
   the `TrainConfig` field holding its config section and whether the policy
   consumes a sampled `z` or `[μ, log σ²]`.

### A new policy algorithm

1. Implement a class exposing `PolicyAlgorithm`'s interface **as staticmethods**
   — `build_model`, `init_state`, `update`, `checkpoint_fields` — plus the
   `name` / `checkpoint_prefix` / `supports_warmstart` class attributes.
   Implementations carry no per-instance state: `Trainer.policy_algo` holds the
   class itself, not an instance.
2. Register it in `_load_policy_algos()` in `src/steppo/training/policy_algos.py`.

`update` may read or mutate exactly five `Trainer` attributes — `profiler`,
`timings`, `env`, `_reward_norm` and `_adv_stats_buffer`. These are
Trainer-private by convention and are the hook's one sanctioned escape hatch;
anything else on `trainer` is off-limits.

## Known inconsistencies

Worth knowing about before trusting a path:

- The `LearnedController` PID fallback is hard-disabled; see
  [deployment.md](deployment.md#control-chart-pid-fallback).
