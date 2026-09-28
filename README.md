# StePPO

**Adaptive Step-Size Control for Initial-Value Problems Using PPO**

Milan Koumans¹, Nicola Pezzotti² and Tristan S.W. Stevens¹

¹ Signal Processing Systems, Eindhoven University of Technology, Eindhoven<br>
² SCAN, ASML, Veldhoven

[![Paper](https://img.shields.io/badge/paper-ICASSP%202027-b31b1b.svg)](#citation)
[![JAX](https://img.shields.io/badge/JAX-Flax%20NNX-blue.svg)](https://github.com/jax-ml/jax)

Reference implementation for the paper, in JAX/Flax NNX.

---

## Overview

ODE solvers are used to model, simulate, and analyse continuous-time systems
found in many signal processing applications. Time integration of these
initial-value problems is traditionally performed using adaptive embedded
Runge--Kutta methods. For proper time-step selection, adaptive solvers accept
or reject each candidate step according to a local error estimate. Their
efficiency is therefore controlled by the rule that selects the next step
size. Classical proportional--integral--derivative (PID) controllers are
computationally inexpensive per decision and reliable, but their reactive
design can lead to costly numerical solves near stiff transitions. We
formulate step-size selection for parametrised IVPs as a Markov decision
process and train a continuous-action policy using proximal policy
optimisation. The policy processes its observation signals using a deep
network while maintaining the embedded solver logic. We evaluate on a decay
function, the Van der Pol system, and the Brusselator system at matched
tolerances, and compare against a standard PID controller and a multi-scale
DeepONet network. We find that the proposed learned policy remains within
prescribed error tolerances while reducing the required number of solve steps
by up to 60\%, achieving a wall-clock time reduction of 58\%, for both in- and
out-of-distribution settings.

The paper evaluates scalar decay, Van der Pol (full range and binned training)
and the Brusselator against a PID controller, an oracle step-size schedule and
a DeepONet surrogate. The configurations of the released models are in
`configs/envs/ode/*/*_icassp2027.yaml`.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/); GPU acceleration
needs CUDA 12 on Linux.

```bash
uv sync
# uv sync --extra logging   # optional: adds Weights & Biases
```

Run commands with `uv run` (as below), or activate the environment with
`source .venv/bin/activate`.

On Windows and macOS `uv sync` installs CPU-only JAX; for a GPU on Windows, the
`steppo.cmd` wrapper runs everything in the dev container. See
[Choosing a setup](docs/installation.md#choosing-a-setup).

A CUDA 12 container is provided (`docker/docker-compose.yml`), and
`scripts/setup/` builds the Apptainer image used on Slurm clusters. Full
instructions, including GPU/JAX caveats, are in
[docs/installation.md](docs/installation.md).

## Quick start

Solve one task with a released model and compare it with PID (see
[REPRODUCE.md](REPRODUCE.md) for the paper figures and
[`examples/quickstart.ipynb`](examples/quickstart.ipynb) for the Python and
`diffrax` API):

```bash
uv run steppo solve van_der_pol --xi 50   # outputs/example/van_der_pol_50.{png,gif}
```

To train:

`PYTHONPATH=.` is required — `research/` is not part of the importable package.
Training and analysis read pre-generated PID datasets (the reward
normalisation and evaluation baselines), which are never built lazily. Generate
them once per config, then train:

```bash
CFG=configs/envs/ode/van_der_pol/van_der_pol_icassp2027.yaml
CONFIG="$CFG" bash scripts/generate-pid-dataset.sh
PYTHONPATH=. uv run python research/ode/run.py --config "$CFG"
scripts/ode-post-run-analysis.sh --run-dir outputs/runs/<DATE>/ode/<system>/<UID>
```

`configs/envs/ode/<system>/<system>_icassp2027.yaml` holds the configuration of
each released paper model; `<system>_default.yaml` is a general starting point.
`bash scripts/generate-pid-datasets-all-ode.sh` generates the datasets for
every system's `*_default.yaml`. Datasets are keyed by config, so generate them
for the exact config you train or evaluate. For batch launching, seed repeats, GPU round-robin and Slurm dispatch,
use the launchers in [`scripts/README.md`](scripts/README.md).

### DeepONet baseline

The DeepONet baseline uses PyTorch/DeepXDE, whose CUDA libraries conflict with
`jax[cuda12]`, so it runs in its own container:

```bash
docker compose -f docker/docker-compose.deeponet.yml up -d
```

See [`research/baselines/deeponet/`](research/baselines/deeponet/).

## Method

Three components are trained jointly:

- **Encoder** — a GRU (or per-step MLP) over embedded
  `(action, state, reward)` tuples, producing a posterior `q(z | τ)` over the
  hidden task.
- **Decoders** — reconstruct reward, next state, task parameters, and solver
  acceptance from `z`, forming the ELBO training signal.
- **Policy** — a PPO actor-critic conditioned on `[state, z]`, with separate
  actor and critic stacks.

An episode is one adaptive ODE solve. The agent emits a scalar
`a ∈ [-1, 1]` applied as `dt ← clip(dt · exp(a · dt_log_gain), dt_min, dt_max)`;
the environment runs an implicit Kvaerno5 step and returns local error and
accept/reject feedback. The hidden task parameter is never observed.

<!-- Ten ODE systems are registered — `scalar_decay`, `van_der_pol`, `robertson`,
`brusselator`, `fitzhugh_nagumo`, `chemical_cascade`, `fosm`, `fosm_smooth`,
`chua`, `chua_smooth` — each a single file plus a registry entry. Three systems
are used in the paper: `scalar_decay`, `van_der_pol` and `brusselator`.  -->
Three systems are used in the paper: `scalar_decay`, `van_der_pol` and `brusselator`.

A trained agent can be deployed as a diffrax
`AbstractAdaptiveStepSizeController`, running inside `diffeqsolve`'s fused XLA
loop as a drop-in replacement for `PIDController`.

## Documentation

| Document | Contents |
| --- | --- |
| [Installation](docs/installation.md) | Prerequisites, uv, Docker, HPC, GPU/JAX notes |
| [Architecture](docs/architecture.md) | POMDP formulation, encoder/decoders/policy, pluggable backbones and policy algorithms |
| [Environments](docs/environments.md) | ODE env, action/observation/reward, registered systems, task splits, pulse forcing |
| [Configuration](docs/configuration.md) | YAML↔dataclass mapping, `extends`/`model`/`pulse_config` resolution, CLI overrides |
| [Training](docs/training.md) | Training loop, loss normalization, imitation warmstart, dataset caches, diagnostics, resuming |
| [Analysis](docs/analysis.md) | Post-run suite, cross-run comparison, seed averaging, baselines, inference benchmarks |
| [Deployment](docs/deployment.md) | Learned controller inside diffrax, PID fallback, Hugging Face model artifacts |
| [Development](docs/development.md) | Tests, JAX gotchas, extension points |
| [`scripts/README.md`](scripts/README.md) | Every launcher and analysis wrapper, in full |

## Repository layout

```
src/steppo/          # Library
├── configs/      #   Dataclass config schema + YAML loader
├── envs/         #   ODEEnv, system registry, learned diffrax controller
├── models/       #   Encoder, decoders, policy, backbone registry, HF artifacts
├── training/     #   Trainer, VAE/PPO updates, rollout, warmstart, eval metrics
└── utils/        #   Buffers, checkpointing, plotting, device selection

research/ode/     # Experiment entrypoints, analysis, baselines, inference
research/baselines/deeponet/  # DeepONet baseline (separate PyTorch container)
configs/          # envs/ (per-system YAML) + models/ (architecture YAML)
scripts/          # Launchers, batch orchestration, dataset generation
tests/            # pytest suite
docs/             # This documentation
```

<!-- ## Citation

```bibtex
@inproceedings{koumans2027steppo,
  title     = {StePPO: Adaptive Step-Size Control for Initial-Value Problems
               Using PPO},
  author    = {Koumans, Milan and Pezzotti, Nicola and Stevens, Tristan S. W.},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal
               Processing (ICASSP)},
  year      = {2027}
}
``` -->

## References

- **VariBAD** — [Zintgraf et al., ICLR 2020](https://arxiv.org/abs/1910.08348)
- **Original PyTorch implementation** — [lmzintgraf/varibad](https://github.com/lmzintgraf/varibad)
- **JAX reference implementation** — [aliang8/varibad_jax](https://github.com/aliang8/varibad_jax)

Built on [JAX](https://github.com/jax-ml/jax),
[Flax NNX](https://flax.readthedocs.io/en/latest/nnx/),
[Optax](https://optax.readthedocs.io/),
[Distrax](https://github.com/google-deepmind/distrax),
[Gymnax](https://github.com/RobertTLange/gymnax) and
[Diffrax](https://github.com/patrick-kidger/diffrax).

## License

Research use.
