# Reproducing StePPO (ICASSP 2027)

This repository accompanies *StePPO: Adaptive Step-Size Control for
Initial-Value Problems Using PPO*. StePPO replaces the PID step-size controller
of an adaptive ODE solver with a policy trained by PPO. The policy is
conditioned on a belief over the unknown system parameter ξ, which a
variational encoder infers from the solver's own error estimates as the solve
runs. The released models control an implicit Kvaerno5 solve of scalar decay
(λ), Van der Pol (μ) and the Brusselator (B).

This page covers three things:

1. [Quickstart](#quickstart): load a model, solve a task, and use StePPO as a
   `diffrax` step-size controller.
2. [Figures from the released models](#figures-from-the-released-models-import)
   (`--mode import`): regenerate Figs. 2–5 in minutes.
3. [Figures from scratch](#figures-from-scratch-execute) (`--mode execute`):
   generate the datasets, train every model yourself, then plot.

[`paper/run.sh`](paper/run.sh) runs route 2 or 3 end to end
([All figures in one command](#all-figures-in-one-command)).

## Setup

Route 2 runs on a CPU; route 3 needs a Linux machine with an NVIDIA GPU.
Install as in the [README](README.md#installation), or use the CUDA 12
container:

```bash
git clone https://github.com/MKoumans/StePPO.git && cd StePPO
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec app bash
```

Scripts take `--gpus <id>` to select a GPU. DeepONet training and timing need
PyTorch, whose CUDA libraries conflict with JAX's, so they run in a second
container ([`docker/docker-compose.deeponet.yml`](docker/docker-compose.deeponet.yml)).

## Released artifacts

| Artifact | Location |
| --- | --- |
| StePPO, scalar decay | [`MKoumans/sd`](https://huggingface.co/MKoumans/sd) |
| StePPO, Van der Pol | [`MKoumans/vdp`](https://huggingface.co/MKoumans/vdp) |
| StePPO, Brusselator | [`MKoumans/bru`](https://huggingface.co/MKoumans/bru) |
| DeepONet (one revision per system and training domain) | [`MKoumans/deeponet`](https://huggingface.co/MKoumans/deeponet) |
| PID and Oracle datasets | [`MKoumans/steppo-icassp2027-data`](https://huggingface.co/datasets/MKoumans/steppo-icassp2027-data) |

Each StePPO model's configuration is
`configs/envs/ode/<system>/<system>_icassp2027.yaml`; the scripts check that it
matches the model they load.

## Quickstart

```bash
steppo solve van_der_pol --xi 50
```
```
[Solve] steppo       175 steps, reached t_end, mean log10 error -4.24, final -6.22
[Solve] pid          378 steps, reached t_end, mean log10 error -3.67, final -4.97
[Output] Saved complete solve to "outputs/example/van_der_pol_50.png"
[Output] Saved solve progress (y, dt, err) to "outputs/example/van_der_pol_50.gif"
```

`steppo solve <system> --xi <value> [--plot traj.png]` downloads the model,
solves task ξ with StePPO and with diffrax's PID controller, and compares both
with a tight-tolerance reference. `--gif y dt err` adds the step sizes and the
error against the reference to the animation; `steppo solve --help` lists all
options. The notebook
[`examples/quickstart.ipynb`](examples/quickstart.ipynb) does the same from
Python, and shows how to pass the model to `diffrax.diffeqsolve` as the
`stepsize_controller` of your own solve.

## Figures from the released models (import)

Output goes to `outputs/paper/`: one PDF per figure plus CSVs of the
plotted numbers. Figs. 3 and 4 need the datasets:

```bash
PYTHONPATH=. python paper/download_data.py
```

| Figure | Command | Contents |
| --- | --- | --- |
| 2 | `PYTHONPATH=. python paper/fig2_trajectory.py` | Trajectory and local error at ξ = 5, PID / StePPO / DeepONet; reference solved on the fly |
| 3 | `PYTHONPATH=. python paper/fig3_sweep.py` | Steps and integrated error vs ξ on the test split, Oracle / PID / StePPO / DeepONet |
| 4 | `PYTHONPATH=. python paper/fig4_robustness.py` | Van der Pol: integrated error vs μ for PID, StePPO and DeepONet trained on the full range and on [5, 15] ∪ [35, 45]; trajectories at μ = 7 and μ = 2 |
| 5 | see below | Wall-clock time per solve vs μ, Van der Pol, one sample on a GPU |

DeepONet was trained and timed in PyTorch (DeepXDE), so its Fig. 5 timing runs
first, in the DeepONet container:

```bash
docker compose -f docker/docker-compose.deeponet.yml up -d
docker compose -f docker/docker-compose.deeponet.yml exec deeponet \
  env PYTHONPATH=. python paper/fig5_deeponet_timing.py
PYTHONPATH=. python paper/fig5_wallclock.py
```

`fig5_deeponet_timing.py` also prints the largest difference between the
PyTorch DeepONet and the NumPy version the other figure scripts use. Absolute
times depend on the hardware.

## Figures from scratch (execute)

This route rebuilds everything the previous one downloads: the datasets, the
StePPO models and the DeepONets. Every figure script also takes
`--mode execute`, which loads the models trained here instead of the released
ones.

### 1. Generate the datasets (JAX container)

```bash
for system in scalar_decay van_der_pol brusselator; do
  bash paper/train.sh $system --data-only --gpus 0
done
```

This solves the tasks of each system's `*_icassp2027` config with PID, the
Oracle and a tight-tolerance reference, and writes the PID training caches and
the test-split evaluation datasets to `data/<system>/`. These are the files
`download_data.py` fetches. The generation is deterministic, so the files
match the published ones.

### 2. Train StePPO (JAX container), several hours per system

```bash
nohup bash paper/train.sh scalar_decay --gpus 0 > train_sd.log 2>&1 &
nohup bash paper/train.sh van_der_pol --gpus 1 > train_vdp.log 2>&1 &
nohup bash paper/train.sh brusselator --gpus 2 > train_bru.log 2>&1 &
```

`train.sh` first runs step 1 for its system, reusing the datasets if they
already exist. It then trains StePPO into `outputs/paper/models/<system>/`.
Extra arguments go to `research/ode/run.py`, e.g. `--seed 1`.

### 3. Train the DeepONets (DeepONet container)

```bash
docker compose -f docker/docker-compose.deeponet.yml exec deeponet bash
for system in scalar_decay van_der_pol brusselator; do
  PYTHONPATH=. python research/baselines/deeponet/train_deeponet_unified.py --profile retrain --system $system --domain complete
done
PYTHONPATH=. python research/baselines/deeponet/train_deeponet_unified.py --profile retrain --system van_der_pol --domain binned
```

The checkpoints land in `research/baselines/deeponet/output-deeponet/<system>/models/`.

### 4. Plot

Run the commands of the [previous section](#figures-from-the-released-models-import)
with `--mode execute`, including `fig5_deeponet_timing.py --mode execute`,
and skip `download_data.py`. A script whose model is missing stops and prints
the command that trains it.

Retrained models reproduce the figures statistically, not exactly: GPU
training is not bit-reproducible.

## All figures in one command

[`paper/run.sh`](paper/run.sh) chains the steps above:

```bash
bash paper/run.sh                  # import: download the datasets, plot Figs. 2–5 from the released models
bash paper/run.sh --mode execute   # execute: generate the datasets, train StePPO and DeepONet, then plot
```

It needs the StePPO environment on `PATH` and the DeepONet (PyTorch)
environment at `$DEEPONET_PYTHON` (default `/opt/deeponet/bin/python`). Without
the latter, import mode plots Fig. 5 without DeepONet and execute mode stops.
`--gpus N` selects the GPU, and `--results DIR` copies the figures and CSVs
(and in execute mode the trained models) to `DIR`.
