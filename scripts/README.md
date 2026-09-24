# Scripts

Launch, batch, analyze, and stop ODE training runs. Every script prints a
short synopsis with `-h/--help`; this file holds the full reference.

All scripts pick their execution mode automatically:

- **HPC (Slurm)** — when `sbatch` is available, work is submitted
  as Slurm jobs that dispatch into the Apptainer image (`steppo.sif`, built by
  `scripts/setup/create-apptainer.sh`).
- **Direct** — otherwise (e.g. inside Docker where GPUs are already visible),
  work runs on the current machine. Long-running launchers detach themselves
  from the terminal (`setsid` + `nohup`) so runs survive the terminal, SSH
  connection, or screen/tmux pane closing; output moves to a log file whose
  path is printed first. Set `_LAUNCH_DETACHED=1` to disable detaching and
  run in the foreground (used internally on re-exec).

Shared helpers live in `scripts/lib/device.sh`. GPU-backed wrappers use its shared `device_extract_gpu_args` helper; CPU-only and administrative scripts do not expose a GPU selector.

---

## run.sh — single training run

```
sbatch scripts/run.sh                # Slurm + apptainer
bash scripts/run.sh                  # direct on device
bash scripts/run.sh --gpus 5
sbatch --job-name=brusselator --export=ALL,CONFIG=configs/envs/ode/brusselator/brusselator_default.yaml scripts/run.sh
```

Runs `research/ode/run.py --config $CONFIG`. `-g, --gpus DEVICES` (or `--gpus=DEVICES`) selects the visible GPU list; all other arguments are forwarded as-is to `run.py`.

Before that, runs a pre-flight check
(`src/steppo/envs/ode/check_oracle_dataset.py`) that verifies the oracle
imitation-warmstart training set when `$CONFIG` has
`warmstart.enabled: true` and `warmstart.expert: oracle`, and also checks the PID
evaluation caches used by the analysis scripts. The check is read-only and
needs no GPU; it never generates anything. On a miss it prints the exact
PID or oracle generation command to run and `run.sh` exits without training.

Checkpoints, outputs, and the config used land under
`outputs/runs/<RUN_DATE>/<RUN_UID>/{checkpoints,outputs}/`, and the log (once
detached) at `outputs/logs/run/<RUN_DATE>/<env>_<system>_<RUN_UID>.log`.

Env vars:

| Var | Meaning | Default |
| --- | --- | --- |
| `CONFIG` | ODE system config to train on | `configs/envs/ode/chemical_cascade/chemical_cascade_default.yaml` |
| `RUN_UID` | Short run ID shared by log file and `outputs/runs/` dir | generated 8-char hex |
| `RUN_DATE` | `YYYYMMDD` the run is grouped under | today |

## run-all-ode.sh — one run per ODE system

```
scripts/run-all-ode.sh                                    # all systems
scripts/run-all-ode.sh van_der_pol brusselator             # only these
scripts/run-all-ode.sh --gpus 6,7                          # 2-wide, round-robin GPUs 6/7
scripts/run-all-ode.sh --env.progress_warp true --env.t_end 50
scripts/run-all-ode.sh --post-run-analysis false
scripts/run-all-ode.sh --repeat 8 --gpus 0,1,2,3
scripts/run-all-ode.sh van_der_pol --gpus 0 --generate-datasets
```

Submits one job (via `run.sh`) per system's default config
(`configs/envs/ode/<system>/<system>_default.yaml`). Systems:
`van_der_pol fitzhugh_nagumo brusselator chemical_cascade robertson
scalar_decay fosm chua_smooth`.

- `-g, --gpus DEVICES` / `--gpus=DEVICES` — direct mode only: systems are assigned one GPU each,
  round-robin across the list, one job per GPU at a time (launching a system
  waits for the previous job on its assigned GPU). Defaults to GPU 0 for
  every system. Under sbatch all jobs are submitted at once; Slurm queues.
- `-r, --repeat N` (default `1`) — launch each system `N` times with seeds
  `0..N-1`. Repeats count as separate jobs for GPU round-robin and Slurm
  scheduling, and the seed is forwarded to `run.py` via `--seed`. Once a
  system's seeds and their post-run analyses all finish, `ode-average-seeds.sh`
  averages `mu_returns`/`efficiency`/`steps_vs_mu`/`z_sensitivity` across them
  into `outputs/runs/<RUN_DATE>/ode/<system>/seed_average/`.
- `--post-run-analysis BOOL` (default `true`) — run
  `ode-post-run-analysis.sh` for each run once it finishes. Under sbatch each
  analysis is a Slurm job dependent on its run's job (cancelled if the run
  fails); in direct mode it runs right after the run on the same GPU, logged
  to `outputs/logs/run/<DATE>/ode_<system>_<UID>_analysis.log`. Analysis
  outputs land in `outputs/runs/<RUN_DATE>/<RUN_UID>/outputs/post_run_analysis/`.
  With `--repeat N > 1`, this flag also gates `ode-repeat-analysis.sh`,
  which runs automatically once all of a system's repeats finish: under
  sbatch as a CPU-only Slurm job (`--partition=rome`) dependent on that
  system's training jobs (`afterany`, so one crashed seed doesn't block the
  rest); in direct mode right after the final wait, once per system.
- `--KEY VALUE` — any other flag is forwarded to `run.py` as a TrainConfig
  override applied to every system, e.g. `--rollout_steps 400` or nested
  `--env.progress_warp true`.

With `--repeat N`, every seed of the invocation is tagged with a shared
`repeat_batch_uid` (bundled into each run's `checkpoints/config.yaml`), so
`ode-repeat-analysis.sh` can group them back together robustly, even for
runs that finish days apart.

## ode-repeat-analysis.sh — average --repeat seeds

```
scripts/ode-repeat-analysis.sh --system van_der_pol
scripts/ode-repeat-analysis.sh --system van_der_pol --dry-run
scripts/ode-repeat-analysis.sh --system robertson --date 20260721
```

Aggregates the seeds of a `run-all-ode.sh --repeat N` batch into mean +/-
std plots and a `summary.txt`, via
`learning_statistics.py aggregate-repeats`. Groups runs by their
shared `repeat_batch_uid` when present (robust); for older runs without that
tag, falls back to a heuristic — matching bundled `config.yaml` content
(ignoring `run_uid`/`seed`) and, if the same seed appears more than once in
a group, splitting by directory mtime order (seeds are launched
sequentially within one invocation). Heuristic groupings are labeled
`legacy_*` and printed in a manifest before aggregating — always check it,
or pass `--dry-run` to inspect the grouping without writing anything.

- `--system SYSTEM` — required.
- `--root DIR` (default `outputs/runs`) / `--date YYYYMMDD` (default: all dates) —
  restrict which `outputs/runs/<DATE>/ode/<system>/*/` dirs are scanned.
- `--out DIR` (default `outputs/repeat_analysis/<system>`).
- `--metrics KEYS` — comma-separated metric keys to aggregate (default: a
  curated set of `eval/*` metrics).
- `--min-seeds N` (default `2`) — skip batches with fewer usable seeds.
- `--dry-run` — print the inferred grouping and exit.

Needs no GPU — it only reads each run's `outputs/<exp_name>_metrics.json`
and bundled `config.yaml`, so it runs the apptainer image directly (no
`--nv`, no Slurm submission), even from a login node.

## launch_experiments.sh — factorial experiment batch

```
scripts/launch_experiments.sh <system> [--gpus DEVICES] [--base CFG --sweep SPEC] [--notes NOTES] [--post-run-analysis BOOL] [--repeat N]
```

Launches a factorial set of experiments for one system, grouped under a fresh
experiment UID:

```
outputs/experiments/<YYYYMMDD>/<UID>/checkpoints/<system>_e<N>/
outputs/experiments/<YYYYMMDD>/<UID>/outputs/<system>_e<N>/
outputs/experiments/<YYYYMMDD>/<UID>/logs/<system>_e<N>.log
outputs/experiments/<YYYYMMDD>/<UID>/configs/<system>_e<N>.yaml
```

Modes: on HPC each experiment is its own 1-GPU Slurm job (the GPU list
is ignored by the launcher and Slurm schedules the allocation). Elsewhere,
configs are split into contiguous chunks,
one chunk per GPU (16 configs on 2 GPUs runs e1–e8 on GPU0, e9–e16 on GPU1);
experiments within a chunk run sequentially, chunks in parallel. With no `--gpus` list, only the first nvidia-smi-visible GPU is used.

Before dispatching anything, every discovered experiment config is checked
with `src/steppo/envs/ode/check_oracle_dataset.py` (read-only, no generation).
It checks both oracle training caches and PID evaluation caches. If any
config is missing a required dataset and `--generate-datasets` is off, every
missing one is listed along with the exact command to generate it, and the
whole batch is aborted without launching a single job — rather than
launching part of a sweep only to have some configs crash during training or
analysis.

- `-g, --gpus DEVICES` / `--gpus=DEVICES` — explicit local GPU list;
  the historical positional GPU-list form remains supported.
- `--base CFG --sweep SPEC` — generate experiment configs from a base config
  plus sweep spec (must be given together) into the batch's `configs/` dir.
  If omitted, defaults to
  `configs/envs/ode/<system>/<system>_default.yaml` +
  `configs/envs/ode/<system>/<system>_experiment.yml` when both exist;
  otherwise hand-authored `configs/envs/ode/<system>/<system>_e<N>.yaml`
  configs are auto-discovered.
- `--notes NOTES` — free text or a path to a text file; copied to
  `.../<UID>/notes.txt`.
- `--post-run-analysis BOOL` (default `true`) — run
  `ode-experiment-analysis.sh` once all experiments finish (per-run analysis
  suite for each run, then the cross-run comparison). On HPC it is submitted
  as a Slurm job dependent on all experiment jobs.
- `--generate-datasets` / `--no-generate-datasets` (default: **on**) — for any
  config missing a required dataset, generate it instead of aborting. On HPC
  each missing config gets its own `generate-pid-dataset.sh` Slurm job
  (`gpu_a100`, proper `--nv`); that config's training job is then submitted
  with `--dependency=afterok:<dataset job>` so it only starts once its data
  exists, without blocking the rest of the batch. Locally, missing datasets
  are generated inline before any training job starts.
- `--repeat N` (default `1`) — run each experiment config `N` times with
  seeds `0..N-1`. Repeat runs are named `<experiment>_r<seed>`; groups of 2+
  repeats sharing an `e<N>` are seed-averaged by `ode-experiment-analysis.sh`
  (see below).

Results can also be analyzed manually with
`research/ode/analyse_experiment.py` (the launcher prints the exact command).

## ode-post-run-analysis.sh — per-run analysis suite

```
scripts/ode-post-run-analysis.sh --checkpoint <path> [--config <path>] [--gpus DEVICES]
scripts/ode-post-run-analysis.sh --config <path>          # PID baseline only
scripts/ode-post-run-analysis.sh --run-dir outputs/runs/<DATE>/<UID>
```

Runs the full analysis suite (`compare.py`, `steps_vs_mu.py`,
`analyse_rollout.py`, `collapse_vs_task.py`, `z_sensitivity.py` from
`research/ode/post_run_analysis/`) against one run. When the checkpoint lives
under an experiment batch's `checkpoints/` tree, the default output directory
is mirrored under the sibling `outputs/` tree for that same run.

`steps_vs_mu.py` and `z_sensitivity.py` (and, from training itself,
`mu_returns.png`/`efficiency.png`) each save a companion `.txt` with the
underlying per-line numbers alongside the plot — this is what
`ode-average-seeds.sh` reads to build seed-averaged versions.

- `--checkpoint <path>` — checkpoint to analyze. The config is auto-detected
  from the checkpoint's bundled `config.yaml`; pass `--config` explicitly
  only for older runs that predate config bundling.
- `--config <path>` — sufficient alone for a PID-baseline-only run with no
  checkpoint.
- `--run-dir <dir>` — a single run's dir as laid out by `run.sh`
  (`outputs/runs/<DATE>/<UID>`); its latest `checkpoints/checkpoint_*` is analyzed
  and outputs default to `<dir>/outputs/post_run_analysis/`.
- `--out-dir <dir>` — override the output location. Without it (and without
  `--run-dir`) outputs go to `outputs/experiment/<run_id>/`.
- `-g, --gpus DEVICES` / `--gpus=DEVICES` — select the GPU(s) used by
  the analysis suite and propagate the choice through nested analysis calls.

`--checkpoint`, `--run-dir`, and `--config`-only are mutually exclusive
modes; one must be chosen.

## ode-experiment-analysis.sh — batch aggregation

```
scripts/ode-experiment-analysis.sh --experiment_uid <UID> [--experiment_date <YYYYMMDD>] [--gpus DEVICES] [extra args...]
```

For an experiment batch produced by `launch_experiments.sh`: first runs
`ode-post-run-analysis.sh` against each run's latest checkpoint, then
aggregates across runs with `research/ode/compare_experiment.py`, writing
comparison plots under `outputs/experiments/<DATE>/<UID>/outputs/comparison/`
(overridable via `--out`), then averages any `--repeat > 1` groups (2+ seeds
sharing an `e<N>`) via `ode-average-seeds.sh`'s `--discover-experiment` mode,
writing `outputs/experiments/<DATE>/<UID>/outputs/comparison/seed_average_e<N>/`.

- `--experiment_uid <UID>` — required; UID assigned by
  `launch_experiments.sh`.
- `--experiment_date <YYYYMMDD>` — auto-discovered by searching
  `outputs/experiments/*/<UID>` if omitted.
- `-g, --gpus DEVICES` / `--gpus=DEVICES` — expose the selected GPU(s) to
  per-run and batch analysis.
- Any additional arguments are passed through to `compare_experiment.py`
  (e.g. `--num_envs`, `--num_task_points`, `--num_step_diff_episodes`,
  `--step_diff_bins`, `--seed`, `--out`,`--workdir`).

## ode-average-seeds.sh — cross-seed averaging

```
scripts/ode-average-seeds.sh --runs <run1>/outputs <run2>/outputs ... --out <dir>
scripts/ode-average-seeds.sh --discover-experiment <outputs_root> [--prefix <prefix>] --out <dir>
scripts/ode-average-seeds.sh --gpus 4 --discover-experiment <outputs_root> --out <dir>
```

Thin wrapper around `learning_statistics.py average-seeds`.
Averages `mu_returns.txt`, `efficiency.txt`, `steps_vs_mu.txt`, and
z_sensitivity/z_sensitivity.txt, and l2_error.txt across a set of seed-repeat runs (same
config, different `--seed`) into combined mean-±-std `_avg.png`/`_avg.txt`
pairs. A run missing one of those files is skipped for that plot only, with
a warning, rather than aborting. The per-run system_meta.txt files are also combined into system_meta.txt and system_meta.tex with per-bin trained labels, PID-relative step improvements, and error differences versus PID.
A single explicit run is valid: its mean is copied and every cross-seed std is zero.

- `--runs <dir>...` — explicit list of run output dirs (each a run's own
  `outputs/` / `cfg.output_path`). Used by `run-all-ode.sh`, whose repeats
  don't share a common experiment root.
- `--discover-experiment <outputs_root>` — auto-discover repeat groups
  (2+ seeds sharing an `e<N>`) under an experiment batch's `outputs/` root,
  the same grouping `compare_experiment.py` uses. `--prefix` is normally
  unnecessary — it's auto-detected per metrics file — pass it explicitly only
  if a batch somehow mixes configs from more than one system.
- Called automatically by `run-all-ode.sh` (`--repeat > 1`) and
  `ode-experiment-analysis.sh` (any `--repeat > 1` groups); run it manually
  to re-average after the fact or for ad-hoc groupings.

## stop_experiments.sh — stop running experiments

```
scripts/stop_experiments.sh                # stop all
scripts/stop_experiments.sh van_der_pol    # stop only this system
```

Stops running factorial experiments by matching and killing
`research/ode/run.py` processes (SIGTERM, then SIGKILL for stragglers).
Optionally restricted to one system (matches `configs/envs/ode/<system>`).

## running.sh — inspect running experiments

```
scripts/running.sh
```

Lists every running `research/ode/run.py` process (found the same way
`stop_experiments.sh` finds them) with the info `nvidia-smi` alone doesn't
give you: GPU index and memory (via `nvidia-smi --query-compute-apps`),
system/experiment id/seed (parsed from the process's `--config`/`--seed`
args), the run's fingerprint (`TrainConfig.run_uid`, recovered from an
explicit `--run_uid` arg or, failing that, from the checkpoint path logged
once a checkpoint has been saved), elapsed time, and the latest
`Iter i/N ... R: ... ETA: ...` line from its log — resolved via
`/proc/<pid>/fd/1`, so it works whether the run was launched directly, via
`run.sh`, or via `launch_experiments.sh`.

## generate-pid-dataset.sh — PID datasets for one config

```
CONFIG=configs/envs/ode/van_der_pol/van_der_pol_icassp2027.yaml bash scripts/generate-pid-dataset.sh --gpus 0
sbatch --export=ALL,CONFIG=<cfg> scripts/generate-pid-dataset.sh
```

Runs `generate_pid_training_dataset.py`, then `generate_pid_eval_dataset.py`
for every split `check_oracle_dataset.py` reports, writing
`data/<system>/training/` and `data/<system>/evaluation/<split>set/`. Training
and analysis both raise if these caches are missing. `FORCE=true` rebuilds
existing entries.

## generate-pid-datasets-all-ode.sh — PID datasets for all systems

```
bash scripts/generate-pid-datasets-all-ode.sh                      # every system
bash scripts/generate-pid-datasets-all-ode.sh van_der_pol --force  # one system, rebuild
```

Runs `generate-pid-dataset.sh` on each system's `*_default.yaml`, round-robin
over `--gpus`.

## upload-model.sh — publish a checkpoint to Hugging Face

```
bash scripts/upload-model.sh outputs/runs/<DATE>/ode/<system>/<UID>/checkpoints/checkpoint_750 <hf_user>/<repo_name>
```

Exports the checkpoint as a model artifact (weights, resolved config, no
optimizer state) and uploads it as a private repository. `HF_TOKEN` is read
from the environment or `.env`. See [docs/deployment.md](../docs/deployment.md#model-artifacts).

## retrain_and_eval_van_der_pol.sh — DeepONet, Van der Pol

```
bash scripts/retrain_and_eval_van_der_pol.sh
```

Run inside the DeepONet container (`docker/docker-compose.deeponet.yml`).
Trains the Van der Pol DeepONet on the complete and binned domains in parallel
(GPUs 1 and 0), then evaluates both with `deeponet_relerr_eval.py`. See
[`research/baselines/deeponet/`](../research/baselines/deeponet/README.md).

## setup/

- `create-apptainer.sh` — pulls the container image and converts it to
  `steppo.sif` for Slurm clusters.
- `setup_device.sh` — checks Docker, Docker Compose and the NVIDIA container
  toolkit, then builds the Docker image.
