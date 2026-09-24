# Training

## Running a training job

```bash
PYTHONPATH=. python research/ode/run.py --config configs/envs/ode/scalar_decay/scalar_decay_default.yaml
PYTHONPATH=. python research/ode/run.py --config configs/envs/ode/van_der_pol/van_der_pol_default.yaml
```

`PYTHONPATH=.` is required — `research/` is not part of the importable package.
The PID dataset caches must exist first; see [Dataset caches](#dataset-caches).
See [configuration.md](configuration.md) for CLI overrides and output layout.

For batch launching, GPU round-robin, Slurm dispatch, seed repeats and automatic
post-run analysis, use the shell launchers instead; they are documented in full
in [`scripts/README.md`](../scripts/README.md):

```bash
bash scripts/run.sh --gpus 5                       # single run
scripts/run-all-ode.sh --repeat 8 --gpus 0,1,2,3   # one run per system, 8 seeds each
scripts/launch_experiments.sh van_der_pol          # factorial sweep
scripts/running.sh                                 # inspect running runs
scripts/stop_experiments.sh                        # stop them
```

## The loop

Each iteration:

1. **Collect** — `training/rollout.py` runs `num_envs` episodes in parallel for
   `rollout_steps` steps, encoding the belief online as it goes.
2. **Update the belief model** — `vae_trainer.py` (ELBO, `vae_loss.py`), for
   `num_updates_per_iter` steps over minibatches of `batch_size` trajectories
   drawn from `utils/replay_buffer.py`.
3. **Update the policy** — through the injected policy algorithm
   (`ppo_algorithm.py` → `ppo_trainer.py` / `ppo.py`, GAE + clipped surrogate)
   over `utils/policy_buffer.py`.

`training/base_trainer.py` owns the loop; the backbone-specific hooks live in
`trainer.py`.

### Loss normalization and stability

`training:` (`TrainingConfig`) holds the process knobs, separate from
architecture:

| Field | Default | Effect |
| --- | --- | --- |
| `ema_norm` | `false` | normalise each VAE and PPO loss term by its running EMA |
| `ema_alpha` | `0.99` | EMA decay (higher = slower adaptation) |
| `ema_clamp` | `5.0` | clamp normalised loss to ±C·EMA before dividing |
| `ema_input_cap` | `10.0` | cap an EMA update input at this × the current EMA (spike protection) |
| `grad_clip_vae` | `10.0` | global grad-norm clip, VAE optimizer |
| `grad_clip_ppo` | `0.5` | global grad-norm clip, PPO optimizer |
| `kl_anneal_frac` | `0.0` | fraction of total VAE steps over which `kl_weight` ramps linearly from 0 |
| `log_grad_norms` | `true` | per-term PPO grad norms; costs 3 extra backward passes per PPO step |

### Reward normalization

`training.reward_normalization` scales PPO rewards per task by the PID baseline's
step count: `scale(μ) = PID_steps(μ) / C`, with `C` the geometric mean over a
log-spaced grid of `grid_points` task values × `num_repeats` draws. Scale is
clipped to `[SCALE_MIN, SCALE_MAX]` (`training/reward_norm.py`); grid points where
PID exhausts its budget are extrapolated to full-episode cost by completion
fraction.

`post_scale_step_penalty` is a flat per-step tax applied **after** the scale
multiplies, unlike `env.survival_penalty` which is applied inside the environment
and so gets diluted or amplified by the scale itself. It is scale-invariant by
construction.

## Imitation warmstart

For stiff systems where learning from scratch is hard, an expert controller
behavior-clones the policy before RL fine-tuning. `warmstart.expert` picks the
demonstrator.

### `expert: pid` (default)

Evaluates the closed-form PID step-size formula directly from the observation
(`training/pid_controller.py`) — cheap enough to re-collect a fresh rollout,
live, every warmstart iteration. No offline step required.

### `expert: oracle`

A real-env local hill-climb search for the best-achievable step at every
trajectory step (`training/error_dist.py`'s oracle search). A stronger
demonstrator than the PID formula, but far too expensive to re-run live every
iteration, so generation is a **separate, explicit step**:

```bash
# Required before training with warmstart.expert=oracle — _warmstart raises otherwise
PYTHONPATH=. python src/steppo/envs/ode/generate_pid_training_dataset.py \
  --config configs/envs/ode/van_der_pol/van_der_pol_default.yaml
```

The script caches full `(state, action, reward, …)` trajectories under
`data/<system>/training/`. `_warmstart` loads that cache once per run
(`training/oracle_dataset.py`'s `load_oracle_training_batch`) and samples a fresh
minibatch every iteration; only the belief encoding is recomputed per iteration
(a cheap GRU replay, no env stepping or search). If nothing is cached for the env
config, it raises and tells you to run the generation script. Requires
`env.immediate_dt_action: true`.

Dataset size is a generation-time-only flag (`--oracle_episodes N`), not a
`WarmstartConfig` field: it does not affect any individual trajectory's content,
only how many are drawn. Putting it in the config would force the cache
fingerprint to either encode it (two configs wanting different sizes for the same
env silently generate near-duplicate caches) or ignore it (whichever config
generates first silently fixes the size for everyone else).

```yaml
warmstart:
  enabled: true
  expert: pid          # "pid" | "oracle"
  num_iters: 100
  bc_lr: 0.001
  bc_epochs: 5
  # PID law coefficients (also used by the deployment-time fallback):
  pid_safety: 0.9
  pid_order: 4
  pid_min_factor: 0.2
  pid_max_factor: 10.0
  pid_kp: 0.0
  pid_ki: 1.0
  pid_kd: 0.0
  # oracle-only:
  oracle_max_iters: 6     # hill-climb iteration cap per trajectory step
  oracle_dataset_seed: 0  # fixed corpus seed, shared across training repeats
```

## Dataset caches

Two separate cache subtrees, deliberately kept apart so training (a deterministic
μ grid) never implicitly sees the evaluation split (a frozen random task sample
used for held-out generalization checks). The shared solve primitive is
`training/pid_solve.py`'s `solve_pid_batch`.

| Cache | Path | Producer | Consumers |
| --- | --- | --- | --- |
| Training calibration | `data/<system>/training/` | `src/steppo/envs/ode/generate_pid_training_dataset.py` | progress-warp grid, PID baseline steps, reward-norm scale grid, oracle warmstart |
| Evaluation | `data/<system>/evaluation/<split>set/` | `src/steppo/envs/ode/generate_pid_eval_dataset.py` | `eval_metrics`, all post-run analysis |

**Neither cache is built lazily.** Training raises on a missing training cache,
and every evaluation consumer loads via `error_dist.load_cached_pid_batch` and
raises on a miss. Generate both before the first run of a config:

```bash
CONFIG=<cfg> bash scripts/generate-pid-dataset.sh     # both caches, one config
bash scripts/generate-pid-datasets-all-ode.sh         # both caches, all systems
PYTHONPATH=. python src/steppo/envs/ode/generate_pid_training_dataset.py --config <cfg>
PYTHONPATH=. python src/steppo/envs/ode/generate_pid_eval_dataset.py --config <cfg> --split train
PYTHONPATH=. python src/steppo/envs/ode/generate_pid_eval_dataset.py --config <cfg> --split val --num_envs 2048
PYTHONPATH=. python src/steppo/envs/ode/generate_pid_eval_dataset.py --config <cfg> --split test --force
```

`src/steppo/envs/ode/check_oracle_dataset.py` is a read-only pre-flight that
verifies both caches for a config and prints the exact generation command on a
miss. `scripts/run.sh` and `scripts/launch_experiments.sh` run it before
dispatching anything.

## In-training diagnostics

| Config | What it computes | Default cadence |
| --- | --- | --- |
| `eval_metrics:` | step-count efficiency (PID-relative) and L2 error (last-step + time-integrated) per train/val/test split, from one diffrax-native rollout per split | every 100 iterations |
| `latent_scatter:` | belief-μ scatter over two latent dims coloured by true task, appended to a CSV and stitched into a GIF at the end of training | every 10 iterations |
| `precompute_baseline:` | PID baseline step counts used as an eval diagnostic | once, at setup |
| `pid_fallback:` | control-chart limits for the deployment-time fallback (see [deployment.md](deployment.md)) | every 100 iterations, after `start_frac` of training |

`eval_metrics` replaces the former separate efficiency and error-distribution
diagnostics. Only the policy side is rolled out live; the PID baseline's own L2
error is read from the pre-generated eval cache. Each call rebuilds the
controller from freshly merged modules and so re-jits the `diffeqsolve` — keep
`interval` coarse.

## Resuming

```bash
PYTHONPATH=. python research/ode/run.py --config <cfg> \
  --resume_checkpoint outputs/runs/<date>/ode/<system>/<uid>/checkpoints/checkpoint_500
```

Model and optimizer state are restored from the Orbax checkpoint. The VAE replay
buffer is not checkpointed, so it is refilled with `--resume_replay_rollouts`
fresh rollouts (default 4) before training continues.
