# Analysis

All analysis scripts live under `research/ode/` and require `PYTHONPATH=.`.
Most read the pre-generated PID evaluation cache; see
[training.md](training.md#dataset-caches) for how to populate it.

The shell wrappers in [`scripts/README.md`](../scripts/README.md) run these in
the right order, with GPU selection and Slurm dispatch handled for you —
`ode-post-run-analysis.sh` (one run), `ode-experiment-analysis.sh` (a batch),
`ode-repeat-analysis.sh` and `ode-average-seeds.sh` (across seeds).

## Vocabulary

The scripts use these terms:

- **Reference trajectory** — the tight-tolerance `PIDController` solve, treated
  as ground truth. Generated once per dataset, never re-solved.
- **PID baseline** — the *standard*-tolerance `PIDController` solve; what the
  policy is compared against. Not itself the ground truth.
- **L2 error** — distance from a solved trajectory to the reference, reported as
  last-step (endpoint) and time-integrated (trapezoidal) values. A monitoring
  metric, never a training objective.

## Per-run analysis

```bash
scripts/ode-post-run-analysis.sh --run-dir outputs/runs/<DATE>/ode/<system>/<UID>
```

Runs the standard suite from `research/ode/post_run_analysis/`:

| Script | Produces |
| --- | --- |
| `compare.py` | diffrax PID baseline vs the learned agent; works from a config alone with no checkpoint |
| `steps_vs_mu.py` | solver steps vs task parameter, PID vs RL, with OOD values beyond the training range marked |
| `analyse_rollout.py` | 9-panel single-episode dashboard (state, action, belief μ/σ over time), with PID overlays |
| `collapse_vs_task.py` | whether belief-variance collapse *timing* depends on the task parameter, including OOD |
| `z_sensitivity.py` | dose-response: return / episode length / success rate as belief μ and log σ² are mixed towards standard-normal noise |

`steps_vs_mu.py` and `z_sensitivity.py` (and `mu_returns.png` / `efficiency.png`
from training itself) each save a companion `.txt` of the underlying per-line
numbers next to the plot — this is what the seed-averaging step reads.

## Latent-space and environment diagnostics

Additional scripts in the same directory, run ad hoc:

| Script | Question |
| --- | --- |
| `reward_belief_heatmap.py` | belief-conditioned reward-decoder landscape over the step-size axis (the ODE analogue of VariBAD Fig. 3a) |
| `ground_truth_action_sweep.py` | the same sweep using the *real* `env.step()` instead of the learned decoder, as a check on decoder extrapolation |
| `vae_noise_injection.py` | how far a burst of observation noise knocks the inferred belief off course, and whether it recovers |
| `obs_noise_injection.py` | local action sensitivity to observation corruption, across one or more models |
| `obs_feature_ranges.py` | empirical per-dimension observation ranges from PID rollouts — a check on `_obs()` normalization; no checkpoint needed |

## Cross-run comparison

```bash
scripts/ode-experiment-analysis.sh --experiment_uid <UID>
```

`research/ode/compare_experiment.py` (a thin CLI over
`research/ode/execute_comparison.py`, which does discovery and all GPU compute,
caching to `<out_dir>/data/` so replotting is a pure load) compares runs sharing
an experiment UID and writes to
`outputs/experiments/<DATE>/<UID>/outputs/comparison/`:

1. `reward_trajectories.png` — mean return per dataset subplot.
2. `latent_<i>_<param><value>.png` — belief mean/variance vs rollout step at each sampled task point.
3. `step_diff[_rel]_<split>.png` — histogram of `pid_steps − rl_steps` per run and split; `_rel` normalizes by `pid_steps` for cross-task comparability.
4. `error_dist_<split>.png` — log₁₀ relative L2 error vs the reference, per run, with the pooled PID distribution and solver `rtol` as references.
5. `step_diff_stats_<split>.png` / `.txt` — per-experiment mean/median/Q1/Q3 of the relative savings.

`research/ode/analyse_experiment.py` is the older factorial dashboard (training
curves, VAE loss breakdown, belief quality, interaction-effect heatmaps).

## Across seeds

```bash
scripts/ode-average-seeds.sh --runs <run1>/outputs <run2>/outputs ... --out <dir>
scripts/ode-repeat-analysis.sh --system van_der_pol --dry-run
```

Both wrap `research/ode/post_run_analysis/learning_statistics.py`:

- `average-seeds` averages the `.txt` companions (`mu_returns`, `efficiency`,
  `steps_vs_mu`, `z_sensitivity`, `l2_error`) across seed repeats into mean ± std
  `_avg.png`/`_avg.txt` pairs, and combines the per-run `system_meta.txt` files
  into `system_meta.txt`/`.tex` with per-bin labels, PID-relative step
  improvements, and error differences versus PID. A run missing one file is
  skipped for that plot only, with a warning.
- `aggregate-repeats` groups a `--repeat N` batch by its shared
  `repeat_batch_uid` and produces mean ± std plots plus a `summary.txt`. For
  older runs without that tag it falls back to a labelled `legacy_*` heuristic —
  always check the printed manifest, or pass `--dry-run` first.

Both are CPU-only.

## Inference and profiling

| Script | Purpose |
| --- | --- |
| `research/ode/inference/plot_custom_trajectories.py` | solve user-specified conditions with a trained controller; accepts `--checkpoint`, `--repo_id` or `--model_dir` |
| `research/ode/inference/benchmark_checkpoint.py` | end-to-end latency of a trained checkpoint |
| `research/ode/inference/benchmark_encoders.py` | initialized-model encoder latency ablation |
| `research/ode/inference/profile_controller_overhead.py` | per-step controller overhead inside `diffeqsolve` |
| `research/ode/diagnostics/wallclock_vs_mu.py` | wall-clock time of PID vs RL controllers across a task sweep |
| `research/ode/diagnostics/wallclock_batch_size_sweep.py` | throughput vs batch size |
