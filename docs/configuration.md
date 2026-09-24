# Configuration

Configs are YAML files mapping onto the nested dataclass hierarchy in
[`src/steppo/configs/base_config.py`](../src/steppo/configs/base_config.py), loaded via
`load_config_from_yaml` / `load_config_from_dict`.

Loading is **strict**: any key that does not match a field of its dataclass
raises. This is deliberate — an unmatched key is otherwise silently dropped, so a
typo'd or wrongly-nested field quietly stops doing anything (a top-level
`entropy_coeff:` sweep axis once did nothing, because `entropy_coeff` lives under
`ppo:`).

## Resolution order

`load_config_from_yaml` applies four transforms before instantiating dataclasses:

1. **`extends: <name>`** — deep-merges a base YAML under the current file.
   Looked up at `<config_dir>/../<name>.yaml`, falling back to a recursive search
   of `configs/envs/`. Recursive: a base may itself extend another.
2. **`model: <name>`** — loads an architecture YAML from
   `<config_dir>/models/<name>.yaml`, falling back to `configs/models/<name>.yaml`.
   It supplies `model_precision`, `backbone`, the arch fields (`latent_dim`,
   `latent_dim_long`, `encoder`, `decoder`) merged into the `vae:` section, and `policy` merged into `ppo.policy`.
3. **`env.pulse_config: <name>`** — deep-merges `configs/envs/ode/<name>.yaml`
   into the `env:` section, so one reusable pulse-settings file can be shared by
   name instead of duplicated inline.
4. **Dotted keys** — `latent_scatter.interval: 25` is expanded into nested dicts.
   Explicit nested sections win over dotted keys on conflict.

In every case the referenced/base file provides defaults and the current file
overrides them, recursively per nested key.

> When changing an architecture default, check whether the value is already set
> in the referenced `configs/models/` file before assuming the dataclass default
> applies.

## Top-level sections

| Section | Dataclass | Covers |
| --- | --- | --- |
| *(top level)* | `TrainConfig` | run identity, iteration/env counts, intervals, output paths |
| `vae:` | `VAEConfig` | VariBAD encoder/decoder arch and ELBO optimization |
| `ppo:` | `PPOConfig` | PPO optimizer, objective, and policy arch |
| `training:` | `TrainingConfig` | EMA loss normalization, grad clipping, KL annealing, reward normalization |
| `env:` | `ODEEnvConfig` | solver, task distribution, observation features, reward shaping |
| `warmstart:` | `WarmstartConfig` | imitation warmstart (see [training.md](training.md)) |
| `precompute_baseline:` | `PrecomputeBaselineConfig` | PID baseline step-count precomputation |
| `eval_metrics:` | `EvalMetricsConfig` | fused per-iteration efficiency + L2-error diagnostic |
| `latent_scatter:` | `LatentScatterConfig` | periodic belief-μ scatter snapshots and GIF |
| `pid_fallback:` | `PidFallbackConfig` | control-chart PID fallback calibration |

Backbone and policy algorithm are selected by two top-level fields:

```yaml
backbone: varibad   # "varibad" (only registered backbone)
algo: ppo           # "ppo" (only registered algorithm today)
```

## A real example

`configs/envs/ode/scalar_decay/scalar_decay_default.yaml`:

```yaml
# ODE: dy/dt = -y, exact solution y(t) = exp(-t) on [0, 1]
extends: ode_default
model: vae_vdp_3

exp_name: "scalar_decay_default"
seed: 0
total_iters: 750
episodes_per_trial: 1
num_envs: 256
num_eval_episodes: 128
rollout_steps: 40
eval_interval: 50
log_interval: 10

env:
  system: scalar_decay
  precision: float64      # step counts are not comparable across precisions
  t_end: 10.0
  dt0: 0.1
  rtol: 1.0e-3
  atol: 1.0e-6
  dt_min: 1.0e-5
  dt_max: 10.0
  dt_log_gain: 2.5        # max exp(2.5) ~= 12.2x change per step
  progress_warp: false
  immediate_dt_action: true
  survival_penalty: 0.0
  margin_penalty: 0.0
  pulse_config: pulse_config
  lam_min: 1.0
  lam_max: 100.0
  task_sample_scheme: "log-binned"
  obs_features: [state, step_context]
  train_bins: [[1.0, 100.0]]
  val_bins: [[1.0, 100.0]]
  test_bins: [[1.0, 100.0]]
```

## CLI overrides

`research/ode/run.py` accepts named flags plus arbitrary dotted overrides.

Named flags:

| Flag | Meaning |
| --- | --- |
| `--config PATH` | config YAML to load |
| `--system NAME` | override `env.system` |
| `--total_iters N` | total training iterations |
| `--seed N` | override `config.seed` |
| `--log_interval N` / `--eval_interval N` | console logging / evaluation cadence |
| `-e, --episodes N` | episodes per trial (`0` = unlimited, bounded only by `rollout_steps`) |
| `--episode_max_steps N` | max steps per episode (`0` = use `rollout_steps`) |
| `--run_uid` / `--run_date` | run identity, for grouping outputs |
| `--checkpoints_dir` / `--outputs_dir` | override artifact locations |
| `--resume_checkpoint DIR` | resume model and optimizer state |
| `--resume_replay_rollouts N` | fresh rollouts used to refill the unsaved replay buffer (default `4`) |
| `--note TEXT` | free text saved to `checkpoint_path/note.txt` |
| `--no_plots` / `--no_json` | suppress plot / metrics-JSON output |

Any other `--key value` is parsed as a dotted `TrainConfig` override:

```bash
PYTHONPATH=. python research/ode/run.py \
  --config configs/envs/ode/van_der_pol/van_der_pol_default.yaml \
  --env.progress_warp true \
  --ppo.entropy_coeff 0.02 \
  --vae.kl_weight 0.05
```

## Run outputs

`TrainConfig` derives its paths from run identity:

```
outputs/runs/<run_date>/<env_family>/<env.system>/<run_uid>/
├── checkpoints/     # Orbax checkpoints + bundled config.yaml + note.txt
└── outputs/         # metrics JSON, plots, post-run analysis
```

`--checkpoints_dir` / `--outputs_dir` override each to
`<dir>/<run_uid>` instead. `repeat_batch_uid` is bundled into each run's
`config.yaml` so seed repeats from one launcher invocation can be grouped back
together robustly, even when they finish days apart.
