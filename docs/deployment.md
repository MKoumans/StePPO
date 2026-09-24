# Deployment

## The learned controller inside diffrax

`src/steppo/envs/ode/learned_controller.py` wraps a trained belief model and policy
as a diffrax `AbstractAdaptiveStepSizeController`, so the agent runs directly
inside `diffeqsolve`'s fused XLA loop — a drop-in replacement for diffrax's
`PIDController`, rather than only through the Gymnax env loop.

Per-step cost versus `PIDController` (from the module docstring):

| | Compute | Carry state |
| --- | --- | --- |
| PID | ~20 scalar ops | 3 scalars (12 B) |
| Learned | ~19.5K FLOPs | ~86 floats (344 B) |

The GRU cell dominates, at ~17K FLOPs (~87% of neural compute).

The controller's carry (`_LearnedState`) holds the previous observation and
action, the recurrent hidden state, belief `(μ, log σ²)`, and the solver
telemetry the observation is built from (accept EMA, log-error EMA and delta,
reject streak, step count, budget-exhausted flag) — so the observation seen
inside `diffeqsolve` matches the one the policy was trained on.

## Control-chart PID fallback

`pid_fallback:` (`PidFallbackConfig`) calibrates an empirical control limit for
each enabled solver statistic — `reject_streak`, `accept_ema`, `log_error_ema` —
over the converged tail of training. Every `interval` iterations, once
`iteration / total_iters >= start_frac`, `num_envs` episodes are rolled out and a
percentile of each episode's worst value is EMA'd into a running limit. The
intent is a Shewhart-SPC-style detector, deliberately independent of any learned
uncertainty estimate: `LearnedController` trips into the PID law
(`pid_action_from_obs`, with coefficients taken from `warmstart:` so the fallback
matches the BC demonstrator) for the rest of an episode once a live statistic
crosses `multiplier ×` its limit — direction-aware, since `reject_streak` and
`log_error_ema` trip on exceeding the limit while `accept_ema` trips on falling
below it.

> **Currently disabled.** `learned_controller.py` hard-codes
> `pid_active = jnp.bool_(False)`, with the trip-wire computation left in place
> but its result unused (commit `cd3f014`, "temporarily disable PID fallback
> mechanism in LearnedController"). Calibration still runs when
> `pid_fallback.enabled: true` and the envelopes are still saved and loaded, but
> no solve will actually fall back until that line is restored.

`pid_fallback` is off by default.

## Model artifacts

A **model artifact** is a deployable package — model weights,
resolved config, and reconstruction metadata — deliberately containing *no*
optimizer or training state. That is what distinguishes it from a **training
checkpoint**. The format is versioned (`ARTIFACT_FORMAT_VERSION`) and weights are
stored as safetensors; `src/steppo/models/huggingface/artifact.py` can export,
load and validate artifacts with no network access.

Hugging Face transport is implemented separately in
`src/steppo/models/huggingface/hub.py`:

```bash
HF_TOKEN=... PYTHONPATH=. python -m steppo.models.huggingface upload \
  --checkpoint outputs/runs/.../checkpoints/checkpoint_500 \
  --repo-id YOUR_ACCOUNT/model-name \
  --revision main

PYTHONPATH=. python -m steppo.models.huggingface download \
  --repo-id YOUR_ACCOUNT/model-name \
  --output-dir downloaded-model
```

- Repositories are **private by default**; `--public` is opt-in and deliberate.
- `--revision` pins a Hub revision (default `main`).
- Standard Hugging Face authentication applies, including `HF_TOKEN`.
- `upload` also takes `--commit-message`; `download` also takes `--cache-dir`
  and `--overwrite`.
- `scripts/upload-model.sh <checkpoint_dir> <hf_user>/<repo_name>` wraps the
  upload (reads `HF_TOKEN` from the environment or `.env`).

## Inference from an artifact

`research/ode/inference/plot_custom_trajectories.py` accepts any of
`--checkpoint`, `--repo_id` (with `--revision`) or `--model_dir`, plus
`--conditions` describing the solves to run:

```bash
PYTHONPATH=. python research/ode/inference/plot_custom_trajectories.py \
  --repo_id YOUR_ACCOUNT/model-name \
  --conditions <spec> \
  --out_dir outputs/inference
```

`tests/integration/test_huggingface_inference_roundtrip.py` covers the
export → upload → download → reconstruct → solve path.
