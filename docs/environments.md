# Environments

## ODE step-size control

`src/steppo/envs/ode_env.py`. One episode is one adaptive ODE solve from `t = 0` to
`env.t_end`. The agent picks relative step sizes; the environment runs the
solver and reports local error and accept/reject feedback.

The solver is a fixed implicit **Kvaerno5** with a `VeryChord` root finder
(`_make_solver`), stepped manually one attempt at a time — the environment owns
the accept/reject decision rather than delegating to a diffrax controller.

### Action

A single scalar `a ∈ [-1, 1]` (`num_actions == 1`), applied multiplicatively in
log space:

```
dt_next = clip(dt · exp(a · dt_log_gain), dt_min, dt_max)
```

`env.dt_log_gain` therefore bounds the per-step change: the default
`0.693147181` (ln 2) allows at most a 2× change per step; `2.5` allows ≈12.2×.

`env.immediate_dt_action: true` applies the scaled `dt` to *this* attempt
(apply-then-step, no feedback delay). `false` is the legacy behaviour where the
scaled `dt` only takes effect on the next call. The oracle warmstart requires
`immediate_dt_action: true`.

### Observation

Assembled from feature groups listed in `env.obs_features`, in list order
(`src/steppo/envs/ode/systems/obs_factory.py`):

| Feature | Dims | Contents |
| --- | --- | --- |
| `state` | system-specific | the system's own transform of its `y` vector |
| `direction` | `y_dim` | current derivative `f(t, y)` |
| `step_context` | 4 | last log error, clipped log-error delta, last accept flag, normalized time `t / t_end` |
| `solver_trend` | 5 | normalized log `dt`, accept EMA, log-error EMA, `tanh(reject_streak/4)`, budget-exhausted flag |

Default: `("state", "step_context", "solver_trend")`. **Task parameters are
always hidden from the observation** — they are the latent the belief model must
infer. Ordering is driven by the config list so that
`pid_controller.compute_step_context_offset` stays correct regardless of how a
config orders or omits features.

### Reward

`ODEEnv._get_reward`:

```
r = progress_frac · reward_factor
  − survival_penalty
  − (margin_penalty · (1 − scaled_error)   if step accepted
     else rejection_penalty)
  + (1 − budget_exhausted) · completion_bonus   on success
```

`env.progress_warp: true` replaces raw time progress with Φ(μ, t), the fraction
of the PID baseline's solver effort spent by time `t`, so progress is measured in
units of expected effort rather than simulated time.

`TrainConfig.__post_init__` rejects `margin_penalty > 0` with
`rejection_penalty <= 0`: with no rejection cost, always rejecting is a free,
zero-reward policy that dominates any accepted step, and PPO has no gradient to
escape that plateau.

## Registered systems

Registered in `src/steppo/envs/ode/systems/__init__.py`. Each is one file plus one
registry entry — `ode_env.py` itself never changes.

| System | Hidden task parameter | Sampling | Config directory |
| --- | --- | --- | --- |
| `scalar_decay` | λ (`lam_min`/`lam_max`) | uniform | `configs/envs/ode/scalar_decay/` |
| `van_der_pol` | μ (`mu_min`/`mu_max`) | log-uniform | `configs/envs/ode/van_der_pol/` |
| `robertson` | k₂ (`k2_min`/`k2_max`) | log-uniform | `configs/envs/ode/robertson/` |
| `brusselator` | B (`B_min`/`B_max`) | log-uniform | `configs/envs/ode/brusselator/` |
| `fitzhugh_nagumo` | ε (`eps_min`/`eps_max`) | log-uniform | `configs/envs/ode/fitzhugh_nagumo/` |
| `chemical_cascade` | λ (`cc_lam_min`/`cc_lam_max`) | log-uniform | `configs/envs/ode/chemical_cascade/` |

`env.task_dim` is 1 for every registered system — a single scalar hidden
parameter per task.

## Task splits

`env.train_bins`, `env.val_bins` and `env.test_bins` are lists of `[lo, hi]`
pairs over the task parameter:

- `train_bins` — sampled during training rollouts.
- `val_bins` — swept periodically during training (`eval_metrics`).
- `test_bins` — swept periodically during training and by post-hoc analysis.
  Only out-of-distribution if the bins lie outside `train_bins`.

`env.task_sample_scheme` is `binned` or `log-binned`. `ODEEnvConfig.__post_init__`
requires every bin to fit inside the active system's task domain, since task
normalization and full-range plots use those bounds.

## Pulse forcing

`src/steppo/envs/ode/systems/pulse.py` is a shared periodic-pulse forcing helper —
smooth Gaussian bumps injected into one state dimension — available to
`chemical_cascade` (`cc_pulse_*`), `fitzhugh_nagumo` (`fhn_pulse_*`) and
`scalar_decay` (`sd_pulse_*`). Off by default for all three.

Each family exposes `_enabled`, `_period`, `_width`, `_amplitude`, `_dim` and
`_random`, where `_random ∈ [0, 1]` jitters the period per episode: `0` is
perfectly periodic, `1` lets a pulse centre land anywhere within the neighbouring
half-periods. Settings can be factored into a shared file and referenced by name
with `env.pulse_config:` (see [configuration.md](configuration.md)).

## Adding a system

1. Create `src/steppo/envs/ode/systems/<name>.py` with a module-level `SPEC`
   (`ODESystemSpec`: `rhs`, `sample_task`, `y0`, `obs`, `feature_dims`). Use
   `obs_factory.base_obs_factory` to wire the system's own `state` feature
   together with the shared solver-telemetry builders.
2. Register it in `_load_systems()` in `src/steppo/envs/ode/systems/__init__.py`.
3. Add its task bounds to `ODEEnvConfig.__post_init__`'s `bounds_by_system`,
   `ODEEnv._TASK_BOUNDS` and `ODEEnv._SAMPLE_ENABLE_FLAG`.
4. If the RHS needs config-dependent construction (e.g. pulse forcing), add a
   `make_rhs(config)` and a branch in `get_rhs`.
