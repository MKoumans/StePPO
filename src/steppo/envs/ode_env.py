"""Generic ODE environment for VariBAD with step-size control."""

import dataclasses

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
from diffrax._root_finder._verychord import VeryChord
from flax import struct
from jax import Array, tree_util
from jax.random import PRNGKey
from lineax import AutoLinearSolver

from steppo.configs.base_config import PRECISION_DTYPES, ODEEnvConfig
from steppo.envs.ode.systems import ODESystemSpec, get_rhs, get_system
from steppo.utils.task_params import task_bounds


def _make_solver(rtol: float, atol: float) -> diffrax.AbstractSolver:
    """Create the implicit Kvaerno5 solver used for ODE episodes."""
    rf = VeryChord(
        rtol=rtol,
        atol=atol,
        norm=optx.rms_norm,
        linear_solver=AutoLinearSolver(well_posed=None),
    )
    return diffrax.Kvaerno5(root_finder=rf)


_EMA_ALPHA: float = 1.0 / 16.0


@struct.dataclass
class ODEState:
    """JAX state containing the ODE solution and solver telemetry."""

    y: Array
    t: Array
    dt: Array
    solver_state_flag: Array
    solver_state_f: Array
    last_log_error: Array
    last_keep_step: Array
    step_count: Array
    budget_exhausted: Array
    accept_ema: Array
    log_error_ema: Array
    log_error_delta: Array
    reject_streak: Array


@dataclasses.dataclass
class ODEParams:
    """Per-task ODE parameters and the episode step budget."""

    lam: any = 1.0
    pulse_phase: any = 0.0
    max_steps: int = 1000


# Register ODEParams as a JAX pytree: lam and pulse_phase are leaves, max_steps is aux.
tree_util.register_pytree_node(
    ODEParams,
    lambda p: (
        [jnp.asarray(p.lam, dtype=jnp.float32), jnp.asarray(p.pulse_phase, dtype=jnp.float32)],
        p.max_steps,
    ),
    lambda max_steps, leaves: ODEParams(lam=leaves[0], pulse_phase=leaves[1], max_steps=max_steps),
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ODEEnv:
    """Gymnax-style environment in which one episode is one adaptive ODE solve.

    `config.system` selects the ODE problem.
    """

    def __init__(self, config: ODEEnvConfig = None, max_steps: int = None) -> None:
        """Initialize the selected ODE system and adaptive solver."""
        self._config = config or ODEEnvConfig(system="scalar_decay")  # default system
        self._spec: ODESystemSpec = get_system(self._config.system)
        self.solver = _make_solver(self._config.rtol, self._config.atol)
        self.terms = diffrax.ODETerm(get_rhs(self._config))

        # The env can be built before apply_precision() runs, so enable x64 here;
        # JAX would otherwise silently truncate float64 to float32.
        if self._config.precision == "float64":
            jax.config.update("jax_enable_x64", True)
        self._dtype = PRECISION_DTYPES[self._config.precision]

        self.t0 = jnp.asarray(0.0, dtype=self._dtype)
        self.t_end = jnp.asarray(self._config.t_end, dtype=self._dtype)
        self.dt0 = jnp.asarray(self._config.dt0, dtype=self._dtype)
        self.rtol = float(self._config.rtol)
        self.atol = float(self._config.atol)
        self.dt_min = jnp.asarray(self._config.dt_min, dtype=self._dtype)
        self.dt_max = jnp.asarray(self._config.dt_max, dtype=self._dtype)
        self.dt_log_gain = jnp.asarray(self._config.dt_log_gain, dtype=self._dtype)
        self.max_steps = max_steps

        # PID effort curve for time-warped progress reward (config.progress_warp).
        # Injected via set_progress_warp() by the trainer; None = linear Δt/t_end.
        self._warp = None

        self._check_init_values()

    def _check_init_values(self):
        """Validate solver bounds required for a well-defined environment."""
        if self.max_steps is None:
            raise ValueError("max_steps must be provided (use config.rollout_steps)")

        if self.dt_min <= 0.0 or self.dt_max <= 0.0:
            raise ValueError("dt_min and dt_max must be positive")

        if self.dt_min > self.dt_max:
            raise ValueError(
                f"dt_min ({self.dt_min}) cannot be greater than dt_max ({self.dt_max})"
            )

        if self.dt0 < self.dt_min or self.dt0 > self.dt_max:
            raise ValueError(
                f"dt0 ({self.dt0}) must be within [dt_min ({self.dt_min}), dt_max ({self.dt_max})]"
            )

        if self.dt_log_gain <= 0.0:
            raise ValueError(f"dt_log_gain ({self.dt_log_gain}) must be positive")

    def _obs_state(self, state: ODEState) -> ODEState:
        """State with zeroed solver internals, matching inference-time controller."""
        return ODEState(
            y=state.y,
            t=state.t,
            dt=state.dt,
            solver_state_flag=jnp.bool_(True),
            solver_state_f=jnp.zeros_like(state.y),
            last_log_error=state.last_log_error,
            last_keep_step=state.last_keep_step,
            step_count=state.step_count,
            budget_exhausted=state.budget_exhausted,
            accept_ema=state.accept_ema,
            log_error_ema=state.log_error_ema,
            log_error_delta=state.log_error_delta,
            reject_streak=state.reject_streak,
        )

    @property
    def num_actions(self) -> int:
        """Return the scalar action dimension used to scale the next step."""
        return 1

    def set_progress_warp(self, log_mu_grid, t_knots, fracs) -> None:
        """Install the PID effort curve Φ(μ, t) for time-warped progress rewards.

        Call before step() is first traced: the env is a static jit argument.
        """
        self._warp = (
            jnp.asarray(log_mu_grid),
            jnp.asarray(t_knots),
            jnp.asarray(fracs),
        )

    def _warp_frac(self, t, mu):
        """Φ(μ, t): fraction of the PID baseline's solver effort spent by time t.

        Knot times are interpolated across the μ grid in log-log space, then Φ
        is piecewise linear in t within the selected row (implicit (0, 0) knot).
        """
        log_mu_grid, t_knots, fracs = self._warp
        num_mus = log_mu_grid.shape[0]
        log_mu = jnp.log(mu)
        i = jnp.clip(jnp.searchsorted(log_mu_grid, log_mu) - 1, 0, num_mus - 2)
        w = jnp.clip(
            (log_mu - log_mu_grid[i]) / (log_mu_grid[i + 1] - log_mu_grid[i]),
            0.0,
            1.0,
        )
        row = jnp.exp((1.0 - w) * jnp.log(t_knots[i]) + w * jnp.log(t_knots[i + 1]))
        zero = jnp.zeros((1,), dtype=row.dtype)
        return jnp.interp(
            t,
            jnp.concatenate([zero, row]),
            jnp.concatenate([zero, fracs.astype(row.dtype)]),
        )

    def _get_reward(
        self, progress_frac, budget_exhausted, success, keep_step, scaled_error=jnp.float32(1.0)
    ):
        """Combine progress, solver-quality, and terminal reward terms."""
        factor = jnp.float32(self._config.reward_factor)  #
        survival_penalty = jnp.float32(self._config.survival_penalty)  #
        rejection_penalty = jnp.float32(self._config.rejection_penalty)  #
        completion_bonus = jnp.float32(self._config.completion_bonus)  # terminal reward
        margin_penalty = jnp.float32(self._config.margin_penalty)  #

        reward = progress_frac * factor  #
        reward -= survival_penalty  #
        reward -= jnp.where(
            keep_step, margin_penalty * (jnp.float32(1.0) - scaled_error), rejection_penalty
        )

        bonus = jnp.where(
            success,
            (jnp.float32(1.0) - budget_exhausted) * completion_bonus,
            jnp.float32(0.0),
        )

        return reward + bonus

    def obs_shape(self, params=None) -> tuple:
        """Return the observation shape for the active feature configuration."""
        return (self._spec.get_obs_dim(self._config),)

    @property
    def task_dim(self) -> int:
        """Number of hidden task parameters (lam is always scalar for ODE envs)."""
        return 1

    _SAMPLE_ENABLE_FLAG = {
        "scalar_decay": "sample_lam",
        "van_der_pol": "sample_mu",
        "robertson": "sample_k2",
        "brusselator": "sample_B",
        "fitzhugh_nagumo": "sample_eps",
        "chemical_cascade": "sample_cc_lam",
        "fosm": "sample_D",
        "fosm_smooth": "sample_D",
        "chua": "sample_m",
        "chua_smooth": "sample_m",
    }

    # Fixed task value when the system's sample_* flag is False: the lower bound
    # ("lo"), the midpoint ("uniform") or the log midpoint ("log-uniform").
    _FIXED_TASK_SCHEME = {
        "scalar_decay": "lo",
        "van_der_pol": "uniform",
        "robertson": "log-uniform",
        "brusselator": "log-uniform",
        "fitzhugh_nagumo": "log-uniform",
        "chemical_cascade": "log-uniform",
        "fosm": "lo",
        "fosm_smooth": "lo",
        "chua": "lo",
        "chua_smooth": "lo",
    }

    def _task_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Raw (lo, hi) task-parameter bounds of the active system."""
        lo, hi = task_bounds(self._config)
        return jnp.float32(lo), jnp.float32(hi)

    def get_task_params(self, params: ODEParams) -> jnp.ndarray:
        """Return hidden task parameters normalised to [0,1] as a (task_dim,) array."""
        raw = jnp.atleast_1d(jnp.asarray(params.lam, dtype=jnp.float32))
        lo, hi = self._task_bounds()
        return (raw - lo) / jnp.maximum(hi - lo, 1e-8)

    @staticmethod
    def _sample_scalar(key: PRNGKey, lo: jnp.ndarray, hi: jnp.ndarray, scheme: str) -> jnp.ndarray:
        """Draw a scalar from [lo, hi] per `scheme` ("uniform"/"log-uniform" — the
        "binned"/"log-binned" schemes reduce to the same inner draw once a bin's
        own [lo, hi] has been picked, see sample_task_bins)."""
        if scheme == "log-binned":
            return jnp.exp(jax.random.uniform(key, (), minval=jnp.log(lo), maxval=jnp.log(hi)))
        return jax.random.uniform(key, (), minval=lo, maxval=hi)

    def sample_task_bins(
        self, key: PRNGKey, bins: jnp.ndarray, scheme: str = "log-binned"
    ) -> ODEParams:
        """Sample λ from a uniformly chosen [lo, hi] row of `bins`, within the bin per `scheme`."""
        key_bin, key_val, key_pulse = jax.random.split(key, 3)
        idx = jax.random.randint(key_bin, (), 0, bins.shape[0])
        lam = self._sample_scalar(key_val, bins[idx, 0], bins[idx, 1], scheme)
        pulse_phase = jax.random.uniform(key_pulse, shape=(), dtype=jnp.float32)
        return ODEParams(lam=lam, pulse_phase=pulse_phase, max_steps=self.max_steps)

    def sample_task(self, key: PRNGKey) -> ODEParams:
        """Sample a training task from `config.train_bins` per `config.task_sample_scheme`
        ("binned" or "log-binned", see sample_task_bins). When the system's own sample_*
        flag is off, replicates that system's fixed-value behavior exactly (see
        _FIXED_TASK_SCHEME) instead."""
        enable_flag = self._SAMPLE_ENABLE_FLAG.get(self._config.system, "sample_lam")
        if not getattr(self._config, enable_flag):
            key_pulse = jax.random.split(key)[1]
            lo, hi = self._task_bounds()
            fixed_scheme = self._FIXED_TASK_SCHEME.get(self._config.system, "log-uniform")
            if fixed_scheme == "lo":
                lam = lo
            elif fixed_scheme == "uniform":
                lam = (lo + hi) / 2.0
            else:
                lam = jnp.exp((jnp.log(lo) + jnp.log(hi)) / 2.0)
            pulse_phase = jax.random.uniform(key_pulse, shape=(), dtype=jnp.float32)
            return ODEParams(lam=lam, pulse_phase=pulse_phase, max_steps=self.max_steps)

        scheme = self._config.task_sample_scheme
        if not self._config.train_bins:
            raise ValueError(f"task_sample_scheme='{scheme}' requires train_bins to be set")
        bins = jnp.asarray(self._config.train_bins, dtype=jnp.float32)
        return self.sample_task_bins(key, bins, scheme)

    def _task_array(self, params: ODEParams) -> Array:
        """Pack task parameters into the solver argument array."""
        return jnp.array(
            [
                jnp.asarray(params.lam, dtype=self._dtype),
                jnp.asarray(params.pulse_phase, dtype=self._dtype),
            ],
            dtype=self._dtype,
        )

    def reset(self, key: PRNGKey, params: ODEParams):
        """Reset one episode and return its initial observation and state."""
        task = self._task_array(params)
        y0 = self._spec.y0(task, key, self._config).astype(self._dtype)
        ss = self.solver.init(self.terms, self.t0, self.t0 + self.dt0, y0, task)

        state = ODEState(
            y=y0,  #
            t=self.t0,  #
            dt=self.dt0,  #
            solver_state_flag=ss[0],  #
            solver_state_f=ss[1],  #
            last_log_error=jnp.float32(0.0),  #
            last_keep_step=jnp.bool_(True),  #
            step_count=jnp.int32(0),  #
            budget_exhausted=jnp.float32(0.0),  #
            accept_ema=jnp.float32(1.0),  #
            log_error_ema=jnp.float32(0.0),  #
            log_error_delta=jnp.float32(0.0),  #
            reject_streak=jnp.int32(0),  #
        )

        return self._spec.obs(self._obs_state(state), self._config), state

    def step(self, key: PRNGKey, state: ODEState, action: Array, params: ODEParams):
        """One solver step. action ∈ [-1, 1] scales the step size — of this
        attempt if config.immediate_dt_action, else of the next call."""
        task = self._task_array(params)
        action = jnp.asarray(action, dtype=jnp.float32).squeeze()

        # ── Apply action to step size (action ∈ [-1, 1] scales dt) ──
        # immediate_dt_action: this attempt uses the scaled dt; otherwise the
        # scaled dt takes effect on the next call.
        dt_scaled = jnp.clip(
            state.dt * jnp.exp(action * self.dt_log_gain),
            self.dt_min,
            self.dt_max,
        )
        dt_attempt = dt_scaled if self._config.immediate_dt_action else state.dt

        # ── ODE step with dt ─────────────────────────────────
        ss = (state.solver_state_flag, state.solver_state_f)
        # Clip the step to land exactly on t_end instead of overshooting it,
        # so success/done trigger with t_reached == t_end rather than t_end + slack.
        t1 = jnp.minimum(state.t + dt_attempt, self.t_end)
        y1, y_err, _dense, ss_new, _result = self.solver.step(
            terms=self.terms,
            t0=state.t,
            t1=t1,
            y0=state.y,
            args=task,
            solver_state=ss,
            made_jump=False,
        )

        # ── Reject non-finite solver output before error check ────
        y1_finite = jnp.all(jnp.isfinite(y1))
        y1_safe = jnp.where(y1_finite, y1, state.y)

        # ── Scaled RMS error, as diffrax's PIDController computes it ────────
        scale = self.atol + jnp.maximum(jnp.abs(state.y), jnp.abs(y1_safe)) * self.rtol
        scaled_error = jnp.sqrt(jnp.mean((y_err / scale) ** 2))
        safe_error = jnp.where(jnp.isfinite(scaled_error), scaled_error, jnp.float32(1e10))
        keep_step = (safe_error <= jnp.float32(1.0)) & y1_finite

        # ── Accept or reject ─────────────────────────────────────────
        new_y = jnp.where(keep_step, y1_safe, state.y)
        new_t = jnp.where(keep_step, t1, state.t)
        new_flag = jnp.where(keep_step, ss_new[0], state.solver_state_flag)
        ss_f_safe = jnp.where(jnp.isfinite(ss_new[1]), ss_new[1], state.solver_state_f)
        new_f = jnp.where(keep_step, ss_f_safe, state.solver_state_f)

        # Keep float32 state: safe_error is float64 in a float64 solve.
        log_error = jnp.clip(jnp.log(safe_error + jnp.float32(1e-10)), -30.0, 30.0).astype(
            jnp.float32
        )
        alpha = jnp.float32(_EMA_ALPHA)

        new_accept_ema = alpha * keep_step.astype(jnp.float32) + (1.0 - alpha) * state.accept_ema
        new_log_error_ema = alpha * log_error + (1.0 - alpha) * state.log_error_ema
        log_error_delta = log_error - state.last_log_error
        new_reject_streak = jnp.where(keep_step, jnp.int32(0), state.reject_streak + jnp.int32(1))
        new_step_count = state.step_count + jnp.int32(1)
        budget_exhausted = new_step_count.astype(jnp.float32) / jnp.float32(params.max_steps)

        # The state stores dt_scaled: the dt just attempted (immediate) or the next one.
        new_state = ODEState(
            y=new_y,
            t=new_t,
            dt=dt_scaled,
            solver_state_flag=new_flag,
            solver_state_f=new_f,
            last_log_error=log_error,
            last_keep_step=keep_step,
            step_count=new_step_count,
            budget_exhausted=budget_exhausted,
            accept_ema=new_accept_ema,
            log_error_ema=new_log_error_ema,
            log_error_delta=log_error_delta,
            reject_streak=new_reject_streak,
        )

        # Progress as a fraction of the episode's total reward budget: warped by
        # the PID effort curve when installed (uniform reward per unit of solver
        # work), otherwise linear in time.
        t_capped = jnp.minimum(new_t, self.t_end)
        if self._warp is not None:
            mu = task[0]
            progress_frac = self._warp_frac(t_capped, mu) - self._warp_frac(state.t, mu)
        else:
            progress_frac = (t_capped - state.t) / self.t_end
        success = new_t >= self.t_end
        reward = self._get_reward(progress_frac, budget_exhausted, success, keep_step, safe_error)

        reward = reward.astype(jnp.float32)
        done = success | (new_step_count >= jnp.int32(params.max_steps))

        info = {
            "success": success,
            "keep_step": keep_step,
            "scaled_error": safe_error,
            "t_reached": new_t,
            "step_count": new_step_count,
        }

        obs = self._spec.obs(self._obs_state(new_state), self._config)
        obs = jnp.where(jnp.isfinite(obs), obs, jnp.zeros_like(obs))

        return obs, new_state, reward, done, info
