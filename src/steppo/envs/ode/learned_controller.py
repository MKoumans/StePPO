"""Learned step-size controller: a trained encoder and policy as a diffrax
AbstractAdaptiveStepSizeController, so inference runs inside diffeqsolve's loop."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from diffrax._custom_types import Args, BoolScalarLike, IntScalarLike, RealScalarLike, Y
from diffrax._solution import RESULTS
from diffrax._step_size_controller.base import AbstractAdaptiveStepSizeController
from diffrax._term import AbstractTerm
from jaxtyping import PyTree

from steppo.envs.ode import ODEState
from steppo.training.pid_controller import pid_action_from_obs

_LN2 = 0.693147181
_EMA_ALPHA = 1.0 / 16.0


class _LearnedState(eqx.Module):
    """Diffrax carry for learned control, belief updates, and fallback state."""

    prev_obs: jax.Array
    prev_action: jax.Array  # action that set the dt of the in-flight attempt
    gru_hidden: PyTree  # recurrent carry: Array for GRU, richer pytree for other encoders
    belief_mu: jax.Array
    belief_logvar: jax.Array
    accept_ema: jax.Array
    log_error_ema: jax.Array
    last_log_error: jax.Array
    last_keep_step: jax.Array
    step_count: jax.Array
    budget_exhausted: jax.Array
    log_error_delta: jax.Array
    reject_streak: jax.Array
    pid_fallback_active: jax.Array  # sticky control-chart trip-wire (cfg.pid_fallback)


class _ZeroLearnedState(eqx.Module):
    """Inference carry for a zero-information encoder, whose belief is a compile-time zero.

    ``prev_obs`` is ``None`` for immediate actions and an array for one-step-delay policies.
    """

    prev_obs: jax.Array | None
    accept_ema: jax.Array
    log_error_ema: jax.Array
    last_log_error: jax.Array
    last_keep_step: jax.Array
    step_count: jax.Array
    budget_exhausted: jax.Array
    log_error_delta: jax.Array
    reject_streak: jax.Array


class LearnedController(
    AbstractAdaptiveStepSizeController["_LearnedState", RealScalarLike | None],
):
    """Diffrax step-size controller backed by a trained belief model and policy."""

    rtol: float
    atol: float
    encoder: eqx.Module = eqx.field(static=True)  # Encoder (GRU + MLP) for belief update
    policy: eqx.Module = eqx.field(static=True)  # Policy (MLP) for action inference
    obs_fn: callable = eqx.field(static=True)  # Function to compute observation from ODEState
    t_end: float = eqx.field(static=True)
    dt0_ref: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    use_latent_sample: bool = eqx.field(
        static=True
    )  # Whether to sample latent z from belief distribution or use mean+logvar
    latent_dim: int = eqx.field(static=True)
    reward_factor: float = eqx.field(static=True)
    step_cost: float = eqx.field(static=True)
    rejection_cost: float = eqx.field(static=True, default=0.0)
    margin_cost: float = eqx.field(static=True, default=0.0)
    # Must mirror the training env: dt *= exp(action * dt_log_gain), and whether
    # the action scales the current attempt (immediate) or the next one (delayed).
    dt_log_gain: float = eqx.field(static=True, default=_LN2)
    immediate_dt_action: bool = eqx.field(static=True, default=False)
    norm: callable = optx.rms_norm
    dtmin: float | None = None
    dtmax: float | None = None
    # (log_mu_grid, t_knots, fracs) for time-warped progress rewards, or None; must match
    # ODEEnv._warp_frac because the reward feeds the encoder.
    warp: tuple[jax.Array, jax.Array, jax.Array] | None = None
    # PID fallback control limits {stat_name: limit}; empty disables the fallback.
    control_envelopes: dict = eqx.field(static=True, default_factory=dict)
    control_multipliers: dict = eqx.field(static=True, default_factory=dict)
    step_context_offset: int = eqx.field(static=True, default=0)
    # PID law of the fallback action, taken from config.warmstart (same as the BC expert).
    pid_q: int = eqx.field(static=True, default=4)
    pid_safety: float = eqx.field(static=True, default=0.9)
    pid_min_factor: float = eqx.field(static=True, default=0.2)
    pid_max_factor: float = eqx.field(static=True, default=10.0)
    pid_kp: float = eqx.field(static=True, default=0.0)
    pid_ki: float = eqx.field(static=True, default=1.0)
    pid_kd: float = eqx.field(static=True, default=0.0)

    # diffrax args carry the raw task [mu, pulse_phase]; the encoder was trained on the
    # normalised task, so the normalisation constants are kept static.
    task_dim: int = eqx.field(static=True, default=0)
    task_min: float = eqx.field(static=True, default=0.0)
    task_max: float = eqx.field(static=True, default=1.0)
    # Static dispatch makes the stripped zero-encoder loop a distinct HLO
    # program. The opt-out supports differential regression tests.
    encoder_type: str = eqx.field(static=True, default="generic")
    specialize_zero_inference: bool = eqx.field(static=True, default=True)
    encoder_uses_reward: bool = eqx.field(static=True, default=True)

    def _task_from_args(self, args):
        """Extract the normalized task vector expected by model inputs."""
        if self.task_dim == 0:
            return None
        if args is None:
            raise ValueError(
                "This LearnedController uses task-conditioned model inputs, but the "
                "diffrax solve did not provide task args. Pass the ODE task as args."
            )
        task_args = jnp.asarray(args)
        if task_args.ndim != 1 or task_args.shape[0] < self.task_dim:
            raise ValueError(
                f"This LearnedController requires at least {self.task_dim} task values in "
                f"diffrax args, got shape {task_args.shape}."
            )
        raw_task = task_args[: self.task_dim]
        return (raw_task - jnp.float32(self.task_min)) / jnp.maximum(
            jnp.float32(self.task_max - self.task_min), jnp.float32(1e-8)
        )

    def _z(self, mu, logvar):
        """Build the policy latent input from posterior parameters."""
        if self.use_latent_sample:
            return mu + jnp.exp(0.5 * logvar) * jax.random.normal(jax.random.PRNGKey(0), mu.shape)
        return jnp.concatenate([mu, logvar], axis=-1)

    def _warp_frac(self, t, mu):
        """Φ(μ, t): fraction of PID solver effort spent by time t. Must match ODEEnv._warp_frac."""
        log_mu_grid, t_knots, fracs = self.warp
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

    @staticmethod
    def _f32(tree):
        """Cast inexact leaves to float32.

        The float64 solve would otherwise promote the carried state, and the
        while_loop carry must keep a fixed dtype.
        """
        return jax.tree.map(lambda a: a.astype(jnp.float32) if eqx.is_inexact_array(a) else a, tree)

    def _scale_dt(self, dt, action):
        """Apply the policy action and clamp the resulting step size."""
        return jnp.clip(
            dt * jnp.exp(action[0] * jnp.float32(self.dt_log_gain)),
            self.dtmin,
            self.dtmax,
        )

    def _zero_latent(self):
        """The exact [mu, logvar] actor input produced by ZeroEncoder."""
        return jnp.zeros((2 * self.latent_dim,), dtype=jnp.float32)

    @property
    def _uses_zero_fast_path(self):
        # Fallback needs policy/telemetry state not represented by
        # _ZeroLearnedState, so retain the generic path whenever it is enabled.
        return (
            self.specialize_zero_inference
            and self.encoder_type == "zero"
            and not self.control_envelopes
        )

    def wrap(self, direction: IntScalarLike):
        """Return this direction-independent controller unchanged."""
        return self

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: RealScalarLike | None,
        args: Args,
        func,
        error_order: RealScalarLike | None,
    ) -> tuple[RealScalarLike, _LearnedState]:
        """Initialize controller carry for a new diffrax solve."""
        del t1, func, error_order

        if dt0 is None:
            dt0 = jnp.float32(self.dt0_ref)
        if self.dtmax is not None:
            dt0 = jnp.minimum(dt0, self.dtmax)
        if self.dtmin is not None:
            dt0 = jnp.maximum(dt0, self.dtmin)

        if self._uses_zero_fast_path:
            return self._init_zero(t0, y0, dt0, args)

        prior_mu, prior_logvar = self._f32(self.encoder.prior())
        task = self._task_from_args(args)

        init_obs = self.obs_fn(
            ODEState(
                y=y0,
                t=t0,
                dt=dt0,
                solver_state_flag=jnp.bool_(True),
                solver_state_f=jnp.zeros_like(y0),
                last_log_error=jnp.float32(0.0),
                last_keep_step=jnp.bool_(True),
                step_count=jnp.int32(0),
                budget_exhausted=jnp.float32(0.0),
                accept_ema=jnp.float32(1.0),
                log_error_ema=jnp.float32(0.0),
                log_error_delta=jnp.float32(0.0),
                reject_streak=jnp.int32(0),
            )
        )

        if self.immediate_dt_action:
            # Matches training: the first env.step scales dt0 by the action the
            # policy takes on the reset observation under the prior belief.
            action0 = self._f32(
                self.policy.infer_action(init_obs, self._z(prior_mu, prior_logvar), task=task)
            )
            dt_first = self._scale_dt(dt0, action0)
        else:
            action0 = jnp.zeros(1, dtype=jnp.float32)
            dt_first = dt0

        state = _LearnedState(
            prev_obs=init_obs,
            prev_action=action0,
            gru_hidden=self._f32(self.encoder.init_hidden()),
            belief_mu=prior_mu,
            belief_logvar=prior_logvar,
            accept_ema=jnp.float32(1.0),
            log_error_ema=jnp.float32(0.0),
            last_log_error=jnp.float32(0.0),
            last_keep_step=jnp.float32(1.0),
            step_count=jnp.int32(0),
            budget_exhausted=jnp.float32(0.0),
            log_error_delta=jnp.float32(0.0),
            reject_streak=jnp.int32(0),
            pid_fallback_active=jnp.bool_(False),
        )
        return t0 + dt_first, state

    def _init_zero(self, t0, y0, dt0, args):
        """Initialise the zero-encoder controller without encoder state."""
        task = self._task_from_args(args)
        init_obs = self.obs_fn(
            ODEState(
                y=y0,
                t=t0,
                dt=dt0,
                solver_state_flag=jnp.bool_(True),
                solver_state_f=jnp.zeros_like(y0),
                last_log_error=jnp.float32(0.0),
                last_keep_step=jnp.bool_(True),
                step_count=jnp.int32(0),
                budget_exhausted=jnp.float32(0.0),
                accept_ema=jnp.float32(1.0),
                log_error_ema=jnp.float32(0.0),
                log_error_delta=jnp.float32(0.0),
                reject_streak=jnp.int32(0),
            )
        )
        if self.immediate_dt_action:
            action0 = self._f32(self.policy.infer_action(init_obs, self._zero_latent(), task=task))
            dt_first = self._scale_dt(dt0, action0)
            prev_obs = None
        else:
            dt_first = dt0
            prev_obs = init_obs
        state = _ZeroLearnedState(
            prev_obs=prev_obs,
            accept_ema=jnp.float32(1.0),
            log_error_ema=jnp.float32(0.0),
            last_log_error=jnp.float32(0.0),
            last_keep_step=jnp.float32(1.0),
            step_count=jnp.int32(0),
            budget_exhausted=jnp.float32(0.0),
            log_error_delta=jnp.float32(0.0),
            reject_streak=jnp.int32(0),
        )
        return t0 + dt_first, state

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        y1_candidate: Y,
        args: Args,
        y_error: Y | None,
        error_order: RealScalarLike,
        controller_state: _LearnedState,
    ) -> tuple[
        BoolScalarLike,
        RealScalarLike,
        RealScalarLike,
        BoolScalarLike,
        _LearnedState,
        RESULTS,
    ]:
        """Accept or reject a candidate step and choose the next step size."""
        if self._uses_zero_fast_path:
            return self._adapt_zero(t0, t1, y0, y1_candidate, args, y_error, controller_state)

        cs = controller_state
        prev_dt = t1 - t0
        task = self._task_from_args(args)

        # ── Accept/reject ─────────────────────────────────────────────
        y1_finite = jnp.all(jnp.isfinite(y1_candidate))
        y1_safe = jnp.where(y1_finite, y1_candidate, y0)
        scale = self.atol + jnp.maximum(jnp.abs(y0), jnp.abs(y1_safe)) * self.rtol
        scaled_error = jnp.sqrt(jnp.mean((y_error / scale) ** 2))
        safe_error = jnp.where(jnp.isfinite(scaled_error), scaled_error, jnp.float32(1e10))
        keep_step = (safe_error <= 1.0) & y1_finite

        # ── Update tracking EMAs ──────────────────────────────────────
        log_error = jnp.clip(jnp.log(safe_error + 1e-10), -30.0, 30.0).astype(jnp.float32)
        alpha = jnp.float32(_EMA_ALPHA)
        new_accept_ema = alpha * keep_step.astype(jnp.float32) + (1 - alpha) * cs.accept_ema
        new_log_error_ema = alpha * log_error + (1 - alpha) * cs.log_error_ema
        new_step_count = cs.step_count + jnp.int32(1)
        new_budget_exhausted = new_step_count.astype(jnp.float32) / jnp.float32(self.max_steps)
        log_error_delta = log_error - cs.last_log_error
        new_reject_streak = jnp.where(keep_step, jnp.int32(0), cs.reject_streak + jnp.int32(1))

        new_y = jnp.where(keep_step, y1_safe, y0)
        new_t = jnp.where(keep_step, t1, t0)

        # Skip the reward computation for encoders that do not consume reward.
        if self.encoder_uses_reward:
            t_capped = jnp.minimum(new_t, self.t_end)
            if self.warp is not None:
                mu = args[0]
                progress_frac = self._warp_frac(t_capped, mu) - self._warp_frac(t0, mu)
            else:
                progress_frac = (t_capped - t0) / self.t_end
            reward = progress_frac * self.reward_factor + self.step_cost
            reward = reward + jnp.where(keep_step, 0.0, jnp.float32(self.rejection_cost))
            reward = reward - jnp.where(
                keep_step,
                jnp.float32(self.margin_cost) * (jnp.float32(1.0) - safe_error),
                jnp.float32(0.0),
            )
            reward = reward.astype(jnp.float32)
        else:
            reward = jnp.zeros((), dtype=jnp.float32)

        def _obs_state(dt):
            return ODEState(
                y=new_y,
                t=new_t,
                dt=dt,
                solver_state_flag=jnp.bool_(True),
                solver_state_f=jnp.zeros_like(new_y),
                last_log_error=log_error,
                last_keep_step=keep_step,
                step_count=new_step_count,
                budget_exhausted=new_budget_exhausted,
                accept_ema=new_accept_ema,
                log_error_ema=new_log_error_ema,
                log_error_delta=log_error_delta,
                reject_streak=new_reject_streak,
            )

        # ── Control-chart PID fallback (config.pid_fallback) ──────────────
        # Needs "step_context" in the observation, so only computed when enabled.
        if self.control_envelopes:
            pid_obs = self.obs_fn(_obs_state(prev_dt))
            # Clipped like a policy action, since it also feeds the encoder as prev_action.
            pid_action = jnp.clip(
                pid_action_from_obs(
                    pid_obs,
                    self.step_context_offset,
                    q=self.pid_q,
                    safety=self.pid_safety,
                    min_factor=self.pid_min_factor,
                    max_factor=self.pid_max_factor,
                    dt_log_gain=self.dt_log_gain,
                    kp=self.pid_kp,
                    ki=self.pid_ki,
                    kd=self.pid_kd,
                ),
                -1.0,
                1.0,
            )
        else:
            pid_action = jnp.zeros(
                (1,), dtype=jnp.float32
            )  # never selected below: trigger stays False

        # The control-chart trigger is disabled: the PID fallback never engages.
        pid_active = jnp.bool_(False)

        if self.immediate_dt_action:
            # Immediate semantics: the new obs and belief reflect the attempt just
            # made, and the action scales the very next attempt.
            new_obs = self.obs_fn(_obs_state(prev_dt))
            new_mu, new_logvar, new_hidden = self._f32(
                self.encoder.encode_step(
                    cs.prev_action,
                    new_obs,
                    jnp.reshape(reward, (1,)),
                    cs.gru_hidden,
                    task=task,
                )
            )
            action = self._f32(
                self.policy.infer_action(new_obs, self._z(new_mu, new_logvar), task=task)
            )
            action = jnp.where(pid_active, pid_action, action)
            dt_next = self._scale_dt(prev_dt, action)
        else:
            # One-step-delay semantics: act on the previous obs/belief; the obs carries dt_next.
            action = self._f32(
                self.policy.infer_action(
                    cs.prev_obs, self._z(cs.belief_mu, cs.belief_logvar), task=task
                )
            )
            action = jnp.where(pid_active, pid_action, action)
            dt_next = self._scale_dt(prev_dt, action)
            new_obs = self.obs_fn(_obs_state(dt_next))
            new_mu, new_logvar, new_hidden = self._f32(
                self.encoder.encode_step(
                    action,
                    new_obs,
                    jnp.reshape(reward, (1,)),
                    cs.gru_hidden,
                    task=task,
                )
            )

        new_state = _LearnedState(
            prev_obs=new_obs,
            prev_action=action,
            gru_hidden=new_hidden,
            belief_mu=new_mu,
            belief_logvar=new_logvar,
            accept_ema=new_accept_ema,
            log_error_ema=new_log_error_ema,
            last_log_error=log_error,
            last_keep_step=keep_step.astype(jnp.float32),
            step_count=new_step_count,
            budget_exhausted=new_budget_exhausted,
            log_error_delta=log_error_delta,
            reject_streak=new_reject_streak,
            pid_fallback_active=pid_active,
        )

        return keep_step, new_t, new_t + dt_next, False, new_state, RESULTS.successful

    def _adapt_zero(self, t0, t1, y0, y1_candidate, args, y_error, controller_state):
        """Adapt a zero-encoder policy without encoder/reward/belief work."""
        cs = controller_state
        prev_dt = t1 - t0
        task = self._task_from_args(args)

        y1_finite = jnp.all(jnp.isfinite(y1_candidate))
        y1_safe = jnp.where(y1_finite, y1_candidate, y0)
        scale = self.atol + jnp.maximum(jnp.abs(y0), jnp.abs(y1_safe)) * self.rtol
        scaled_error = jnp.sqrt(jnp.mean((y_error / scale) ** 2))
        safe_error = jnp.where(jnp.isfinite(scaled_error), scaled_error, jnp.float32(1e10))
        keep_step = (safe_error <= 1.0) & y1_finite

        log_error = jnp.clip(jnp.log(safe_error + 1e-10), -30.0, 30.0).astype(jnp.float32)
        alpha = jnp.float32(_EMA_ALPHA)
        new_accept_ema = alpha * keep_step.astype(jnp.float32) + (1 - alpha) * cs.accept_ema
        new_log_error_ema = alpha * log_error + (1 - alpha) * cs.log_error_ema
        new_step_count = cs.step_count + jnp.int32(1)
        new_budget_exhausted = new_step_count.astype(jnp.float32) / jnp.float32(self.max_steps)
        log_error_delta = log_error - cs.last_log_error
        new_reject_streak = jnp.where(keep_step, jnp.int32(0), cs.reject_streak + jnp.int32(1))
        new_y = jnp.where(keep_step, y1_safe, y0)
        new_t = jnp.where(keep_step, t1, t0)

        def obs_state(dt):
            return ODEState(
                y=new_y,
                t=new_t,
                dt=dt,
                solver_state_flag=jnp.bool_(True),
                solver_state_f=jnp.zeros_like(new_y),
                last_log_error=log_error,
                last_keep_step=keep_step,
                step_count=new_step_count,
                budget_exhausted=new_budget_exhausted,
                accept_ema=new_accept_ema,
                log_error_ema=new_log_error_ema,
                log_error_delta=log_error_delta,
                reject_streak=new_reject_streak,
            )

        if self.immediate_dt_action:
            new_obs = self.obs_fn(obs_state(prev_dt))
            action = self._f32(self.policy.infer_action(new_obs, self._zero_latent(), task=task))
            next_prev_obs = None
        else:
            action = self._f32(
                self.policy.infer_action(cs.prev_obs, self._zero_latent(), task=task)
            )
            dt_next = self._scale_dt(prev_dt, action)
            new_obs = self.obs_fn(obs_state(dt_next))
            next_prev_obs = new_obs

        dt_next = self._scale_dt(prev_dt, action)
        new_state = _ZeroLearnedState(
            prev_obs=next_prev_obs,
            accept_ema=new_accept_ema,
            log_error_ema=new_log_error_ema,
            last_log_error=log_error,
            last_keep_step=keep_step.astype(jnp.float32),
            step_count=new_step_count,
            budget_exhausted=new_budget_exhausted,
            log_error_delta=log_error_delta,
            reject_streak=new_reject_streak,
        )
        return keep_step, new_t, new_t + dt_next, False, new_state, RESULTS.successful

    @staticmethod
    def from_checkpoint(checkpoint_path, config, env):
        """Load models from a checkpoint and build a matching controller."""
        from steppo.utils.checkpoint import build_models, load_checkpoint, load_control_envelopes

        vae, policy = build_models(config, env, config.seed)
        vae, policy = load_checkpoint(
            vae, policy, checkpoint_path, backbone=config.backbone, algo=config.algo
        )
        control_envelopes = (
            load_control_envelopes(checkpoint_path) if config.pid_fallback.enabled else {}
        )
        return LearnedController.from_models(
            vae, policy, config, env, control_envelopes=control_envelopes
        )

    @staticmethod
    def from_models(
        vae,
        policy,
        config,
        env,
        control_envelopes: dict | None = None,
        specialize_zero_inference: bool = True,
    ):
        """Build a controller around already-instantiated inference models."""
        from steppo.envs.ode.systems import get_system
        from steppo.models.backbones import get_backbone
        from steppo.training.pid_controller import compute_step_context_offset

        backbone_spec = get_backbone(config.backbone)
        model_config = getattr(config, backbone_spec.config_attr)

        spec = get_system(config.env.system)

        def obs_fn(state):
            return spec.obs(state, config.env)

        needs_task = getattr(vae.encoder, "task_embed", None) is not None or getattr(
            policy, "_use_task", False
        )
        task_dim = int(getattr(env, "task_dim", 0)) if needs_task else 0
        if needs_task:
            task_min, task_max = env._task_bounds()
            task_min, task_max = float(task_min), float(task_max)
        else:
            task_min, task_max = 0.0, 1.0

        control_envelopes = control_envelopes or {}
        control_multipliers = {
            "reject_streak": float(config.pid_fallback.reject_streak_multiplier),
            "accept_ema": float(config.pid_fallback.accept_ema_multiplier),
            "log_error_ema": float(config.pid_fallback.log_error_ema_multiplier),
        }
        step_context_offset = (
            compute_step_context_offset(config.env.obs_features, spec.feature_dims)
            if control_envelopes
            else 0  # unused placeholder — adapt_step_size gates on control_envelopes too
        )

        warp = getattr(env, "_warp", None)
        if warp is None and getattr(config.env, "progress_warp", False):
            from steppo.training.reward_norm import build_pid_warp_grid

            rn = config.training.reward_normalization
            lo, hi = env._task_bounds()
            warp = build_pid_warp_grid(
                config.env,
                float(lo),
                float(hi),
                grid_points=rn.grid_points,
                num_repeats=rn.num_repeats,
                max_steps=rn.max_steps or config.rollout_steps,
                require_cache=True,
            )

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return LearnedController(
                rtol=float(config.env.rtol),
                atol=float(config.env.atol),
                dtmin=float(config.env.dt_min),
                dtmax=float(config.env.dt_max),
                encoder=vae.encoder,
                policy=policy,
                obs_fn=obs_fn,
                t_end=float(config.env.t_end),
                dt0_ref=float(config.env.dt0),
                max_steps=int(config.rollout_steps),
                # use_latent_sample=bool(policy.use_latent_sample),
                use_latent_sample=False,  # Use mean+logvar for deterministic inference, as described in paper.
                latent_dim=int(model_config.total_latent_dim),
                reward_factor=float(config.env.reward_factor),
                step_cost=-float(config.env.survival_penalty),
                rejection_cost=-float(getattr(config.env, "rejection_penalty", 0.0)),
                margin_cost=float(getattr(config.env, "margin_penalty", 0.0)),
                dt_log_gain=float(config.env.dt_log_gain),
                immediate_dt_action=bool(getattr(config.env, "immediate_dt_action", False)),
                warp=warp,
                control_envelopes=control_envelopes,
                control_multipliers=control_multipliers,
                step_context_offset=step_context_offset,
                pid_q=int(config.warmstart.pid_order),
                pid_safety=float(config.warmstart.pid_safety),
                pid_min_factor=float(config.warmstart.pid_min_factor),
                pid_max_factor=float(config.warmstart.pid_max_factor),
                pid_kp=float(config.warmstart.pid_kp),
                pid_ki=float(config.warmstart.pid_ki),
                pid_kd=float(config.warmstart.pid_kd),
                task_dim=task_dim,
                task_min=task_min,
                task_max=task_max,
                encoder_type=str(config.vae.encoder.encoder_type),
                specialize_zero_inference=specialize_zero_inference,
                encoder_uses_reward=("reward" in config.vae.encoder.encoder_inputs),
            )
