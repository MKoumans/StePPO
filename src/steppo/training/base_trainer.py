"""Shared training loop for pluggable belief models and policy algorithms."""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

from steppo.configs.base_config import TrainConfig, apply_precision
from steppo.training.eval import (
    collect_control_stat_envelope,
    collect_latent_scatter,
    compute_pid_baseline_steps,
    eval_mu_table,
    eval_policy,
    print_mu_table,
    run_cached_diagnostic_episode,
)
from steppo.training.pid_controller import bc_loss, compute_step_context_offset
from steppo.training.policy_algos import PolicyAlgorithm, get_policy_algo
from steppo.training.ppo_algorithm import total_ppo_steps_for
from steppo.training.ppo_trainer import PPOTrainState, create_ppo_train_state
from steppo.training.reward_norm import build_pid_scale_grid, build_pid_warp_grid
from steppo.training.rollout import (
    RolloutState,
    collect_pid_rollout,
    collect_rollout,
    encode_oracle_batch,
    init_rollout_state,
)
from steppo.utils.logger import MetricsLogger, TimingProfiler, Timings
from steppo.utils.replay_buffer import get_trajectory_batch
from steppo.utils.summary import get_metrics, get_summary_str


@dataclass
class Trainer:
    """Training loop: rollouts, belief-model and policy updates, evaluation, checkpoints.

    Subclasses provide the belief model through `_init_belief_model`,
    `_update_belief_model` and `_belief_checkpoint_fields`.
    """

    config: TrainConfig
    vae: Any
    policy: Any
    env: Any
    env_params: Any
    # Defaults to the algorithm named by `config.algo`.
    policy_algo: Optional[type[PolicyAlgorithm]] = None

    def _init_belief_model(self, cfg: TrainConfig):
        """Create the belief-model state and replay buffer for this trainer."""
        raise NotImplementedError

    def _update_belief_model(
        self, cfg: TrainConfig, vae_state, vae_buffer, key_vae: PRNGKey, batch: Any
    ):
        """Update the belief model from one rollout and return its metrics."""
        raise NotImplementedError

    def _belief_checkpoint_fields(self, vae_state) -> dict:
        """Return belief-model state fields in the checkpoint schema."""
        raise NotImplementedError

    def __post_init__(self):
        """Resolve the policy algorithm from `config.algo` when none was given."""
        if self.policy_algo is None:
            self.policy_algo = get_policy_algo(self.config.algo).algo_cls
        self._control_envelopes: dict[str, float] = {}

    def _init(self, cfg: TrainConfig):
        self.env_dtype, self.model_dtype = apply_precision(cfg)

        vae_state, vae_buffer = self._init_belief_model(cfg)
        policy_state, policy_buffer = self.policy_algo.init_state(self.policy, cfg)

        logger = MetricsLogger(use_wandb=cfg.use_wandb)

        # Timing trackers
        self.timings = Timings()
        self.profiler = TimingProfiler(enabled=cfg.extensive_timing_logging)

        self._return_buffer: list[dict] = []
        self._adv_stats_buffer: tuple[list, list] = ([], [])
        self._reward_norm: Optional[tuple] = None
        self._progress_warp: bool = False
        self._efficiency_datasets: dict = {}

        self._all_metrics: list[dict] = []
        self._mu_history: list[tuple[int, list[dict]]] = []
        self._efficiency_history: list[tuple[int, dict[str, dict]]] = []
        self._l2_error_history: list[tuple[int, dict[str, dict]]] = []

        return (vae_state, vae_buffer), (policy_state, policy_buffer), logger

    def _save_hires_diagnostics(self, cfg: TrainConfig):
        """Writes individual high-res figures + a text snapshot of the hyperparameter
        values actually applied (lr, entropy coeff, grad norms, KL weight) to a
        `hires_diagnostics/` subfolder of the run's output directory."""
        from steppo.utils.plotting import _write_hyperparam_snapshot, save_hires_diagnostics

        hires_dir = os.path.join(cfg.output_path, "hires_diagnostics")
        save_hires_diagnostics(self._all_metrics, hires_dir)
        _write_hyperparam_snapshot(
            self._all_metrics, os.path.join(hires_dir, "hyperparams_latest.txt")
        )

    def _record_rollout_returns(self, batch):
        """Extract per-env episode return from a rollout batch and buffer it."""
        rewards = jnp.squeeze(batch.rewards, axis=-1)  # (T, num_envs)
        env_returns = float(jnp.sum(rewards, axis=0).mean())
        env_min = float(jnp.sum(rewards, axis=0).min())
        env_max = float(jnp.sum(rewards, axis=0).max())
        self._return_buffer.append({"mean": env_returns, "min": env_min, "max": env_max})

    def _flush_return_buffer(self) -> dict | None:
        """Aggregate buffered returns and clear. Returns None if empty."""
        if not self._return_buffer:
            return None
        means = [r["mean"] for r in self._return_buffer]
        mins = [r["min"] for r in self._return_buffer]
        maxs = [r["max"] for r in self._return_buffer]
        result = {
            "train/mean_return": sum(means) / len(means),
            "train/min_return": min(mins),
            "train/max_return": max(maxs),
        }
        self._return_buffer.clear()
        return result

    def _rollout(
        self, states: dict, keys: dict[str, PRNGKey], cfg: TrainConfig
    ) -> tuple[RolloutState, Any]:
        t_rollout_start = time.perf_counter()
        p = self.profiler

        vae_state = states.get("vae_state")
        policy_state = states.get("policy_state")

        key_task = keys.get("key_task")
        key_reset = keys.get("key_reset")
        key_rollout = keys.get("key_rollout")

        with p.section("rollout.nnx_merge"):
            self.vae = nnx.merge(vae_state.graphdef, vae_state.params)
            self.policy = nnx.merge(policy_state.graphdef, policy_state.params)

        with p.section("rollout.sample_task"):
            task_keys = jax.random.split(key_task, cfg.num_envs)
            env_params_batch = jax.vmap(self.env.sample_task)(task_keys)

        with p.section("rollout.init_state"):
            rollout_state = init_rollout_state(
                self.vae, self.env, env_params_batch, cfg.num_envs, key_reset
            )

        with p.section("rollout.collect"):
            rollout_state, batch = collect_rollout(
                rollout_state,
                self.vae,
                self.policy,
                self.env,
                env_params_batch,
                cfg,
                key_rollout,
            )

        p.block_and_record("rollout.collect", batch.rewards)

        self.timings.update(rollout_time=time.perf_counter() - t_rollout_start)

        return rollout_state, batch

    def _warmstart(
        self,
        vae_state,
        vae_buffer,
        ppo_state: PPOTrainState,
        rng_key: PRNGKey,
    ):
        """Imitation warmstart: collect expert rollouts, train belief model + BC on policy.

        `cfg.warmstart.expert` selects the demonstrator ("pid" or "oracle",
        see WarmstartConfig)."""
        import optax

        cfg = self.config
        ws_cfg = cfg.warmstart
        assert ws_cfg.expert in ("pid", "oracle"), (
            f"warmstart.expert must be 'pid' or 'oracle', got {ws_cfg.expert!r}"
        )
        if ws_cfg.expert == "oracle":
            assert cfg.env.immediate_dt_action, (
                "warmstart.expert='oracle' requires config.env.immediate_dt_action=True "
                "— the oracle's search candidate must affect the step it's testing."
            )

        feature_dims = self.env._spec.feature_dims
        self._step_context_offset = compute_step_context_offset(cfg.env.obs_features, feature_dims)

        bc_optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.training.grad_clip_ppo),
            optax.adam(ws_cfg.bc_lr),
        )
        bc_opt_state = bc_optimizer.init(ppo_state.params)

        oracle_dataset = None
        if ws_cfg.expert == "oracle":
            from steppo.training.oracle_dataset import load_oracle_training_batch

            raw = load_oracle_training_batch(
                cfg.env,
                rollout_steps=cfg.rollout_steps,
                episodes_per_trial=cfg.episodes_per_trial,
                seed=ws_cfg.oracle_dataset_seed,
                oracle_max_iters=ws_cfg.oracle_max_iters,
            )
            oracle_dataset = jax.tree.map(jnp.asarray, raw)
            print(
                f"[+] Oracle training set: {oracle_dataset['states'].shape[1]} episodes "
                f"x {oracle_dataset['states'].shape[0]} steps"
            )

        print(f"[*] Warmstart (WS): {ws_cfg.num_iters} iterations of {ws_cfg.expert} imitation")

        for ws_iter in range(ws_cfg.num_iters):
            key_iter = jax.random.fold_in(rng_key, ws_iter)
            key_task, key_rollout, key_vae, key_bc = jax.random.split(key_iter, 4)

            # 1. Collect expert rollouts
            self.vae = nnx.merge(vae_state.graphdef, vae_state.params)

            if ws_cfg.expert == "oracle":
                dataset_size = oracle_dataset["states"].shape[1]
                idx = jax.random.choice(
                    key_task,
                    dataset_size,
                    shape=(cfg.num_envs,),
                    replace=dataset_size < cfg.num_envs,
                )
                # task_params has no leading time axis (num_envs, task_dim) — every
                # other field is (rollout_steps, num_envs, ...).
                minibatch = {
                    k: (v[idx] if k == "task_params" else v[:, idx])
                    for k, v in oracle_dataset.items()
                }
                expert_batch = encode_oracle_batch(self.vae, minibatch, cfg)
            else:
                task_keys = jax.random.split(key_task, cfg.num_envs)
                env_params_batch = jax.vmap(self.env.sample_task)(task_keys)
                expert_batch = collect_pid_rollout(
                    self.vae,
                    self.env,
                    env_params_batch,
                    cfg,
                    ws_cfg,
                    self._step_context_offset,
                    key_rollout,
                )

            # 2. Train belief model on expert trajectories
            vae_state, vae_metrics = self._update_belief_model(
                cfg, vae_state, vae_buffer, key_vae, expert_batch
            )

            # 3. Behavioral cloning on policy (exclude frozen trial steps)
            T, num_envs = expert_batch.states.shape[0], expert_batch.states.shape[1]
            flat_states = expert_batch.states.reshape(T * num_envs, -1)
            flat_z = jnp.concatenate(
                [expert_batch.beliefs_mu, expert_batch.beliefs_logvar], axis=-1
            ).reshape(T * num_envs, -1)
            flat_actions = expert_batch.actions.reshape(T * num_envs, -1)
            flat_task = jnp.broadcast_to(
                expert_batch.task_params[None], (T,) + expert_batch.task_params.shape
            ).reshape(T * num_envs, -1)

            if cfg.episodes_per_trial > 0:
                frozen = (expert_batch.keep_steps < 0.5) & (expert_batch.dones > 0.5)
                bc_mask = ~frozen.reshape(-1)
                flat_states = flat_states[bc_mask]
                flat_z = flat_z[bc_mask]
                flat_actions = flat_actions[bc_mask]
                flat_task = flat_task[bc_mask]

            for bc_epoch in range(ws_cfg.bc_epochs):

                def loss_fn(params):
                    return bc_loss(
                        ppo_state.graphdef, params, flat_states, flat_z, flat_actions, flat_task
                    )

                (loss_val, bc_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    ppo_state.params
                )
                updates, bc_opt_state = bc_optimizer.update(grads, bc_opt_state, ppo_state.params)
                new_params = optax.apply_updates(ppo_state.params, updates)
                ppo_state = ppo_state.replace(params=new_params)

            if ws_iter % cfg.log_interval == 0:
                bc_val = float(bc_metrics["bc_loss"])
                vae_val = float(vae_metrics.get("total_loss", 0.0)) if vae_metrics else 0.0
                print(
                    f"[WS: {ws_iter:04d}/{ws_cfg.num_iters:04d}] bc_loss: {bc_val:.2E} | vae_loss: {vae_val:.2E}"
                )

        total_ppo_steps = total_ppo_steps_for(cfg)
        ppo_state = create_ppo_train_state(
            nnx.merge(ppo_state.graphdef, ppo_state.params), cfg.ppo, cfg.training, total_ppo_steps
        )

        print("[+] Warmstart complete.")

        return vae_state, vae_buffer, ppo_state

    def _log_and_evaluate(
        self,
        info: dict,
        states: dict,
        keys: dict[str, PRNGKey],
        logger: MetricsLogger,
        cfg: TrainConfig,
    ) -> dict:
        iteration = info.get("iteration")
        vae_metrics_iter = info.get("vae_metrics_iter", {})
        ppo_metrics_iter = info.get("ppo_metrics_iter", {})
        iter_time = info.get("iter_time", 0.0)

        vae_state = states.get("vae_state")
        policy_state = states.get("policy_state")
        key_eval = keys.get("key_eval")

        metrics = {
            **{f"vae/{k}": float(v) for k, v in vae_metrics_iter.items() if v is not None},
            **{
                f"{self.policy_algo.name}/{k}": float(v)
                for k, v in ppo_metrics_iter.items()
                if v is not None
            },
        }

        self._all_metrics.append(metrics)

        p = self.profiler

        # Latent-space scatter snapshot cadence is decoupled from eval_interval (cheaper
        # per-call than the full eval block, so it can run much more frequently).
        run_latent_scatter = (
            hasattr(self.env, "t_end")
            and cfg.latent_scatter.enabled
            and iteration % cfg.latent_scatter.interval == 0
        )
        run_eval_metrics = (
            hasattr(self.env, "t_end")
            and cfg.eval_metrics.enabled
            and iteration % cfg.eval_metrics.interval == 0
        )
        run_pid_fallback_calib = (
            hasattr(self.env, "t_end")
            and cfg.pid_fallback.enabled
            and iteration % cfg.pid_fallback.interval == 0
            and iteration / cfg.total_iters >= cfg.pid_fallback.start_frac
        )
        run_eval = cfg.eval_interval > 0 and iteration % cfg.eval_interval == 0

        if run_eval or run_latent_scatter or run_eval_metrics or run_pid_fallback_calib:
            with p.section("eval.nnx_merge"):
                self.vae = nnx.merge(vae_state.graphdef, vae_state.params)
                self.policy = nnx.merge(policy_state.graphdef, policy_state.params)

        # Latent-space scatter snapshot (belief μ colored by task λ, ODE envs only)
        if run_latent_scatter:
            with p.section("eval.latent_scatter"):
                try:
                    key_latent = jax.random.fold_in(key_eval, 555)
                    ls_cfg = cfg.latent_scatter
                    ls_max_steps = ls_cfg.max_steps or cfg.eval_max_steps
                    z, task_values = collect_latent_scatter(
                        self.vae,
                        self.policy,
                        self.env,
                        key_latent,
                        num_samples=ls_cfg.num_samples,
                        max_steps=ls_max_steps,
                    )
                    from steppo.utils.plotting import save_latent_scatter_snapshot

                    scatter_dir = os.path.join(cfg.output_path, "latent_scatter")
                    os.makedirs(scatter_dir, exist_ok=True)
                    save_latent_scatter_snapshot(
                        z,
                        task_values,
                        iteration,
                        ls_cfg.dims,
                        png_path=os.path.join(scatter_dir, f"iter_{iteration:07d}.png"),
                        csv_path=os.path.join(cfg.output_path, "latent_scatter_history.csv"),
                    )
                except Exception as e:
                    print(f"[-] Warning: Failed to save latent scatter snapshot: {e}")

        # Efficiency and L2 error per split against the cached PID datasets (ODE envs only).
        if run_eval_metrics:
            with p.section("eval.eval_metrics"):
                try:
                    from steppo.envs.ode.learned_controller import LearnedController
                    from steppo.training.eval_metrics import eval_metrics as compute_eval_metrics

                    em_cfg = cfg.eval_metrics
                    em_max_steps = em_cfg.max_steps or cfg.rollout_steps

                    controller = LearnedController.from_models(self.vae, self.policy, cfg, self.env)
                    em_results = compute_eval_metrics(
                        self.env._config,
                        controller,
                        self._efficiency_datasets,
                        num_envs=em_cfg.num_envs,
                        max_steps=em_max_steps,
                    )
                    for subset, r in em_results["efficiency"].items():
                        metrics[f"eval/efficiency_{subset}"] = r["mean"]
                    for subset, r in em_results["l2_error"].items():
                        metrics[f"eval/l2_error_{subset}"] = r["policy_err_mean"]
                        metrics[f"eval/l2_error_integrated_{subset}"] = r[
                            "policy_err_integrated_mean"
                        ]

                    from steppo.utils.plotting import (
                        plot_efficiency_history,
                        plot_l2_error_history,
                        save_efficiency_txt,
                        save_l2_error_txt,
                    )

                    os.makedirs(cfg.output_path, exist_ok=True)

                    self._efficiency_history.append((iteration, em_results["efficiency"]))
                    plot_efficiency_history(
                        self._efficiency_history,
                        os.path.join(cfg.output_path, "efficiency.png"),
                    )
                    save_efficiency_txt(
                        self._efficiency_history,
                        os.path.join(cfg.output_path, "efficiency.txt"),
                    )

                    self._l2_error_history.append((iteration, em_results["l2_error"]))
                    plot_l2_error_history(
                        self._l2_error_history,
                        os.path.join(cfg.output_path, "l2_error.png"),
                    )
                    save_l2_error_txt(
                        self._l2_error_history,
                        os.path.join(cfg.output_path, "l2_error.txt"),
                    )
                except Exception as e:
                    print(f"[-] Warning: Failed to compute eval_metrics: {e}")

        # PID-fallback calibration late in training: EMA each statistic's percentile
        # into self._control_envelopes (saved with the checkpoint).
        if run_pid_fallback_calib:
            with p.section("eval.pid_fallback_calib"):
                try:
                    pf_cfg = cfg.pid_fallback
                    pf_max_steps = pf_cfg.max_steps or cfg.eval_max_steps
                    key_calib = jax.random.fold_in(key_eval, 333)
                    alpha = pf_cfg.ema_alpha
                    for name, enabled, pct in [
                        (
                            "reject_streak",
                            pf_cfg.track_reject_streak,
                            pf_cfg.reject_streak_percentile,
                        ),
                        ("accept_ema", pf_cfg.track_accept_ema, pf_cfg.accept_ema_percentile),
                        (
                            "log_error_ema",
                            pf_cfg.track_log_error_ema,
                            pf_cfg.log_error_ema_percentile,
                        ),
                    ]:
                        if not enabled:
                            continue
                        key_stat = jax.random.fold_in(key_calib, hash(name) % (2**31))
                        stat = collect_control_stat_envelope(
                            self.vae,
                            self.policy,
                            self.env,
                            key_stat,
                            pf_cfg.num_envs,
                            pf_max_steps,
                            name,
                            pct,
                        )
                        prev = self._control_envelopes.get(name)
                        self._control_envelopes[name] = (
                            stat if prev is None else alpha * prev + (1 - alpha) * stat
                        )
                except Exception as e:
                    print(f"[-] Warning: Failed to update PID-fallback control envelopes: {e}")

        # Run evaluation at specific intervals
        if run_eval:
            # ODE environments replay the cached validation/test tasks; others sample tasks.
            eval_datasets = (
                {
                    split: self._efficiency_datasets[split]
                    for split in ("val", "test")
                    if split in self._efficiency_datasets
                }
                if hasattr(self.env, "t_end")
                else {None: None}
            )
            eval_metrics = {}
            for split, dataset in eval_datasets.items():
                with p.section("eval.eval_policy"):
                    eval_m = eval_policy(
                        self.vae,
                        self.policy,
                        self.env,
                        self.env_params,
                        key_eval,
                        cfg.num_eval_episodes,
                        num_envs=cfg.num_eval_episodes,
                        max_steps=cfg.eval_max_steps,
                        episodes_per_trial=cfg.episodes_per_trial,
                        reset_belief_between_episodes=cfg.reset_belief_between_episodes,
                        dataset=dataset,
                    )
                prefix = "eval" if split is None else f"eval/{split}"
                for k, v in eval_m.items():
                    try:
                        value = float(v)
                    except (TypeError, ValueError):
                        value = v
                    eval_metrics[f"{prefix}/{k}"] = value
            metrics.update(eval_metrics)

            # Diagnostic rollout plot (ODE envs only)
            if cfg.save_plots and hasattr(self.env, "t_end") and eval_datasets:
                with p.section("eval.diagnostic_plot"):
                    try:
                        key_diag = jax.random.fold_in(key_eval, 999)
                        from steppo.utils.plotting import save_rollout_diagnostic_plot

                        os.makedirs(cfg.output_path, exist_ok=True)
                        for split, dataset in eval_datasets.items():
                            if split is None:
                                continue
                            diag = run_cached_diagnostic_episode(
                                self.vae,
                                self.policy,
                                self.env,
                                dataset,
                                key_diag,
                                cfg.eval_max_steps,
                            )
                            plot_path = os.path.join(
                                cfg.output_path, f"rollout_{split}_iter_{iteration}.png"
                            )
                            save_rollout_diagnostic_plot(diag, plot_path, t_end=self.env.t_end)
                    except Exception as e:
                        print(f"[-] Warning: Failed to save rollout diagnostic: {e}")

            # Per-mu breakdown table (ODE envs only)
            if cfg.save_plots and hasattr(self.env, "t_end"):
                with p.section("eval.mu_table"):
                    try:
                        key_mu = jax.random.fold_in(key_eval, 888)
                        mu_results = eval_mu_table(
                            self.vae,
                            self.policy,
                            self.env,
                            mus=list(cfg.precompute_baseline.eval_mus),
                            rng_key=key_mu,
                            max_steps=cfg.eval_max_steps,
                            num_repeats=16,
                            episodes_per_trial=cfg.episodes_per_trial,
                            reset_belief_between_episodes=cfg.reset_belief_between_episodes,
                        )
                        print_mu_table(
                            mu_results,
                            float(self.env.t_end),
                            cfg.eval_max_steps,
                            pid_baseline=self._pid_baseline,
                        )

                        # Hypothesis 1 diagnostic plots
                        from steppo.utils.plotting import (
                            plot_mu_advantage_scatter,
                            plot_mu_return_history,
                            save_mu_return_txt,
                        )

                        os.makedirs(cfg.output_path, exist_ok=True)
                        self._mu_history.append((iteration, mu_results))
                        plot_mu_return_history(
                            self._mu_history,
                            os.path.join(cfg.output_path, "mu_returns.png"),
                        )
                        save_mu_return_txt(
                            self._mu_history,
                            os.path.join(cfg.output_path, "mu_returns.txt"),
                        )
                        if self._adv_stats_buffer[0]:
                            plot_mu_advantage_scatter(
                                np.concatenate(self._adv_stats_buffer[0]),
                                np.concatenate(self._adv_stats_buffer[1]),
                                os.path.join(cfg.output_path, f"mu_adv_{iteration}.png"),
                                iteration,
                            )
                            self._adv_stats_buffer = ([], [])
                    except Exception as e:
                        print(f"[-] Warning: Failed to run per-μ table: {e}")

            # Checkpoint periodically
            with p.section("eval.checkpoint"):
                try:
                    self.save_checkpoint(vae_state, policy_state, iteration)
                except Exception as e:
                    print(f"[-] Warning: Failed to save checkpoint at iteration {iteration}: {e}")

            # Hyperparameter diagnostics (applied lr, entropy coeff, grad norms, KL weight)
            if cfg.save_plots:
                with p.section("eval.hires_diagnostics"):
                    try:
                        self._save_hires_diagnostics(cfg)
                    except Exception as e:
                        print(
                            f"[-] Warning: Failed to save hires diagnostics at iteration {iteration}: {e}"
                        )

        # Print live training loss progress
        if iteration % cfg.log_interval == 0:
            return_stats = self._flush_return_buffer()
            if return_stats:
                metrics.update(return_stats)
            logger.log_training(iteration, cfg.total_iters, metrics, iter_time)

            if p.enabled and iteration > 0:
                print(p.report(last_n=cfg.log_interval))

        if iteration != 0:
            self.timings.update(iter_time=iter_time)

        return metrics

    def _compute_baseline(self, cfg: TrainConfig, verbose: int = 0):
        self._pid_baseline = None

        if hasattr(self.env, "t_end") and cfg.precompute_baseline.enabled:
            try:
                eval_mus = cfg.precompute_baseline.eval_mus
                max_steps = cfg.precompute_baseline.max_steps or cfg.rollout_steps * 10
                if verbose > 0:
                    print("[*] Computing PID baseline steps per μ ...")
                # Use the eval env's config so the baseline matches whatever
                # environment eval_mu_table actually runs under.
                baseline_env_config = self.env._config
                self._pid_baseline = compute_pid_baseline_steps(
                    baseline_env_config,
                    mus=eval_mus,
                    num_repeats=cfg.precompute_baseline.num_repeats,
                    max_steps=max_steps,
                    require_cache=True,
                    verbose=verbose,
                )
                pid_str = ", ".join(
                    f"μ={mu}: {v['steps']:.0f} ({v['accepted']:.0f}/{v['rejected']:.0f})"
                    for mu, v in self._pid_baseline.items()
                )
                if verbose > 0:
                    print(f"[+] PID baseline: {pid_str}")
            except FileNotFoundError:
                raise
            except Exception as e:
                if verbose > 0:
                    print(f"[-] Warning: Failed to compute PID baseline: {e}")

    def _setup_progress_warp(self, cfg: TrainConfig, verbose: int = 0):
        """Build PID effort warp and install it on the training env before tracing."""
        self._progress_warp = False
        if not getattr(cfg.env, "progress_warp", False):
            return
        if not (hasattr(self.env, "t_end") and getattr(self.env, "task_dim", 0) == 1):
            if verbose > 0:
                print(
                    "[-] Warning: progress_warp enabled but env is not a scalar-task ODE env; skipping."
                )
            return

        rn = cfg.training.reward_normalization
        lo, hi = self.env._task_bounds()
        max_steps = rn.max_steps or cfg.rollout_steps
        if verbose > 0:
            print("[*] Building PID progress-warp grid ...")
        log_mu_grid, t_knots, fracs = build_pid_warp_grid(
            cfg.env,
            float(lo),
            float(hi),
            grid_points=rn.grid_points,
            num_repeats=rn.num_repeats,
            max_steps=max_steps,
            require_cache=True,
            verbose=verbose,
        )
        self.env.set_progress_warp(log_mu_grid, t_knots, fracs)
        self._progress_warp = True
        if verbose > 0:
            print(
                f"[+] Progress warp: μ∈[{float(lo):.4g}, {float(hi):.4g}], "
                f"{t_knots.shape[0]} μ pts × {t_knots.shape[1]} knots, "
                f"median-μ half-effort time t(Φ=0.5)="
                f"{float(t_knots[t_knots.shape[0] // 2, t_knots.shape[1] // 2]):.4g} "
                f"of t_end={float(cfg.env.t_end):.4g}"
            )

    def _per_mu_reward_normalization(self, cfg: TrainConfig, verbose: int = 0) -> float:
        self._reward_norm = None
        rn = cfg.training.reward_normalization

        if rn.enabled and self._progress_warp:
            if verbose > 0:
                print(
                    "[*] Per-μ reward normalization skipped: progress_warp already equalizes per-episode reward totals across μ."
                )
            return

        if rn.enabled and hasattr(self.env, "t_end") and getattr(self.env, "task_dim", 0) == 1:
            try:
                lo, hi = self.env._task_bounds()
                max_steps = rn.max_steps or cfg.rollout_steps
                if verbose > 0:
                    print("[*] Building per-μ reward normalization grid ...")
                self._reward_norm = build_pid_scale_grid(
                    cfg.env,
                    float(lo),
                    float(hi),
                    grid_points=rn.grid_points,
                    num_repeats=rn.num_repeats,
                    max_steps=max_steps,
                    require_cache=True,
                    verbose=verbose,
                )
                log_mu_grid, log_pid_steps, log_c = self._reward_norm
                if verbose > 0:
                    print(
                        f"[+] Reward norm grid: μ∈[{float(lo):.4g}, {float(hi):.4g}], "
                        f"{rn.grid_points} pts, PID steps "
                        f"[{float(jnp.exp(log_pid_steps[0])):.0f} … {float(jnp.exp(log_pid_steps[-1])):.0f}], "
                        f"C={float(jnp.exp(log_c)):.1f}"
                    )
            except FileNotFoundError:
                raise
            except Exception as e:
                if verbose > 0:
                    print(f"[-] Warning: Failed to build reward normalization grid: {e}")
        elif rn.enabled:
            if verbose > 0:
                print(
                    "[-] Warning: reward_normalization enabled but env is not a scalar-task ODE env; skipping."
                )

    def _setup_efficiency_datasets(self, cfg: TrainConfig, verbose: int = 0):
        """Load the cached train/val/test PID evaluation datasets used by eval_metrics.

        These are the canonical datasets checked by check_oracle_dataset.py and
        must exist before training starts.
        """
        self._efficiency_datasets = {}
        if not hasattr(self.env, "t_end"):
            return
        env_config = self.env._config
        has_eval_splits = bool(env_config.val_bins or env_config.test_bins)
        if cfg.eval_interval > 0 and not has_eval_splits:
            raise ValueError(
                "cfg.eval_interval > 0 but env.val_bins and env.test_bins are both "
                "empty — periodic eval would silently run zero eval/* metrics and "
                "produce no diagnostic plots for the whole run. Set at least one of "
                "val_bins/test_bins, or set eval_interval=0 to disable periodic eval."
            )
        needs_cached_eval = cfg.eval_metrics.enabled or has_eval_splits
        if not needs_cached_eval:
            return

        from steppo.training.error_dist import (
            PID_EVAL_NUM_ENVS,
            PID_EVAL_SEED,
            load_cached_pid_batch,
        )

        for split in ("train", "val", "test"):
            bins = getattr(env_config, f"{split}_bins")
            if not bins:
                continue
            self._efficiency_datasets[split] = load_cached_pid_batch(
                env_config,
                max_steps=cfg.rollout_steps,
                bins=list(bins),
                split=split,
                num_envs=PID_EVAL_NUM_ENVS,
                seed=PID_EVAL_SEED,
            )
        if verbose > 0:
            print(f"[+] Loaded PID efficiency-metric datasets: {sorted(self._efficiency_datasets)}")

    @staticmethod
    def _remap_checkpoint_tree(template: Any, restored: Any) -> Any:
        """Give a restored Orbax tree the structure of a fresh state tree."""
        return jax.tree.unflatten(jax.tree.structure(template), jax.tree.leaves(restored))

    def _restore_training_states(self, vae_state, policy_state, checkpoint_dir: str):
        """Restore parameters, optimiser states and step counters from a checkpoint.

        Replay buffers, RNG state and diagnostic EMAs are not checkpointed.
        """
        restored = self.load_checkpoint(checkpoint_dir)

        def restore_state(state, field_templates: dict, label: str):
            replacements = {}
            for key in field_templates:
                if key not in restored:
                    continue
                try:
                    state_field = key.split("_", 1)[1]
                    template = getattr(state, state_field)
                except (IndexError, AttributeError) as exc:
                    raise KeyError(
                        f"Checkpoint field {key!r} has no matching {label} state field"
                    ) from exc
                if key.endswith(("_params", "_opt_state")):
                    replacements[state_field] = self._remap_checkpoint_tree(template, restored[key])
                else:
                    replacements[state_field] = restored[key]
            missing = [key for key in field_templates if key not in restored]
            if missing:
                raise KeyError(
                    f"Checkpoint {checkpoint_dir!r} is missing {label} fields: {missing}"
                )
            return state.replace(**replacements)

        vae_state = restore_state(
            vae_state, self._belief_checkpoint_fields(vae_state), "belief-model"
        )
        policy_state = restore_state(
            policy_state, self.policy_algo.checkpoint_fields(policy_state), "policy"
        )

        if "control_envelopes" in restored:
            self._control_envelopes = {
                str(k): float(v) for k, v in restored["control_envelopes"].items()
            }

        iteration = int(restored["iteration"])
        print(
            f"[+] Resuming from checkpoint iteration {iteration}; "
            f"unsaved replay/RNG state will be reinitialized"
        )
        return vae_state, policy_state, iteration

    def _refill_replay_buffer(
        self, vae_state, policy_state, vae_buffer, key_train: PRNGKey, count: int
    ) -> None:
        """Populate a fresh belief-model buffer with non-training rollouts."""
        if count <= 0:
            return
        cfg = self.config
        for refill_idx in range(count):
            key_refill = jax.random.fold_in(key_train, 10_000 + refill_idx)
            key_task, key_reset, key_rollout = jax.random.split(key_refill, 3)
            _, batch = self._rollout(
                states={
                    "vae_state": vae_state,
                    "policy_state": policy_state,
                    "vae_buffer": vae_buffer,
                },
                keys={
                    "key_task": key_task,
                    "key_reset": key_reset,
                    "key_rollout": key_rollout,
                },
                cfg=cfg,
            )
            vae_buffer.insert(get_trajectory_batch(batch, self.policy.action_dim))
        print(f"[+] Refilled replay buffer with {count} fresh rollout(s)")

    def train(
        self,
        rng_key: PRNGKey,
        resume_checkpoint: str | None = None,
        resume_replay_rollouts: int = 4,
    ) -> dict:
        """Run training, optionally continuing from a checkpoint.

        A continuation keeps absolute iteration numbers; the replay buffer is
        refilled and RNG keys are derived from the seed and iteration.
        """
        key_train = rng_key
        cfg = self.config

        # ── Init ──────────────────────────────────────────────────
        (vae_state, vae_buffer), (policy_state, policy_buffer), logger = self._init(cfg)

        # Must run before warmstart/rollouts that jit-trace env.step.
        self._setup_progress_warp(cfg, verbose=1)

        start_iteration = 0
        if resume_checkpoint is not None:
            if resume_replay_rollouts < 0:
                raise ValueError("resume_replay_rollouts must be non-negative")
            vae_state, policy_state, checkpoint_iteration = self._restore_training_states(
                vae_state, policy_state, resume_checkpoint
            )
            start_iteration = checkpoint_iteration + 1
            if start_iteration >= cfg.total_iters:
                raise ValueError(
                    f"Checkpoint iteration {checkpoint_iteration} is already at or beyond "
                    f"configured total_iters={cfg.total_iters}"
                )
            self._refill_replay_buffer(
                vae_state, policy_state, vae_buffer, key_train, resume_replay_rollouts
            )
        elif cfg.warmstart.enabled:
            if self.policy_algo.supports_warmstart:
                vae_state, vae_buffer, policy_state = self._warmstart(
                    vae_state, vae_buffer, policy_state, jax.random.fold_in(key_train, 999999)
                )
            else:
                print(
                    f"[-] Warmstart requested but policy_algo '{self.policy_algo.name}' "
                    "doesn't support it; skipping."
                )

        self._compute_baseline(cfg, verbose=1)

        self._per_mu_reward_normalization(cfg, verbose=1)

        self._setup_efficiency_datasets(cfg, verbose=1)

        start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[*] Starting training loop at {start_time_str}. Total iterations: {cfg.total_iters} (log interval: {cfg.log_interval}, eval interval: {cfg.eval_interval})"
        )
        for iteration in range(start_iteration, cfg.total_iters):
            self.profiler.start_iteration()
            t_start = time.perf_counter()
            key_iter = jax.random.fold_in(key_train, iteration)
            key_task, key_reset, key_rollout, key_vae, key_ppo, key_eval = jax.random.split(
                key_iter, 6
            )

            # -- Rollout --
            rollout_state, batch = self._rollout(
                states={
                    "vae_state": vae_state,
                    "policy_state": policy_state,
                    "vae_buffer": vae_buffer,
                },
                keys={"key_task": key_task, "key_reset": key_reset, "key_rollout": key_rollout},
                cfg=cfg,
            )

            with self.profiler.section("record_returns"):
                self._record_rollout_returns(batch)

            # -- Belief model update --
            vae_state, vae_metrics_iter = self._update_belief_model(
                cfg, vae_state, vae_buffer, key_vae, batch
            )
            # Expose the freshly updated belief model to the policy update.
            self.vae = nnx.merge(vae_state.graphdef, vae_state.params)

            # -- Policy update --
            policy_state, ppo_metrics_iter = self.policy_algo.update(
                self, cfg, policy_state, policy_buffer, key_ppo, batch, iteration
            )

            self.profiler.end_iteration()

            # -- Logging & Evaluation --
            self._log_and_evaluate(
                info={
                    "iteration": iteration,
                    "vae_metrics_iter": vae_metrics_iter,
                    "ppo_metrics_iter": ppo_metrics_iter,
                    "iter_time": time.perf_counter() - t_start,
                },
                states={"vae_state": vae_state, "policy_state": policy_state},
                keys={"key_eval": key_eval},
                logger=logger,
                cfg=cfg,
            )

        # ── Performance Summary Report ──────────────────────────
        summary_metrics = get_metrics(
            cfg,
            self.timings.rollout_times,
            self.timings.vae_times,
            self.timings.ppo_times,
            self.timings.iter_times,
            jax.devices(),
        )

        self.output(
            states={"vae_state": vae_state, "policy_state": policy_state},
            metrics=summary_metrics,
            all_metrics=self._all_metrics,
            cfg=cfg,
        )

        return {"metrics": self._all_metrics, "vae_state": vae_state, "policy_state": policy_state}

    def evaluate(self, vae_state, policy_state, rng_key: PRNGKey, num_episodes: int = 64) -> dict:
        """Evaluate the current belief model and policy on held-out tasks."""
        vae = nnx.merge(vae_state.graphdef, vae_state.params)
        policy = nnx.merge(policy_state.graphdef, policy_state.params)
        dataset = None
        if hasattr(self.env, "t_end"):
            dataset = self._efficiency_datasets.get("val") or self._efficiency_datasets.get("test")
        cfg = self.config
        return eval_policy(
            vae,
            policy,
            self.env,
            self.env_params,
            rng_key,
            num_episodes,
            episodes_per_trial=cfg.episodes_per_trial,
            reset_belief_between_episodes=cfg.reset_belief_between_episodes,
            dataset=dataset,
        )

    def _save_config_yaml(self):
        """Write the resolved config to the run's checkpoint directory (once)."""
        from dataclasses import asdict

        import yaml

        config_path = os.path.join(self.config.checkpoint_path, "config.yaml")
        if os.path.exists(config_path):
            return
        os.makedirs(self.config.checkpoint_path, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.safe_dump(asdict(self.config), f, default_flow_style=False, sort_keys=False)
        self._save_note()

    def _save_note(self):
        """Write the user-provided --note to the run's checkpoint directory (once)."""
        if not self.config.note:
            return
        note_path = os.path.join(self.config.checkpoint_path, "note.txt")
        if os.path.exists(note_path):
            return
        with open(note_path, "w") as f:
            f.write(self.config.note + "\n")

    def save_checkpoint(self, vae_state, policy_state, iteration: int) -> str:
        """Saves a checkpoint of the current training states."""
        import orbax.checkpoint as ocp

        ckpt_dir = os.path.join(self.config.checkpoint_path, f"checkpoint_{iteration}")
        os.makedirs(self.config.checkpoint_path, exist_ok=True)

        self._save_config_yaml()

        state_dict = {
            **self._belief_checkpoint_fields(vae_state),
            **self.policy_algo.checkpoint_fields(policy_state),
            "iteration": iteration,
        }
        if self._control_envelopes:
            state_dict["control_envelopes"] = {
                k: jnp.float32(v) for k, v in self._control_envelopes.items()
            }

        ckptr = ocp.StandardCheckpointer()
        ckptr.save(ckpt_dir, state_dict)
        ckptr.wait_until_finished()
        print(f"[+] Saved model checkpoint to: {ckpt_dir}")
        return ckpt_dir

    def load_checkpoint(self, checkpoint_dir: str) -> dict:
        """Restores training states from a checkpoint directory."""
        import orbax.checkpoint as ocp

        ckptr = ocp.StandardCheckpointer()
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        restored = ckptr.restore(checkpoint_dir, strict=False)
        print(f"[+] Restored training state from: {checkpoint_dir}")
        return restored

    def _save_steps_vs_mu(self, vae_state, policy_state, cfg: TrainConfig) -> None:
        """Write the final cache-backed PID/Policy/Oracle steps comparison."""
        if not hasattr(self.env, "t_end"):
            return
        if not cfg.env.test_bins:
            print("[*] Skipping steps-vs-μ comparison (config.env.test_bins is empty)")
            return
        from steppo.envs.ode.learned_controller import LearnedController
        from steppo.training.steps_vs_mu import build_steps_vs_mu_data_by_split
        from steppo.utils.figures.ode import (
            get_param_range,
            plot_steps_vs_mu_splits,
            save_convergence_table,
            save_steps_vs_mu_splits_txt,
            write_steps_vs_mu_error_outputs,
        )

        vae = nnx.merge(vae_state.graphdef, vae_state.params)
        policy = nnx.merge(policy_state.graphdef, policy_state.params)
        controller = LearnedController.from_models(vae, policy, cfg, self.env)
        split_data = build_steps_vs_mu_data_by_split(
            cfg.env,
            max_steps=cfg.rollout_steps,
            split_bins={
                "train": cfg.env.train_bins,
                "val": cfg.env.val_bins,
                "test": cfg.env.test_bins,
            },
            policy_controller=controller,
            with_error=True,
        )
        mu_min, mu_max, param_label = get_param_range(cfg)
        output_path = os.path.join(cfg.output_path, "steps_vs_mu.png")
        if mu_min == mu_max:
            table_path = os.path.splitext(output_path)[0] + ".txt"
            save_convergence_table(split_data, table_path, param_label=param_label)
            print(f"[+] Saved convergence table (single {param_label}={mu_min:g}) to: {table_path}")
            return
        plot_steps_vs_mu_splits(
            split_data,
            mu_min,
            mu_max,
            output_path,
            param_label=param_label,
            train_bins=cfg.env.train_bins,
        )
        txt_path = os.path.splitext(output_path)[0] + ".txt"
        save_steps_vs_mu_splits_txt(split_data, txt_path, mu_min, mu_max)
        print(f"[+] Saved steps-vs-μ comparison to: {output_path}")
        for err_txt_path in write_steps_vs_mu_error_outputs(
            split_data,
            mu_min,
            mu_max,
            output_path,
            param_label=param_label,
            train_bins=cfg.env.train_bins,
        ):
            print(f"[+] Saved error-vs-μ comparison to: {err_txt_path}")

    def output(self, states: dict, metrics: dict, all_metrics: list, cfg: TrainConfig) -> dict:
        """Persist final artifacts and return the collected training results."""
        vae_state = states.get("vae_state")
        policy_state = states.get("policy_state")

        summary_str = get_summary_str(metrics, cfg)

        # Save final checkpoint at the end of training
        try:
            self.save_checkpoint(vae_state, policy_state, cfg.total_iters)
        except Exception as e:
            print(f"[-] Warning: Failed to save final checkpoint: {e}")

        # Save to benchmark_summary.txt in the output_path directory
        output_dir = cfg.output_path
        try:
            os.makedirs(output_dir, exist_ok=True)
            summary_path = os.path.join(output_dir, "benchmark_summary.txt")
            with open(summary_path, "w") as f:
                f.write(summary_str)
            print(f"[+] Saved performance benchmark to: {summary_path}")
        except Exception as e:
            print(f"[-] Warning: Failed to save benchmark summary to file in '{output_dir}': {e}")
            fallback_dirs = ["/tmp", os.path.expanduser("~")]
            saved = False
            for f_dir in fallback_dirs:
                try:
                    os.makedirs(f_dir, exist_ok=True)
                    fallback_path = os.path.join(f_dir, "benchmark_summary.txt")
                    with open(fallback_path, "w") as f:
                        f.write(summary_str)
                    print(f"[+] Saved performance benchmark to fallback: {fallback_path}")
                    saved = True
                    break
                except Exception:
                    continue
            if not saved:
                print("[-] Error: Could not save benchmark summary to any fallback directory.")

        # Save full-run section timing breakdown (eval.*, rollout/vae/ppo, ...)
        if cfg.extensive_timing_logging:
            try:
                os.makedirs(output_dir, exist_ok=True)
                timings_path = os.path.join(output_dir, "timings.txt")
                with open(timings_path, "w") as f:
                    f.write(self.profiler.report())
                print(f"[+] Saved timing profile to: {timings_path}")
            except Exception as e:
                print(f"[-] Warning: Failed to save timing profile in '{output_dir}': {e}")

        # ── Export & Plotting ───────────────────────────────────
        if cfg.save_json or cfg.save_plots:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                print(f"[-] Warning: Could not create directory '{output_dir}': {e}")

        if cfg.save_json:
            json_path = os.path.join(output_dir, f"{cfg.exp_name}_metrics.json")
            try:
                with open(json_path, "w") as f:
                    json.dump(all_metrics, f, indent=2)
                print(f"[+] Saved metrics JSON to: {json_path}")
            except Exception as e:
                print(f"[-] Warning: Failed to save metrics JSON in '{output_dir}': {e}")
                fallback_dirs = ["/tmp", os.path.expanduser("~")]
                saved = False
                for f_dir in fallback_dirs:
                    try:
                        os.makedirs(f_dir, exist_ok=True)
                        fallback_path = os.path.join(f_dir, f"{cfg.exp_name}_metrics.json")
                        with open(fallback_path, "w") as f:
                            json.dump(all_metrics, f, indent=2)
                        print(f"[+] Saved metrics JSON to fallback: {fallback_path}")
                        saved = True
                        break
                    except Exception:
                        continue
                if not saved:
                    print("[-] Error: Could not save metrics JSON to any fallback directory.")

        if cfg.save_plots:
            plot_path = os.path.join(output_dir, f"{cfg.exp_name}_metrics.png")
            try:
                from steppo.utils.plotting import save_training_plots

                save_training_plots(all_metrics, plot_path)
                print(f"[+] Saved training progress plot to: {plot_path}")
            except Exception as e:
                print(f"[-] Warning: Failed to save training plots in '{output_dir}': {e}")
                fallback_dirs = ["/tmp", os.path.expanduser("~")]
                saved = False
                for f_dir in fallback_dirs:
                    try:
                        os.makedirs(f_dir, exist_ok=True)
                        from steppo.utils.plotting import save_training_plots

                        fallback_path = os.path.join(f_dir, f"{cfg.exp_name}_metrics.png")
                        save_training_plots(all_metrics, fallback_path)
                        print(f"[+] Saved training progress plot to fallback: {fallback_path}")
                        saved = True
                        break
                    except Exception:
                        continue
                if not saved:
                    print(
                        "[-] Error: Could not save training progress plot to any fallback directory."
                    )

            try:
                self._save_hires_diagnostics(cfg)
                print(
                    f"[+] Saved hires diagnostics to: {os.path.join(output_dir, 'hires_diagnostics')}"
                )
            except Exception as e:
                print(f"[-] Warning: Failed to save hires diagnostics in '{output_dir}': {e}")
        if cfg.save_plots:
            self._save_steps_vs_mu(vae_state, policy_state, cfg)

        # Stitch per-iteration latent scatter snapshots into a GIF
        if cfg.latent_scatter.enabled:
            try:
                from steppo.utils.plotting import assemble_latent_scatter_gif

                assemble_latent_scatter_gif(
                    os.path.join(output_dir, "latent_scatter"),
                    os.path.join(output_dir, "latent_scatter.gif"),
                    fps=cfg.latent_scatter.gif_fps,
                    csv_path=os.path.join(output_dir, "latent_scatter_history.csv"),
                    dims=cfg.latent_scatter.dims,
                )
            except Exception as e:
                print(f"[-] Warning: Failed to assemble latent scatter GIF: {e}")
