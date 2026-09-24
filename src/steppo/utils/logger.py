"""Console, W&B, and timing loggers used by the training loop."""

import collections
import time
from dataclasses import dataclass, field
from typing import Optional


def fmt_num(x: float) -> str:
    """Format a number with at most 3 digits before and 2 after the decimal point.

    Falls back to scientific notation (e.g. 1.23e+16) for very large/small values.
    """
    ax = abs(x)
    if ax != 0 and (ax >= 1000 or ax < 0.01):
        return f"{x:.2e}"
    return f"{x:.2f}"


@dataclass
class MetricsLogger:
    """Track running metrics and emit training and evaluation summaries."""

    use_wandb: bool = False
    _wandb_run: Optional[object] = field(default=None, init=False, repr=False)
    _window_size: int = 50
    _history: dict = field(
        default_factory=lambda: collections.defaultdict(lambda: collections.deque(maxlen=50)),
        init=False,
    )
    _start_time: float = field(default_factory=time.time, init=False)

    def __post_init__(self):
        if self.use_wandb:
            try:
                import wandb

                self._wandb_run = wandb.run
            except ImportError:
                self.use_wandb = False
        self._start_time = time.time()

    def update_history(self, metrics: dict):
        """Append non-null metric values to their bounded history windows."""
        for k, v in metrics.items():
            if v is not None:
                self._history[k].append(v)

    def get_running_avg(self, key: str, default: float = 0.0) -> float:
        """Return a metric's running mean or ``default`` when it is unseen."""
        history = self._history[key]
        if not history:
            return default
        return sum(history) / len(history)

    def log_training(self, step: int, total_steps: int, metrics: dict, step_time: float):
        """Logs training losses at a short interval (e.g. 10 steps)."""
        # Update running history
        self.update_history(metrics)

        # --- Weighted (what the optimizer sees) ---
        w_tot = self.get_running_avg("vae/total_loss")
        w_kl = self.get_running_avg("vae/w_kl_loss")
        w_rew = self.get_running_avg("vae/w_rew_loss")
        w_st = self.get_running_avg("vae/w_state_loss", None)
        w_task = self.get_running_avg("vae/w_task_loss", None)
        w_accept = self.get_running_avg("vae/w_accept_loss", None)
        w_endrew = self.get_running_avg("vae/w_end_reward_loss", None)

        # --- Raw (unweighted, pre-EMA-norm) ---
        r_kl = self.get_running_avg("vae/kl_loss")
        r_rew = self.get_running_avg("vae/rew_loss")
        r_st = self.get_running_avg("vae/state_loss", None)
        r_task = self.get_running_avg("vae/task_loss", None)
        r_accept = self.get_running_avg("vae/accept_loss", None)
        r_endrew = self.get_running_avg("vae/end_reward_loss", None)

        ppo_tot = self.get_running_avg("ppo/total_loss")
        ppo_act = self.get_running_avg("ppo/actor_loss")
        ppo_val = self.get_running_avg("ppo/value_loss")
        ppo_ent = self.get_running_avg("ppo/entropy")
        ppo_kl = self.get_running_avg("ppo/approx_kl", None)

        # ETA calculation
        elapsed = time.time() - self._start_time
        if step > 0:
            avg_time = elapsed / step
            eta_sec = avg_time * (total_steps - step)
            if eta_sec > 3600:
                eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
            elif eta_sec > 60:
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
            else:
                eta_str = f"{int(eta_sec)}s"
        else:
            eta_str = "Calculating..."

        # Line 1: weighted losses (optimizer view)
        parts = [
            f"Iter {step}/{total_steps}",
            f"VAE: {fmt_num(w_tot)} (kl:{fmt_num(w_kl)}, rew:{fmt_num(w_rew)}"
            + (f", st:{fmt_num(w_st)}" if w_st is not None else "")
            + (f", task:{fmt_num(w_task)}" if w_task is not None else "")
            + (f", acc:{fmt_num(w_accept)}" if w_accept is not None else "")
            + (f", endrew:{fmt_num(w_endrew)}" if w_endrew is not None else "")
            + ")",
        ]

        if "ppo/total_loss" in metrics:
            parts.append(
                f"PPO: {fmt_num(ppo_tot)} (act:{fmt_num(ppo_act)}, val:{fmt_num(ppo_val)}, ent:{fmt_num(ppo_ent)}"
                + (f", kl:{fmt_num(ppo_kl)}" if ppo_kl is not None else "")
                + ")"
            )
        else:
            parts.append("PPO: Pre-training VAE")

        train_ret = metrics.get("train/mean_return")
        if train_ret is not None:
            train_min = metrics.get("train/min_return", train_ret)
            train_max = metrics.get("train/max_return", train_ret)
            parts.append(f"R: {fmt_num(train_ret)} [{fmt_num(train_min)}, {fmt_num(train_max)}]")

        parts.append(f"Step: {step_time:.2f}s | ETA: {eta_str}")
        print(" | ".join(parts))

        # Line 2: raw losses (pre-normalisation)
        raw_parts = [
            f"  raw VAE: kl:{fmt_num(r_kl)}, rew:{fmt_num(r_rew)}"
            + (f", st:{fmt_num(r_st)}" if r_st is not None else "")
            + (f", task:{fmt_num(r_task)}" if r_task is not None else "")
            + (f", acc:{fmt_num(r_accept)}" if r_accept is not None else "")
            + (f", endrew:{fmt_num(r_endrew)}" if r_endrew is not None else ""),
        ]
        print("".join(raw_parts))

        # Log raw metrics (not running averages) to wandb if enabled
        if self.use_wandb and self._wandb_run is not None:
            import wandb

            wandb.log({"step": step, **metrics})

    def log_evaluation(self, step: int, metrics: dict):
        """Logs evaluation results in a distinct, emphasized block."""
        mean_ret = metrics.get("eval/mean_return", 0.0)
        std_ret = metrics.get("eval/std_return", None)
        ep_len = metrics.get("eval/mean_length", None)
        std_len = metrics.get("eval/std_length", None)
        sigma_ratio = metrics.get("eval/sigma_reduction_ratio", None)
        std_sigma = metrics.get("eval/std_sigma_reduction_ratio", None)
        task_mae = metrics.get("eval/task_embedding_mae", None)
        std_mae = metrics.get("eval/std_task_embedding_mae", None)

        print("\n" + "=" * 65)
        t_per_ep = metrics.get("eval/t_per_episode", None)
        success_per_ep = metrics.get("eval/success_per_episode", None)

        ret_str = fmt_num(mean_ret)
        if std_ret is not None:
            ret_str += f" ± {fmt_num(std_ret)}"
        eval_parts = [f"EVALUATION AT ITERATION {step}", f"Return: {ret_str}"]
        if success_per_ep is not None:
            success_strs = [f"{s:.0f}" for s in success_per_ep]
            while len(success_strs) > 1 and success_strs[-1] == "0":
                success_strs.pop()
            eval_parts.append(f"Success: {'-'.join(success_strs)}")
        if t_per_ep is not None:
            eval_parts.append(f"t: {t_per_ep}")
        if ep_len is not None:
            len_str = f"{ep_len:.0f}"
            if std_len is not None:
                len_str += f" ± {std_len:.0f}"
            eval_parts.append(f"Steps: {len_str}")
        print(" | ".join(eval_parts))

        belief_parts = []
        if sigma_ratio is not None:
            sr_str = f"{sigma_ratio:.3f}x"
            if std_sigma is not None:
                sr_str += f" ± {std_sigma:.3f}"
            belief_parts.append(f"sigma reduction: {sr_str}")
        if task_mae is not None:
            mae_str = fmt_num(task_mae)
            if std_mae is not None:
                mae_str += f" ± {fmt_num(std_mae)}"
            belief_parts.append(f"task embedding MAE: {mae_str}")
        if belief_parts:
            print("  Belief: " + " | ".join(belief_parts))

        mean_acc = metrics.get("eval/mean_accepted", None)
        mean_rej = metrics.get("eval/mean_rejected", None)
        if mean_acc is not None:
            print(
                f"  Solver: accepted: {mean_acc:.1f} | rejected: {mean_rej:.1f} | rate: {mean_acc / max(mean_acc + mean_rej, 1):.1%}"
            )
        print("=" * 65 + "\n")

        # Also log eval metrics to wandb if enabled
        if self.use_wandb and self._wandb_run is not None:
            import wandb

            wandb.log({"step": step, **metrics})


@dataclass
class Timings:
    """Collected wall-clock samples for major training components."""

    rollout_times: list = field(default_factory=list)
    vae_times: list = field(default_factory=list)
    ppo_times: list = field(default_factory=list)
    iter_times: list = field(default_factory=list)

    def update(self, rollout_time=None, vae_time=None, ppo_time=None, iter_time=None):
        """Append any supplied component timings to their respective lists."""
        if rollout_time is not None:
            self.rollout_times.append(rollout_time)
        if vae_time is not None:
            self.vae_times.append(vae_time)
        if ppo_time is not None:
            self.ppo_times.append(ppo_time)
        if iter_time is not None:
            self.iter_times.append(iter_time)


class TimingProfiler:
    """Fine-grained wall-clock profiler activated by extensive_timing_logging.

    Each ``section`` context-manager times a block. JAX async dispatch means
    wall-clock != compute unless you synchronise, so use ``block_and_record``
    after JAX calls for true GPU time.
    """

    def __init__(self, enabled: bool = False):
        """Create a profiler, optionally recording section timings."""
        self.enabled = enabled
        self._records: dict = collections.defaultdict(list)
        self._iter_start: float = 0.0

    class _Section:
        __slots__ = ("profiler", "name", "t0")

        def __init__(self, profiler, name):
            self.profiler = profiler
            self.name = name
            self.t0 = 0.0

        def __enter__(self):
            if self.profiler.enabled:
                self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            if self.profiler.enabled:
                self.profiler._records[self.name].append(time.perf_counter() - self.t0)

    def section(self, name: str):
        """Return a context manager that records a named code section."""
        return self._Section(self, name)

    def block_and_record(self, name: str, *jax_outputs):
        """Synchronize JAX outputs and record the resulting wall-clock time."""
        if not self.enabled:
            return
        import jax

        t0 = time.perf_counter()
        for out in jax_outputs:
            jax.block_until_ready(out)
        self._records[name + ".block_until_ready"].append(time.perf_counter() - t0)

    def start_iteration(self):
        """Start timing the current training iteration."""
        self._iter_start = time.perf_counter()

    def end_iteration(self):
        """Record the elapsed time since ``start_iteration``."""
        if self.enabled:
            self._records["iteration.total"].append(time.perf_counter() - self._iter_start)

    def report(self, last_n: int = 0) -> str:
        """Return a formatted timing report, optionally limited to recent samples."""
        if not self._records:
            return ""

        lines = [
            "",
            "=" * 78,
            "  TIMING PROFILE  (wall-clock, synchronised per section)",
            "=" * 78,
            f"  {'Section':<42s} {'Mean':>8s} {'Std':>8s} "
            f"{'Min':>8s} {'Max':>8s} {'N':>5s} {'Pct':>6s}",
            "-" * 78,
        ]

        iter_vals = self._records.get("iteration.total", [1.0])
        iter_slice = iter_vals[-last_n:] if last_n else iter_vals
        mean_iter = sum(iter_slice) / max(len(iter_slice), 1)

        sorted_sections = sorted(
            self._records.items(),
            key=lambda kv: -(sum(kv[1][-last_n:]) / max(len(kv[1][-last_n:]), 1)),
        )

        def _ms(v):
            return f"{v * 1000:.1f}ms" if v < 1.0 else f"{v:.2f}s"

        for name, all_vals in sorted_sections:
            vals = all_vals[-last_n:] if last_n else all_vals
            n = len(vals)
            if n == 0:
                continue
            mean = sum(vals) / n
            mn, mx = min(vals), max(vals)
            std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
            pct = mean / mean_iter * 100 if mean_iter > 0 else 0.0
            lines.append(
                f"  {name:<42s} {_ms(mean):>8s} {_ms(std):>8s} "
                f"{_ms(mn):>8s} {_ms(mx):>8s} {n:>5d} {pct:>5.1f}%"
            )

        lines.append("=" * 78)
        return "\n".join(lines)

    def clear(self):
        """Discard all recorded timing samples."""
        self._records.clear()
