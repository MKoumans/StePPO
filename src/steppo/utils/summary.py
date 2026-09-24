"""Performance-summary calculations for completed training runs."""

from steppo.configs.base_config import TrainConfig


def get_metrics(cfg: TrainConfig, rollout_times, vae_times, ppo_times, iter_times, devices) -> dict:
    """Aggregate timing samples and throughput into a serializable dictionary."""
    total_time = sum(rollout_times) + sum(vae_times) + sum(ppo_times)
    total_steps = cfg.total_iters * cfg.num_envs * cfg.rollout_steps
    avg_rollout = sum(rollout_times) / len(rollout_times) if rollout_times else 0

    # VAE Compile vs Steady State
    vae_compile = vae_times[0] if vae_times else 0
    vae_steady = (
        sum(vae_times[1:]) / len(vae_times[1:])
        if len(vae_times) > 1
        else (vae_times[0] if vae_times else 0)
    )

    # PPO Compile vs Steady State
    ppo_compile = ppo_times[0] if ppo_times else 0
    ppo_steady = (
        sum(ppo_times[1:]) / len(ppo_times[1:])
        if len(ppo_times) > 1
        else (ppo_times[0] if ppo_times else 0)
    )

    avg_iter = sum(iter_times) / len(iter_times) if iter_times else 0
    steps_per_sec = total_steps / total_time if total_time > 0 else 0

    device_names = [f"{d.device_kind}:{d.id}" for d in devices]

    metrics = {
        "total_time": total_time,
        "total_steps": total_steps,
        "steps_per_sec": steps_per_sec,
        "avg_rollout": avg_rollout,
        "vae_compile": vae_compile,
        "vae_steady": vae_steady,
        "ppo_compile": ppo_compile,
        "ppo_steady": ppo_steady,
        "avg_iter": avg_iter,
        "device_names": device_names,
    }

    return metrics


def get_summary_str(metrics: dict, cfg: TrainConfig) -> str:
    """Format and print the human-readable performance summary."""
    summary = []
    summary.append("=" * 60)
    summary.append("            VariBAD JAX Performance Summary             ")
    summary.append("=" * 60)
    summary.append(f" Experiment Name:     {cfg.exp_name}")
    summary.append(f" Total Iterations:    {cfg.total_iters}")
    summary.append(f" Vectorised Envs:     {cfg.num_envs} (Steps per rollout: {cfg.rollout_steps})")
    summary.append(f" Total Environment Steps: {metrics['total_steps']:,}")
    summary.append("-" * 60)
    summary.append(f" Active GPU Devices:  {len(metrics['device_names'])}")
    summary.append(f" Device Names:        {', '.join(metrics['device_names'])}")
    summary.append("-" * 60)
    summary.append(f" Total Elapsed Time:  {metrics['total_time']:.2f} seconds")
    summary.append(f" Overall Throughput:  {metrics['steps_per_sec']:.2f} env steps/sec")
    summary.append("-" * 60)
    summary.append(" Component Runtimes (Mean / Compile):")
    summary.append(f"  - Rollout Collection:  {metrics['avg_rollout'] * 1000:.2f} ms")
    summary.append(
        f"  - VAE Update:          {metrics['vae_steady'] * 1000:.2f} ms  (First/Compile: {metrics['vae_compile'] * 1000:.2f} ms)"
    )
    summary.append(
        f"  - PPO Update:          {metrics['ppo_steady'] * 1000:.2f} ms  (First/Compile: {metrics['ppo_compile'] * 1000:.2f} ms)"
    )
    summary.append(f"  - Steady State Iter:   {metrics['avg_iter'] * 1000:.2f} ms")
    summary.append("=" * 60)

    summary_str = "\n".join(summary)
    print("\n" + summary_str + "\n")

    return summary_str
