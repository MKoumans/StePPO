"""Sweep PID/RL batched wallclock over several batch sizes and integrate the AUC.

Each parameter value is timed as a complete batch (wallclock_vs_mu.py's
report_batch_time mode). AUC integrates ms over the parameter axis, so
AUC_PID - AUC_RL is the total saved wallclock area across the sweep.

Usage: --config points to a WallclockBatchSizeSweepConfig YAML
(configs/diagnostics/wallclock_batch_size_sweep/*.yaml); any field can be
overridden ad hoc, e.g. --reuse true.
    uv run python research/ode/diagnostics/wallclock_batch_size_sweep.py \\
        --config configs/diagnostics/wallclock_batch_size_sweep/van_der_pol.yaml
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from utils import plot_batch_size_auc

from configs import WallclockBatchSizeSweepConfig

try:
    from ..cli_config import parse_config_args
    from ..output_dirs import resolve_output_dir
except ImportError:
    from research.ode.cli_config import parse_config_args
    from research.ode.output_dirs import resolve_output_dir

SIZES = (1, 2, 4, 8, 16, 32, 64)
# Get repo root by finding .git or pyproject.toml
_HERE = Path(__file__).resolve()
ROOT = _HERE.parent.parent.parent  # research/ode/diagnostics -> workspace root
BENCH = _HERE.with_name("wallclock_vs_mu.py")  # sibling script


def _paths(bench_dir: Path, system: str, batch_size: int, device: str = "gpu"):
    prefix = "vdp" if system == "van_der_pol" else system
    return (
        bench_dir / f"{prefix}_complete_empty_batch{batch_size}_{device}_sweep.npz",
        bench_dir / f"{prefix}_pid_batch{batch_size}_{device}_sweep.npz",
    )


def run_one(
    bench_dir: Path,
    system: str,
    checkpoint: str,
    batch_size: int,
    repeats: int,
    n_sweep: int,
    max_steps: int,
    reuse: bool,
    device: str,
) -> None:
    """Run or reuse one batch-size benchmark."""
    rl_path, pid_path = _paths(bench_dir, system, batch_size, device)
    if reuse and rl_path.exists() and pid_path.exists():
        print(f"[*] Reusing batch size {batch_size}: {rl_path}", flush=True)
        return
    rl_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "JAX_PYTHON_CLIENT_MEM_FRACTION": "0.90",
            "JAX_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONPATH": str(ROOT),
        }
    )
    if device == "gpu":
        env["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        env["GPUS"] = "cpu"
    cmd = [
        sys.executable,
        str(BENCH),
        "--checkpoint",
        checkpoint,
        "--mode",
        "batched",
        "--device",
        device,
        "--n_sweep",
        str(n_sweep),
        "--batch_size",
        str(batch_size),
        "--report_batch_time",
        "true",
        "--max_steps",
        str(max_steps),
        "--repeats",
        str(repeats),
        "--output",
        str(rl_path),
        "--pid_output",
        str(pid_path),
    ]
    print(f"[*] Running {system}, batch size {batch_size}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def aggregate(bench_dir: Path, system: str, sizes: tuple[int, ...], device: str):
    """Aggregate timing curves and write AUC summaries for each batch size."""
    records = []
    for batch_size in sizes:
        rl_path, pid_path = _paths(bench_dir, system, batch_size, device)
        pid = np.load(pid_path)
        rl = np.load(rl_path)
        mus = np.asarray(pid["mus"], dtype=float)
        pid_ms = np.asarray(pid["pid_batch_ms"], dtype=float)
        rl_ms = np.asarray(rl["rl_batch_ms"], dtype=float)
        auc_pid = float(np.trapezoid(pid_ms, mus))
        auc_rl = float(np.trapezoid(rl_ms, mus))
        saved = auc_pid - auc_rl
        records.append(
            (
                batch_size,
                auc_pid,
                auc_rl,
                saved,
                saved / auc_pid * 100.0,
                auc_pid / auc_rl,
                saved / (mus[-1] - mus[0]),
            )
        )
    arr = np.asarray(records, dtype=float)
    out = bench_dir / f"{system}_batch_size_auc_{device}.npz"
    np.savez(
        out,
        batch_size=arr[:, 0].astype(int),
        auc_pid_ms_param=arr[:, 1],
        auc_rl_ms_param=arr[:, 2],
        auc_saved_ms_param=arr[:, 3],
        area_reduction_percent=arr[:, 4],
        auc_speedup=arr[:, 5],
        mean_saved_batch_ms=arr[:, 6],
    )
    csv = bench_dir / f"{system}_batch_size_auc_{device}.csv"
    with csv.open("w") as f:
        f.write(
            "batch_size,auc_pid_ms_param,auc_rl_ms_param,auc_saved_ms_param,area_reduction_percent,auc_speedup,mean_saved_batch_ms\n"
        )
        for row in records:
            f.write(",".join(f"{x:.9g}" for x in row) + "\n")
    md = bench_dir / f"{system}_batch_size_auc_{device}.md"
    with md.open("w") as f:
        f.write(f"# {system}: {device} batch-size AUC sweep\n\n")
        f.write(
            "AUC is trapezoidal integration of complete batch time (ms) over the actual parameter axis. "
            "`AUC saved = AUC PID - AUC RL`; positive values mean RL saves time.\n\n"
        )
        f.write(
            "| Batch size | PID AUC (ms·μ) | RL AUC (ms·μ) | Saved AUC (ms·μ) | Area reduction | AUC speedup | Mean saved batch ms |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in records:
            f.write(
                f"| {int(row[0])} | {row[1]:.3f} | {row[2]:.3f} | {row[3]:.3f} | {row[4]:.2f}% | {row[5]:.3f}x | {row[6]:.3f} |\n"
            )
        f.write(
            "\nEach curve uses the same log-spaced parameter sweep and complete batch times; no division by batch size.\n"
        )
    return arr


def main():
    """Run the batch-size sweep and save its summaries and figure."""
    args = parse_config_args(
        description=__doc__,
        config_help="WallclockBatchSizeSweepConfig YAML, e.g. "
        "configs/diagnostics/wallclock_batch_size_sweep/<name>.yaml",
    )

    from steppo.configs.base_config import apply_dotted_overrides, load_config_from_yaml

    cfg = (
        load_config_from_yaml(WallclockBatchSizeSweepConfig, args.config)
        if args.config
        else WallclockBatchSizeSweepConfig()
    )
    cfg = apply_dotted_overrides(cfg, args.overrides)

    if cfg.system not in ("van_der_pol", "brusselator"):
        raise SystemExit(f"system must be 'van_der_pol' or 'brusselator', got {cfg.system!r}")
    if not cfg.checkpoint:
        raise SystemExit("config needs checkpoint")

    bench_dir = Path(
        resolve_output_dir("diagnostics", cfg, name=cfg.name, exclude=("name", "reuse"))
    )

    for size in SIZES:
        run_one(
            bench_dir,
            cfg.system,
            cfg.checkpoint,
            size,
            cfg.repeats,
            cfg.n_sweep,
            cfg.max_steps,
            cfg.reuse,
            cfg.device,
        )
    arr = aggregate(bench_dir, cfg.system, SIZES, cfg.device)
    print(f"[+] Wrote {bench_dir}/{cfg.system}_batch_size_auc_{cfg.device}.csv", flush=True)
    print("batch_size saved_auc area_reduction auc_speedup", flush=True)
    for row in arr:
        print(f"{int(row[0]):>5} {row[3]:>12.3f} {row[4]:>8.2f}% {row[5]:>8.3f}x", flush=True)

    plot_path = plot_batch_size_auc(
        {cfg.system: arr}, bench_dir / "batch_size_auc_pid_vs_rl.png", SIZES
    )
    print(f"[+] Wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
