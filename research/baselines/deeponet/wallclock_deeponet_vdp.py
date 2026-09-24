"""Wall-clock time per task parameter for the van_der_pol DeepONet, the
counterpart of research/ode/diagnostics/wallclock_vs_mu.py for PID and RL (Fig. 5).

--mode single   one mu per predict call (batch size 1)
--mode batched  all mus in one call; ms per mu = total time / number of mus
--device        cpu or gpu (default gpu)

Run in the DeepONet container:
    python wallclock_deeponet_vdp.py
    python wallclock_deeponet_vdp.py --mode batched
"""

import os
import sys


def _scan_flag(name, default=None):
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return default


_device = _scan_flag("--device", "gpu")
if _device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("DDE_BACKEND", "pytorch")

import argparse
import time

import numpy as np
import yaml
from deeponet_relerr_eval import _SYSTEMS, load_model_for_system

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..", "..")
SYSTEM = "van_der_pol"
N_SWEEP = 30
N_REPEATS = 20
N_T = 200


def time_deeponet(
    model, mus, t_valid, mode="single", repeats=N_REPEATS, verbose=False, progress_label=None
):
    """Warm wall-clock time (ms) per mu, predicting one mu per call ("single") or all at once ("batched")."""
    t_col = t_valid.astype(np.float32)[:, None]

    if mode == "single":

        def solve_one(mu):
            return model.predict((np.array([[mu]], dtype=np.float32), t_col))

        for _ in range(3):
            solve_one(mus[0])

        means = []
        for mu in mus:
            times = []
            for r in range(repeats):
                t0 = time.perf_counter()
                solve_one(mu)
                dt = (time.perf_counter() - t0) * 1000.0
                times.append(dt)
                if verbose:
                    print(f"    mu={mu:8.2f} repeat={r:2d} time={dt:8.2f} ms")
            means.append(np.mean(times))
            if progress_label:
                print(f"    [{progress_label}] mu={mu:8.2f} mean={means[-1]:8.3f} ms", flush=True)
        return np.asarray(means)

    if mode == "batched":
        branch = mus.astype(np.float32)[:, None]

        def solve_all():
            return model.predict((branch, t_col))

        t0 = time.perf_counter()
        for _ in range(3):
            solve_all()
        if progress_label:
            print(
                f"    [{progress_label}] warmup done ({(time.perf_counter() - t0):.1f}s)",
                flush=True,
            )

        times = []
        for r in range(repeats):
            t0 = time.perf_counter()
            solve_all()
            elapsed = time.perf_counter() - t0
            dt = elapsed * 1000.0 / len(mus)
            times.append(dt)
            if verbose:
                print(f"    batch(n={len(mus)}) repeat={r:2d} ms/mu={dt:8.2f}")
            if progress_label:
                print(
                    f"    [{progress_label}] repeat {r + 1}/{repeats} ({elapsed:.1f}s)", flush=True
                )
        return np.full(len(mus), np.mean(times))

    raise ValueError(f"unknown mode {mode!r}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", choices=["retrain", "legacy"], default="retrain")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="optional explicit checkpoint path relative to the repository root",
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--mode", choices=["single", "batched"], default="single")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument(
        "--repeats",
        type=int,
        default=N_REPEATS,
        help=f"Timing repeats per mu (single) or per batch (batched). Default {N_REPEATS}.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.device = _device  # the device actually used (it may be forced by the environment)

    spec = _SYSTEMS[SYSTEM]
    with open(os.path.join(REPO_ROOT, spec["config"])) as f:
        cfg = yaml.safe_load(f)["env"]
    test_bins = cfg["test_bins"]
    lo = min(b[0] for b in test_bins)
    hi = max(b[1] for b in test_bins)
    mus = np.logspace(np.log10(lo), np.log10(hi), N_SWEEP)
    t_end = float(cfg["t_end"])
    t_valid = np.linspace(0.0, t_end, N_T)

    print(
        f"[*] System={SYSTEM}  param range=[{lo:.3g}, {hi:.3g}]  n_sweep={N_SWEEP}  "
        f"mode={args.mode}  device={args.device}  repeats={args.repeats}"
    )

    model, _, _ = load_model_for_system(SYSTEM, profile=args.profile, checkpoint=args.checkpoint)

    print("[*] DeepONet timen ...")
    deeponet_ms = time_deeponet(
        model,
        mus,
        t_valid,
        mode=args.mode,
        repeats=args.repeats,
        verbose=args.verbose,
        progress_label="DeepONet",
    )
    print(f"  DeepONet mean over sweep: {deeponet_ms.mean():.3f} ms")

    out_dir = os.path.join(HERE, "output-deeponet", SYSTEM, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, f"wallclock_deeponet_vdp_{args.mode}.npz")
    np.savez(
        out_path,
        mus=mus,
        mean_times_ms=deeponet_ms,
        param_label=spec["param_label"],
        mode=args.mode,
        device=args.device,
        profile=args.profile,
    )
    print(f"[+] Saved: {out_path}")


if __name__ == "__main__":
    main()
