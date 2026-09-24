"""Fig. 5: wall-clock time per solve vs μ on Van der Pol, one sample per solve on a GPU.

Times PID and StePPO with a step budget of WALLCLOCK_MAX_STEPS (large enough
for PID to reach t_end), as the mean of WALLCLOCK_REPEATS warm runs per μ.
DeepONet was trained and timed in PyTorch (DeepXDE); its timings come from
paper/fig5_deeponet_timing.py, run first in the DeepONet container, and
are added to the plot when outputs/paper/fig5_deeponet_wallclock.csv
exists. Absolute times depend on the hardware.

    PYTHONPATH=. python paper/fig5_wallclock.py [--mode import|execute] [--gpus 0]
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import csv
import os

import matplotlib

import steppo.envs.ode  # noqa: F401 — import order avoids a circular import

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper.utils import (
    COLORS,
    WALLCLOCK_MAX_STEPS,
    WALLCLOCK_REPEATS,
    WALLCLOCK_XI,
    add_mode_argument,
    load_config,
    load_steppo,
    out_path,
    save_csv,
)

SYSTEM = "van_der_pol"
DEEPONET_CSV = "fig5_deeponet_wallclock.csv"


def read_deeponet_timings() -> dict | None:
    """{"xi", "ms"} from fig5_deeponet_timing.py, or None if it has not been run."""
    path = out_path(DEEPONET_CSV)
    if not os.path.isfile(path):
        print(
            f"[!] {path} not found: run paper/fig5_deeponet_timing.py in the DeepONet container "
            "to add DeepONet"
        )
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        "xi": np.array([float(r["xi"]) for r in rows]),
        "ms": np.array([float(r["ms"]) for r in rows]),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_mode_argument(parser)
    args = parser.parse_args()

    import jax

    from research.ode.diagnostics.utils import time_controller
    from steppo.training.pid_solve import env_pid_controller

    print(f"[*] Timing on {jax.devices()[0]}")
    config = load_config(SYSTEM)
    env = config.env
    controllers = {"PID": env_pid_controller(env), "StePPO": load_steppo(SYSTEM, config, args.mode)}
    timings = {}
    for name, controller in controllers.items():
        print(f"[*] Timing {name} ...")
        ms = time_controller(
            env,
            controller,
            WALLCLOCK_XI,
            WALLCLOCK_MAX_STEPS,
            mode="single",
            repeats=WALLCLOCK_REPEATS,
        )
        timings[name] = {"xi": WALLCLOCK_XI, "ms": ms}
    save_csv(out_path("fig5_wallclock.csv"), timings)
    deeponet = read_deeponet_timings()
    if deeponet is not None:
        timings["DeepONet"] = deeponet

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for name, t in timings.items():
        ax.plot(t["xi"], t["ms"], "o-", ms=3, color=COLORS[name], label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("μ")
    ax.set_ylabel("wall-clock per solve (ms)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path("fig5_wallclock.pdf"))
    print(f"[+] {out_path('fig5_wallclock.pdf')}")


if __name__ == "__main__":
    main()
