"""Fig. 5, DeepONet part: wall-clock time per prediction vs μ in PyTorch (DeepXDE), one sample per call.

Run in the DeepONet container before paper/fig5_wallclock.py; writes
outputs/paper/fig5_deeponet_wallclock.csv. Uses the same μ grid and
repeats as the PID/StePPO timings, predicting on the 200-point training grid.
Also checks that the NumPy DeepONet used by the other figure scripts matches
PyTorch.

    docker compose -f docker/docker-compose.deeponet.yml exec deeponet \\
        env PYTHONPATH=. python paper/fig5_deeponet_timing.py [--mode import|execute]
"""

import argparse
import os
import sys

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np
import yaml

from paper.utils import (
    REPO_ROOT,
    SYSTEMS,
    WALLCLOCK_REPEATS,
    WALLCLOCK_XI,
    DeepONet,
    add_mode_argument,
    deeponet_checkpoint,
    deeponet_grid,
    out_path,
    read_torch_state_dict,
    save_csv,
)

sys.path.insert(0, os.path.join(REPO_ROOT, "research", "baselines", "deeponet"))
from wallclock_deeponet_vdp import time_deeponet  # noqa: E402

SYSTEM = "van_der_pol"


def load_torch_model(checkpoint: str, t_grid: np.ndarray):
    """DeepXDE model restored from `checkpoint`, with the architecture read from its weights."""
    import deepxde as dde
    import torch
    from train_deeponet_unified import build_deeponet

    shape = DeepONet(read_torch_state_dict(checkpoint))
    n_out = shape.num_outputs
    t_col = t_grid[:, None].astype(np.float32)
    dummy_y = np.zeros((1, len(t_grid), n_out) if n_out > 1 else (1, len(t_grid)), dtype=np.float32)
    data = dde.data.TripleCartesianProd(
        X_train=(np.zeros((1, 1), np.float32), t_col),
        y_train=dummy_y,
        X_test=(np.zeros((1, 1), np.float32), t_col),
        y_test=dummy_y,
    )
    branch_layers = shape.layer_sizes("branch")
    if shape.is_msffn_trunk():
        n_sigmas, trunk_layers = shape.msffn_spec()
        net, _ = build_deeponet(
            branch_layers, trunk_layers, n_out, msffn_sigmas=[1.0] * n_sigmas, verbose=False
        )
    else:
        kwargs = dict(num_outputs=n_out, multi_output_strategy="independent") if n_out > 1 else {}
        net = dde.nn.DeepONetCartesianProd(
            branch_layers, shape.layer_sizes("trunk"), "relu", "Glorot normal", **kwargs
        )
    net.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    return model


def main():
    import torch

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_mode_argument(parser)
    args = parser.parse_args()

    checkpoint = deeponet_checkpoint(SYSTEM, "complete", args.mode)
    with open(os.path.join(REPO_ROOT, SYSTEMS[SYSTEM]["config"])) as f:
        t_grid = deeponet_grid(float(yaml.safe_load(f)["env"]["t_end"]))

    model = load_torch_model(checkpoint, t_grid)
    xi = WALLCLOCK_XI.astype(np.float32)[:, None]
    diff = np.abs(
        model.predict((xi, t_grid[:, None].astype(np.float32)))
        - DeepONet(read_torch_state_dict(checkpoint))(xi, t_grid)
    ).max()
    print(f"[*] NumPy vs PyTorch DeepONet: max |difference| = {diff:.2e}")

    print(f"[*] Timing DeepONet on {'cuda' if torch.cuda.is_available() else 'cpu'} ...")
    ms = time_deeponet(model, WALLCLOCK_XI, t_grid, mode="single", repeats=WALLCLOCK_REPEATS)
    save_csv(out_path("fig5_deeponet_wallclock.csv"), {"DeepONet": {"xi": WALLCLOCK_XI, "ms": ms}})


if __name__ == "__main__":
    main()
