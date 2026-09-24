"""Shared helpers for the ODE post-run-analysis scripts.

Kept separate (rather than defined in analyse_rollout.py) so other analysis
scripts — e.g. vae_noise_injection.py — can import them without creating a
circular import back into analyse_rollout.py, which itself imports from
vae_noise_injection.py to expose the noise-injection panel.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from steppo.utils.checkpoint import load_checkpoint
from steppo.utils.task_params import task_label

_BLUE = "#2176AE"
_ORANGE = "#F4873C"
_GREEN = "#3BAA6E"
_ALPHA = 0.25  # shading alpha


def get_task_param(env_params_batch, system_name: str):
    """Return (param_array, label) for the primary stiffness parameter of the system.

    All ODE systems store their hidden parameter in ODEParams.lam.
    """
    return np.array(env_params_batch.lam).ravel(), task_label(system_name)


def try_load_checkpoint(
    vae, policy, checkpoint_dir: str, backbone: str = "varibad", algo: str = "ppo"
):
    """Restore model parameters when possible, falling back to random weights."""
    try:
        return load_checkpoint(vae, policy, checkpoint_dir, backbone=backbone, algo=algo)
    except Exception as exc:
        print(f"[!] Checkpoint load failed ({exc}); using random weights.")
        return vae, policy


def freeze_inactive(active, new, old):
    """Keep `old` where `active` is False, pinning a scanned trajectory's
    values once its episode is done instead of letting them drift further."""
    mask = active.reshape((-1,) + (1,) * (new.ndim - 1))
    return jnp.where(mask, new, old)


def _savefig(fig, path: str):
    """Save and close one analysis figure."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def col_edges(dt_grid: np.ndarray) -> np.ndarray:
    """Per-row column edges for pcolormesh from per-row candidate dt centers
    (dt_grid is (T, G), monotone along axis=1). Returns (T, G+1), extrapolating
    the first/last half-cell.
    """
    T, G = dt_grid.shape
    mids = 0.5 * (dt_grid[:, :-1] + dt_grid[:, 1:])  # (T, G-1)
    first = dt_grid[:, :1] - (mids[:, :1] - dt_grid[:, :1])
    last = dt_grid[:, -1:] + (dt_grid[:, -1:] - mids[:, -1:])
    return np.concatenate([first, mids, last], axis=1)  # (T, G+1)
