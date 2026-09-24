"""GPU-batched trajectory generation (torchdiffeq DOPRI5) for DeepONet training data.

The CPU path is the SciPy Radau code in the system-specific modules.
"""

from __future__ import annotations

import numpy as np


def _cuda_runtime():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU data generation was requested, but CUDA is unavailable. "
            "Check that the container was started with --gpus all."
        )
    try:
        from torchdiffeq import odeint
    except ImportError as exc:
        raise RuntimeError(
            "GPU data generation requires torchdiffeq; rebuild the DeepONet image."
        ) from exc
    return torch, odeint


def analytic_scalar_decay_gpu(lams: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Evaluate scalar decay analytically on CUDA, then return NumPy arrays."""
    torch, _ = _cuda_runtime()
    with torch.inference_mode():
        lam = torch.as_tensor(lams, dtype=torch.float64, device="cuda")
        t = torch.as_tensor(t_grid, dtype=torch.float64, device="cuda")
        y = torch.exp(-lam[:, None] * t[None, :])
        return y.cpu().numpy()


def van_der_pol_gpu(
    tasks: np.ndarray,
    t_grid: np.ndarray,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> np.ndarray:
    """Integrate a batch of Van der Pol tasks on CUDA with DOPRI5."""
    torch, odeint = _cuda_runtime()
    with torch.inference_mode():
        task = torch.as_tensor(tasks, dtype=torch.float64, device="cuda")
        t = torch.as_tensor(t_grid, dtype=torch.float64, device="cuda")
        mu = task[:, 0]
        y0 = task[:, 1:3]

        def rhs(_t, y):
            y1, y2 = y[:, 0], y[:, 1]
            return torch.stack((y2, mu * (1.0 - y1.square()) * y2 - y1), dim=1)

        # odeint returns (N_t, N_tasks, state_dim); transpose to (N_tasks, N_t, state_dim).
        y = odeint(rhs, y0, t, method="dopri5", rtol=rtol, atol=atol)
        return y.permute(1, 0, 2).cpu().numpy()


def brusselator_gpu(
    tasks: np.ndarray,
    t_grid: np.ndarray,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> np.ndarray:
    """Integrate a batch of Brusselator tasks on CUDA with DOPRI5."""
    torch, odeint = _cuda_runtime()
    with torch.inference_mode():
        task = torch.as_tensor(tasks, dtype=torch.float64, device="cuda")
        t = torch.as_tensor(t_grid, dtype=torch.float64, device="cuda")
        B = task[:, 0]
        y0 = task[:, 1:3]

        def rhs(_t, y):
            y1, y2 = y[:, 0], y[:, 1]
            nonlinear = y1.square() * y2
            return torch.stack(
                (1.0 + nonlinear - (B + 1.0) * y1, B * y1 - nonlinear),
                dim=1,
            )

        y = odeint(rhs, y0, t, method="dopri5", rtol=rtol, atol=atol)
        return y.permute(1, 0, 2).cpu().numpy()
