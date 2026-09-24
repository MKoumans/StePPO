"""Generate training trajectories shared by the unified DeepONet trainer."""

import numpy as np
from scipy.integrate import solve_ivp


def sample_binned_values(
    rng: np.random.Generator,
    bins,
    n: int,
    scheme: str = "log-binned",
) -> np.ndarray:
    """Sample scalar parameters from a union of intervals."""
    bins = np.asarray(bins, dtype=np.float64)
    bin_idx = rng.integers(0, len(bins), size=n)
    lo, hi = bins[bin_idx, 0], bins[bin_idx, 1]
    if scheme == "log-binned":
        return np.exp(rng.uniform(np.log(lo), np.log(hi)))
    return rng.uniform(lo, hi)


def sample_tasks(system: str, rng: np.random.Generator, cfg: dict, bins, n: int) -> np.ndarray:
    """Sample each task parameter with the fixed initial state used by retraining."""
    parameter = sample_binned_values(
        rng, bins, n, scheme=cfg.get("task_sample_scheme", "log-binned")
    )
    if system == "van_der_pol":
        initial_state = (0.0, -2.0)
    elif system == "brusselator":
        initial_state = (float(cfg["y0_x"]), float(cfg["y0_y"]))
    else:
        raise ValueError(f"Task sampling is not defined for {system!r}")
    states = np.broadcast_to(initial_state, (n, len(initial_state)))
    return np.column_stack((parameter, states))


def scalar_decay_trajectories(parameters: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Evaluate the scalar-decay solution for each parameter on a shared grid."""
    return np.exp(-np.outer(parameters, t_grid))


def scalar_decay_trajectories_on_device(
    parameters: np.ndarray,
    t_grid: np.ndarray,
    device: str,
) -> np.ndarray:
    if device == "cuda":
        from deeponet_gpu_data import analytic_scalar_decay_gpu

        return analytic_scalar_decay_gpu(parameters, t_grid)
    return scalar_decay_trajectories(parameters, t_grid)


def _van_der_pol_rhs(_t, y, mu):
    y1, y2 = y
    return [y2, mu * (1.0 - y1**2) * y2 - y1]


def _brusselator_rhs(_t, y, B):
    y1, y2 = y
    nonlinear = y1**2 * y2
    return [1.0 + nonlinear - (B + 1.0) * y1, B * y1 - nonlinear]


def generate_trajectories(
    system: str,
    tasks: np.ndarray,
    t_grid: np.ndarray,
    t_end: float,
    device: str = "cpu",
) -> np.ndarray:
    """Integrate Van der Pol or Brusselator tasks on a shared time grid."""
    if device == "cuda":
        from deeponet_gpu_data import brusselator_gpu, van_der_pol_gpu

        generator = {"van_der_pol": van_der_pol_gpu, "brusselator": brusselator_gpu}.get(system)
        if generator is None:
            raise ValueError(f"GPU trajectory generation is not defined for {system!r}")
        return generator(tasks, t_grid)

    rhs = {"van_der_pol": _van_der_pol_rhs, "brusselator": _brusselator_rhs}.get(system)
    if rhs is None:
        raise ValueError(f"Trajectory generation is not defined for {system!r}")

    trajectories = np.empty((len(tasks), len(t_grid), 2))
    for i, (parameter, *initial_state) in enumerate(tasks):
        solution = solve_ivp(
            rhs,
            (0.0, t_end),
            initial_state,
            args=(parameter,),
            method="Radau",
            rtol=1e-3,
            atol=1e-6,
            max_step=5.0,
            t_eval=t_grid,
        )
        trajectories[i] = solution.y.T
    return trajectories
