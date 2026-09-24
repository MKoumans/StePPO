"""Error metrics copied from src/steppo/training/trajectory_compare.py.

The DeepONet scripts run without JAX, and importing steppo imports JAX, so the
definitions are duplicated here. Keep both in sync.
"""

import numpy as np

LOG10_ERR_FLOOR = -16.0
LOG10_ERR_CEIL = 3.0


def relative_l2_error(y: np.ndarray, y_ref: np.ndarray, atol: float) -> np.ndarray:
    """Relative L2 error ||y - y_ref|| / (||y_ref|| + atol) over the last axis."""
    diff = np.linalg.norm(y - y_ref, axis=-1)
    denom = np.linalg.norm(y_ref, axis=-1) + atol
    return diff / denom


def to_log10_clipped(err: np.ndarray) -> np.ndarray:
    """log10 of a non-negative error, clipped to [LOG10_ERR_FLOOR, LOG10_ERR_CEIL]."""
    err = np.where(np.isfinite(err), err, 10.0**LOG10_ERR_CEIL)
    return np.clip(np.log10(err + 10.0**LOG10_ERR_FLOOR), LOG10_ERR_FLOOR, LOG10_ERR_CEIL)


def log10_rel_error(y: np.ndarray, y_ref: np.ndarray, atol: float) -> np.ndarray:
    """log10 relative L2 error of final states."""
    return to_log10_clipped(relative_l2_error(y, y_ref, atol))


def pid_scaled_error(y: np.ndarray, y_ref: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    """PID-style scaled error ||y - y_ref|| / (atol + rtol * max|y_ref|) at each time.

    Uses the maximum over state dimensions at each time, not diffrax's per-dimension
    max(|y0|, |y1|) of a single step.
    """
    diff = np.linalg.norm(y - y_ref, axis=-1)
    scale = atol + rtol * np.max(np.abs(y_ref), axis=-1)
    return diff / scale


def time_integrated_log10_rel_error(
    t_valid: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, atol: float
) -> float:
    """Trapezoidal time average of the log10 relative error over `t_valid` (inputs already aligned)."""
    pointwise_rel = relative_l2_error(y_pred, y_true, atol)
    pointwise_log10 = to_log10_clipped(pointwise_rel)
    duration = t_valid[-1] - t_valid[0]
    if duration <= 0:
        return float(pointwise_log10[-1])
    return float(np.trapezoid(pointwise_log10, t_valid) / duration)
