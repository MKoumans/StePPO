"""Errors of a solved trajectory (PID, learned controller or external prediction)
against the cached reference trajectory.

Trajectories use diffrax's padded layout: `ts` is (..., T) and `ys` is
(..., T, y_dim), both +inf past each episode's last accepted step.
"""

import numpy as np

LOG10_ERR_FLOOR = -16.0
LOG10_ERR_CEIL = 3.0


def relative_l2_error(y: np.ndarray, y_ref: np.ndarray, atol: float) -> np.ndarray:
    """Linear-space relative L2 error ||y - y_ref|| / (||y_ref|| + atol),
    reduced over the last (state) axis. Un-logged so trajectory-level
    aggregates (e.g. time integration) happen in linear space, not log
    space."""
    diff = np.linalg.norm(y - y_ref, axis=-1)
    denom = np.linalg.norm(y_ref, axis=-1) + atol
    return diff / denom


def to_log10_clipped(err: np.ndarray) -> np.ndarray:
    """log10 of a linear-space (non-negative) error, non-finite values mapped
    to the ceiling, clipped to [LOG10_ERR_FLOOR, LOG10_ERR_CEIL] — the shared
    tail end of every log10-space error metric in this module and in
    `error_dist.log10_rel_error`."""
    err = np.where(np.isfinite(err), err, 10.0**LOG10_ERR_CEIL)
    return np.clip(np.log10(err + 10.0**LOG10_ERR_FLOOR), LOG10_ERR_FLOOR, LOG10_ERR_CEIL)


def final_state_from_trajectory(ts: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Extract y at the last finite (accepted) save-point per episode from a
    `(N, T)` / `(N, T, y_dim)` padded trajectory pair."""
    finite = np.isfinite(ts)
    last_idx = np.clip(finite.sum(axis=-1) - 1, 0, ts.shape[-1] - 1)
    return ys[np.arange(ts.shape[0]), last_idx]


def interp_at(ref_ts: np.ndarray, ref_ys: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Interpolate each padded reference episode at `query_ts` (padded queries give junk to discard)."""
    n, y_dim = ref_ys.shape[0], ref_ys.shape[-1]
    out = np.empty(query_ts.shape + (y_dim,), dtype=ref_ys.dtype)
    for i in range(n):
        finite = np.isfinite(ref_ts[i])
        t_valid = ref_ts[i, finite]
        y_valid = ref_ys[i, finite]
        q = np.where(np.isfinite(query_ts[i]), query_ts[i], t_valid[-1])
        for d in range(y_dim):
            out[i, :, d] = np.interp(q, t_valid, y_valid[:, d])
    return out


def last_valid_value(ts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Value at each episode's last finite (accepted) time in a `(N, T)`
    padded per-step scalar array — the scalar-signal counterpart of
    `final_state_from_trajectory` (which does the same thing for `(N, T,
    y_dim)` state arrays)."""
    finite = np.isfinite(ts)
    last_idx = np.clip(finite.sum(axis=-1) - 1, 0, ts.shape[-1] - 1)
    return values[np.arange(ts.shape[0]), last_idx]


def time_average(ts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Trapezoidal time average of a padded (N, T) array over each episode's valid range."""
    n = ts.shape[0]
    result = np.empty(n, dtype=float)
    for i in range(n):
        finite = np.isfinite(ts[i])
        t_valid = ts[i, finite]
        v_valid = values[i, finite]
        duration = t_valid[-1] - t_valid[0]
        result[i] = np.trapezoid(v_valid, t_valid) / duration
    return result


def local_error(
    ref_ts: np.ndarray,
    ref_ys: np.ndarray,
    ts: np.ndarray,
    ys: np.ndarray,
    atol: float,
    kind: str = "linear",
) -> np.ndarray:
    """Relative L2 error at every step against the interpolated reference, shape (N, T).

    `kind="linear"` returns the ratio (padding stays +inf); `kind="log10"`
    applies to_log10_clipped (padding maps to the ceiling).
    """
    n = ts.shape[0]
    out = np.full(ts.shape, np.inf, dtype=float)
    for i in range(n):
        finite = np.isfinite(ts[i])
        t_valid = ts[i, finite]
        y_valid = ys[i, finite]
        y_ref = interp_at(ref_ts[i : i + 1], ref_ys[i : i + 1], t_valid[None, :])[0]
        out[i, finite] = relative_l2_error(y_valid, y_ref, atol)
    if kind == "linear":
        return out
    if kind == "log10":
        return to_log10_clipped(out)
    raise ValueError(f"unknown local_error kind {kind!r}; expected 'linear' or 'log10'")


def log10_time_integrated_error(
    ref_ts: np.ndarray,
    ref_ys: np.ndarray,
    ts: np.ndarray,
    ys: np.ndarray,
    atol: float,
) -> np.ndarray:
    """Time average of the log10 local error.

    Averaging in log space keeps a near-zero reference value from dominating.
    """
    return time_average(ts, local_error(ref_ts, ref_ys, ts, ys, atol, kind="log10"))


def log10_trajectory_error(
    reference: dict, ts: np.ndarray, ys: np.ndarray, atol: float
) -> np.ndarray:
    """log10 local error of one unpadded trajectory (ts, ys) against `reference` {"ts", "ys"}."""
    return local_error(
        reference["ts"][None], reference["ys"][None], ts[None], ys[None], atol, kind="log10"
    )[0]
