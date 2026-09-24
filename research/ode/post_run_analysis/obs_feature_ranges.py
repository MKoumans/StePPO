"""Empirical range of every observation feature under PID rollouts (no checkpoint needed).

Per feature, plots the values on a linear axis clipped near the 99.9th
percentile and log10|x| split by sign, overlaid with the feature's transform
from `_obs()`. Writes one PNG and one .xlsx (transform curves) per system and a
text summary of min/max/percentiles.

Usage:
    PYTHONPATH=. python research/ode/post_run_analysis/obs_feature_ranges.py \\
        --systems van_der_pol robertson --num_envs 256 --out outputs/analysis/obs_ranges
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import os
import re

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv
from steppo.training.pid_controller import compute_step_context_offset, pid_action_from_obs

_BLUE = "#2176AE"
_ORANGE = "#F4873C"

ALL_FEATURES = ("state", "direction", "step_context", "solver_trend")

# Default config per system — chosen to match training (t_end, dt0, rtol/atol,
# task range) rather than reinventing plausible-looking values.
SYSTEM_CONFIGS = {
    "scalar_decay": "configs/envs/ode/scalar_decay/scalar_decay_default.yaml",
    "van_der_pol": "configs/envs/ode/van_der_pol/van_der_pol_default.yaml",
    "robertson": "configs/envs/ode/robertson/robertson_default.yaml",
    "brusselator": "configs/envs/ode/brusselator/brusselator_default.yaml",
    "fitzhugh_nagumo": "configs/envs/ode/fitzhugh_nagumo/fitzhugh_nagumo_default.yaml",
    "chemical_cascade": "configs/envs/ode/chemical_cascade/chemical_cascade_default.yaml",
}

# step_context / solver_trend are identical (same code, same normalization
# constants) across every system — see obs_factory.py.
_TAIL_LABELS = [
    "step_context: last_log_error/5",
    "step_context: clip(log_error_delta,-10,10)/5",
    "step_context: last_keep_step",
    "step_context: t/t_end",
    "solver_trend: log(dt/dt0)/10",
    "solver_trend: accept_ema",
    "solver_trend: log_error_ema/5",
    "solver_trend: tanh(reject_streak/4)",
    "solver_trend: budget_exhausted",
]

# (state_labels, direction_labels) per system, in the order each _obs() emits
# them. chemical_cascade's state/direction are 50 structurally-identical
# per-chemical entries, so they're aggregated into one pooled row each.
_HEAD_LABELS = {
    "scalar_decay": (
        ["state: clip(y,-10,10)"],
        ["direction: sign(k1)*log(|k1|+1)"],
    ),
    "van_der_pol": (
        ["state: y0/2", "state: sign(y1)*log(|y1|+1)"],
        ["direction: sign(k1_0)*log(|k1_0|+1)", "direction: sign(k1_1)*log(|k1_1|+1)"],
    ),
    "robertson": (
        ["state: y1", "state: sign(y2)*log(|y2|+1e-10)/10", "state: y3"],
        [f"direction: sign(f{i})*log(|f{i}|+1)" for i in range(3)],
    ),
    "brusselator": (
        ["state: y1/5", "state: sign(y2)*log(|y2|+1)"],
        [f"direction: sign(f{i})*log(|f{i}|+1)" for i in range(2)],
    ),
    "fitzhugh_nagumo": (
        ["state: v/2", "state: w/2"],
        [f"direction: sign(f{i})*log(|f{i}|+1)" for i in range(2)],
    ),
    "chemical_cascade": (
        ["state: clip(y,-5,5)/5 (all 50 layer x chemical dims, pooled)"],
        ["direction: sign(k1)*log(|k1|+1) (all 50 dims, pooled)"],
    ),
}

# label -> name of the raw variable that's f(x)'s argument (the x-column
# header in the exported xlsx). None for the recursive EMAs (no formula).
_TAIL_VAR_NAMES = {
    "step_context: last_log_error/5": "scaled_error",
    "step_context: clip(log_error_delta,-10,10)/5": "log_error_delta",
    "step_context: last_keep_step": "keep_step",
    "step_context: t/t_end": "t",
    "solver_trend: log(dt/dt0)/10": "dt/dt0",
    "solver_trend: accept_ema": None,
    "solver_trend: log_error_ema/5": "scaled_error_ema",
    "solver_trend: tanh(reject_streak/4)": "reject_streak",
    "solver_trend: budget_exhausted": "step_count",
}

_HEAD_VAR_NAMES = {
    "scalar_decay": {
        "state: clip(y,-10,10)": "y",
        "direction: sign(k1)*log(|k1|+1)": "k1",
    },
    "van_der_pol": {
        "state: y0/2": "y0",
        "state: sign(y1)*log(|y1|+1)": "y1",
        "direction: sign(k1_0)*log(|k1_0|+1)": "k1_0",
        "direction: sign(k1_1)*log(|k1_1|+1)": "k1_1",
    },
    "robertson": {
        "state: y1": "y1",
        "state: sign(y2)*log(|y2|+1e-10)/10": "y2",
        "state: y3": "y3",
        **{f"direction: sign(f{i})*log(|f{i}|+1)": f"f{i}" for i in range(3)},
    },
    "brusselator": {
        "state: y1/5": "y1",
        "state: sign(y2)*log(|y2|+1)": "y2",
        **{f"direction: sign(f{i})*log(|f{i}|+1)": f"f{i}" for i in range(2)},
    },
    "fitzhugh_nagumo": {
        "state: v/2": "v",
        "state: w/2": "w",
        **{f"direction: sign(f{i})*log(|f{i}|+1)": f"f{i}" for i in range(2)},
    },
    "chemical_cascade": {
        "state: clip(y,-5,5)/5 (all 50 layer x chemical dims, pooled)": "y",
        "direction: sign(k1)*log(|k1|+1) (all 50 dims, pooled)": "k1",
    },
}


def make_var_names(system: str) -> dict:
    """label -> raw variable name (x-column header). Companion to
    make_formulas/make_inverses, same keys."""
    names = dict(_TAIL_VAR_NAMES)
    names.update(_HEAD_VAR_NAMES[system])
    return names


def _y_column_name(label: str) -> str:
    """The formula expression itself (label with its 'state: '/'direction: '/
    etc. category prefix stripped), used as the y-column header."""
    return label.split(": ", 1)[1] if ": " in label else label


def parse_args():
    """Parse system, rollout, plotting, and export options."""
    p = argparse.ArgumentParser(description="Empirical ODE observation-feature ranges")
    p.add_argument(
        "--systems",
        nargs="*",
        default=list(SYSTEM_CONFIGS),
        choices=list(SYSTEM_CONFIGS),
        help="Systems to analyze",
    )
    p.add_argument("-n", "--num_envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=None, help="Override rollout_steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="outputs/analysis/obs_ranges")
    return p.parse_args()


def collect_pid_obs(env, params_batch, num_envs, T, step_context_offset, rng_key):
    """PID-controller rollout, recording every raw observation and its
    active (not-yet-done) mask. No checkpoint / VAE / policy required."""
    keys_reset = jax.random.split(rng_key, num_envs)
    obs0, env_state0 = jax.vmap(env.reset)(keys_reset, params_batch)
    done0 = jnp.zeros(num_envs, dtype=jnp.bool_)

    def scan_step(carry, key_i):
        obs, env_state, done_so_far = carry
        active = ~done_so_far

        actions = jax.vmap(
            lambda o: pid_action_from_obs(o, step_context_offset, dt_log_gain=env.dt_log_gain)
        )(obs)

        key_envs = jax.random.split(key_i, num_envs)
        next_obs, next_env_state, _rewards, dones, _ = jax.vmap(
            lambda k, s, a, p: env.step(k, s, a, p)
        )(key_envs, env_state, actions, params_batch)

        mask = active.reshape(-1, 1)
        final_obs = jnp.where(mask, next_obs, obs)
        final_env_state = jax.tree.map(
            lambda na, oa: jnp.where(active.reshape((-1,) + (1,) * (na.ndim - 1)), na, oa),
            next_env_state,
            env_state,
        )
        final_dones = jnp.logical_or(done_so_far, dones)

        carry = (final_obs, final_env_state, final_dones)
        ys = {"obs": final_obs, "active": active}
        return carry, ys

    init_carry = (obs0, env_state0, done0)
    keys_scan = jax.random.split(rng_key, T)
    _, traces = jax.lax.scan(scan_step, init_carry, keys_scan)
    return traces  # obs: (T, num_envs, obs_dim), active: (T, num_envs)


def _logsign(x: np.ndarray) -> np.ndarray:
    """sign(x)*log(|x|+1) — the shape used for every `direction` feature and
    several `state` features (van_der_pol/brusselator/fitzhugh_nagumo)."""
    return np.sign(x) * np.log(np.abs(x) + 1.0)


def _safe_log_div(divisor: float, eps: float = 1e-10):
    """log(x+eps)/divisor, defined only for x>0 (NaN elsewhere) — used for the
    step-size-ratio and error-ratio features, which are physically positive."""

    def f(x):
        x = np.asarray(x, dtype=np.float64)
        y = np.full_like(x, np.nan)
        mask = x > 0
        y[mask] = np.log(x[mask] + eps) / divisor
        return y

    return f


def make_formulas(system: str, t_end: float, max_steps: int) -> dict:
    """label -> vectorized f(x), the literal forward formula from each
    system's `_obs()`, keyed to line up with build_rows' labels. None where
    the feature is a recursive EMA and has no closed-form single-input f(x)."""
    formulas = {
        "step_context: last_log_error/5": _safe_log_div(5.0),
        "step_context: clip(log_error_delta,-10,10)/5": lambda x: np.clip(x, -10.0, 10.0) / 5.0,
        "step_context: last_keep_step": lambda x: x,
        "step_context: t/t_end": lambda x: x / t_end,
        "solver_trend: log(dt/dt0)/10": _safe_log_div(10.0),
        "solver_trend: accept_ema": None,  # recursive EMA, no static f(x)
        "solver_trend: log_error_ema/5": _safe_log_div(5.0),
        "solver_trend: tanh(reject_streak/4)": lambda x: np.tanh(x / 4.0),
        "solver_trend: budget_exhausted": lambda x: x / max_steps,
    }
    formulas.update(
        {
            "scalar_decay": {
                "state: clip(y,-10,10)": lambda x: np.clip(x, -10.0, 10.0),
                "direction: sign(k1)*log(|k1|+1)": _logsign,
            },
            "van_der_pol": {
                "state: y0/2": lambda x: x / 2.0,
                "state: sign(y1)*log(|y1|+1)": _logsign,
                "direction: sign(k1_0)*log(|k1_0|+1)": _logsign,
                "direction: sign(k1_1)*log(|k1_1|+1)": _logsign,
            },
            "robertson": {
                "state: y1": lambda x: x,
                "state: sign(y2)*log(|y2|+1e-10)/10": lambda x: (
                    np.sign(x) * np.log(np.abs(x) + 1e-10) / 10.0
                ),
                "state: y3": lambda x: x,
                **{f"direction: sign(f{i})*log(|f{i}|+1)": _logsign for i in range(3)},
            },
            "brusselator": {
                "state: y1/5": lambda x: x / 5.0,
                "state: sign(y2)*log(|y2|+1)": _logsign,
                **{f"direction: sign(f{i})*log(|f{i}|+1)": _logsign for i in range(2)},
            },
            "fitzhugh_nagumo": {
                "state: v/2": lambda x: x / 2.0,
                "state: w/2": lambda x: x / 2.0,
                **{f"direction: sign(f{i})*log(|f{i}|+1)": _logsign for i in range(2)},
            },
            "chemical_cascade": {
                "state: clip(y,-5,5)/5 (all 50 layer x chemical dims, pooled)": lambda x: (
                    np.clip(x, -5.0, 5.0) / 5.0
                ),
                "direction: sign(k1)*log(|k1|+1) (all 50 dims, pooled)": _logsign,
            },
        }[system]
    )
    return formulas


def _logsign_inv(y: np.ndarray) -> np.ndarray:
    """Inverse of sign(x)*log(|x|+1): sign(y)*(exp(|y|)-1)."""
    y = np.asarray(y, dtype=np.float64)
    return np.sign(y) * np.expm1(np.abs(y))


def _safe_log_div_inv(divisor: float, eps: float = 1e-10):
    """Inverse of log(x+eps)/divisor: exp(divisor*y) - eps."""

    def inv(y):
        y = np.asarray(y, dtype=np.float64)
        return np.exp(divisor * y) - eps

    return inv


def make_inverses(system: str, t_end: float, max_steps: int) -> dict:
    """label -> inverse of the corresponding make_formulas() entry. Used to
    recover the true raw quantity (e.g. the physical dt/dt0 ratio, which is
    routinely >>1 even though log(dt/dt0)/10 itself stays near [0,1]) from
    the observed/normalized value's min/max, so exported sweeps cover the
    range actually seen rather than the compressed observed-value range."""
    inverses = {
        "step_context: last_log_error/5": _safe_log_div_inv(5.0),
        "step_context: clip(log_error_delta,-10,10)/5": lambda y: y * 5.0,
        "step_context: last_keep_step": lambda y: y,
        "step_context: t/t_end": lambda y: y * t_end,
        "solver_trend: log(dt/dt0)/10": _safe_log_div_inv(10.0),
        "solver_trend: accept_ema": None,
        "solver_trend: log_error_ema/5": _safe_log_div_inv(5.0),
        "solver_trend: tanh(reject_streak/4)": lambda y: (
            4.0 * np.arctanh(np.clip(y, -0.999999, 0.999999))
        ),
        "solver_trend: budget_exhausted": lambda y: y * max_steps,
    }
    inverses.update(
        {
            "scalar_decay": {
                "state: clip(y,-10,10)": lambda y: y,
                "direction: sign(k1)*log(|k1|+1)": _logsign_inv,
            },
            "van_der_pol": {
                "state: y0/2": lambda y: y * 2.0,
                "state: sign(y1)*log(|y1|+1)": _logsign_inv,
                "direction: sign(k1_0)*log(|k1_0|+1)": _logsign_inv,
                "direction: sign(k1_1)*log(|k1_1|+1)": _logsign_inv,
            },
            "robertson": {
                "state: y1": lambda y: y,
                "state: sign(y2)*log(|y2|+1e-10)/10": lambda y: (
                    np.sign(y) * np.exp(10.0 * np.abs(y))
                ),
                "state: y3": lambda y: y,
                **{f"direction: sign(f{i})*log(|f{i}|+1)": _logsign_inv for i in range(3)},
            },
            "brusselator": {
                "state: y1/5": lambda y: y * 5.0,
                "state: sign(y2)*log(|y2|+1)": _logsign_inv,
                **{f"direction: sign(f{i})*log(|f{i}|+1)": _logsign_inv for i in range(2)},
            },
            "fitzhugh_nagumo": {
                "state: v/2": lambda y: y * 2.0,
                "state: w/2": lambda y: y * 2.0,
                **{f"direction: sign(f{i})*log(|f{i}|+1)": _logsign_inv for i in range(2)},
            },
            "chemical_cascade": {
                "state: clip(y,-5,5)/5 (all 50 layer x chemical dims, pooled)": lambda y: y * 5.0,
                "direction: sign(k1)*log(|k1|+1) (all 50 dims, pooled)": _logsign_inv,
            },
        }[system]
    )
    return inverses


def _nice_bound(abs_p999: float) -> float:
    """Snap to the smallest of {1,2,5,10,20,50,100,200,500,1000,...} >= abs_p999."""
    for b in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000):
        if abs_p999 <= b:
            return float(b)
    return float(abs_p999)


def build_rows(
    system: str, obs_flat: np.ndarray, t_end: float, max_steps: int
) -> list[tuple[str, np.ndarray, object, object]]:
    """Return [(label, values, formula_fn_or_None, inverse_fn_or_None)] —
    values pooled across dims for aggregated (chemical_cascade) groups, one
    column otherwise."""
    from steppo.envs.ode.systems import get_system

    feature_dims = get_system(system).feature_dims
    formulas = make_formulas(system, t_end, max_steps)
    inverses = make_inverses(system, t_end, max_steps)

    state_labels, direction_labels = _HEAD_LABELS[system]
    rows = []
    idx = 0

    n_state = feature_dims["state"]
    if len(state_labels) == 1 and n_state > 1:
        lbl = state_labels[0]
        rows.append(
            (lbl, obs_flat[:, idx : idx + n_state].reshape(-1), formulas[lbl], inverses[lbl])
        )
    else:
        for j, lbl in enumerate(state_labels):
            rows.append((lbl, obs_flat[:, idx + j], formulas[lbl], inverses[lbl]))
    idx += n_state

    n_dir = feature_dims["direction"]
    if len(direction_labels) == 1 and n_dir > 1:
        lbl = direction_labels[0]
        rows.append((lbl, obs_flat[:, idx : idx + n_dir].reshape(-1), formulas[lbl], inverses[lbl]))
    else:
        for j, lbl in enumerate(direction_labels):
            rows.append((lbl, obs_flat[:, idx + j], formulas[lbl], inverses[lbl]))
    idx += n_dir

    for j, lbl in enumerate(_TAIL_LABELS):
        rows.append((lbl, obs_flat[:, idx + j], formulas[lbl], inverses[lbl]))
    idx += len(_TAIL_LABELS)

    assert idx == obs_flat.shape[1], f"{system}: consumed {idx} dims, obs has {obs_flat.shape[1]}"
    return rows


_LOG_LO, _LOG_HI = -5, 5  # right panel covers |x| in [1e-5, 1e5]


_CURVE_COLOR = "#333333"

_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")


def _sheet_name(label: str, used: set) -> str:
    """Excel sheet names: <=31 chars, no [ ] : * ? / \\, unique within the workbook."""
    name = _INVALID_SHEET_CHARS.sub("_", label)[:31]
    base, i = name, 1
    while name in used:
        suffix = f"_{i}"
        name = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(name)
    return name


def export_formula_curves(
    system: str, rows: list[tuple[str, np.ndarray, object, object]], out_path: str
):
    """Write one .xlsx sheet per feature with a static f(x), columns named after
    the actual variable/formula (e.g. "dt/dt0", "log(dt/dt0)/10"). The x column
    is the raw quantity recovered by inverting the observed min/max through the
    feature's exact inverse, not the normalized value's own min/max (see module
    docstring). Features with no closed-form f(x) (recursive EMAs) are skipped."""
    var_names = make_var_names(system)
    used_names = set()
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        wrote_any = False
        for label, values, formula, inverse in rows:
            if formula is None:
                continue
            finite = values[np.isfinite(values)]
            vmin, vmax = float(finite.min()), float(finite.max())
            if vmin == vmax:
                vmax = vmin + 1e-9
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_lo, raw_hi = float(inverse(vmin)), float(inverse(vmax))
            if raw_lo == raw_hi:
                raw_hi = raw_lo + 1e-9
            x = np.linspace(raw_lo, raw_hi, 500)
            with np.errstate(divide="ignore", invalid="ignore"):
                y = formula(x)
            x_col = var_names[label]
            y_col = _y_column_name(label)
            df = pd.DataFrame({x_col: x, y_col: y})
            df.to_excel(writer, sheet_name=_sheet_name(label, used_names), index=False)
            wrote_any = True
        if not wrote_any:
            pd.DataFrame({"note": ["no feature in this system has a closed-form f(x)"]}).to_excel(
                writer, sheet_name="empty", index=False
            )
    print(f"  saved -> {out_path}")


def plot_system(system: str, rows: list[tuple[str, np.ndarray, object, object]], out_path: str):
    """Plot observed feature distributions for one ODE system."""
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(11, 1.9 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    summary_lines = [
        f"{'feature':<55}{'min':>12}{'p0.1':>12}{'p99.9':>12}{'max':>12}{'bound':>8}{'%outside':>10}"
    ]

    for i, (label, values, formula, _inverse) in enumerate(rows):
        values = values[np.isfinite(values)]
        vmin, vmax = float(values.min()), float(values.max())
        p01, p999 = np.percentile(values, [0.1, 99.9])
        bound = _nice_bound(max(abs(p01), abs(p999), 1e-6))
        frac_outside = float(np.mean(np.abs(values) > bound)) * 100.0

        summary_lines.append(
            f"{label:<55}{vmin:>12.4g}{p01:>12.4g}{p999:>12.4g}{vmax:>12.4g}{bound:>8.3g}{frac_outside:>9.3f}%"
        )

        ax_l, ax_r = axes[i, 0], axes[i, 1]

        # ── left: linear, auto-bounded ──────────────────────────────
        counts_l, _bins_l, _ = ax_l.hist(
            np.clip(values, -bound, bound), bins=80, color=_BLUE, alpha=0.85
        )
        ax_l.set_xlim(-bound, bound)
        ax_l.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
        title = f"±{bound:g}"
        if frac_outside > 0:
            title += f"  ({frac_outside:.2f}% clipped)"
        ax_l.set_title(title, fontsize=8)

        # ── right: log10(|x|) magnitude, colored by sign, zeros called out ──
        pos = values[values > 0]
        neg = values[values < 0]
        zero_frac = float(np.mean(values == 0)) * 100.0
        log_pos = np.clip(np.log10(pos), _LOG_LO, _LOG_HI) if pos.size else np.array([])
        log_neg = np.clip(np.log10(-neg), _LOG_LO, _LOG_HI) if neg.size else np.array([])

        bin_edges = np.linspace(_LOG_LO, _LOG_HI, 101)
        counts_pos = counts_neg = np.array([0.0])
        if log_pos.size:
            counts_pos, _, _ = ax_r.hist(
                log_pos, bins=bin_edges, color=_BLUE, alpha=0.6, label="positive"
            )
        if log_neg.size:
            counts_neg, _, _ = ax_r.hist(
                log_neg, bins=bin_edges, color=_ORANGE, alpha=0.6, label="negative"
            )
        ax_r.set_xlim(_LOG_LO, _LOG_HI)
        ax_r.set_xticks(range(_LOG_LO, _LOG_HI + 1))
        ax_r.set_xticklabels([f"1e{k}" for k in range(_LOG_LO, _LOG_HI + 1)], fontsize=7)

        title = f"min={vmin:.3g}  max={vmax:.3g}"
        if zero_frac > 0:
            title += f"  ({zero_frac:.1f}% exactly 0)"
        if formula is None:
            title += "  [f(x): n/a, recursive EMA]"
        ax_r.set_title(title, fontsize=8)

        # Primary axes (histogram counts) always tinted blue, so they can
        # never be confused with the grey f(x) overlay axes below.
        ax_l.tick_params(axis="y", labelcolor=_BLUE)
        ax_r.tick_params(axis="y", labelcolor=_BLUE)

        # Overlay: sweep x through the literal forward formula on a secondary
        # y-axis, to show where the data sits on the transfer function's shape.
        if formula is not None:
            # Sweep only the range actually observed, never extrapolating into a
            # sign/region that can't occur (e.g. budget_exhausted is never negative).
            ax_l2 = ax_l.twinx()
            x_sweep = np.linspace(vmin, vmax, 400)
            with np.errstate(divide="ignore", invalid="ignore"):
                y_sweep = formula(x_sweep)
            ax_l2.plot(x_sweep, y_sweep, color=_CURVE_COLOR, lw=1.3)
            ax_l2.tick_params(axis="y", labelsize=6, labelcolor=_CURVE_COLOR)
            ax_l2.set_ylabel("f(x)", fontsize=7, color=_CURVE_COLOR, rotation=-90, labelpad=9)

            ax_r2 = ax_r.twinx()
            if pos.size:
                t_pos = np.linspace(np.log10(pos.min()), np.log10(pos.max()), 200)
                with np.errstate(divide="ignore", invalid="ignore"):
                    y_pos_curve = formula(10.0**t_pos)
                ax_r2.plot(
                    t_pos, y_pos_curve, color=_CURVE_COLOR, lw=1.3, ls="-", label="f(x), x>0"
                )
            if neg.size:
                t_neg = np.linspace(np.log10(-neg.max()), np.log10(-neg.min()), 200)
                with np.errstate(divide="ignore", invalid="ignore"):
                    y_neg_curve = formula(-(10.0**t_neg))
                ax_r2.plot(
                    t_neg, y_neg_curve, color=_CURVE_COLOR, lw=1.3, ls="--", label="f(x), x<0"
                )
            ax_r2.tick_params(axis="y", labelsize=6, labelcolor=_CURVE_COLOR)
            ax_r2.set_ylabel("f(x)", fontsize=7, color=_CURVE_COLOR, rotation=-90, labelpad=9)
            if i == 0:
                ax_r2.legend(fontsize=6, loc="lower right")
        else:
            ax_l2 = ax_r2 = None

        # ── shared, visible log-scale y-axis (counts) across both panels ──
        # Log-scale so rare tail bins (a handful of samples) stay visible
        # next to the dominant peak bins (tens of thousands of samples).
        ymax = max(counts_l.max(), counts_pos.max(), counts_neg.max(), 1) * 2
        ax_l.set_yscale("log")
        ax_r.set_yscale("log")
        ax_l.set_ylim(0.8, ymax)
        ax_r.set_ylim(0.8, ymax)
        ax_l.tick_params(axis="y", labelsize=6)
        ax_r.tick_params(axis="y", labelsize=6)

    axes[0, 0].set_title("linear (auto bound)\n" + axes[0, 0].get_title(), fontsize=8)
    axes[0, 1].set_title(
        "log10(|x|) by sign, x in [1e-5, 1e5]\n" + axes[0, 1].get_title(), fontsize=8
    )
    axes[0, 1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{system} — observation feature ranges", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")

    return "\n".join(summary_lines)


def main():
    """Collect PID observations and report empirical feature ranges."""
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)

    all_summaries = []
    for system in args.systems:
        print(f"[*] {system}")
        config = load_config_from_yaml(TrainConfig, SYSTEM_CONFIGS[system])
        config.env.obs_features = ALL_FEATURES
        T = args.steps or config.rollout_steps

        env = ODEEnv(config.env, T)
        rng, key_tasks, key_roll = jax.random.split(rng, 3)
        task_keys = jax.random.split(key_tasks, args.num_envs)
        params_batch = jax.vmap(env.sample_task)(task_keys)

        offset = compute_step_context_offset(config.env.obs_features, env._spec.feature_dims)
        traces = collect_pid_obs(env, params_batch, args.num_envs, T, offset, key_roll)

        obs = np.asarray(traces["obs"])  # (T, num_envs, obs_dim)
        active = np.asarray(traces["active"])  # (T, num_envs)
        obs_flat = obs[active]  # (num_valid, obs_dim)
        print(
            f"    {obs_flat.shape[0]} valid obs out of {obs.shape[0] * obs.shape[1]} "
            f"({T} steps x {args.num_envs} envs)"
        )

        rows = build_rows(system, obs_flat, config.env.t_end, T)
        out_path = os.path.join(args.out, f"{system}.png")
        summary = plot_system(system, rows, out_path)
        export_formula_curves(system, rows, os.path.join(args.out, f"{system}_fx.xlsx"))
        all_summaries.append(f"=== {system} ===\n{summary}\n")

    summary_path = os.path.join(args.out, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(all_summaries))
    print(f"[*] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
