"""Helpers shared by the DeepONet comparison plots: parsers for the
'# split: <name>' tables written by save_steps_vs_mu_splits_txt and for the
DeepONet eval tables, --train-bins arguments, and find_rl_wallclock. NumPy only.
"""

import glob
import os

import numpy as np

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..", "..")
OVERLEAF_OUTPUTS = os.path.join(REPO_ROOT, "overleaf", "outputs")

SYSTEMS = {
    "scalar_decay": dict(
        overleaf_dir="scalar-decay-single-run", param_label="lambda", deeponet_dir="scalar_decay"
    ),
    "van_der_pol": dict(
        overleaf_dir="vanderpol-single-run", param_label="mu", deeponet_dir="van_der_pol"
    ),
    "brusselator": dict(
        overleaf_dir="brusselator-single-run", param_label="B", deeponet_dir="brusselator"
    ),
}

# Training bins depend on the checkpoint, so callers pass them via --train-bins.

TRAIN_BINS_HELP = (
    "Training-bin range to shade, e.g. --train-bins 5,15 --train-bins 25,35 "
    "(repeatable). Omit for no shading."
)
TRAIN_BINS_BY_SYSTEM_HELP = (
    "Training-bin range to shade for one system, e.g. "
    "--train-bins van_der_pol:5,15 --train-bins van_der_pol:25,35 "
    "(repeatable). Omit for no shading."
)


def add_train_bins_arg(parser):
    parser.add_argument("--train-bins", action="append", metavar="LO,HI", help=TRAIN_BINS_HELP)


def parse_train_bins(values):
    """values: repeated "lo,hi" strings -> [[lo, hi], ...], or None if empty."""
    if not values:
        return None
    bins = []
    for v in values:
        lo_s, hi_s = v.split(",")
        bins.append([float(lo_s), float(hi_s)])
    return bins


def add_train_bins_by_system_arg(parser):
    parser.add_argument(
        "--train-bins", action="append", metavar="SYSTEM:LO,HI", help=TRAIN_BINS_BY_SYSTEM_HELP
    )


def parse_train_bins_by_system(values):
    """values: repeated "system:lo,hi" strings -> {system: [[lo, hi], ...]}."""
    out = {}
    for v in values or []:
        system, rng = v.split(":", 1)
        lo_s, hi_s = rng.split(",")
        out.setdefault(system, []).append([float(lo_s), float(hi_s)])
    return out


def add_rl_tag_arg(parser):
    parser.add_argument(
        "--rl-tag",
        default=None,
        help="Which wallclock_rl_{system}_<tag>_{mode}.npz variant to use "
        "(the `tag` wallclock_vs_mu.py --tag/--hf_repo/--checkpoint wrote it under). "
        "Omit to use the most-recently-written variant.",
    )


def find_rl_wallclock(results_dir: str, system: str, mode: str, tag: str = None):
    """Find wallclock_vs_mu.py's RL timing file wallclock_rl_{system}_{tag}_{mode}.npz.

    Returns (mus, rl_ms, tag, mode, device), or five Nones if none exists. Without
    `tag`, the most recently written match is used.
    """
    if tag:
        pattern = os.path.join(results_dir, f"wallclock_rl_{system}_{tag}_{mode}.npz")
    else:
        pattern = os.path.join(results_dir, f"wallclock_rl_{system}_*_{mode}.npz")
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not paths:
        return None, None, None, None, None
    d = np.load(paths[-1])
    found_tag = (
        str(d["tag"]) if "tag" in d.files else os.path.splitext(os.path.basename(paths[-1]))[0]
    )
    rl_mode = str(d["mode"]) if "mode" in d.files else None
    rl_device = str(d["device"]) if "device" in d.files else None
    return d["mus"], d["rl_ms"], found_tag, rl_mode, rl_device


def parse_split_table(path: str, split: str = "test") -> dict:
    """Parse one '# split: <split>' section into {column_name: array}; empty bins are kept."""
    with open(path) as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"# split: {split}":
            start = i + 1
            break
    if start is None:
        raise ValueError(f"No '# split: {split}' section in {path}")
    header = lines[start].split()
    rows = []
    for line in lines[start + 1 :]:
        if line.strip() == "" or line.startswith("#"):
            break
        rows.append([float(x) for x in line.split()])
    rows = np.asarray(rows)
    return {name: rows[:, i] for i, name in enumerate(header)}


def parse_flat_table(path: str) -> dict:
    """Parse a whitespace-separated table without split sections (DeepONet eval tables)."""
    table = load_eval_table(path)
    return {name: np.atleast_1d(table[name]) for name in table.dtype.names}


def load_eval_table(path: str, *, missing_ok: bool = False) -> np.ndarray | None:
    """Load a whitespace-separated evaluation table with a named header."""
    if missing_ok and not os.path.exists(path):
        return None
    return np.genfromtxt(path, names=True)


def deeponet_dir(system: str) -> str:
    return os.path.join(HERE, "output-deeponet", SYSTEMS[system]["deeponet_dir"])


def overleaf_outputs_dir(system: str) -> str:
    return os.path.join(OVERLEAF_OUTPUTS, SYSTEMS[system]["overleaf_dir"], "outputs")


def load_oracle_testset(data_dir: str):
    """Pick the largest cached split with ref_ts, the newest on ties."""
    files = glob.glob(os.path.join(data_dir, "testset", "*.npz"))
    candidates = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "ref_ts" in d.files:
            candidates.append((f, len(d["task_params"]), os.path.getmtime(f)))
    if not candidates:
        raise FileNotFoundError(f"No test-set file with ref_ts found in {data_dir}/testset")
    best_file, best_n, _ = max(candidates, key=lambda x: (x[1], x[2]))
    print(f"[*] Loaded test set (largest, newest on ties): {best_file} (n={best_n})")
    return np.load(best_file, allow_pickle=True)


def bin_stats(x, values, bin_edges):
    """Per-bin nanmean, nanstd and count of `values` binned by `x`."""
    idx = np.digitize(x, bin_edges) - 1
    means, stds, counts = [], [], []
    for b in range(len(bin_edges) - 1):
        mask = idx == b
        if mask.sum() == 0:
            means.append(np.nan)
            stds.append(np.nan)
            counts.append(0)
        else:
            means.append(np.nanmean(values[mask]))
            stds.append(np.nanstd(values[mask]))
            counts.append(int(mask.sum()))
    return np.asarray(means), np.asarray(stds), np.asarray(counts)


def align_to_reference(
    t_ref: np.ndarray,
    ts: np.ndarray,
    ys: np.ndarray,
    *,
    mask_beyond_end: bool = False,
) -> np.ndarray:
    """Interpolate a padded trajectory onto reference times.

    By default, values beyond the last solved time use ``np.interp``'s endpoint
    value. Set ``mask_beyond_end`` to leave those values as NaN for early-stopped
    trajectories.
    """
    aligned = np.full((len(t_ref), ys.shape[-1]), np.nan, dtype=float)
    valid = np.isfinite(ts)
    if valid.sum() >= 2:
        t_valid = ts[valid].astype(float)
        y_valid = ys[valid]
        for dim in range(ys.shape[-1]):
            aligned[:, dim] = np.interp(t_ref, t_valid, y_valid[:, dim])
        if mask_beyond_end:
            aligned[t_ref > t_valid[-1]] = np.nan
    return aligned


def trajectory_column_names(prefix: str, state_dim: int) -> list[str]:
    """Return CSV column names for a trajectory's state dimensions."""
    suffixes = ["_y"] if state_dim == 1 else [f"_y{dim}" for dim in range(state_dim)]
    return [f"{prefix}{suffix}" for suffix in suffixes]
