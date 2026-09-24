"""Generic content-addressed cache for per-run analysis heavy-compute stages.
Stores an arbitrary pickled payload (each script's output shape differs) under
`data/analysis_cache/<script_name>/<fingerprint>.pkl`.

IMPORTANT: the fingerprint covers checkpoint path, config, and CLI params only
— it can't detect model/eval code changes (encoder.py, vae.py, eval.py,
ode_env.py etc.), which would leave a stale-but-wrong cache silently served.
Bump ANALYSIS_CACHE_VERSION whenever such a change could alter numeric output
for an unchanged checkpoint.
"""

import hashlib
import json
import os
import pickle
import tempfile

DATA_ROOT = "data"
ANALYSIS_CACHE_VERSION = 1


def analysis_cache_dir(script_name: str, root: str = DATA_ROOT) -> str:
    """Return the cache directory reserved for one analysis script."""
    return os.path.join(root, "analysis_cache", script_name)


def analysis_fingerprint(payload: dict) -> str:
    """Stable hash over a JSON-serializable payload; namespaced by ANALYSIS_CACHE_VERSION."""
    blob = json.dumps(
        {"analysis_cache_version": ANALYSIS_CACHE_VERSION, **payload}, sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_or_compute(script_name: str, payload: dict, compute_fn):
    """Return compute_fn()'s result, cached under data/analysis_cache/<script_name>/<fingerprint>.pkl.

    `payload` must include every input that affects the result — omitting one
    lets unrelated inputs collide on the same cache entry.
    """
    cache_dir = analysis_cache_dir(script_name)
    fp = analysis_fingerprint(payload)
    path = os.path.join(cache_dir, f"{fp}.pkl")

    if os.path.exists(path):
        print(f"  [cache hit] {path}", flush=True)
        with open(path, "rb") as f:
            return pickle.load(f)

    result = compute_fn()

    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp.pkl")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(result, f)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return result
