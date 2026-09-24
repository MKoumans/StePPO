"""outputs/<category>/<date>/<name-or-fingerprint>/ layout for diagnostics
and inference scripts — same date/identifier shape as TrainConfig's
outputs/runs/<date>/... and generate_experiments.py's
outputs/experiments/<date>/<uuid>/, but keyed by a content fingerprint
instead of a random uuid so re-running an unchanged config reuses its dir.
"""

import dataclasses
import datetime
import hashlib
import json
import os


def _fingerprint(payload: dict) -> str:
    """Short stable hash of a JSON-serializable payload (reimplemented from
    steppo.training.pid_solve.fingerprint to skip its jax/diffrax import cost
    here, where callers may just be naming a directory)."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def resolve_output_dir(
    category: str,
    cfg,
    name: str | None = None,
    exclude: tuple[str, ...] = (),
    root: str = "outputs",
) -> str:
    """Return (and create) outputs/<category>/<YYYYMMDD>/<name-or-fingerprint>/.

    `name`, if given, is used directly (reruns under the same name overwrite
    in place). Otherwise the dir is a content fingerprint of `cfg` (fields in
    `exclude` dropped first) so identical configs share a dir and changed
    ones get a fresh one.
    """
    date = datetime.datetime.now().strftime("%Y%m%d")
    if name:
        slug = name
    else:
        payload = dataclasses.asdict(cfg)
        for key in exclude:
            payload.pop(key, None)
        slug = _fingerprint(payload)
    out_dir = os.path.join(root, category, date, slug)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
