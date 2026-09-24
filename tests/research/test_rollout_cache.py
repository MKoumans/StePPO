"""Contract tests for the cache used by expensive post-run analyses."""

import pickle

import pytest

from research.ode.post_run_analysis import rollout_cache


def test_analysis_cache_dir_uses_script_namespace_and_supplied_root(tmp_path):
    assert rollout_cache.analysis_cache_dir("compare", str(tmp_path)) == str(
        tmp_path / "analysis_cache" / "compare"
    )


def test_analysis_fingerprint_is_stable_across_mapping_order():
    assert rollout_cache.analysis_fingerprint({"a": 1, "b": [2, 3]}) == (
        rollout_cache.analysis_fingerprint({"b": [2, 3], "a": 1})
    )


def test_analysis_fingerprint_changes_when_cache_version_changes(monkeypatch):
    payload = {"checkpoint": "checkpoint_1", "seed": 3}
    original = rollout_cache.analysis_fingerprint(payload)
    monkeypatch.setattr(
        rollout_cache, "ANALYSIS_CACHE_VERSION", rollout_cache.ANALYSIS_CACHE_VERSION + 1
    )

    assert rollout_cache.analysis_fingerprint(payload) != original


def test_load_or_compute_reuses_matching_payload_and_recomputes_on_change(tmp_path, monkeypatch):
    monkeypatch.setattr(rollout_cache, "analysis_cache_dir", lambda script: str(tmp_path / script))
    calls = []

    def compute():
        calls.append("compute")
        return {"steps": [10, 20]}

    first = rollout_cache.load_or_compute("compare", {"seed": 1}, compute)
    cached = rollout_cache.load_or_compute(
        "compare", {"seed": 1}, lambda: pytest.fail("cache hit must skip compute")
    )
    changed = rollout_cache.load_or_compute("compare", {"seed": 2}, lambda: {"steps": [30]})

    assert first == cached == {"steps": [10, 20]}
    assert changed == {"steps": [30]}
    assert calls == ["compute"]
    assert len(list((tmp_path / "compare").glob("*.pkl"))) == 2


def test_failed_serialization_removes_temporary_file_and_allows_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(rollout_cache, "analysis_cache_dir", lambda script: str(tmp_path / script))

    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        rollout_cache.load_or_compute("compare", {"seed": 1}, lambda: lambda: None)

    cache_dir = tmp_path / "compare"
    assert list(cache_dir.iterdir()) == []
    assert rollout_cache.load_or_compute("compare", {"seed": 1}, lambda: {"recovered": True}) == {
        "recovered": True
    }
