"""Tests for research/ode/diagnostics/model_sweep.py's config-driven model dispatch
and per-mu aggregation — the pure-logic pieces that don't require a GPU solve.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_DIAGNOSTICS_DIR = str(Path(__file__).resolve().parents[2] / "research" / "ode" / "diagnostics")
if _DIAGNOSTICS_DIR not in sys.path:
    sys.path.insert(0, _DIAGNOSTICS_DIR)

model_sweep = importlib.import_module("model_sweep")


def _env_config():
    return SimpleNamespace(
        system="scalar_decay",
        rtol=1e-3,
        atol=1e-6,
        dt_min=1e-8,
        dt_max=1.0,
        t_end=1.0,
    )


def test_resolve_models_skips_checkpoint_when_unset():
    cfg = SimpleNamespace(models=[{"type": "pid", "label": "PID"}], checkpoint_steppo=None)
    assert model_sweep.resolve_models(cfg) == [{"type": "pid", "label": "PID"}]


def test_resolve_models_appends_checkpoint_when_set():
    cfg = SimpleNamespace(
        models=[{"type": "pid", "label": "PID"}], checkpoint_steppo="outputs/runs/x/checkpoint_1"
    )
    resolved = model_sweep.resolve_models(cfg)
    assert resolved == [
        {"type": "pid", "label": "PID"},
        {"type": "checkpoint", "label": "RL (StePPO)", "checkpoint": "outputs/runs/x/checkpoint_1"},
    ]
    # cfg.models itself is untouched — resolve_models returns a new list.
    assert cfg.models == [{"type": "pid", "label": "PID"}]


def test_build_model_rejects_unknown_type():
    with pytest.raises(SystemExit, match="unknown/missing model 'type'"):
        model_sweep.build_model(
            {"type": "not-a-type", "label": "x"}, _env_config(), None, max_steps=10
        )


def test_build_model_requires_label():
    with pytest.raises(SystemExit, match="missing 'label'"):
        model_sweep.build_model({"type": "pid"}, _env_config(), None, max_steps=10)


def test_build_model_deeponet_not_implemented():
    with pytest.raises(NotImplementedError):
        model_sweep.build_model(
            {"type": "deeponet", "label": "DeepONet"}, _env_config(), None, max_steps=10
        )


def test_build_model_pid_has_dense_solve():
    handle = model_sweep.build_model(
        {"type": "pid", "label": "PID"}, _env_config(), None, max_steps=10
    )
    assert handle.label == "PID"
    assert handle.dense_solve is not None
    assert callable(handle.cheap_solve)


def test_build_model_pid_oracle_has_no_dense_solve():
    handle = model_sweep.build_model(
        {"type": "pid_oracle", "label": "PID (oracle)", "max_iters": 3},
        _env_config(),
        None,
        max_steps=10,
    )
    assert handle.label == "PID (oracle)"
    assert handle.dense_solve is None
    assert callable(handle.cheap_solve)


def test_build_model_pid_tuned_passes_gains_to_controller(monkeypatch):
    captured = {}

    def fake_solve_pid_batch(env_config, controller, mus, keys, max_steps, **kwargs):
        captured["pcoeff"] = controller.pcoeff
        captured["icoeff"] = controller.icoeff
        captured["dcoeff"] = controller.dcoeff
        return {"steps": np.ones(len(mus))}

    monkeypatch.setattr(model_sweep, "solve_pid_batch", fake_solve_pid_batch)
    handle = model_sweep.build_model(
        {"type": "pid_tuned", "label": "PID (tuned)", "kp": 0.06, "ki": 1.9, "kd": 0.02},
        _env_config(),
        None,
        max_steps=10,
    )
    handle.cheap_solve(np.array([1.0]), np.zeros((1, 2), dtype=np.uint32))

    assert captured == {"pcoeff": 0.06, "icoeff": 1.9, "dcoeff": 0.02}


def test_build_model_pid_tuned_rejects_negative_gain():
    with pytest.raises(ValueError, match="non-negative"):
        model_sweep.build_model(
            {"type": "pid_tuned", "label": "PID (tuned)", "kp": -1.0},
            _env_config(),
            None,
            max_steps=10,
        )


def test_sweep_one_model_aggregates_across_repeats(monkeypatch):
    monkeypatch.setattr(
        model_sweep, "load_or_compute", lambda name, payload, compute_fn: compute_fn()
    )

    mus = np.array([1.0, 2.0])
    num_repeats = 2
    mu_grid = np.repeat(mus, num_repeats)

    def cheap_solve(mus_batch, keys_batch):
        # steps == 10x the mu value, deterministic so the per-mu mean is exact.
        return {"steps": 10.0 * np.asarray(mus_batch)}

    handle = model_sweep.ModelHandle(label="fake", cheap_solve=cheap_solve, dense_solve=None)

    def fake_time_solve_fn(solve_fn, mus_arg, **kwargs):
        return np.full(len(mus_arg), 5.0)

    monkeypatch.setattr(model_sweep, "time_solve_fn", fake_time_solve_fn)

    cfg = SimpleNamespace(
        n_sweep=len(mus),
        num_repeats=num_repeats,
        seed=0,
        max_steps=10,
        ref_tol_factor=0.01,
        wallclock_repeats=1,
        wallclock_batch_size=1,
    )
    spec = {"type": "pid", "label": "fake"}
    result = model_sweep.sweep_one_model(
        spec,
        handle,
        mus=mus,
        mu_grid=mu_grid,
        episode_keys=np.zeros((len(mu_grid), 2), dtype=np.uint32),
        ref_out=None,
        cfg=cfg,
        atol=1e-6,
    )

    assert np.allclose(result["steps_mean"], [10.0, 20.0])
    assert result["err_int_mean"] is None
    assert np.allclose(result["wallclock_ms"], [5.0, 5.0])


def test_sweep_one_model_computes_e_int_when_dense_solve_present(monkeypatch):
    monkeypatch.setattr(
        model_sweep, "load_or_compute", lambda name, payload, compute_fn: compute_fn()
    )
    monkeypatch.setattr(
        model_sweep, "time_solve_fn", lambda solve_fn, mus_arg, **kwargs: np.zeros(len(mus_arg))
    )

    def fake_log10_time_integrated_error(ref_ts, ref_ys, ts, ys, atol):
        return np.full(len(ts), -2.5)

    monkeypatch.setattr(
        model_sweep, "log10_time_integrated_error", fake_log10_time_integrated_error
    )

    mus = np.array([1.0])
    num_repeats = 3
    mu_grid = np.repeat(mus, num_repeats)

    handle = model_sweep.ModelHandle(
        label="fake",
        cheap_solve=lambda mus_batch, keys_batch: {"steps": np.ones(len(mus_batch))},
        dense_solve=lambda mus_batch, keys_batch: {
            "ts": np.zeros((len(mus_batch), 1)),
            "ys": np.zeros((len(mus_batch), 1, 1)),
        },
    )
    cfg = SimpleNamespace(
        n_sweep=len(mus),
        num_repeats=num_repeats,
        seed=0,
        max_steps=10,
        ref_tol_factor=0.01,
        wallclock_repeats=1,
        wallclock_batch_size=1,
    )
    result = model_sweep.sweep_one_model(
        {"type": "pid", "label": "fake"},
        handle,
        mus=mus,
        mu_grid=mu_grid,
        episode_keys=np.zeros((len(mu_grid), 2), dtype=np.uint32),
        ref_out={"ts": None, "ys": None},
        cfg=cfg,
        atol=1e-6,
    )

    assert np.allclose(result["err_int_mean"], [-2.5])


def test_plot_and_save_outputs(tmp_path):
    mus = np.array([1.0, 10.0])
    results = {
        "PID": {
            "steps_mean": np.array([9.0, 12.0]),
            "err_int_mean": np.array([-2.0, -1.5]),
            "wallclock_ms": np.array([1.0, 1.2]),
        },
        "PID (oracle)": {
            "steps_mean": np.array([7.0, 10.0]),
            "err_int_mean": None,
            "wallclock_ms": np.array([2.0, 2.3]),
        },
    }
    out_png = tmp_path / "model_sweep.png"
    model_sweep.plot_model_sweep(mus, results, out_png, param_label="mu", system="scalar_decay")
    assert out_png.is_file() and out_png.stat().st_size > 0

    out_txt = tmp_path / "model_sweep.txt"
    model_sweep.save_model_sweep_txt(mus, results, out_txt)
    lines = out_txt.read_text().splitlines()
    assert "PID_steps" in lines[0] and "PID (oracle)_e_int" in lines[0]
    # "PID (oracle)"'s e_int is None -> its e_int column is blank (two adjacent tabs) on every data row.
    assert all("\t\t" in line for line in lines[1:])
