"""Regression tests for factorial experiment config generation."""

import itertools
import sys

import pytest
import yaml

from research.ode import generate_experiments


def _write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def _run_generator(monkeypatch, tmp_path, base, sweep):
    base_path = _write_yaml(tmp_path / "base.yaml", base)
    sweep_path = _write_yaml(tmp_path / "sweep.yaml", sweep)
    output_dir = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_experiments.py",
            "--base",
            str(base_path),
            "--sweep",
            str(sweep_path),
            "--prefix",
            "test",
            "--out-dir",
            str(output_dir),
        ],
    )
    generate_experiments.main()
    return output_dir


def test_generator_writes_cartesian_product_without_mutating_base(monkeypatch, tmp_path):
    base = {"env": {"system": "scalar_decay", "rtol": 0.001}, "ppo": {"entropy": 0.01}}
    sweep = {"env.rtol": [0.001, 0.01], "ppo.entropy": "0.0, 0.02"}

    output_dir = _run_generator(monkeypatch, tmp_path, base, sweep)
    generated = [yaml.safe_load((output_dir / f"test_e{i}.yaml").read_text()) for i in range(1, 5)]

    assert [(data["env"]["rtol"], data["ppo"]["entropy"]) for data in generated] == list(
        itertools.product([0.001, 0.01], [0.0, 0.02])
    )
    assert [data["exp_name"] for data in generated] == [f"test_e{i}" for i in range(1, 5)]
    assert all(data["env"]["system"] == "scalar_decay" for data in generated)
    assert yaml.safe_load((tmp_path / "base.yaml").read_text()) == base


def test_generator_maps_model_fields_and_keeps_linked_keys_as_one_axis(monkeypatch, tmp_path):
    base = {"vae": {"latent_dim": 8}, "ppo": {"policy": {}}}
    sweep = {
        "model.latent_dim": [2, 4],
        "ppo.entropy_coeff_start,ppo.entropy_coeff": [[0.0, 0.005], [0.1, 0.02]],
    }

    output_dir = _run_generator(monkeypatch, tmp_path, base, sweep)
    generated = [yaml.safe_load((output_dir / f"test_e{i}.yaml").read_text()) for i in range(1, 5)]
    paired_coeffs = [
        (data["ppo"]["entropy_coeff_start"], data["ppo"]["entropy_coeff"]) for data in generated
    ]

    assert [data["vae"]["latent_dim"] for data in generated] == [2, 2, 4, 4]
    assert paired_coeffs == [
        (0.0, 0.005),
        (0.1, 0.02),
        (0.0, 0.005),
        (0.1, 0.02),
    ]
    assert "model.latent_dim = 2  (-> vae.latent_dim)" in (output_dir / "test_e1.yaml").read_text()


def test_generator_rejects_linked_values_with_wrong_arity(monkeypatch, tmp_path):
    base = {"ppo": {}}
    sweep = {"ppo.a,ppo.b": [[1]]}
    base_path = _write_yaml(tmp_path / "base.yaml", base)
    sweep_path = _write_yaml(tmp_path / "sweep.yaml", sweep)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_experiments.py",
            "--base",
            str(base_path),
            "--sweep",
            str(sweep_path),
            "--prefix",
            "test",
        ],
    )

    with pytest.raises(ValueError, match="expects each value to be a list of 2 elements"):
        generate_experiments.main()
