"""Default and strict YAML-loading regressions for diagnostics config schemas."""

import pytest

from research.ode.diagnostics.configs import (
    ModelSweepConfig,
    WallclockBatchSizeSweepConfig,
    WallclockVsMuConfig,
)
from steppo.configs.base_config import load_config_from_yaml


def test_config_collection_defaults_are_independent_between_runs():
    first = ModelSweepConfig()
    second = ModelSweepConfig()
    first.models.append({"type": "pid", "label": "PID"})

    assert first.models == [{"type": "pid", "label": "PID"}]
    assert second.models == []
    assert first.n_sweep == 30
    assert first.device is None


def test_wallclock_config_defaults_match_timing_modes():
    single = WallclockVsMuConfig()
    batch = WallclockBatchSizeSweepConfig()

    assert (single.mode, single.device, single.n_sweep, single.repeats) == (
        "single",
        None,
        30,
        20,
    )
    assert (batch.system, batch.device, batch.repeats, batch.reuse) == (
        "van_der_pol",
        "gpu",
        3,
        False,
    )


def test_model_sweep_yaml_preserves_model_specific_options(tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text(
        "env_config: configs/ode.yaml\n"
        "n_sweep: 12\n"
        "models:\n"
        "  - type: pid_tuned\n"
        "    label: tuned\n"
        "    kp: 0.06\n"
        "    ki: 1.9\n"
        "    kd: 0.02\n"
    )

    config = load_config_from_yaml(ModelSweepConfig, str(path))

    assert config.n_sweep == 12
    assert config.models == [
        {
            "type": "pid_tuned",
            "label": "tuned",
            "kp": 0.06,
            "ki": 1.9,
            "kd": 0.02,
        }
    ]


def test_diagnostic_yaml_rejects_unknown_fields(tmp_path):
    path = tmp_path / "wallclock.yaml"
    path.write_text("mode: single\nunrecognized: true\n")

    with pytest.raises((TypeError, ValueError), match="unrecognized"):
        load_config_from_yaml(WallclockVsMuConfig, str(path))
