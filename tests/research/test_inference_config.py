"""Schema/default tests for the custom-trajectory inference config."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.ode.inference.configs import PlotCustomTrajectoriesConfig
from steppo.configs.base_config import ODEEnvConfig, load_config_from_yaml

_INFERENCE_DIR = str(Path(__file__).resolve().parents[2] / "research" / "ode" / "inference")
sys.path.insert(0, str(_INFERENCE_DIR))
_saved_modules = {
    name: sys.modules.pop(name, None) for name in ("configs", "plot_custom_trajectories")
}
try:
    inference = importlib.import_module("plot_custom_trajectories")
finally:
    sys.path.remove(_INFERENCE_DIR)
    for _name, _module in _saved_modules.items():
        sys.modules.pop(_name, None)
        if _module is not None:
            sys.modules[_name] = _module


def test_conditions_have_per_instance_defaults():
    first = PlotCustomTrajectoriesConfig()
    second = PlotCustomTrajectoriesConfig()
    first.conditions.append([10.0])

    assert first.conditions == [[10.0]]
    assert second.conditions == []
    assert first.step_interval == 1
    assert first.max_steps == 2000


def test_yaml_loader_builds_custom_trajectory_config(tmp_path):
    path = tmp_path / "inference.yaml"
    path.write_text(
        "repo_id: org/model\n"
        "revision: release\n"
        "conditions:\n  - [10.0]\n  - [50.0, [0.0, -2.0]]\n"
        "step_interval: 4\n"
        "show_rejected: true\n"
    )

    config = load_config_from_yaml(PlotCustomTrajectoriesConfig, str(path))

    assert config.repo_id == "org/model"
    assert config.revision == "release"
    assert config.conditions == [[10.0], [50.0, [0.0, -2.0]]]
    assert config.step_interval == 4
    assert config.show_rejected is True


def test_yaml_loader_rejects_unknown_inference_options(tmp_path):
    path = tmp_path / "inference.yaml"
    path.write_text("repo_id: org/model\nunrecognized: true\n")

    with pytest.raises((TypeError, ValueError), match="unrecognized"):
        load_config_from_yaml(PlotCustomTrajectoriesConfig, str(path))


def test_condition_parser_normalizes_task_only_and_explicit_initial_state():
    parsed = inference._parse_conditions([[10], [50, [0, -2.5]]])

    assert parsed == [(10.0, None), (50.0, (0.0, -2.5))]
    with pytest.raises(ValueError, match="Each condition must be"):
        inference._parse_conditions([[1.0, 2.0, 3.0]])


def test_env_for_condition_rebuilds_two_dimensional_system_with_pinned_y0(monkeypatch):
    base_env = SimpleNamespace(_spec=SimpleNamespace(y_dim=2), max_steps=123)
    base_config = SimpleNamespace(
        env=ODEEnvConfig(
            system="van_der_pol",
            sample_y0=True,
            y0_x=0.0,
            y0_y=-2.0,
        )
    )
    observed = {}

    def make_env(config, max_steps):
        observed.update(config=config, max_steps=max_steps)
        return "rebuilt"

    monkeypatch.setattr(inference, "ODEEnv", make_env)
    result = inference._env_for_condition(base_env, base_config, (1.25, 3.5))

    assert result == "rebuilt"
    assert observed["max_steps"] == 123
    assert observed["config"].sample_y0 is False
    assert (observed["config"].y0_x, observed["config"].y0_y) == (1.25, 3.5)


@pytest.mark.parametrize(
    ("system", "y_dim", "y0", "config_fields", "error"),
    [
        ("van_der_pol", 2, (1.0,), {"sample_y0": True}, "has 1 dims"),
        ("fixed_initial_state", 2, (1.0, 2.0), {}, "no sample_y0"),
        ("scalar_decay", 1, (1.0,), {"sample_y0": True}, "only wired up for 2-D"),
    ],
)
def test_env_for_condition_rejects_unsupported_initial_state_overrides(
    system,
    y_dim,
    y0,
    config_fields,
    error,
):
    base_env = SimpleNamespace(_spec=SimpleNamespace(y_dim=y_dim), max_steps=10)
    env_config = SimpleNamespace(system=system, **config_fields)

    with pytest.raises(ValueError, match=error):
        inference._env_for_condition(base_env, SimpleNamespace(env=env_config), y0)
