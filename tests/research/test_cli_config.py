"""Tests for shared dotted-override parsing used by research CLIs."""

import sys

import pytest

from research.ode.cli_config import parse_config_args, parse_dotted_overrides


def test_parse_dotted_overrides_supports_both_forms_and_last_value_wins():
    overrides = parse_dotted_overrides(
        [
            "--env.rtol",
            "1e-5",
            "--vae.latent_dim=4",
            "--env.rtol",
            "2e-5",
        ]
    )

    assert overrides == {"env.rtol": "2e-5", "vae.latent_dim": "4"}


@pytest.mark.parametrize(
    "unknown, message",
    [
        (["orphan"], "expected '--field value'"),
        (["--env.rtol"], "missing a value"),
    ],
)
def test_parse_dotted_overrides_rejects_malformed_arguments(unknown, message):
    with pytest.raises(ValueError, match=message):
        parse_dotted_overrides(unknown)


def test_parse_config_args_keeps_config_and_collects_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool.py",
            "--config",
            "train.yaml",
            "--env.rtol",
            "1e-4",
        ],
    )

    args = parse_config_args("description", "config help")

    assert args.config == "train.yaml"
    assert args.overrides == {"env.rtol": "1e-4"}
