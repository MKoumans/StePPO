"""Shared --config + dotted-CLI-override parsing for research/ode's
config-driven scripts (run.py and the diagnostics/inference scripts) —
one `--config <yaml>` flag plus arbitrary `--field value` overrides applied
via steppo.configs.base_config.apply_dotted_overrides.
"""

import argparse


def parse_dotted_overrides(unknown: list[str]) -> dict:
    """Parse trailing `--field value` / `--field=value` CLI args into a dict."""
    overrides = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unrecognized argument '{tok}' (expected '--field value')")
        key = tok[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            i += 1
        else:
            if i + 1 >= len(unknown):
                raise ValueError(f"Override '--{key}' is missing a value")
            value = unknown[i + 1]
            i += 2
        overrides[key] = value
    return overrides


def parse_config_args(description: str, config_help: str) -> argparse.Namespace:
    """argparse with a single --config flag; returns args with args.overrides populated
    from any other `--field value` pairs on the command line."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=str, default=None, help=config_help)
    args, unknown = parser.parse_known_args()
    args.overrides = parse_dotted_overrides(unknown)
    return args
