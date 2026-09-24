"""Command-line lookups that must run before JAX is imported (e.g. to pick the device)."""

import sys


def scan_flag(name, default=None):
    """Return the value of `name value` or `name=value` in sys.argv, else `default`."""
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return default
