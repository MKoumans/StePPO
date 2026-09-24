"""Debugging helpers for disabling JIT during interactive investigation."""

import os

import jax

DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


def maybe_jit(fun=None, **jit_kwargs):
    """Like jax.jit, but a no-op when DEBUG=True so breakpoints/pdb work."""

    def decorator(f):
        if DEBUG:
            return f
        return jax.jit(f, **jit_kwargs)

    if fun is not None:
        return decorator(fun)
    return decorator
