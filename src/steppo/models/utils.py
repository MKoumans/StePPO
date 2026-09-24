"""Shared validation, activation, and normalization helpers for models."""

from typing import Callable, Optional

import flax.nnx as nnx
import jax.numpy as jnp

_ACTIVATIONS: dict[str, Callable] = {
    "relu": nnx.relu,
    "tanh": jnp.tanh,
    "gelu": nnx.gelu,
    "elu": nnx.elu,
    "silu": nnx.silu,
}

_VALID_ACTIVATIONS = sorted(_ACTIVATIONS)
_VALID_NORMALIZATIONS = ("none", "layer_norm", "instance_norm")
# Canonical order: encoder embed/concat order is fixed regardless of the order
# fields are listed in config, so embed_norms indices stay in sync.
_CANONICAL_ENCODER_INPUTS = ("action", "state", "reward", "task")
# Canonical order for the policy's input concatenation (ActorCritic.__call__).
_CANONICAL_POLICY_INPUTS = ("state", "z", "task")


def get_activation(name: str) -> Callable:
    """Resolve a configured activation name to a callable."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {_VALID_ACTIVATIONS}")
    return _ACTIVATIONS[name]


def build_mlp(in_dim: int, hidden_dims: tuple[int, ...], rngs: nnx.Rngs) -> nnx.List:
    """Build a stack of hidden linear layers, sized in_dim -> hidden_dims[0] -> ... -> hidden_dims[-1]."""
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nnx.Linear(prev, h, rngs=rngs))
        prev = h
    return nnx.List(layers)


def make_norm_layers(
    normalize: str,
    dims: tuple[int, ...],
    rngs: nnx.Rngs,
) -> Optional[nnx.List]:
    """Build one normalization module per hidden layer, if requested.

    ``instance_norm`` uses affine-free layer normalization because these inputs
    have no spatial axis.
    """
    if normalize == "none":
        return None
    if normalize == "layer_norm":
        return nnx.List([nnx.LayerNorm(d, rngs=rngs) for d in dims])
    if normalize == "instance_norm":
        return nnx.List(
            [nnx.LayerNorm(d, use_scale=False, use_bias=False, rngs=rngs) for d in dims]
        )
    raise ValueError(f"Unknown normalization '{normalize}'. Choose from: {_VALID_NORMALIZATIONS}")


def validate_encoder_inputs(encoder_inputs: tuple[str, ...]) -> tuple[str, ...]:
    """Validate encoder inputs and return them in canonical order."""
    unknown = set(encoder_inputs) - set(_CANONICAL_ENCODER_INPUTS)
    if unknown:
        raise ValueError(
            f"Unknown encoder input(s) {sorted(unknown)}. Choose from: {_CANONICAL_ENCODER_INPUTS}"
        )
    if not encoder_inputs:
        raise ValueError(
            f"encoder_inputs must include at least one of: {_CANONICAL_ENCODER_INPUTS}"
        )
    return tuple(name for name in _CANONICAL_ENCODER_INPUTS if name in encoder_inputs)


def encoder_embed_dim(arch) -> int:
    """Return the total embedding width for the active encoder inputs."""
    inputs = validate_encoder_inputs(arch.encoder_inputs)
    return sum(getattr(arch, f"{name}_embed_dim") for name in inputs)


def validate_policy_inputs(policy_inputs: tuple[str, ...]) -> tuple[str, ...]:
    """Validate policy inputs and return them in canonical order."""
    unknown = set(policy_inputs) - set(_CANONICAL_POLICY_INPUTS)
    if unknown:
        raise ValueError(
            f"Unknown policy input(s) {sorted(unknown)}. Choose from: {_CANONICAL_POLICY_INPUTS}"
        )
    missing = {"state", "z"} - set(policy_inputs)
    if missing:
        raise ValueError(f"policy_inputs must include {sorted(missing)}")
    return tuple(name for name in _CANONICAL_POLICY_INPUTS if name in policy_inputs)
