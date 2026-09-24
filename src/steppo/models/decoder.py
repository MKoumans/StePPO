"""Neural decoders used by the belief-model backbones."""

from typing import Optional

import flax.nnx as nnx
import jax.numpy as jnp
from jax import Array

from steppo.configs.base_config import DecoderArchConfig
from steppo.models.utils import build_mlp, get_activation, make_norm_layers


class RewardDecoder(nnx.Module):
    """Reconstruct reward from (z, s_t, a_t)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        arch: Optional[DecoderArchConfig] = None,
        rngs: nnx.Rngs = None,
        heteroscedastic: bool = False,
    ):
        """Initialize a decoder for scalar reward reconstruction."""
        if arch is None:
            arch = DecoderArchConfig()
        self._act = get_activation(arch.activation)
        self.heteroscedastic = heteroscedastic

        in_dim = latent_dim + state_dim + action_dim
        self.layers = build_mlp(in_dim, arch.hidden_dims, rngs)
        self.norms = make_norm_layers(arch.normalize, arch.hidden_dims, rngs)
        out_dim = 2 if heteroscedastic else 1

        self.fc_out = nnx.Linear(arch.hidden_dims[-1], out_dim, rngs=rngs)

    def __call__(
        self,
        z: Array,
        state: Array,
        action: Array,
    ):
        """Predict reward, or reward mean and log-variance when enabled."""
        x = jnp.concatenate([z, state, action], axis=-1)
        for i, layer in enumerate(self.layers):
            x = self._act(layer(x))
            if self.norms is not None:
                x = self.norms[i](x)
        out = self.fc_out(x)

        if self.heteroscedastic:
            return out[..., 0], out[..., 1]

        return out.squeeze(-1)


class StateDecoder(nnx.Module):
    """Reconstruct next state from (z, s_t, a_t)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        arch: Optional[DecoderArchConfig] = None,
        rngs: nnx.Rngs = None,
        heteroscedastic: bool = False,
    ):
        """Initialize a decoder for next-state reconstruction."""
        if arch is None:
            arch = DecoderArchConfig()
        self._act = get_activation(arch.activation)
        self.heteroscedastic = heteroscedastic
        self._state_dim = state_dim

        in_dim = latent_dim + state_dim + action_dim
        self.layers = build_mlp(in_dim, arch.hidden_dims, rngs)
        self.norms = make_norm_layers(arch.normalize, arch.hidden_dims, rngs)
        out_dim = 2 * state_dim if heteroscedastic else state_dim
        self.fc_out = nnx.Linear(arch.hidden_dims[-1], out_dim, rngs=rngs)

    def __call__(self, z: Array, state: Array, action: Array):
        """Predict the next state, optionally with per-dimension log-variance."""
        x = jnp.concatenate([z, state, action], axis=-1)
        for i, layer in enumerate(self.layers):
            x = self._act(layer(x))
            if self.norms is not None:
                x = self.norms[i](x)
        out = self.fc_out(x)
        if self.heteroscedastic:
            return out[..., : self._state_dim], out[..., self._state_dim :]
        return out


class StepAcceptDecoder(nnx.Module):
    """Predict step acceptance from (z, s_t, a_t)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        arch: Optional[DecoderArchConfig] = None,
        rngs: nnx.Rngs = None,
    ):
        """Initialize a decoder for solver-step acceptance."""
        if arch is None:
            arch = DecoderArchConfig()
        self._act = get_activation(arch.activation)

        in_dim = latent_dim + state_dim + action_dim
        self.layers = build_mlp(in_dim, arch.hidden_dims, rngs)
        self.norms = make_norm_layers(arch.normalize, arch.hidden_dims, rngs)
        self.fc_out = nnx.Linear(arch.hidden_dims[-1], 1, rngs=rngs)

    def __call__(self, z: Array, state: Array, action: Array) -> Array:
        """Return the acceptance logit for the proposed solver step."""
        x = jnp.concatenate([z, state, action], axis=-1)
        for i, layer in enumerate(self.layers):
            x = self._act(layer(x))
            if self.norms is not None:
                x = self.norms[i](x)
        return self.fc_out(x).squeeze(-1)


class TaskDecoder(nnx.Module):
    """Reconstruct task parameters from z."""

    def __init__(
        self,
        latent_dim: int,
        task_dim: int,
        arch: Optional[DecoderArchConfig] = None,
        rngs: nnx.Rngs = None,
        heteroscedastic: bool = False,
    ):
        """Initialize a decoder for task-parameter reconstruction."""
        if arch is None:
            arch = DecoderArchConfig()
        self._act = get_activation(arch.activation)
        self.heteroscedastic = heteroscedastic
        self._task_dim = task_dim

        self.layers = build_mlp(latent_dim, arch.hidden_dims, rngs)
        self.norms = make_norm_layers(arch.normalize, arch.hidden_dims, rngs)
        out_dim = 2 * task_dim if heteroscedastic else task_dim
        self.fc_out = nnx.Linear(arch.hidden_dims[-1], out_dim, rngs=rngs)

    def __call__(self, z: Array) -> Array:
        """Predict task parameters, optionally with their log-variances."""
        x = z
        for i, layer in enumerate(self.layers):
            x = self._act(layer(x))
            if self.norms is not None:
                x = self.norms[i](x)
        out = self.fc_out(x)
        if self.heteroscedastic:
            return out[..., : self._task_dim], out[..., self._task_dim :]
        return out
