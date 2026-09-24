"""Trajectory encoders that infer a latent belief over hidden tasks."""

from typing import Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.configs.base_config import EncoderArchConfig
from steppo.models.utils import (
    build_mlp,
    encoder_embed_dim,
    get_activation,
    make_norm_layers,
    validate_encoder_inputs,
)

_LOG_VAR_MIN = -10.0
_LOG_VAR_MAX = 10.0


class _BeliefEncoderMixin:
    """Shared mu/logvar head + embedding logic for trajectory encoders."""

    @staticmethod
    def _bound_logvar(raw: Array) -> Array:
        """Bound predicted log-variances to the numerically safe range."""
        return _LOG_VAR_MIN + 0.5 * (_LOG_VAR_MAX - _LOG_VAR_MIN) * (jnp.tanh(raw) + 1.0)

    def _heads(self, hidden: Array) -> tuple[Array, Array]:
        """Apply both latent heads, concatenating [short, long]."""
        mu = self.fc_mu(hidden)
        logvar = (
            jnp.zeros_like(mu) if self.deterministic else self._bound_logvar(self.fc_logvar(hidden))
        )
        if self.fc_mu_long is not None:
            mu_long = self.fc_mu_long(hidden)
            logvar_long = (
                jnp.zeros_like(mu_long)
                if self.deterministic
                else self._bound_logvar(self.fc_logvar_long(hidden))
            )
            mu = jnp.concatenate([mu, mu_long], axis=-1)
            logvar = jnp.concatenate([logvar, logvar_long], axis=-1)
        return mu, logvar

    def _embed(
        self, action: Array, state: Array, reward: Array, task: Optional[Array] = None
    ) -> Array:
        """Embed inputs configured in arch.encoder_inputs."""
        parts = []
        i = 0
        if self.action_embed is not None:
            ea = self._act(self.action_embed(action))
            parts.append(self.embed_norms[i](ea) if self.embed_norms is not None else ea)
            i += 1
        if self.state_embed is not None:
            es = self._act(self.state_embed(state))
            parts.append(self.embed_norms[i](es) if self.embed_norms is not None else es)
            i += 1
        if self.reward_embed is not None:
            reward_1d = reward.reshape(-1) if reward.ndim > 1 else reward.reshape(1)
            er = self._act(self.reward_embed(reward_1d))
            parts.append(self.embed_norms[i](er) if self.embed_norms is not None else er)
            i += 1
        if self.task_embed is not None:
            et = self._act(self.task_embed(task))
            parts.append(self.embed_norms[i](et) if self.embed_norms is not None else et)
            i += 1
        return jnp.concatenate(parts, axis=-1)

    def init_hidden(self, batch_shape: tuple = ()) -> Array:
        """Initial recurrent carry, optionally batched (e.g. `(num_envs,)`)."""
        return jnp.zeros(batch_shape + (self.hidden_size,), dtype=jnp.float32)

    def prior(self) -> tuple[Array, Array]:
        """Prior: pass zeros hidden state through output heads."""
        h0 = jnp.zeros(self.hidden_size, dtype=jnp.float32)
        return self._heads(h0)

    def sample(self, mu: Array, logvar: Array, key: PRNGKey) -> Array:
        """Reparameterised sample z ~ N(mu, sigma^2)."""
        logvar = jnp.clip(logvar, _LOG_VAR_MIN, _LOG_VAR_MAX)
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, shape=mu.shape)
        return mu + std * eps

    def encode_trajectory(
        self,
        actions: Array,
        states: Array,
        rewards: Array,
        task: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        """Encode a trajectory; the leading output row is the prior."""
        rewards = rewards.reshape(rewards.shape[0], 1) if rewards.ndim == 1 else rewards

        h0 = jnp.zeros(self.hidden_size, dtype=jnp.float32)
        prior_mu, prior_logvar = self.prior()

        def scan_fn(hidden, inputs):
            action, state, reward = inputs
            mu, logvar, new_hidden = self.encode_step(action, state, reward, hidden, task)
            return new_hidden, (mu, logvar)

        _, (mus, logvars) = jax.lax.scan(scan_fn, h0, (actions, states, rewards))

        mus = jnp.concatenate([prior_mu[None], mus], axis=0)
        logvars = jnp.concatenate([prior_logvar[None], logvars], axis=0)
        return mus, logvars


class ZeroEncoder(nnx.Module):
    """Stateless encoder with fixed zero-information latent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        arch: EncoderArchConfig,
        rngs: nnx.Rngs,
        latent_dim_long: int = 0,
        deterministic: bool = False,
        task_dim: int = 0,
    ):
        del state_dim, action_dim, arch, rngs, deterministic, task_dim
        self.latent_dim = latent_dim
        self.latent_dim_long = latent_dim_long
        self.total_latent_dim = latent_dim + latent_dim_long
        self.hidden_size = 0

    def init_hidden(self, batch_shape: tuple = ()) -> Array:
        return jnp.zeros(batch_shape + (self.hidden_size,), dtype=jnp.float32)

    def prior(self) -> tuple[Array, Array]:
        zeros = jnp.zeros((self.total_latent_dim,), dtype=jnp.float32)
        return zeros, jnp.zeros_like(zeros)

    def encode_step(
        self,
        action: Array,
        state: Array,
        reward: Array,
        hidden: Array,
        task: Optional[Array] = None,
    ) -> tuple[Array, Array, Array]:
        """Return the zero belief for one step without changing ``hidden``."""
        del action, state, reward, task
        mu, logvar = self.prior()
        return mu, logvar, hidden

    def encode_trajectory(
        self,
        actions: Array,
        states: Array,
        rewards: Array,
        task: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        """Return zero beliefs for a trajectory, including its prior."""
        del states, rewards, task
        T = actions.shape[0]
        zeros = jnp.zeros((T + 1, self.total_latent_dim), dtype=jnp.float32)
        return zeros, jnp.zeros_like(zeros)

    def sample(self, mu: Array, logvar: Array, key: PRNGKey) -> Array:
        """Return the deterministic zero latent with the shape of ``mu``."""
        del logvar, key
        return jnp.zeros_like(mu)


def _init_embed_and_heads(
    encoder: nnx.Module,
    state_dim: int,
    action_dim: int,
    latent_dim: int,
    latent_dim_long: int,
    hidden_size: int,
    arch: EncoderArchConfig,
    rngs: nnx.Rngs,
    task_dim: int = 0,
) -> None:
    """Attach input embeddings and latent heads to an encoder module."""
    inputs = validate_encoder_inputs(arch.encoder_inputs)
    if "task" in inputs and task_dim == 0:
        raise ValueError("encoder_inputs includes 'task' but task_dim is 0")
    encoder.action_embed = (
        nnx.Linear(action_dim, arch.action_embed_dim, rngs=rngs) if "action" in inputs else None
    )
    encoder.state_embed = (
        nnx.Linear(state_dim, arch.state_embed_dim, rngs=rngs) if "state" in inputs else None
    )
    encoder.reward_embed = (
        nnx.Linear(1, arch.reward_embed_dim, rngs=rngs) if "reward" in inputs else None
    )
    encoder.task_embed = (
        nnx.Linear(task_dim, arch.task_embed_dim, rngs=rngs) if "task" in inputs else None
    )
    encoder.embed_norms = make_norm_layers(
        arch.normalize,
        tuple(getattr(arch, f"{name}_embed_dim") for name in inputs),
        rngs,
    )

    encoder.fc_mu = nnx.Linear(hidden_size, latent_dim, rngs=rngs)
    encoder.fc_logvar = nnx.Linear(hidden_size, latent_dim, rngs=rngs)

    if latent_dim_long > 0:
        encoder.fc_mu_long = nnx.Linear(hidden_size, latent_dim_long, rngs=rngs)
        encoder.fc_logvar_long = nnx.Linear(hidden_size, latent_dim_long, rngs=rngs)
    else:
        encoder.fc_mu_long = None
        encoder.fc_logvar_long = None


class LSTMEncoder(_BeliefEncoderMixin, nnx.Module):
    """LSTM-based trajectory encoder."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        arch: EncoderArchConfig,
        rngs: nnx.Rngs,
        latent_dim_long: int = 0,
        deterministic: bool = False,
        task_dim: int = 0,
    ):
        """Initialize an LSTM-style recurrent encoder."""
        self.latent_dim = latent_dim
        self.latent_dim_long = latent_dim_long
        self.deterministic = deterministic
        self.total_latent_dim = latent_dim + latent_dim_long
        self.hidden_size = arch.hidden_size
        self._act = get_activation(arch.activation)

        _init_embed_and_heads(
            self,
            state_dim,
            action_dim,
            latent_dim,
            latent_dim_long,
            arch.hidden_size,
            arch,
            rngs,
            task_dim=task_dim,
        )
        self.lstm_cell = nnx.GRUCell(encoder_embed_dim(arch), arch.hidden_size, rngs=rngs)

    def encode_step(
        self,
        action: Array,
        state: Array,
        reward: Array,
        hidden: Array,
        task: Optional[Array] = None,
    ) -> tuple[Array, Array, Array]:
        """Encode one transition and return its posterior plus new carry."""
        x = self._embed(action, state, reward, task)
        new_hidden, _ = self.lstm_cell(hidden, x)
        mu, logvar = self._heads(new_hidden)
        return mu, logvar, new_hidden


class GRUEncoder(_BeliefEncoderMixin, nnx.Module):
    """GRU-based trajectory encoder."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        arch: EncoderArchConfig,
        rngs: nnx.Rngs,
        latent_dim_long: int = 0,
        deterministic: bool = False,
        task_dim: int = 0,
    ):
        """Initialize a GRU-based recurrent encoder."""
        self.latent_dim = latent_dim
        self.latent_dim_long = latent_dim_long
        self.deterministic = deterministic
        self.total_latent_dim = latent_dim + latent_dim_long
        self.hidden_size = arch.hidden_size
        self._act = get_activation(arch.activation)

        _init_embed_and_heads(
            self,
            state_dim,
            action_dim,
            latent_dim,
            latent_dim_long,
            arch.hidden_size,
            arch,
            rngs,
            task_dim=task_dim,
        )
        self.gru_cell = nnx.GRUCell(encoder_embed_dim(arch), arch.hidden_size, rngs=rngs)

    def encode_step(
        self,
        action: Array,
        state: Array,
        reward: Array,
        hidden: Array,
        task: Optional[Array] = None,
    ) -> tuple[Array, Array, Array]:
        """Encode one transition and return its posterior plus new carry."""
        x = self._embed(action, state, reward, task)
        new_hidden, _ = self.gru_cell(hidden, x)
        mu, logvar = self._heads(new_hidden)
        return mu, logvar, new_hidden


class LinearEncoder(_BeliefEncoderMixin, nnx.Module):
    """Memoryless trajectory encoder; ablation variant without recurrence."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        arch: EncoderArchConfig,
        rngs: nnx.Rngs,
        latent_dim_long: int = 0,
        deterministic: bool = False,
        task_dim: int = 0,
    ):
        """Initialize a memoryless per-transition encoder."""
        self.latent_dim = latent_dim
        self.latent_dim_long = latent_dim_long
        self.deterministic = deterministic
        self.total_latent_dim = latent_dim + latent_dim_long
        self._act = get_activation(arch.activation)

        embed_dim = encoder_embed_dim(arch)
        hidden_dims = arch.linear_hidden_dims
        self.layers = build_mlp(embed_dim, hidden_dims, rngs)
        self.norms = make_norm_layers(arch.normalize, hidden_dims, rngs)
        self.head_hidden_size = hidden_dims[-1]
        self.hidden_size = 0

        _init_embed_and_heads(
            self,
            state_dim,
            action_dim,
            latent_dim,
            latent_dim_long,
            self.head_hidden_size,
            arch,
            rngs,
            task_dim=task_dim,
        )

    def init_hidden(self, batch_shape: tuple = ()) -> Array:
        """Return the empty carry used by the memoryless encoder."""
        return jnp.zeros(batch_shape + (0,), dtype=jnp.float32)

    def prior(self) -> tuple[Array, Array]:
        """Return the prior produced by the encoder's output heads."""
        return self._heads(jnp.zeros(self.head_hidden_size, dtype=jnp.float32))

    def _encode_single(
        self, action: Array, state: Array, reward: Array, task: Optional[Array] = None
    ) -> tuple[Array, Array]:
        h = self._embed(action, state, reward, task)
        for i, layer in enumerate(self.layers):
            h = self._act(layer(h))
            if self.norms is not None:
                h = self.norms[i](h)
        return self._heads(h)

    def encode_step(
        self,
        action: Array,
        state: Array,
        reward: Array,
        hidden: Array,
        task: Optional[Array] = None,
    ) -> tuple[Array, Array, Array]:
        """Encode one transition and preserve the empty carry."""
        mu, logvar = self._encode_single(action, state, reward, task)
        return mu, logvar, hidden

    def encode_trajectory(
        self,
        actions: Array,
        states: Array,
        rewards: Array,
        task: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        """Encode each transition independently; the leading row is the prior."""
        rewards = rewards.reshape(rewards.shape[0], 1) if rewards.ndim == 1 else rewards

        prior_mu, prior_logvar = self.prior()
        mus, logvars = jax.vmap(self._encode_single, in_axes=(0, 0, 0, None))(
            actions, states, rewards, task
        )

        mus = jnp.concatenate([prior_mu[None], mus], axis=0)
        logvars = jnp.concatenate([prior_logvar[None], logvars], axis=0)
        return mus, logvars
