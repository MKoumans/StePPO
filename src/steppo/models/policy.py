"""Actor-critic policies conditioned on observations and task beliefs."""

from typing import Optional, Tuple

import distrax
import flax.nnx as nnx
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from steppo.configs.base_config import PolicyArchConfig
from steppo.models.utils import build_mlp, get_activation, make_norm_layers, validate_policy_inputs


class ActorCritic(nnx.Module):
    """Actor-Critic policy conditioned on latent task belief."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        arch: Optional[PolicyArchConfig] = None,
        action_space: str = "discrete",
        use_latent_sample: bool = False,
        task_dim: int = 0,
        rngs: nnx.Rngs = None,
    ):
        """Initialize the actor and critic for the configured action space."""
        if arch is None:
            arch = PolicyArchConfig()

        self.action_space = action_space
        self.action_dim = action_dim
        self._act = get_activation(arch.activation)

        assert use_latent_sample is False, "Currently only use_latent_sample=False is supported."
        self.use_latent_sample = use_latent_sample

        self._use_task = "task" in validate_policy_inputs(arch.policy_inputs)
        if self._use_task and task_dim == 0:
            raise ValueError("policy_inputs includes 'task' but task_dim is 0")

        z_input_dim = latent_dim if use_latent_sample else latent_dim * 2
        in_dim = state_dim + z_input_dim + (task_dim if self._use_task else 0)

        self._build_neural_components(in_dim, arch, rngs)

    def _build_neural_components(self, in_dim: int, arch: PolicyArchConfig, rngs: nnx.Rngs) -> None:
        self.layers_critic = build_mlp(in_dim, arch.hidden_dims, rngs)
        self.layers_actor = build_mlp(in_dim, arch.hidden_dims, rngs)

        self.norms_actor = make_norm_layers(arch.normalize, arch.hidden_dims, rngs)
        self.norms_critic = make_norm_layers(arch.normalize, arch.hidden_dims, rngs)

        last_layer = arch.hidden_dims[-1] if arch.hidden_dims else in_dim

        self.critic_head = nnx.Linear(last_layer, 1, rngs=rngs)
        self.actor_head = nnx.Linear(
            last_layer,
            self.action_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(0.01),
        )

        if self.action_space == "continuous":
            self.log_std = nnx.Param(jnp.full(self.action_dim, 0.0))

    def _forward_actor(self, x: Array) -> Array:
        for i, layer in enumerate(self.layers_actor):
            x = self._act(layer(x))
            if self.norms_actor is not None:
                x = self.norms_actor[i](x)
        logits = self.actor_head(x)
        return logits

    def _forward_critic(self, x: Array) -> Array:
        for i, layer in enumerate(self.layers_critic):
            x = self._act(layer(x))
            if self.norms_critic is not None:
                x = self.norms_critic[i](x)
        value = self.critic_head(x).squeeze(-1)
        return value

    def __call__(
        self, state: Array, z: Array, task: Optional[Array] = None
    ) -> Tuple[distrax.Distribution, Array]:
        """Return the action distribution and scalar value estimate."""
        parts = [state, z, task] if self._use_task else [state, z]
        x = jnp.concatenate(parts, axis=-1)
        logits = self._forward_actor(x)
        value = self._forward_critic(x)

        if self.action_space == "discrete":
            dist = distrax.Categorical(logits=logits)
        else:
            std = jnp.exp(jnp.clip(self.log_std[...], -6.0, 2.0))
            base = distrax.MultivariateNormalDiag(loc=logits, scale_diag=std)
            dist = distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))

        return dist, value

    def act(
        self,
        state: Array,
        z: Array,
        key: PRNGKey,
        deterministic: bool = False,
        task: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array]:
        """Sample or select an action and return its log-probability and value."""
        dist, value = self(state, z, task)
        if deterministic:
            if self.action_space == "discrete":
                action = jnp.argmax(dist.logits, axis=-1)
                log_prob = dist.log_prob(action)
            else:
                action = jnp.tanh(dist.distribution.loc)
                log_prob = dist.log_prob(action)
        else:
            action = dist.sample(seed=key)
            log_prob = dist.log_prob(action)

        return action, log_prob, value

    def infer_action(self, state: Array, z: Array, task: Optional[Array] = None) -> Array:
        """Select the deterministic action used by inference-time control."""
        parts = [state, z, task] if self._use_task else [state, z]
        x = jnp.concatenate(parts, axis=-1)
        x = self._forward_actor(x)

        if self.action_space == "discrete":
            return jnp.argmax(x, axis=-1)
        return jnp.tanh(x)
