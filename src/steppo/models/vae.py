"""VariBAD variational belief model and its decoder collection."""

from typing import Optional

import flax.nnx as nnx
from jax import Array
from jax.random import PRNGKey

from steppo.configs.base_config import VAEConfig
from steppo.models.decoder import RewardDecoder, StateDecoder, StepAcceptDecoder, TaskDecoder
from steppo.models.encoder import GRUEncoder, LinearEncoder, LSTMEncoder, ZeroEncoder


class VariBADVAE(nnx.Module):
    """Container: LSTMEncoder + RewardDecoder + optional StateDecoder/TaskDecoder."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: VAEConfig,
        rngs: nnx.Rngs,
        task_dim: Optional[int] = None,
    ):
        """Initialize the encoder and configured reconstruction decoders."""
        self.config = config
        if config.encoder.encoder_type == "linear":
            encoder_cls = LinearEncoder
        elif config.encoder.encoder_type == "zero":
            encoder_cls = ZeroEncoder
        elif config.encoder.encoder_type == "gru":
            encoder_cls = GRUEncoder
        elif config.encoder.encoder_type == "lstm":
            encoder_cls = LSTMEncoder
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder.encoder_type}")
        self.encoder = encoder_cls(
            state_dim,
            action_dim,
            config.latent_dim,
            config.encoder,
            rngs,
            latent_dim_long=config.latent_dim_long,
            deterministic=config.deterministic_latent,
            task_dim=task_dim or 0,
        )
        # Task-identity decoders read the long latent when it exists, else the full z.
        long_latent_dim = (
            config.latent_dim_long if config.latent_dim_long > 0 else config.latent_dim
        )

        self.reward_decoder = RewardDecoder(
            state_dim,
            action_dim,
            config.latent_dim,
            arch=config.decoder,
            rngs=rngs,
            heteroscedastic=config.heteroscedastic_recon,
        )
        self.state_decoder = (
            StateDecoder(
                state_dim,
                action_dim,
                config.latent_dim,
                arch=config.decoder,
                rngs=rngs,
                heteroscedastic=config.heteroscedastic_recon,
            )
            if config.state_loss_coeff > 0
            else None
        )
        self.task_decoder = (
            TaskDecoder(
                long_latent_dim,
                task_dim,
                arch=config.decoder,
                rngs=rngs,
                heteroscedastic=config.heteroscedastic_recon,
            )
            if (config.task_loss_coeff > 0 and task_dim is not None and task_dim > 0)
            else None
        )
        self.accept_decoder = (
            StepAcceptDecoder(
                state_dim, action_dim, config.latent_dim, arch=config.decoder, rngs=rngs
            )
            if config.accept_loss_coeff > 0
            else None
        )
        self.end_reward_decoder = (
            TaskDecoder(long_latent_dim, 1, arch=config.decoder, rngs=rngs)
            if config.end_reward_loss_coeff > 0
            else None
        )

    def get_prior(self) -> tuple[Array, Array]:
        """Return the encoder prior as ``(mu, logvar)``."""
        return self.encoder.prior()

    def encode(
        self,
        actions: Array,  # (T, action_dim)
        states: Array,  # (T, state_dim)
        rewards: Array,  # (T, 1)
        task_params: Optional[Array] = None,  # (task_dim,) — constant for the whole trajectory
    ) -> tuple[Array, Array]:
        """Encode a trajectory and include the prior at time index zero."""
        return self.encoder.encode_trajectory(actions, states, rewards, task_params)

    def decode(
        self,
        z: Array,
        states: Array,
        actions: Array,
    ) -> dict[str, Optional[Array]]:
        """Decode latent samples into the enabled reconstruction targets.

        With latent_dim_long > 0, z = [z_short, z_long]: reward/state/accept decoders
        read z_short and task/end_reward decoders read z_long; otherwise all read z.
        """
        if self.config.latent_dim_long > 0:
            z_short = z[..., : self.config.latent_dim]
            z_long = z[..., self.config.latent_dim :]
        else:
            z_short = z
            z_long = z

        result: dict[str, Optional[Array]] = {
            "reward": None,
            "reward_logvar": None,
            "state": None,
            "state_logvar": None,
            "task": None,
            "task_logvar": None,
            "accept_logit": None,
            "end_reward": None,
        }
        if self.config.heteroscedastic_recon:
            result["reward"], result["reward_logvar"] = self.reward_decoder(
                z_short, states, actions
            )
        else:
            result["reward"] = self.reward_decoder(z_short, states, actions)
        if self.state_decoder is not None:
            if self.config.heteroscedastic_recon:
                result["state"], result["state_logvar"] = self.state_decoder(
                    z_short, states, actions
                )
            else:
                result["state"] = self.state_decoder(z_short, states, actions)
        if self.task_decoder is not None:
            if self.config.heteroscedastic_recon:
                result["task"], result["task_logvar"] = self.task_decoder(z_long)
            else:
                result["task"] = self.task_decoder(z_long)
        if self.accept_decoder is not None:
            result["accept_logit"] = self.accept_decoder(z_short, states, actions)
        if self.end_reward_decoder is not None:
            result["end_reward"] = self.end_reward_decoder(z_long)
        return result

    def sample_z(self, mu: Array, logvar: Array, key: PRNGKey) -> Array:
        """Sample latent beliefs using the encoder's reparameterization rule."""
        if self.config.deterministic_latent:
            return mu
        return self.encoder.sample(mu, logvar, key)
