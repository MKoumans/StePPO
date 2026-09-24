"""Registry of belief-model backbones (VariBAD VAE, ...).

Adding a new backbone:
  1. Implement a model class exposing the same interface as VariBADVAE
     (get_prior, encode, decode, sample_z, plus an `.encoder` with prior/init_hidden/
     encode_step/encode_trajectory/sample).
  2. Implement a Trainer subclass (see src/steppo/training/base_trainer.py) with
     `_init_belief_model`, `_update_belief_model`, `_belief_checkpoint_fields`.
  3. Register both below.
"""

from dataclasses import dataclass


@dataclass
class BackboneSpec:
    """Registration record connecting a belief model to its trainer."""

    name: str
    model_cls: type  #
    trainer_cls: type  #
    config_attr: str  # TrainConfig field holding this backbone's model config (e.g. "vae")
    use_latent_sample: (
        bool  # policy consumes sampled/deterministic z (True) vs [mu, logvar] concat (False)
    )
    checkpoint_prefix: str = "vae"


def _load_backbones() -> dict[str, BackboneSpec]:
    from steppo.models.vae import VariBADVAE
    from steppo.training.trainer import VariBADTrainer

    return {
        "varibad": BackboneSpec(
            name="varibad",
            model_cls=VariBADVAE,
            trainer_cls=VariBADTrainer,
            config_attr="vae",
            use_latent_sample=False,
            checkpoint_prefix="vae",
        ),
    }


def get_backbone(name: str) -> BackboneSpec:
    """Return the registered backbone specification named by ``name``."""
    backbones = _load_backbones()
    if name not in backbones:
        raise ValueError(f"Unknown backbone '{name}'. Available: {sorted(backbones)}")
    return backbones[name]
