"""Model building and checkpoint loading utilities."""

import glob
import os

import flax.nnx as nnx
import jax

from steppo.configs.base_config import TrainConfig
from steppo.models.backbones import get_backbone
from steppo.training.policy_algos import get_policy_algo


def build_models(config: TrainConfig, env, seed: int):
    """Construct initialized belief and policy models from a config."""
    obs_dim = env.obs_shape()[0]
    action_dim = env.num_actions
    rngs = nnx.Rngs(seed)

    spec = get_backbone(config.backbone)
    model_config = getattr(config, spec.config_attr)
    needs_task_dim = (
        model_config.task_loss_coeff > 0
        or "task" in model_config.encoder.encoder_inputs
        or "task" in config.ppo.policy.policy_inputs
    )
    task_dim = getattr(env, "task_dim", 0) if needs_task_dim else None

    vae = spec.model_cls(obs_dim, action_dim, model_config, rngs, task_dim=task_dim)

    algo_spec = get_policy_algo(config.algo)
    policy = algo_spec.algo_cls.build_model(
        obs_dim,
        action_dim,
        model_config.total_latent_dim,
        task_dim or 0,
        rngs=nnx.Rngs(seed + 1),
        cfg=config,
        use_latent_sample=spec.use_latent_sample,
    )
    return vae, policy


def resolve_checkpoint(path: str) -> str:
    """Resolve a checkpoint step dir, or a run dir to its latest checkpoint_N step."""
    path = os.path.abspath(path)
    if os.path.basename(path).startswith("checkpoint_"):
        return path
    steps = sorted(
        glob.glob(os.path.join(path, "checkpoint_*")),
        key=lambda p: int(p.rsplit("_", 1)[-1]),
    )
    if not steps:
        raise FileNotFoundError(f"No checkpoint_* steps found in {path}")
    return steps[-1]


def _restore_raw_checkpoint(checkpointer, checkpoint_dir: str):
    """Restore raw arrays, targeting a single CPU device when running without a GPU.

    Orbax rejects GPU-sharded checkpoints on CPU without an explicit target.
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if jax.default_backend() != "cpu":
        return checkpointer.restore(checkpoint_dir, strict=False)

    metadata = checkpointer.metadata(checkpoint_dir).item_metadata.tree
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    target = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
        metadata,
    )
    return checkpointer.restore(checkpoint_dir, target=target)


def load_checkpoint(vae, policy, checkpoint_dir: str, backbone: str = "varibad", algo: str = "ppo"):
    """Load VAE and policy weights from an Orbax checkpoint into freshly built models.

    `backbone` and `algo` select the checkpoint key prefixes ("vae_params",
    "<algo>_params", falling back to "ppo_params").
    """
    import orbax.checkpoint as ocp

    spec = get_backbone(backbone)
    belief_key = f"{spec.checkpoint_prefix}_params"

    algo_spec = get_policy_algo(algo)
    policy_key = f"{algo_spec.checkpoint_prefix}_params"

    vae_graphdef, vae_params = nnx.split(vae)
    policy_graphdef, policy_params = nnx.split(policy)

    ckptr = ocp.StandardCheckpointer()
    raw = _restore_raw_checkpoint(ckptr, checkpoint_dir)

    _, vae_treedef = jax.tree.flatten(vae_params)
    _, policy_treedef = jax.tree.flatten(policy_params)

    vae_leaves = jax.tree.leaves(raw[belief_key])
    if policy_key in raw:
        policy_raw = raw[policy_key]
    elif "ppo_params" in raw:
        policy_raw = raw["ppo_params"]
    else:
        raise KeyError(
            f"Checkpoint {checkpoint_dir!r} has no policy weights under "
            f"{policy_key!r} or the legacy 'ppo_params' key."
        )
    policy_leaves = jax.tree.leaves(policy_raw)

    restored_vae = jax.tree.unflatten(vae_treedef, vae_leaves)
    restored_policy = jax.tree.unflatten(policy_treedef, policy_leaves)

    vae = nnx.merge(vae_graphdef, restored_vae)
    policy = nnx.merge(policy_graphdef, restored_policy)
    print(f"[+] Loaded checkpoint: {checkpoint_dir}")
    return vae, policy


def load_control_envelopes(checkpoint_dir: str) -> dict[str, float]:
    """Return the saved PID-fallback control limits, or {} if the checkpoint has none."""
    import orbax.checkpoint as ocp

    ckptr = ocp.StandardCheckpointer()
    raw = _restore_raw_checkpoint(ckptr, checkpoint_dir)
    envelopes = raw.get("control_envelopes")
    return {k: float(v) for k, v in envelopes.items()} if envelopes else {}
