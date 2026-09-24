"""Registry of policy-update algorithms.

To add one, implement `PolicyAlgorithm` as a class of staticmethods (see
ppo_algorithm.PPOAlgorithm) and register it in `_load_policy_algos`.
"""

from dataclasses import dataclass


@dataclass
class PolicyAlgoSpec:
    """Registration record for a policy-update algorithm."""

    name: str
    algo_cls: type  # implements PolicyAlgorithm
    config_attr: str  # TrainConfig field holding this algo's config (e.g. "ppo")
    checkpoint_prefix: str = "policy"


class PolicyAlgorithm:
    """Interface of a policy-update algorithm; `Trainer.policy_algo` holds the class itself."""

    name: str = ""
    checkpoint_prefix: str = "policy"
    supports_warmstart: bool = False

    @staticmethod
    def build_model(
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        task_dim: int,
        rngs,
        cfg,
        use_latent_sample: bool,
    ):
        """Builds the policy model. The only contract `rollout.py`/`LearnedController`
        require of the returned model is the existing `.act(...)`/`.infer_action(...)`
        interface."""
        raise NotImplementedError

    @staticmethod
    def init_state(policy, cfg):
        """Returns (policy_state, policy_buffer) built from an already-constructed
        `policy` model (as returned by `build_model`)."""
        raise NotImplementedError

    @staticmethod
    def update(trainer, cfg, policy_state, policy_buffer, key, batch, iteration):
        """Update the policy on one rollout; return (new_policy_state, metrics).

        May use trainer.vae, .policy, .profiler, .timings, .env, ._reward_norm
        and ._adv_stats_buffer.
        """
        raise NotImplementedError

    @staticmethod
    def checkpoint_fields(policy_state) -> dict:
        """Returns this algorithm's entries for the checkpoint state dict, keyed
        under `f"{checkpoint_prefix}_..."`."""
        raise NotImplementedError


def _load_policy_algos() -> dict[str, PolicyAlgoSpec]:
    from steppo.training.ppo_algorithm import PPOAlgorithm

    return {
        "ppo": PolicyAlgoSpec(
            name="ppo",
            algo_cls=PPOAlgorithm,
            config_attr="ppo",
            checkpoint_prefix="ppo",
        ),
    }


def get_policy_algo(name: str) -> PolicyAlgoSpec:
    """Return the registered policy-algorithm specification named by ``name``."""
    algos = _load_policy_algos()
    if name not in algos:
        raise ValueError(f"Unknown policy algorithm '{name}'. Available: {sorted(algos)}")
    return algos[name]
