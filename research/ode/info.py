"""Print a summary of an ODE VariBAD experiment: observation layout, env/task
dimensions, and model parameter counts — without running any training.

Examples:
    python research/ode/info.py --config configs/envs/ode/chemical_cascade/chemical_cascade_default.yaml
    python research/ode/info.py --system van_der_pol
"""

from steppo.utils.device import setup_devices

setup_devices()

import argparse

import flax.nnx as nnx
import jax

from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv
from steppo.models.policy import ActorCritic
from steppo.models.vae import VariBADVAE


def parse_args():
    """Parse the config used for the structural experiment summary."""
    parser = argparse.ArgumentParser(description="Show ODE VariBAD env/model info")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Override env.system (e.g. scalar_decay, van_der_pol, chemical_cascade)",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> TrainConfig:
    """Load the selected config and apply command-line overrides."""
    config = (
        TrainConfig() if args.config is None else load_config_from_yaml(TrainConfig, args.config)
    )
    if args.system is not None:
        config.env.system = args.system
    if args.seed is not None:
        config.seed = args.seed
    return config


def param_count(module: nnx.Module) -> int:
    """Return the number of scalar parameters in an NNX module."""
    _, state = nnx.split(module)
    return sum(leaf.size for leaf in jax.tree.leaves(state))


def print_obs_layout(config: TrainConfig, env: ODEEnv) -> None:
    """Print observation-feature widths for the configured ODE environment."""
    feature_dims = env._spec.feature_dims or {}
    print(
        f"\n[env] system={config.env.system}  t_end={config.env.t_end}  "
        f"dt0={config.env.dt0}  precision={config.env.precision}"
    )
    print("[obs_features]")
    total = 0
    for feat in config.env.obs_features:
        dim = feature_dims.get(feat, 0)
        total += dim
        print(f"    {feat:<14} {dim:>4}D")
    print(f"    {'total':<14} {total:>4}D")
    print(f"[task] task_dim={env.task_dim}  action_dim={env.num_actions}")


def print_param_counts(config: TrainConfig, vae: VariBADVAE, policy: ActorCritic) -> None:
    """Print parameter counts for the belief model and policy."""
    print(
        "\n[vae] latent_dim=%d  latent_dim_long=%d  total_latent_dim=%d"
        % (
            config.vae.latent_dim,
            config.vae.latent_dim_long,
            config.vae.total_latent_dim,
        )
    )
    vae_total = 0
    for name in (
        "encoder",
        "reward_decoder",
        "state_decoder",
        "task_decoder",
        "accept_decoder",
        "end_reward_decoder",
    ):
        sub = getattr(vae, name)
        if sub is None:
            print(f"    {name:<16} (disabled)")
            continue
        n = param_count(sub)
        vae_total += n
        print(f"    {name:<16} {n:>10,}")
    print(f"    {'vae total':<16} {vae_total:>10,}")

    policy_n = param_count(policy)
    print(
        f"\n[policy] hidden_dims={config.ppo.policy.hidden_dims}  "
        f"input_dim={policy.layers_actor[0].in_features}"
    )
    print(f"    {'policy total':<16} {policy_n:>10,}")

    print(f"\n[total trainable params] {vae_total + policy_n:>10,}")


def main():
    """Print model and environment dimensions without running training."""
    args = parse_args()
    config = create_config(args)

    env = ODEEnv(config.env, config.rollout_steps)
    obs_dim = env.obs_shape()[0]
    action_dim = env.num_actions
    rngs = nnx.Rngs(config.seed)

    task_dim = env.task_dim if config.vae.task_loss_coeff > 0 else None
    vae = VariBADVAE(obs_dim, action_dim, config.vae, rngs, task_dim=task_dim)
    policy = ActorCritic(
        obs_dim,
        action_dim,
        config.vae.total_latent_dim,
        arch=config.ppo.policy,
        action_space="continuous",
        use_latent_sample=False,
        rngs=nnx.Rngs(config.seed + 1),
    )

    print_obs_layout(config, env)
    print_param_counts(config, vae, policy)

    print(
        f"\n[rollout] num_envs={config.num_envs}  rollout_steps={config.rollout_steps}  "
        f"episodes_per_trial={config.episodes_per_trial}  "
        f"episode_max_steps={config.episode_max_steps or config.rollout_steps}"
    )


if __name__ == "__main__":
    main()
