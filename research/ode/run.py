"""Entrypoint for ODE VariBAD experiments."""

from steppo.utils.device import setup_devices

setup_devices()

import argparse
import os

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from cli_config import parse_dotted_overrides

from steppo.configs.base_config import TrainConfig, apply_dotted_overrides, load_config_from_yaml
from steppo.envs.ode import ODEEnv, ODEParams
from steppo.models.backbones import get_backbone
from steppo.training.policy_algos import get_policy_algo


def parse_args():
    """Parse the training config path and CLI overrides."""
    parser = argparse.ArgumentParser(description="Run ODE VariBAD experiment")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Override env.system (e.g. scalar_decay, van_der_pol)",
    )
    parser.add_argument("--total_iters", type=int, default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (overrides config.seed)"
    )

    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--run_uid", type=str, default=None)
    parser.add_argument("--run_date", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--outputs_dir", type=str, default=None)
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Resume model/optimizer state from a checkpoint directory",
    )
    parser.add_argument(
        "--resume_replay_rollouts",
        type=int,
        default=4,
        help="Fresh rollouts used to refill the unsaved replay buffer",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=None,
        help="Episodes per trial (0 = unlimited, rollout_steps is the only limit)",
    )
    parser.add_argument(
        "--episode_max_steps",
        type=int,
        default=None,
        help="Max steps per episode (0 = use rollout_steps)",
    )
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_json", action="store_true")
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Free-text message saved to checkpoint_path/note.txt (e.g. 'run without warmstart')",
    )
    args, unknown = parser.parse_known_args()
    args.overrides = parse_dotted_overrides(unknown)
    return args


def create_config(config: str | None, args: argparse.Namespace) -> TrainConfig:
    """Load a training config and apply command-line overrides."""
    if config is None:
        config = TrainConfig()
    else:
        config = load_config_from_yaml(TrainConfig, config)

    # Apply CLI overrides (only if specified)
    if args.system is not None:
        config.env.system = args.system
    if args.total_iters is not None:
        config.total_iters = args.total_iters
    if args.seed is not None:
        config.seed = args.seed

    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.run_uid is not None:
        config.run_uid = args.run_uid
    if args.run_date is not None:
        config.run_date = args.run_date
    if args.checkpoints_dir is not None:
        config.checkpoints_dir = args.checkpoints_dir
    if args.outputs_dir is not None:
        config.outputs_dir = args.outputs_dir
    if args.episodes is not None:
        config.episodes_per_trial = args.episodes
    if args.episode_max_steps is not None:
        config.episode_max_steps = args.episode_max_steps
    if args.no_plots:
        config.save_plots = False
    if args.no_json:
        config.save_json = False
    if args.note is not None:
        config.note = args.note

    if args.overrides:
        config = apply_dotted_overrides(config, args.overrides)

    return config


def _init(config: TrainConfig):
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

    rng = jax.random.PRNGKey(config.seed)
    env = ODEEnv(config.env, config.rollout_steps)
    params = ODEParams(lam=jnp.zeros((1,), dtype=jnp.float32))

    obs_dim = env.obs_shape()[0]
    action_dim = env.num_actions

    rngs = nnx.Rngs(config.seed)

    return rng, env, params, obs_dim, action_dim, rngs


def main():
    """Build the configured models and run ODE training."""
    args = parse_args()
    config = create_config(args.config, args)

    rng, env, params, obs_dim, action_dim, rngs = _init(config)

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
        rngs=nnx.Rngs(config.seed + 1),
        cfg=config,
        use_latent_sample=spec.use_latent_sample,
    )

    print(
        f"[*] ODE system: {config.env.system}  obs_dim={obs_dim}  t_end={config.env.t_end}  "
        f"backbone={config.backbone}  algo={config.algo}"
    )

    trainer = spec.trainer_cls(config, vae, policy, env, params, policy_algo=algo_spec.algo_cls)
    metrics = trainer.train(
        rng,
        resume_checkpoint=args.resume_checkpoint,
        resume_replay_rollouts=args.resume_replay_rollouts,
    )

    eval_returns = {}
    for entry in reversed(metrics["metrics"]):
        eval_returns = {
            k: v
            for k, v in entry.items()
            if k == "eval/mean_return" or (k.startswith("eval/") and k.endswith("/mean_return"))
        }
        if eval_returns:
            break
    print(f"Final eval return: {eval_returns or 'N/A'}")


if __name__ == "__main__":
    main()
