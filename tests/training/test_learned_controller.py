import jax
import jax.numpy as jnp

from steppo.configs.base_config import TrainConfig
from steppo.envs.ode import ODEEnv
from steppo.envs.ode.learned_controller import LearnedController
from steppo.training.pid_solve import solve_pid_batch
from steppo.utils.checkpoint import build_models


def test_task_conditioned_model_builds_a_learned_controller():
    config = TrainConfig()
    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, config.seed)

    controller = LearnedController.from_models(vae, policy, config, env)

    assert controller.encoder.task_embed is not None


def test_task_conditioned_model_runs_through_diffrax_controller():
    config = TrainConfig()
    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, config.seed)
    controller = LearnedController.from_models(vae, policy, config, env)

    result = solve_pid_batch(
        config.env,
        controller,
        jnp.asarray([1.0], dtype=jnp.float32),
        jax.random.split(jax.random.PRNGKey(7), 1),
        max_steps=8,
    )

    assert result["steps"].shape == (1,)


def test_zero_controller_specialization_matches_generic_decisions():
    """The fast path must preserve the old zero-encoder policy exactly."""
    config = TrainConfig()
    config.vae.encoder.encoder_type = "zero"
    config.rollout_steps = 64
    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, config.seed)
    fast = LearnedController.from_models(vae, policy, config, env)
    generic = LearnedController.from_models(
        vae, policy, config, env, specialize_zero_inference=False
    )
    mus = jnp.asarray([1.0], dtype=jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(11), 1)

    fast_result = solve_pid_batch(
        config.env, fast, mus, keys, max_steps=64, save_steps=True, batch_size=1
    )
    generic_result = solve_pid_batch(
        config.env, generic, mus, keys, max_steps=64, save_steps=True, batch_size=1
    )

    for name in ("accepted", "rejected", "ts"):
        assert jnp.array_equal(fast_result[name], generic_result[name])


def test_linear_encoder_has_an_empty_recurrent_carry():
    config = TrainConfig()
    config.vae.encoder.encoder_type = "linear"
    config.vae.encoder.linear_hidden_dims = (8, 4)
    env = ODEEnv(config.env, config.rollout_steps)
    vae, _ = build_models(config, env, config.seed)

    assert vae.encoder.init_hidden().shape == (0,)
    assert vae.encoder.init_hidden((3,)).shape == (3, 0)
