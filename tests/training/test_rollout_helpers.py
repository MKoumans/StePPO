"""CPU-sized regressions for rollout state transitions."""

import flax.struct
import jax
import jax.numpy as jnp

from steppo.training.rollout import (
    _broadcast_params,
    _build_z,
    _encode_action_for_encoder,
    _env_step_and_reset,
    _policy_step,
    _task_batch_for,
)


@flax.struct.dataclass
class _EnvState:
    steps: jnp.ndarray


class _TinyEnv:
    """A deterministic vectorizable environment with reset sentinels."""

    @staticmethod
    def step(key, state, action, params):
        del key, params
        next_state = _EnvState(steps=state.steps + 1)
        next_obs = jnp.stack([state.steps.astype(jnp.float32), action.astype(jnp.float32)])
        reward = action.astype(jnp.float32) + 2.0
        done = jnp.asarray(True)
        info = {"keep_step": (action > 0)}
        return next_obs, next_state, reward, done, info

    @staticmethod
    def reset(key, params):
        del key, params
        return jnp.array([-99.0, -99.0]), _EnvState(steps=jnp.array(0))


def test_broadcast_params_only_repeats_unbatched_parameter_trees():
    unbatched = {"scale": jnp.array(2.0), "offset": jnp.array([1.0, -1.0])}
    broadcast = _broadcast_params(unbatched, num_envs=3)
    already_batched = {"scale": jnp.array([1.0, 2.0, 3.0])}

    assert broadcast["scale"].tolist() == [2.0, 2.0, 2.0]
    assert broadcast["offset"].tolist() == [[1.0, -1.0]] * 3
    assert _broadcast_params(already_batched, num_envs=3) is already_batched


def test_latent_policy_input_supports_mean_logvar_and_reparameterized_sample():
    mu = jnp.array([[1.0, 2.0], [-1.0, 0.5]])
    logvar = jnp.array([[0.0, 1.0], [2.0, -1.0]])
    deterministic = _build_z(mu, logvar, use_latent_sample=False, vae_encoder=None, key=None)

    class _Encoder:
        @staticmethod
        def sample(one_mu, one_logvar, key):
            return one_mu + one_logvar + jax.random.uniform(key, one_mu.shape)

    sampled = _build_z(
        mu,
        logvar,
        use_latent_sample=True,
        vae_encoder=_Encoder(),
        key=jax.random.PRNGKey(3),
    )

    assert deterministic.shape == (2, 4)
    assert jnp.array_equal(deterministic, jnp.concatenate([mu, logvar], axis=-1))
    assert sampled.shape == mu.shape
    assert jnp.all(sampled >= mu + logvar)
    assert jnp.all(sampled < mu + logvar + 1.0)


def test_action_encoding_distinguishes_discrete_one_hot_from_continuous_cast():
    discrete = _encode_action_for_encoder(jnp.array([0, 2]), "discrete", action_dim=3)
    continuous = _encode_action_for_encoder(jnp.array([[0, 1]], dtype=jnp.int32), "continuous", 1)

    assert jnp.array_equal(discrete, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert continuous.dtype == jnp.float32
    assert jnp.array_equal(continuous, [[0.0, 1.0]])


def test_policy_step_clamps_nonfinite_log_probability_and_preserves_batches():
    class _Policy:
        @staticmethod
        def act(obs, z, key, task=None):
            del z, key, task
            return obs, jnp.array(jnp.nan), jnp.sum(obs)

    obs = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    actions, log_probs, values = _policy_step(
        _Policy(),
        obs,
        jnp.zeros((2, 1)),
        jax.random.PRNGKey(4),
        2,
        task=None,
    )

    assert jnp.array_equal(actions, obs)
    assert jnp.array_equal(log_probs, [-100.0, -100.0])
    assert jnp.array_equal(values, [3.0, 7.0])


def test_task_batch_uses_environment_task_params_or_empty_task_vectors():
    class _TaskEnv:
        @staticmethod
        def get_task_params(params):
            return jnp.array([params["mu"], params["sigma"]])

    params = {"mu": jnp.array([1.0, 2.0]), "sigma": jnp.array([0.1, 0.2])}

    assert jnp.array_equal(_task_batch_for(_TaskEnv(), params, 2), [[1.0, 0.1], [2.0, 0.2]])
    assert jnp.array_equal(_task_batch_for(object(), params, 2), jnp.zeros((2, 0)))


def test_trial_boundary_resets_intermediate_episode_and_freezes_completed_trials():
    obs = jnp.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    states = _EnvState(steps=jnp.array([4, 7, 9]))

    (
        final_obs,
        final_state,
        encoder_obs,
        rewards,
        dones,
        keep_steps,
        episode_count,
        trial_done,
        episode_dones,
    ) = _env_step_and_reset(
        _TinyEnv(),
        states,
        obs,
        actions=jnp.array([1.0, -1.0, 1.0]),
        params_batch=None,
        key_env=jnp.array([0, 1], dtype=jnp.uint32),
        key_reset=jnp.array([2, 3], dtype=jnp.uint32),
        num_envs=3,
        episode_count=jnp.array([0, 1, 2]),
        trial_done=jnp.array([False, False, True]),
        episodes_per_trial=2,
    )

    # Env 0 ends an episode but has more episodes remaining, so reset it.
    assert jnp.array_equal(final_obs[0], [-99.0, -99.0])
    assert final_state.steps[0] == 0
    # Env 1 completes the configured trial, so retain its terminal transition.
    assert jnp.array_equal(final_obs[1], [7.0, -1.0])
    assert final_state.steps[1] == 8
    # Env 2 was already frozen: preserve its carry and don't leak its step.
    assert jnp.array_equal(final_obs[2], obs[2])
    assert final_state.steps[2] == states.steps[2]

    assert jnp.array_equal(encoder_obs[:2], [[4.0, 1.0], [7.0, -1.0]])
    assert jnp.array_equal(encoder_obs[2], obs[2])
    assert jnp.array_equal(rewards, [[3.0], [1.0], [0.0]])
    assert jnp.array_equal(dones, [True, True, True])
    assert jnp.array_equal(keep_steps, [1.0, 0.0, 0.0])
    assert jnp.array_equal(episode_count, [1, 2, 3])
    assert jnp.array_equal(trial_done, [False, True, True])
    assert jnp.array_equal(episode_dones, [True, True, True])
