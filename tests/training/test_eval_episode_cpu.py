"""CPU-only contract tests for evaluation accounting and episode resets."""

from types import SimpleNamespace

import flax.struct
import jax
import jax.numpy as jnp
import pytest

from steppo.training.eval import _dataset_ode_params, eval_episode, eval_episode_early_exit


@flax.struct.dataclass
class _State:
    t: jnp.ndarray


class _TinyEnv:
    def reset(self, key, params):
        del key, params
        return jnp.zeros((1,)), _State(t=jnp.array(0.0))

    def step(self, key, state, action, params):
        del key, action, params
        next_state = _State(t=state.t + 1.0)
        return (
            jnp.array([next_state.t]),
            next_state,
            jnp.array(1.0),
            jnp.array(True),
            {"keep_step": jnp.array(True)},
        )


class _TinyEncoder:
    @staticmethod
    def init_hidden():
        return jnp.array(0.0)

    @staticmethod
    def encode_step(action, obs, reward, hidden, task):
        del action, obs, task
        updated = hidden + reward[0]
        return updated[None], jnp.zeros((1,)), updated


class _TinyVAE:
    encoder = _TinyEncoder()
    task_decoder = None

    @staticmethod
    def get_prior():
        return jnp.zeros((1,)), jnp.zeros((1,))


class _TinyPolicy:
    use_latent_sample = False
    action_space = "continuous"
    action_dim = 1

    @staticmethod
    def act(obs, z, key, deterministic, task):
        del obs, z, key, deterministic, task
        return jnp.zeros((1,)), jnp.array(0.0), jnp.array(0.0)


@pytest.mark.parametrize(
    ("reset_belief", "expected_final_mu", "expected_trajectory"),
    [
        (False, 2.0, [0.0, 1.0, 2.0, 2.0, 2.0]),
        (True, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_eval_episode_counts_trial_episodes_and_optionally_resets_belief(
    reset_belief, expected_final_mu, expected_trajectory
):
    result = eval_episode(
        _TinyVAE(),
        _TinyPolicy(),
        _TinyEnv(),
        env_params=None,
        rng_key=jnp.array([0, 1], dtype=jnp.uint32),
        max_steps=4,
        episodes_per_trial=2,
        reset_belief_between_episodes=reset_belief,
        return_trajectories=True,
    )

    assert result["total_return"] == pytest.approx(2.0)
    assert result["episode_length"] == 2
    assert result["ep_t_reached"][:2].tolist() == [1.0, 1.0]
    assert result["num_episodes"] == 2
    assert result["accepted_steps"] == 2
    assert result["rejected_steps"] == 0
    assert result["final_mu"].tolist() == [expected_final_mu]
    assert result["belief_mu_trajectory"][:, 0].tolist() == expected_trajectory


def test_early_exit_evaluation_stops_at_the_first_terminal_step():
    result = eval_episode_early_exit(
        _TinyVAE(),
        _TinyPolicy(),
        _TinyEnv(),
        env_params=None,
        rng_key=jnp.array([4, 5], dtype=jnp.uint32),
        max_steps=4,
    )

    assert result["total_return"] == pytest.approx(1.0)
    assert result["episode_length"] == 1
    assert result["t_reached"] == pytest.approx(1.0)


def test_cached_ode_task_rebuilds_deterministic_pulse_phase_and_budget():
    key = jax.random.PRNGKey(21)

    params = _dataset_ode_params(SimpleNamespace(max_steps=37), lam=4.5, key=key)

    expected_phase = jax.random.uniform(jax.random.split(key)[1], shape=(), dtype=jnp.float32)
    assert params.lam == pytest.approx(4.5)
    assert params.pulse_phase == pytest.approx(float(expected_phase))
    assert params.max_steps == 37
