"""Opt-in end-to-end parity test for deployable ODE model artifacts.

Run this module with a real checkpoint, a writable Hugging Face repository, and
``-m pipeline``. The test deliberately uses one fixed task and reset key for
each path. Exact comparisons are made between checkpoint and downloaded models
within the same path; ``env.step`` and compiled ``diffeqsolve`` are separate
numerical implementations and are not expected to be bitwise identical.
"""

import os
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from steppo.configs.base_config import TrainConfig, load_config_from_yaml
from steppo.envs.ode import ODEEnv
from steppo.envs.ode.learned_controller import LearnedController
from steppo.models.huggingface import download_model, upload_model
from steppo.utils.checkpoint import build_models, load_checkpoint, resolve_checkpoint

pytestmark = pytest.mark.pipeline


def _checkpoint_config(checkpoint: Path) -> tuple[Path, Path]:
    resolved = Path(resolve_checkpoint(str(checkpoint)))
    for candidate in (checkpoint / "config.yaml", resolved.parent / "config.yaml"):
        if candidate.is_file():
            return resolved, candidate
    raise FileNotFoundError(f"Could not find config.yaml next to {resolved}")


def _load_checkpoint_bundle(checkpoint: Path):
    resolved, config_path = _checkpoint_config(checkpoint)
    config = load_config_from_yaml(TrainConfig, str(config_path))
    if config.pid_fallback.enabled:
        pytest.fail(
            "Exact checkpoint-to-Hub parity requires pid_fallback.enabled=false: "
            "control envelopes are intentionally not included in the current "
            "Hugging Face artifact format."
        )

    env = ODEEnv(config.env, config.rollout_steps)
    vae, policy = build_models(config, env, config.seed)
    vae, policy = load_checkpoint(vae, policy, str(resolved), backbone=config.backbone)
    controller = LearnedController.from_models(vae, policy, config, env)
    if controller.warp is not None:
        env.set_progress_warp(*controller.warp)
    return resolved, config, env, vae, policy, controller


def _raw_task(env: ODEEnv, params) -> jax.Array:
    return jnp.asarray([params.lam, params.pulse_phase], dtype=env.t0.dtype)


def _run_env_step_inference(vae, policy, env: ODEEnv, params, reset_key):
    """Run deterministic inference through the public env.step() boundary."""
    obs, state = env.reset(reset_key, params)
    task = env.get_task_params(params)
    belief_mu, belief_logvar = vae.get_prior()
    gru_hidden = vae.encoder.init_hidden()

    accepted_ts = []
    accepted_ys = []
    attempts = 0

    for _ in range(int(env.max_steps)):
        z = jnp.concatenate([belief_mu, belief_logvar], axis=-1)
        action = policy.infer_action(obs, z, task=task)
        obs_next, state_next, reward, done, info = env.step(
            jax.random.PRNGKey(0), state, action, params
        )

        attempts += 1
        if bool(np.asarray(info["keep_step"])):
            accepted_ts.append(np.asarray(state_next.t))
            accepted_ys.append(np.asarray(state_next.y))

        action_for_encoder = action.astype(jnp.float32)
        reward_for_encoder = jnp.reshape(reward, (1,)).astype(jnp.float32)
        belief_mu, belief_logvar, gru_hidden = vae.encoder.encode_step(
            action_for_encoder,
            obs_next,
            reward_for_encoder,
            gru_hidden,
            task,
        )
        obs, state = obs_next, state_next

        if bool(np.asarray(done)):
            break

    y_dim = np.asarray(state.y).shape[-1]
    accepted_ys = (
        np.stack(accepted_ys)
        if accepted_ys
        else np.empty((0, y_dim), dtype=np.asarray(state.y).dtype)
    )
    accepted_ts = np.asarray(accepted_ts, dtype=np.asarray(state.t).dtype)
    return {
        "ts": accepted_ts,
        "ys": accepted_ys,
        "accepted_steps": len(accepted_ts),
        "rejected_steps": attempts - len(accepted_ts),
        "t_reached": np.asarray(state.t),
        "final_y": np.asarray(state.y),
    }


def _run_controller_inference(controller, env: ODEEnv, params, y0):
    """Run the same task and initial state through diffrax LearnedController."""
    task = _raw_task(env, params)
    solution = diffrax.diffeqsolve(
        env.terms,
        env.solver,
        t0=env.t0,
        t1=env.t_end,
        dt0=env.dt0,
        y0=jnp.asarray(y0, dtype=env.t0.dtype),
        args=task,
        stepsize_controller=controller,
        max_steps=int(params.max_steps),
        saveat=diffrax.SaveAt(steps=True),
        throw=False,
    )

    ts = np.asarray(solution.ts)
    ys = np.asarray(solution.ys)
    valid = np.isfinite(ts)
    ts = ts[valid]
    ys = ys[valid]
    accepted_steps = int(np.asarray(solution.stats["num_accepted_steps"]))
    rejected_steps = int(np.asarray(solution.stats["num_rejected_steps"]))
    final_y = ys[-1] if len(ys) else np.asarray(y0)
    t_reached = ts[-1] if len(ts) else np.asarray(env.t0)
    return {
        "ts": ts,
        "ys": ys,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "t_reached": np.asarray(t_reached),
        "final_y": np.asarray(final_y),
    }


def _assert_exact_inference(left: dict, right: dict) -> None:
    np.testing.assert_array_equal(left["ts"], right["ts"])
    np.testing.assert_array_equal(left["ys"], right["ys"])
    assert left["accepted_steps"] == right["accepted_steps"]
    assert left["rejected_steps"] == right["rejected_steps"]
    np.testing.assert_array_equal(left["t_reached"], right["t_reached"])
    np.testing.assert_array_equal(left["final_y"], right["final_y"])


def test_checkpoint_and_hub_inference_are_reproducible(tmp_path):
    if os.environ.get("VARIBAD_HF_RUN_INTEGRATION") != "1":
        pytest.skip("set VARIBAD_HF_RUN_INTEGRATION=1 to run the Hub integration test")

    checkpoint_text = os.environ.get("VARIBAD_CHECKPOINT")
    repo_id = os.environ.get("VARIBAD_HF_REPO_ID")
    revision = os.environ.get("VARIBAD_HF_REVISION")
    if not checkpoint_text or not repo_id or not revision:
        pytest.skip(
            "set VARIBAD_CHECKPOINT, VARIBAD_HF_REPO_ID, and "
            "VARIBAD_HF_REVISION to run the Hub integration test"
        )

    checkpoint = Path(checkpoint_text)
    _, _, env, vae, policy, checkpoint_controller = _load_checkpoint_bundle(checkpoint)

    task_key = jax.random.PRNGKey(2026)
    reset_key = jax.random.PRNGKey(2027)
    params = env.sample_task(task_key)
    _, reset_state = env.reset(reset_key, params)

    checkpoint_env_output = _run_env_step_inference(vae, policy, env, params, reset_key)
    checkpoint_controller_output = _run_controller_inference(
        checkpoint_controller, env, params, reset_state.y
    )

    upload_model(checkpoint, repo_id, revision=revision)
    loaded = download_model(
        repo_id,
        revision=revision,
        local_dir=tmp_path / "downloaded-model",
        cache_dir=tmp_path / "hf-cache",
    )
    if loaded.controller.warp is not None:
        loaded.env.set_progress_warp(*loaded.controller.warp)

    downloaded_env_output = _run_env_step_inference(
        loaded.vae, loaded.policy, loaded.env, params, reset_key
    )
    downloaded_controller_output = _run_controller_inference(
        loaded.controller, loaded.env, params, reset_state.y
    )

    _assert_exact_inference(checkpoint_env_output, downloaded_env_output)
    _assert_exact_inference(checkpoint_controller_output, downloaded_controller_output)
