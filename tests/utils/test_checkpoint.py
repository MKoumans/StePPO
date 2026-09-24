"""Tests for steppo.utils.checkpoint."""

import os
import shutil
import tempfile

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest

from steppo.configs.base_config import TrainConfig, VAEConfig
from steppo.models.policy import ActorCritic
from steppo.models.vae import VariBADVAE
from steppo.training.policy_algos import PolicyAlgoSpec
from steppo.utils.checkpoint import (
    load_checkpoint,
    load_control_envelopes,
    resolve_checkpoint,
)


def _save_raw(ckpt_dir: str, state_dict: dict):
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(ckpt_dir, state_dict)
    ckptr.wait_until_finished()


def test_resolve_checkpoint_uses_numeric_iteration_order(tmp_path):
    run_dir = tmp_path / "run"
    for step in (2, 10, 3):
        (run_dir / f"checkpoint_{step}").mkdir(parents=True, exist_ok=True)

    assert resolve_checkpoint(str(run_dir)) == str(run_dir / "checkpoint_10")


def test_resolve_checkpoint_accepts_a_step_directory_and_rejects_empty_run(tmp_path):
    checkpoint = tmp_path / "checkpoint_4"
    checkpoint.mkdir()
    empty_run = tmp_path / "empty"
    empty_run.mkdir()

    assert resolve_checkpoint(str(checkpoint)) == str(checkpoint)
    with pytest.raises(FileNotFoundError, match=r"No checkpoint_\* steps"):
        resolve_checkpoint(str(empty_run))


def test_load_control_envelopes_round_trip():
    """load_control_envelopes (src/steppo/utils/checkpoint.py) must return exactly
    what Trainer.save_checkpoint wrote under 'control_envelopes'."""
    tmp_dir = tempfile.mkdtemp()
    try:
        ckpt_dir = os.path.join(tmp_dir, "checkpoint_0")
        envelopes = {"reject_streak": jnp.float32(12.5), "accept_ema": jnp.float32(0.72)}
        _save_raw(ckpt_dir, {"iteration": 0, "control_envelopes": envelopes})

        loaded = load_control_envelopes(ckpt_dir)
        assert loaded == pytest.approx({"reject_streak": 12.5, "accept_ema": 0.72})
        assert all(isinstance(v, float) for v in loaded.values())
    finally:
        shutil.rmtree(tmp_dir)


def test_load_control_envelopes_missing_key_returns_empty_dict():
    """A checkpoint saved without cfg.pid_fallback.enabled (including every
    checkpoint saved before this feature existed) must not raise — the
    fallback mechanism should just come up disabled."""
    tmp_dir = tempfile.mkdtemp()
    try:
        ckpt_dir = os.path.join(tmp_dir, "checkpoint_0")
        _save_raw(ckpt_dir, {"iteration": 0})  # no control_envelopes key at all

        loaded = load_control_envelopes(ckpt_dir)
        assert loaded == {}
    finally:
        shutil.rmtree(tmp_dir)


def test_load_checkpoint_falls_back_to_legacy_ppo_params_key(monkeypatch):
    """Checkpoints written before policy-algorithm checkpoint prefixes were
    configurable used the literal key "ppo_params". If a future policy
    algorithm's spec used a different checkpoint_prefix, load_checkpoint must
    still find such a checkpoint's policy weights under the legacy key."""
    from steppo.training import policy_algos as policy_algos_module

    STATE_DIM, ACTION_DIM, LATENT_DIM = 3, 2, 4
    cfg = TrainConfig(vae=VAEConfig(latent_dim=LATENT_DIM))
    vae = VariBADVAE(STATE_DIM, ACTION_DIM, cfg.vae, nnx.Rngs(0))
    policy = ActorCritic(
        STATE_DIM,
        ACTION_DIM,
        cfg.vae.total_latent_dim,
        action_space="continuous",
        use_latent_sample=False,
        rngs=nnx.Rngs(1),
    )
    _, vae_params = nnx.split(vae)
    _, policy_params = nnx.split(policy)

    tmp_dir = tempfile.mkdtemp()
    try:
        ckpt_dir = os.path.join(tmp_dir, "checkpoint_0")
        # Legacy shape: literal "ppo_params", no algo-specific prefix at all.
        _save_raw(
            ckpt_dir,
            {
                "vae_params": vae_params,
                "ppo_params": policy_params,
                "iteration": 0,
            },
        )

        # Simulate a future algo whose spec uses a different checkpoint_prefix
        # than "ppo" — load_checkpoint must still fall back to "ppo_params".
        fake_spec = PolicyAlgoSpec(
            name="future_algo",
            algo_cls=object,
            config_attr="ppo",
            checkpoint_prefix="future_algo",
        )
        monkeypatch.setattr(
            policy_algos_module, "_load_policy_algos", lambda: {"future_algo": fake_spec}
        )

        restored_vae, restored_policy = load_checkpoint(
            vae,
            policy,
            ckpt_dir,
            backbone="varibad",
            algo="future_algo",
        )

        _, restored_vae_params = nnx.split(restored_vae)
        _, restored_policy_params = nnx.split(restored_policy)
        assert jax.tree.structure(restored_vae_params) == jax.tree.structure(vae_params)
        assert jax.tree.structure(restored_policy_params) == jax.tree.structure(policy_params)
    finally:
        shutil.rmtree(tmp_dir)
