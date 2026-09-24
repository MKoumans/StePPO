"""Tests for the model architecture config system.

Covers: EncoderArchConfig, DecoderArchConfig, PolicyArchConfig,
nested YAML loading, activation/normalization variants, variable hidden_dims.
"""

import os
import tempfile

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

STATE_DIM = 2
ACTION_DIM = 4
LATENT_DIM = 5
T = 8


# ── Config dataclass defaults ─────────────────────────────────────────────────


def test_encoder_arch_defaults():
    from steppo.configs.base_config import EncoderArchConfig

    arch = EncoderArchConfig()
    assert arch.hidden_size == 64
    assert arch.activation == "relu"
    assert arch.normalize == "none"
    assert arch.state_embed_dim == 10
    assert arch.action_embed_dim == 10
    assert arch.reward_embed_dim == 5


def test_decoder_arch_defaults():
    from steppo.configs.base_config import DecoderArchConfig

    arch = DecoderArchConfig()
    assert arch.hidden_dims == (32, 32)
    assert arch.activation == "relu"
    assert arch.normalize == "none"


def test_policy_arch_defaults():
    from steppo.configs.base_config import PolicyArchConfig

    arch = PolicyArchConfig()
    assert arch.hidden_dims == (32, 32)
    assert arch.activation == "tanh"
    assert arch.normalize == "none"


def test_vae_config_nested_arch():
    from steppo.configs.base_config import DecoderArchConfig, EncoderArchConfig, VAEConfig

    cfg = VAEConfig(
        latent_dim=8,
        encoder=EncoderArchConfig(hidden_size=128, activation="gelu"),
        decoder=DecoderArchConfig(hidden_dims=(64, 64, 64)),
    )
    assert cfg.latent_dim == 8
    assert cfg.encoder.hidden_size == 128
    assert cfg.encoder.activation == "gelu"
    assert cfg.decoder.hidden_dims == (64, 64, 64)


def test_ppo_config_policy_arch():
    from steppo.configs.base_config import PolicyArchConfig, PPOConfig

    cfg = PPOConfig(policy=PolicyArchConfig(hidden_dims=(128,), activation="silu"))
    assert cfg.policy.hidden_dims == (128,)
    assert cfg.policy.activation == "silu"


# ── Config loading from dict ──────────────────────────────────────────────────


def test_load_nested_arch_from_dict():
    from steppo.configs.base_config import VAEConfig, load_config_from_dict

    data = {
        "latent_dim": 8,
        "encoder": {"hidden_size": 128, "activation": "gelu", "normalize": "layer_norm"},
        "decoder": {"hidden_dims": [64, 64, 64], "activation": "elu"},
        "kl_weight": 0.5,
    }
    cfg = load_config_from_dict(VAEConfig, data)
    assert cfg.latent_dim == 8
    assert cfg.encoder.hidden_size == 128
    assert cfg.encoder.activation == "gelu"
    assert cfg.encoder.normalize == "layer_norm"
    assert cfg.decoder.hidden_dims == (64, 64, 64)  # list → tuple
    assert cfg.decoder.activation == "elu"
    assert cfg.kl_weight == 0.5


def test_load_ppo_policy_from_dict():
    from steppo.configs.base_config import PPOConfig, load_config_from_dict

    data = {
        "lr": 3e-4,
        "policy": {"hidden_dims": [128, 128], "activation": "tanh", "normalize": "layer_norm"},
    }
    cfg = load_config_from_dict(PPOConfig, data)
    assert cfg.lr == 3e-4
    assert cfg.policy.hidden_dims == (128, 128)
    assert cfg.policy.normalize == "layer_norm"


def test_load_full_train_config_from_yaml():
    """model: ref is resolved and merged; fields come from the model YAML."""
    import yaml

    from steppo.configs.base_config import TrainConfig, load_config_from_yaml

    model_data = {
        "latent_dim": 7,
        "encoder": {"hidden_size": 48, "activation": "gelu", "normalize": "none"},
        "decoder": {"hidden_dims": [32, 16], "activation": "relu"},
    }
    env_data = {"model": "test_arch", "vae": {"kl_weight": 0.2}}

    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        os.makedirs(models_dir)
        model_path = os.path.join(models_dir, "test_arch.yaml")
        env_path = os.path.join(tmpdir, "env.yaml")
        with open(model_path, "w") as f:
            yaml.dump(model_data, f)
        with open(env_path, "w") as f:
            yaml.dump(env_data, f)

        cfg = load_config_from_yaml(TrainConfig, env_path)

    assert cfg.vae.latent_dim == 7
    assert cfg.vae.encoder.hidden_size == 48
    assert cfg.vae.encoder.activation == "gelu"
    assert cfg.vae.decoder.hidden_dims == (32, 16)
    assert cfg.vae.kl_weight == 0.2


def test_margin_penalty_without_rejection_penalty_raises():
    """margin_penalty>0 with rejection_penalty<=0 makes 'always reject' a free,
    zero-reward policy PPO can't gradient its way out of (see the van_der_pol
    margin-penalty sweep collapse, outputs/experiments/20260824/22193876)."""
    from steppo.configs.base_config import TrainConfig, load_config_from_dict

    with pytest.raises(ValueError, match="margin_penalty"):
        load_config_from_dict(
            TrainConfig, {"env": {"margin_penalty": 0.25, "rejection_penalty": 0.0}}
        )


def test_margin_penalty_with_rejection_penalty_is_fine():
    from steppo.configs.base_config import TrainConfig, load_config_from_dict

    cfg = load_config_from_dict(
        TrainConfig, {"env": {"margin_penalty": 0.25, "rejection_penalty": 0.1}}
    )
    assert cfg.env.margin_penalty == 0.25
    assert cfg.env.rejection_penalty == 0.1


def test_margin_penalty_default_zero_is_fine():
    from steppo.configs.base_config import TrainConfig, load_config_from_dict

    cfg = load_config_from_dict(TrainConfig, {})
    assert cfg.env.margin_penalty == 0.0


def test_load_config_hidden_dims_list_to_tuple():
    """YAML lists for hidden_dims must become tuples (hashable)."""
    import yaml

    from steppo.configs.base_config import TrainConfig, load_config_from_yaml

    data = {"vae": {"decoder": {"hidden_dims": [16, 32, 64]}}}
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config_from_yaml(TrainConfig, path)
        assert isinstance(cfg.vae.decoder.hidden_dims, tuple)
        assert cfg.vae.decoder.hidden_dims == (16, 32, 64)
    finally:
        os.unlink(path)


def test_load_config_nested_bins_list_to_tuple_of_tuples():
    """YAML list-of-lists for train_bins must become a tuple of tuples, not a
    tuple of lists — the latter is unhashable and breaks ODEEnvConfig's
    @dataclass(unsafe_hash=True) (used as a static jit arg)."""
    import yaml

    from steppo.configs.base_config import TrainConfig, load_config_from_yaml

    data = {"env": {"mu_min": 1.0, "mu_max": 100.0, "train_bins": [[1.0, 10.0], [20.0, 30.0]]}}
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config_from_yaml(TrainConfig, path)
        assert cfg.env.train_bins == ((1.0, 10.0), (20.0, 30.0))
        assert all(isinstance(b, tuple) for b in cfg.env.train_bins)
        hash(cfg.env)  # must not raise
    finally:
        os.unlink(path)


def test_model_ref_env_values_override_model_defaults():
    """Env config values override model defaults when both specify a field."""
    import yaml

    from steppo.configs.base_config import TrainConfig, load_config_from_yaml

    model_data = {
        "latent_dim": 5,
        "encoder": {"hidden_size": 64, "activation": "tanh"},
    }
    env_data = {
        "model": "test_arch",
        "vae": {"kl_weight": 0.5, "encoder": {"activation": "gelu"}},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        os.makedirs(models_dir)
        with open(os.path.join(models_dir, "test_arch.yaml"), "w") as f:
            yaml.dump(model_data, f)
        env_path = os.path.join(tmpdir, "env.yaml")
        with open(env_path, "w") as f:
            yaml.dump(env_data, f)

        cfg = load_config_from_yaml(TrainConfig, env_path)

    # env override wins
    assert cfg.vae.encoder.activation == "gelu"
    # model default for other fields still applied
    assert cfg.vae.encoder.hidden_size == 64
    assert cfg.vae.kl_weight == 0.5


def test_extends_resolves_for_config_nested_below_env_family_root():
    """Regression test: generate_experiments.py writes sweep configs under
    configs/experiments/ode/<system>/experiment_<uuid>/ -- two directories
    below configs/envs/ode/, not one. `extends: ode_default` (copied verbatim
    from the base config) must still resolve via the configs/envs/ fallback
    search, not just the one-directory-up path used for hand-authored configs.
    """
    import yaml

    from steppo.configs.base_config import TrainConfig, load_config_from_yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = os.path.join(
            tmpdir, "experiments", "ode", "van_der_pol", "experiment_deadbeef"
        )
        os.makedirs(nested_dir)
        data = {"extends": "ode_default", "model": "vae_vdp_2", "exp_name": "regress_e1"}
        path = os.path.join(nested_dir, "regress_e1.yaml")
        with open(path, "w") as f:
            yaml.dump(data, f)

        cfg = load_config_from_yaml(TrainConfig, path)

    assert cfg.exp_name == "regress_e1"


def test_load_config_from_tempfile():
    import yaml

    from steppo.configs.base_config import TrainConfig, load_config_from_yaml

    data = {
        "vae": {
            "latent_dim": 3,
            "encoder": {"hidden_size": 32, "activation": "silu", "normalize": "layer_norm"},
            "decoder": {"hidden_dims": [16, 16], "activation": "gelu"},
        },
        "ppo": {
            "policy": {"hidden_dims": [32], "activation": "tanh"},
        },
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = load_config_from_yaml(TrainConfig, path)
        assert cfg.vae.latent_dim == 3
        assert cfg.vae.encoder.hidden_size == 32
        assert cfg.vae.encoder.activation == "silu"
        assert cfg.vae.encoder.normalize == "layer_norm"
        assert cfg.vae.decoder.hidden_dims == (16, 16)
        assert cfg.ppo.policy.hidden_dims == (32,)
    finally:
        os.unlink(path)


# ── get_activation helper ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["relu", "tanh", "gelu", "elu", "silu"])
def test_get_activation_valid(name):
    from steppo.models.utils import get_activation

    act = get_activation(name)
    x = jnp.array([0.5, -0.5])
    out = act(x)
    assert out.shape == x.shape


def test_get_activation_invalid():
    from steppo.models.utils import get_activation

    with pytest.raises(ValueError, match="Unknown activation"):
        get_activation("sigmoid")


# ── Encoder with various activations ─────────────────────────────────────────


@pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "silu"])
def test_encoder_activation_variants(activation):
    from steppo.configs.base_config import EncoderArchConfig
    from steppo.models.encoder import LSTMEncoder

    arch = EncoderArchConfig(hidden_size=32, activation=activation)
    enc = LSTMEncoder(STATE_DIM, ACTION_DIM, LATENT_DIM, arch, nnx.Rngs(0))
    actions = jnp.zeros((T, ACTION_DIM))
    states = jnp.zeros((T, STATE_DIM))
    rewards = jnp.zeros((T, 1))
    mus, logvars = enc.encode_trajectory(actions, states, rewards)
    assert mus.shape == (T + 1, LATENT_DIM)
    assert jnp.all(jnp.isfinite(mus))


def test_encoder_layer_norm():
    from steppo.configs.base_config import EncoderArchConfig
    from steppo.models.encoder import LSTMEncoder

    arch = EncoderArchConfig(hidden_size=32, normalize="layer_norm")
    enc = LSTMEncoder(STATE_DIM, ACTION_DIM, LATENT_DIM, arch, nnx.Rngs(0))
    assert enc.embed_norms is not None
    assert len(enc.embed_norms) == 3
    actions = jnp.zeros((T, ACTION_DIM))
    states = jnp.zeros((T, STATE_DIM))
    rewards = jnp.zeros((T, 1))
    mus, _ = enc.encode_trajectory(actions, states, rewards)
    assert mus.shape == (T + 1, LATENT_DIM)


# ── Decoder with variable depth and activations ───────────────────────────────


@pytest.mark.parametrize("hidden_dims", [(16,), (32, 32), (64, 32, 16)])
def test_reward_decoder_variable_depth(hidden_dims):
    from steppo.configs.base_config import DecoderArchConfig
    from steppo.models.decoder import RewardDecoder

    arch = DecoderArchConfig(hidden_dims=hidden_dims)
    dec = RewardDecoder(STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, rngs=nnx.Rngs(0))
    assert len(dec.layers) == len(hidden_dims)
    z = jnp.zeros(LATENT_DIM)
    s = jnp.zeros(STATE_DIM)
    a = jnp.zeros(ACTION_DIM)
    r_hat = dec(z, s, a)
    assert r_hat.shape == ()


@pytest.mark.parametrize("hidden_dims", [(16,), (32, 32), (64, 32, 16)])
def test_state_decoder_variable_depth(hidden_dims):
    from steppo.configs.base_config import DecoderArchConfig
    from steppo.models.decoder import StateDecoder

    arch = DecoderArchConfig(hidden_dims=hidden_dims)
    dec = StateDecoder(STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, rngs=nnx.Rngs(0))
    z = jnp.zeros(LATENT_DIM)
    s = jnp.zeros(STATE_DIM)
    a = jnp.zeros(ACTION_DIM)
    s_hat = dec(z, s, a)
    assert s_hat.shape == (STATE_DIM,)


def test_decoder_layer_norm():
    from steppo.configs.base_config import DecoderArchConfig
    from steppo.models.decoder import RewardDecoder

    arch = DecoderArchConfig(hidden_dims=(32, 32), normalize="layer_norm")
    dec = RewardDecoder(STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, rngs=nnx.Rngs(0))
    assert dec.norms is not None
    assert len(dec.norms) == 2
    z = jnp.zeros(LATENT_DIM)
    s = jnp.zeros(STATE_DIM)
    a = jnp.zeros(ACTION_DIM)
    r_hat = dec(z, s, a)
    assert r_hat.shape == ()


@pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "elu"])
def test_decoder_activation_variants(activation):
    from steppo.configs.base_config import DecoderArchConfig
    from steppo.models.decoder import RewardDecoder

    arch = DecoderArchConfig(hidden_dims=(32, 32), activation=activation)
    dec = RewardDecoder(STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, rngs=nnx.Rngs(0))
    z = jnp.zeros(LATENT_DIM)
    s = jnp.zeros(STATE_DIM)
    a = jnp.zeros(ACTION_DIM)
    r_hat = dec(z, s, a)
    assert jnp.isfinite(r_hat)


# ── Policy with variable depth, activations, and normalization ────────────────


@pytest.mark.parametrize("hidden_dims", [(16,), (32, 32), (64, 32, 16)])
def test_policy_variable_depth(hidden_dims):
    from steppo.configs.base_config import PolicyArchConfig
    from steppo.models.policy import ActorCritic

    arch = PolicyArchConfig(hidden_dims=hidden_dims)
    policy = ActorCritic(
        STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, action_space="discrete", rngs=nnx.Rngs(0)
    )
    assert len(policy.layers_actor) == len(hidden_dims)
    assert len(policy.layers_critic) == len(hidden_dims)
    z = jnp.zeros(LATENT_DIM * 2)
    dist, value = policy(jnp.zeros(STATE_DIM), z)
    assert value.shape == ()


@pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "silu"])
def test_policy_activation_variants(activation):
    from steppo.configs.base_config import PolicyArchConfig
    from steppo.models.policy import ActorCritic

    arch = PolicyArchConfig(hidden_dims=(32, 32), activation=activation)
    policy = ActorCritic(
        STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, action_space="discrete", rngs=nnx.Rngs(0)
    )
    z = jnp.zeros(LATENT_DIM * 2)
    dist, value = policy(jnp.zeros(STATE_DIM), z)
    assert jnp.isfinite(value)


def test_policy_layer_norm():
    from steppo.configs.base_config import PolicyArchConfig
    from steppo.models.policy import ActorCritic

    arch = PolicyArchConfig(hidden_dims=(32, 32), normalize="layer_norm")
    policy = ActorCritic(
        STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, action_space="discrete", rngs=nnx.Rngs(0)
    )
    assert policy.norms_actor is not None
    assert len(policy.norms_actor) == 2
    assert policy.norms_critic is not None
    assert len(policy.norms_critic) == 2
    z = jnp.zeros(LATENT_DIM * 2)
    dist, value = policy(jnp.zeros(STATE_DIM), z)
    assert value.shape == ()


def test_policy_continuous_layer_norm():
    from steppo.configs.base_config import PolicyArchConfig
    from steppo.models.policy import ActorCritic

    arch = PolicyArchConfig(hidden_dims=(32, 32), normalize="layer_norm")
    policy = ActorCritic(
        STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, action_space="continuous", rngs=nnx.Rngs(0)
    )
    z = jnp.zeros(LATENT_DIM * 2)
    key = jax.random.PRNGKey(0)
    action, log_prob, value = policy.act(jnp.zeros(STATE_DIM), z, key)
    assert action.shape == (ACTION_DIM,)
    assert jnp.isfinite(value)


# ── End-to-end VAE with new config structure ──────────────────────────────────


def test_vae_full_forward_pass_with_arch():
    from steppo.configs.base_config import DecoderArchConfig, EncoderArchConfig, VAEConfig
    from steppo.models.vae import VariBADVAE

    cfg = VAEConfig(
        latent_dim=LATENT_DIM,
        encoder=EncoderArchConfig(
            hidden_size=32,
            activation="gelu",
            normalize="layer_norm",
        ),
        decoder=DecoderArchConfig(
            hidden_dims=(32, 16),
            activation="relu",
            normalize="layer_norm",
        ),
        state_loss_coeff=1.0,
    )
    vae = VariBADVAE(STATE_DIM, ACTION_DIM, cfg, nnx.Rngs(0))
    actions = jnp.zeros((T, ACTION_DIM))
    states = jnp.zeros((T, STATE_DIM))
    rewards = jnp.zeros((T, 1))
    mus, logvars = vae.encode(actions, states, rewards)
    assert mus.shape == (T + 1, LATENT_DIM)
    z = mus[1]
    result = vae.decode(z, states[0], actions[0])
    assert result["reward"].shape == ()
    assert result["state"].shape == (STATE_DIM,)


def test_vae_jit_compatible_with_arch():
    from steppo.configs.base_config import DecoderArchConfig, EncoderArchConfig, VAEConfig
    from steppo.models.vae import VariBADVAE

    cfg = VAEConfig(
        latent_dim=LATENT_DIM,
        encoder=EncoderArchConfig(hidden_size=32, activation="silu"),
        decoder=DecoderArchConfig(hidden_dims=(32, 32)),
    )
    vae = VariBADVAE(STATE_DIM, ACTION_DIM, cfg, nnx.Rngs(0))
    graphdef, params = nnx.split(vae)

    def encode(params, actions, states, rewards):
        v = nnx.merge(graphdef, params)
        return v.encode(actions, states, rewards)

    jit_encode = jax.jit(encode)
    actions = jnp.zeros((T, ACTION_DIM))
    states = jnp.zeros((T, STATE_DIM))
    rewards = jnp.zeros((T, 1))
    mus, _ = jit_encode(params, actions, states, rewards)
    assert mus.shape == (T + 1, LATENT_DIM)


def test_invalid_normalize_raises():
    from steppo.models.utils import make_norm_layers

    with pytest.raises(ValueError, match="Unknown normalization"):
        make_norm_layers("batch_norm", (32,), nnx.Rngs(0))


# ── Instance norm ──────────────────────────────────────────────────────────────


def test_make_norm_layers_instance_norm_no_params():
    from steppo.models.utils import make_norm_layers

    norms = make_norm_layers("instance_norm", (16,), nnx.Rngs(0))
    assert norms is not None
    params = nnx.state(norms[0], nnx.Param)
    assert len(jax.tree.leaves(params)) == 0
    x = jnp.arange(16.0)
    y = norms[0](x)
    assert not jnp.allclose(y, 0.0)
    assert jnp.allclose(y.mean(), 0.0, atol=1e-5)
    assert jnp.allclose(y.std(), 1.0, atol=1e-2)


def test_make_norm_layers_unknown_raises():
    from steppo.models.utils import make_norm_layers

    with pytest.raises(ValueError, match="Unknown normalization"):
        make_norm_layers("batch_norm", (16,), nnx.Rngs(0))


def test_encoder_instance_norm():
    from steppo.configs.base_config import EncoderArchConfig
    from steppo.models.encoder import LSTMEncoder

    arch = EncoderArchConfig(hidden_size=32, normalize="instance_norm")
    enc = LSTMEncoder(STATE_DIM, ACTION_DIM, LATENT_DIM, arch, nnx.Rngs(0))
    assert enc.embed_norms is not None
    assert len(enc.embed_norms) == 3
    actions = jnp.ones((T, ACTION_DIM))
    states = jnp.ones((T, STATE_DIM))
    rewards = jnp.ones((T, 1))
    mus, _ = enc.encode_trajectory(actions, states, rewards)
    assert mus.shape == (T + 1, LATENT_DIM)
    assert jnp.all(jnp.isfinite(mus))


def test_decoder_instance_norm():
    from steppo.configs.base_config import DecoderArchConfig
    from steppo.models.decoder import RewardDecoder

    arch = DecoderArchConfig(hidden_dims=(32, 32), normalize="instance_norm")
    dec = RewardDecoder(STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, rngs=nnx.Rngs(0))
    assert dec.norms is not None
    assert len(dec.norms) == 2
    z = jnp.ones(LATENT_DIM)
    s = jnp.ones(STATE_DIM)
    a = jnp.ones(ACTION_DIM)
    r_hat = dec(z, s, a)
    assert r_hat.shape == ()
    assert jnp.isfinite(r_hat)


def test_policy_instance_norm():
    from steppo.configs.base_config import PolicyArchConfig
    from steppo.models.policy import ActorCritic

    arch = PolicyArchConfig(hidden_dims=(32, 32), normalize="instance_norm")
    policy = ActorCritic(
        STATE_DIM, ACTION_DIM, LATENT_DIM, arch=arch, action_space="continuous", rngs=nnx.Rngs(0)
    )
    assert policy.norms_actor is not None
    assert policy.norms_critic is not None
    z = jnp.ones(LATENT_DIM * 2)
    key = jax.random.PRNGKey(0)
    action, log_prob, value = policy.act(jnp.ones(STATE_DIM), z, key)
    assert action.shape == (ACTION_DIM,)
    assert jnp.isfinite(value)


def test_zero_encoder_returns_zero_latent_for_all_inputs():
    import flax.nnx as nnx
    import jax.numpy as jnp

    from steppo.configs.base_config import EncoderArchConfig, VAEConfig
    from steppo.models.vae import VariBADVAE

    config = VAEConfig(
        latent_dim=3,
        encoder=EncoderArchConfig(encoder_type="zero", hidden_size=0),
    )
    vae = VariBADVAE(2, 1, config, nnx.Rngs(0), task_dim=1)

    prior_mu, prior_logvar = vae.get_prior()
    assert prior_mu.shape == (3,)
    assert jnp.all(prior_mu == 0)
    assert jnp.all(prior_logvar == 0)

    mus, logvars = vae.encode(jnp.ones((4, 1)), jnp.ones((4, 2)), jnp.ones((4, 1)), jnp.ones((1,)))
    assert mus.shape == (5, 3)
    assert jnp.all(mus == 0)
    assert jnp.all(logvars == 0)

    sampled = vae.sample_z(mus[-1], logvars[-1], jnp.array([0, 1]))
    assert jnp.all(sampled == 0)


def test_retired_sections_in_saved_configs_are_ignored():
    """Configs saved with older checkpoints still carry the removed tft/tqc sections."""
    from steppo.configs.base_config import TrainConfig, load_config_from_dict

    config = load_config_from_dict(TrainConfig, {"tft": {"latent_dim": 5}, "tqc": {"lr": 3e-4}})
    assert config.vae == TrainConfig().vae and config.ppo == TrainConfig().ppo


def test_unknown_top_level_key_still_raises():
    from steppo.configs.base_config import TrainConfig, load_config_from_dict

    with pytest.raises(ValueError, match="Unrecognized config key"):
        load_config_from_dict(TrainConfig, {"not_a_field": 1})
