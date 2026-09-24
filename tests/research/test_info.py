"""Fast checks for research/ode/info.py's configuration and report helpers."""

from types import SimpleNamespace

from flax import nnx

from research.ode import info


def test_create_config_applies_only_provided_system_and_seed_overrides():
    args = SimpleNamespace(config=None, system="chua", seed=17)

    config = info.create_config(args)

    assert config.env.system == "chua"
    assert config.seed == 17


def test_parameter_count_sums_all_scalar_leaves():
    linear = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    assert info.param_count(linear) == 9


def test_print_obs_layout_reports_unknown_features_as_zero_width(capsys):
    config = SimpleNamespace(
        env=SimpleNamespace(
            system="scalar_decay",
            t_end=4.0,
            dt0=0.01,
            precision="float32",
            obs_features=("state", "unregistered"),
        )
    )
    env = SimpleNamespace(
        _spec=SimpleNamespace(feature_dims={"state": 2}),
        task_dim=1,
        num_actions=1,
    )

    info.print_obs_layout(config, env)

    lines = capsys.readouterr().out.splitlines()
    assert any("state" in line and line.rstrip().endswith("2D") for line in lines)
    assert any("unregistered" in line and line.rstrip().endswith("0D") for line in lines)
    assert any("total" in line and line.rstrip().endswith("2D") for line in lines)
    assert any("task_dim=1  action_dim=1" in line for line in lines)


def test_print_param_counts_reports_optional_decoders_and_total(monkeypatch, capsys):
    modules = {
        "encoder": SimpleNamespace(count=10),
        "reward_decoder": SimpleNamespace(count=5),
        "state_decoder": None,
        "task_decoder": SimpleNamespace(count=2),
        "accept_decoder": None,
        "end_reward_decoder": None,
    }
    vae = SimpleNamespace(**modules)
    policy = SimpleNamespace(
        layers_actor=[SimpleNamespace(in_features=12)],
        count=7,
    )
    config = SimpleNamespace(
        vae=SimpleNamespace(latent_dim=3, latent_dim_long=2, total_latent_dim=5),
        ppo=SimpleNamespace(policy=SimpleNamespace(hidden_dims=(32, 32))),
    )
    monkeypatch.setattr(info, "param_count", lambda module: module.count)

    info.print_param_counts(config, vae, policy)

    output = capsys.readouterr().out
    assert "state_decoder" in output and "(disabled)" in output
    assert "vae total" in output and "17" in output
    assert "policy total" in output and "7" in output
    assert "total trainable params" in output and "24" in output
