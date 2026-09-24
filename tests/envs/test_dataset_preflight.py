from types import SimpleNamespace

from steppo.envs.ode import check_oracle_dataset as preflight


def test_evaluation_requests_cover_configured_splits():
    config = SimpleNamespace(
        env=SimpleNamespace(
            train_bins=((1.0, 10.0),),
            val_bins=(),
            test_bins=((1.0, 10.0),),
        )
    )

    requests = preflight._evaluation_requests(config)

    assert [request["label"] for request in requests] == [
        "train_bins",
        "test_bins",
        "compare_table",
    ]
    assert [request["generator_args"][1] for request in requests] == [
        "train",
        "test",
        "test",
    ]
    assert all(
        request["generator_args"]
        == [
            "--split",
            split,
            "--ref_tol_factor",
            repr(preflight.PID_EVAL_REF_TOL_FACTOR),
        ]
        for request, split in zip(requests, ("train", "test", "test"))
    )


def test_missing_training_cache_does_not_skip_evaluation_cache_check(monkeypatch):
    config = SimpleNamespace(env=SimpleNamespace(progress_warp=False))
    calls = []
    monkeypatch.setattr(preflight, "load_config_from_yaml", lambda *_: config)
    monkeypatch.setattr(preflight, "_check_oracle_dataset", lambda *_: False)
    monkeypatch.setattr(preflight, "_check_progress_warp_cache", lambda *_: True)
    monkeypatch.setattr(
        preflight,
        "_check_pid_eval_datasets",
        lambda *args: calls.append(args) or True,
    )

    assert preflight.check_oracle_dataset("config.yaml") is False
    assert calls == [("config.yaml", config)]


def test_missing_evaluation_cache_prints_exact_split_command(monkeypatch, capsys):
    config = SimpleNamespace(
        env=SimpleNamespace(
            train_bins=((1.0, 10.0),),
            val_bins=(),
            test_bins=(),
        ),
        rollout_steps=50,
    )
    monkeypatch.setattr(preflight, "load_config_from_yaml", lambda *_: config)
    monkeypatch.setattr(preflight, "_check_oracle_dataset", lambda *_: True)
    monkeypatch.setattr(
        preflight,
        "load_cached_pid_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert preflight.check_oracle_dataset("outputs/scalar_decay_e2.yaml") is False
    output = capsys.readouterr().out
    assert "--split train" in output
    assert "--config outputs/scalar_decay_e2.yaml" in output
    assert "--force" in output


def test_pid_cache_preflight_checks_every_split_and_distinguishes_invalid_from_missing(
    monkeypatch,
    capsys,
):
    config = SimpleNamespace(env=object(), rollout_steps=25)
    requests = [
        {
            "label": f"{split}_bins",
            "split": split,
            "num_envs": 8,
            "seed": 4,
            "bins": [(1.0, 10.0)],
            "generator_args": ["--split", split, "--ref_tol_factor", "1.0"],
        }
        for split in ("train", "val", "test")
    ]
    monkeypatch.setattr(preflight, "_evaluation_requests", lambda _config: requests)
    calls = []

    def fake_load(env_config, **kwargs):
        calls.append((env_config, kwargs))
        if kwargs["split"] == "val":
            raise ValueError("cache schema mismatch")
        if kwargs["split"] == "test":
            raise FileNotFoundError("cache absent")
        return {"task_params": [1.0]}

    monkeypatch.setattr(preflight, "load_cached_pid_batch", fake_load)

    assert not preflight._check_pid_eval_datasets("configs/system eval.yaml", config)

    assert [kwargs["split"] for _, kwargs in calls] == ["train", "val", "test"]
    assert all(env is config.env for env, _ in calls)
    assert all(kwargs["max_steps"] == 25 for _, kwargs in calls)
    output = capsys.readouterr().out
    assert "train_bins PID evaluation cache already cached" in output
    assert "invalid val_bins PID evaluation cache: cache schema mismatch" in output
    assert "missing test_bins PID evaluation cache" in output
    assert "--config 'configs/system eval.yaml'" in output
    assert "--split val" in output and "--split test" in output


def test_progress_warp_preflight_requires_cache_for_scalar_task_env(monkeypatch):
    from steppo.training import reward_norm

    normalization = SimpleNamespace(grid_points=5, num_repeats=3, max_steps=17)
    env_config = SimpleNamespace(progress_warp=True)
    config = SimpleNamespace(
        env=env_config,
        training=SimpleNamespace(reward_normalization=normalization),
        rollout_steps=25,
    )

    class _ScalarTaskEnv:
        t_end = 7.0
        task_dim = 1

        @staticmethod
        def _task_bounds():
            return 2.0, 20.0

    monkeypatch.setattr(preflight, "ODEEnv", lambda *_args, **_kwargs: _ScalarTaskEnv())
    observed = {}

    def fake_build(env_cfg, lo, hi, **kwargs):
        observed.update(env_cfg=env_cfg, lo=lo, hi=hi, **kwargs)
        return object()

    monkeypatch.setattr(reward_norm, "build_pid_warp_grid", fake_build)

    assert preflight._check_progress_warp_cache("config.yaml", config)
    assert observed == {
        "env_cfg": env_config,
        "lo": 2.0,
        "hi": 20.0,
        "grid_points": 5,
        "num_repeats": 3,
        "max_steps": 17,
        "require_cache": True,
    }


def test_progress_warp_preflight_reports_a_missing_required_cache(monkeypatch, capsys):
    from steppo.training import reward_norm

    config = SimpleNamespace(
        env=SimpleNamespace(progress_warp=True),
        training=SimpleNamespace(
            reward_normalization=SimpleNamespace(
                grid_points=2,
                num_repeats=1,
                max_steps=0,
            )
        ),
        rollout_steps=19,
    )

    class _ScalarTaskEnv:
        t_end = 1.0
        task_dim = 1

        @staticmethod
        def _task_bounds():
            return 1.0, 4.0

    monkeypatch.setattr(preflight, "ODEEnv", lambda *_args, **_kwargs: _ScalarTaskEnv())
    monkeypatch.setattr(
        reward_norm,
        "build_pid_warp_grid",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("warp cache missing")),
    )

    assert not preflight._check_progress_warp_cache("config.yaml", config)
    assert "config.yaml: warp cache missing" in capsys.readouterr().out
