from dataclasses import replace

import numpy as np
import pytest

from steppo.configs.base_config import ODEEnvConfig
from steppo.training.pid_solve import (
    PID_BASELINE_SEED,
    PID_WARP_SEED,
    fingerprint,
    pid_env_payload,
)


def _artifact_key(config, kind, pid_seed):
    return fingerprint(
        {
            "kind": kind,
            "env_config": pid_env_payload(config),
            "pid_seed": pid_seed,
        }
    )


def test_pid_payload_ignores_consumer_only_environment_fields():
    base = ODEEnvConfig(system="scalar_decay")
    variant = replace(
        base,
        progress_warp=True,
        immediate_dt_action=True,
        dt_log_gain=3.0,
        obs_features=("state",),
        reward_factor=100.0,
        survival_penalty=0.25,
        rejection_penalty=0.1,
    )

    assert pid_env_payload(base) == pid_env_payload(variant)
    assert _artifact_key(base, "progress_warp_grid", PID_WARP_SEED) == _artifact_key(
        variant, "progress_warp_grid", PID_WARP_SEED
    )


def test_pid_payload_changes_for_solver_and_dynamics_inputs():
    base = ODEEnvConfig(system="scalar_decay")

    assert pid_env_payload(base) != pid_env_payload(replace(base, t_end=2.0))
    assert pid_env_payload(base) != pid_env_payload(replace(base, rtol=1.0e-4))
    assert pid_env_payload(base) != pid_env_payload(replace(base, precision="float32"))
    assert pid_env_payload(base) != pid_env_payload(replace(base, sd_pulse_enabled=True))
    assert pid_env_payload(base) != pid_env_payload(replace(base, sd_pulse_period=0.2))
    assert pid_env_payload(base) != pid_env_payload(replace(base, sample_y0=True))


def test_pid_sampling_seed_is_artifact_specific_and_training_seed_is_absent():
    config = ODEEnvConfig(system="scalar_decay")
    payload = pid_env_payload(config)

    assert "seed" not in payload
    assert PID_BASELINE_SEED != PID_WARP_SEED
    assert _artifact_key(config, "pid_baseline_steps", PID_BASELINE_SEED) != _artifact_key(
        config, "progress_warp_grid", PID_WARP_SEED
    )
    assert _artifact_key(config, "pid_baseline_steps", PID_BASELINE_SEED) != _artifact_key(
        config, "pid_baseline_steps", PID_WARP_SEED
    )


def test_warp_builder_loads_cache_across_warp_ablation(tmp_path, monkeypatch):
    base = ODEEnvConfig(system="scalar_decay", progress_warp=False)
    warped = replace(base, progress_warp=True)
    cache_payload = {
        "kind": "progress_warp_grid",
        "mu_lo": 1.0,
        "mu_hi": 100.0,
        "grid_points": 24,
        "num_repeats": 8,
        "max_steps": 50,
        "num_knots": 64,
        "pid_seed": PID_WARP_SEED,
    }
    cache_path = (
        tmp_path
        / "data"
        / "scalar_decay"
        / "training"
        / f"warp_grid_{fingerprint({**cache_payload, 'env_config': pid_env_payload(base)})}.npz"
    )
    cache_path.parent.mkdir(parents=True)
    np.savez_compressed(
        cache_path,
        log_mu_grid=np.array([0.0, 1.0], dtype=np.float32),
        t_knots=np.ones((2, 2), dtype=np.float32),
        fracs=np.array([0.5, 1.0], dtype=np.float32),
    )
    monkeypatch.chdir(tmp_path)

    def fail_if_solver_runs(*args, **kwargs):
        raise AssertionError("warp cache miss: PID solver ran instead of loading the canonical key")

    monkeypatch.setattr("steppo.training.pid_solve.solve_pid_batch", fail_if_solver_runs)

    from steppo.training.reward_norm import build_pid_warp_grid

    loaded = build_pid_warp_grid(
        warped,
        1.0,
        100.0,
        grid_points=24,
        num_repeats=8,
        max_steps=50,
        num_knots=64,
    )

    assert loaded[0].shape == (2,)
    assert np.allclose(loaded[1], 1.0)


def test_baseline_builder_loads_cache_across_consumer_ablation(tmp_path, monkeypatch):
    base = ODEEnvConfig(system="scalar_decay", progress_warp=False)
    variant = replace(base, progress_warp=True, reward_factor=100.0)
    cache_payload = {
        "kind": "pid_baseline_steps",
        "env_config": pid_env_payload(base),
        "pid_seed": PID_BASELINE_SEED,
        "mus": [1.0],
        "num_repeats": 1,
        "max_steps": 50,
    }
    cache_path = (
        tmp_path
        / "data"
        / "scalar_decay"
        / "training"
        / f"baseline_{fingerprint(cache_payload)}.npz"
    )
    cache_path.parent.mkdir(parents=True)
    np.savez_compressed(
        cache_path,
        steps=np.array([3.0]),
        accepted=np.array([3.0]),
        rejected=np.array([0.0]),
        t_reached=np.array([1.0]),
    )
    monkeypatch.chdir(tmp_path)

    def fail_if_solver_runs(*args, **kwargs):
        raise AssertionError(
            "baseline cache miss: PID solver ran instead of loading the canonical key"
        )

    monkeypatch.setattr("steppo.training.pid_solve.solve_pid_batch", fail_if_solver_runs)

    from steppo.training.eval import compute_pid_baseline_steps

    loaded = compute_pid_baseline_steps(
        variant,
        mus=[1.0],
        num_repeats=1,
        max_steps=50,
    )

    assert loaded[1.0]["steps"] == 3.0


def test_baseline_builder_require_cache_fails_without_cache(tmp_path, monkeypatch):
    from steppo.training.eval import compute_pid_baseline_steps

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "steppo.training.pid_solve.solve_pid_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("baseline solver ran on a strict cache miss")
        ),
    )

    with pytest.raises(FileNotFoundError, match="training"):
        compute_pid_baseline_steps(
            ODEEnvConfig(system="scalar_decay"),
            mus=[1.0],
            num_repeats=1,
            max_steps=50,
            require_cache=True,
        )


def test_warp_builder_require_cache_fails_without_cache(tmp_path, monkeypatch):
    from steppo.training.reward_norm import build_pid_warp_grid

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "steppo.training.pid_solve.solve_pid_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("warp solver ran on a strict cache miss")
        ),
    )

    with pytest.raises(FileNotFoundError, match="training"):
        build_pid_warp_grid(
            ODEEnvConfig(system="scalar_decay"),
            1.0,
            100.0,
            grid_points=2,
            num_repeats=1,
            max_steps=50,
            num_knots=2,
            require_cache=True,
        )


def test_pid_generation_fingerprint_ignores_warp_consumer_flag():
    from steppo.configs.base_config import TrainConfig
    from steppo.envs.ode.generate_pid_training_dataset import pid_generation_fingerprint

    base = TrainConfig(env=ODEEnvConfig(system="scalar_decay"))
    variant = replace(base, env=replace(base.env, progress_warp=not base.env.progress_warp))

    assert pid_generation_fingerprint(base, 1.0, 100.0) == pid_generation_fingerprint(
        variant, 1.0, 100.0
    )


def test_pid_generation_fingerprint_changes_with_generation_knob():
    from steppo.configs.base_config import TrainConfig
    from steppo.envs.ode.generate_pid_training_dataset import pid_generation_fingerprint

    base = TrainConfig(env=ODEEnvConfig(system="scalar_decay"))
    variant = replace(
        base,
        precompute_baseline=replace(
            base.precompute_baseline,
            num_repeats=base.precompute_baseline.num_repeats + 1,
        ),
    )

    assert pid_generation_fingerprint(base, 1.0, 100.0) != pid_generation_fingerprint(
        variant, 1.0, 100.0
    )


def test_pid_plot_data_keeps_ood_baseline_points():
    from steppo.envs.ode.generate_pid_training_dataset import _scale_grid_plot_data

    mu = np.geomspace(1.0, 100.0, 24)
    steps = np.linspace(3.0, 31.0, 24)
    plot_data = _scale_grid_plot_data(
        (np.log(mu), np.log(steps), 0.0),
        {200.0: {"steps": 32.0}},
    )

    assert plot_data["mu"].shape == (25,)
    assert plot_data["mu"][-1] == 200.0
    assert plot_data["steps"][-1] == 32.0


def test_pid_plot_data_uses_dense_scale_grid():
    from steppo.envs.ode.generate_pid_training_dataset import _scale_grid_plot_data

    mu = np.geomspace(1.0, 100.0, 24)
    steps = np.linspace(3.0, 31.0, 24)
    plot_data = _scale_grid_plot_data((np.log(mu), np.log(steps), 0.0))

    assert plot_data["mu"].shape == (24,)
    assert plot_data["steps"].shape == (24,)
    assert np.allclose(plot_data["mu"], mu)
    assert np.allclose(plot_data["steps"], steps)


def test_oracle_plot_data_uses_first_done_step_and_budget_for_incomplete_episodes():
    from steppo.envs.ode.generate_pid_training_dataset import _oracle_scale_grid_plot_data

    oracle_batch = {
        "dones_ep": np.array(
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
                [False, False, False],
            ]
        ),
        "task_params": np.array([[0.8], [0.2], [0.5]]),
    }

    plot_data = _oracle_scale_grid_plot_data(oracle_batch, lo=2.0, hi=12.0)

    assert np.allclose(plot_data["mu"], [4.0, 7.0, 10.0])
    assert np.array_equal(plot_data["steps"], [1.0, 4.0, 2.0])


def test_pid_eval_fingerprint_includes_dataset_split():
    from steppo.training.error_dist import pid_eval_fingerprint

    config = ODEEnvConfig(system="scalar_decay", train_bins=((1.0, 10.0),))
    common = dict(
        env_config=config,
        num_envs=4,
        seed=0,
        max_steps=50,
        bins=[(1.0, 10.0)],
    )

    assert pid_eval_fingerprint(split="train", **common) != pid_eval_fingerprint(
        split="val", **common
    )


def test_pid_eval_cache_path_preserves_evaluation_and_split_names(tmp_path):
    from steppo.training.error_dist import _cache_path

    train_path = _cache_path(str(tmp_path), "scalar_decay", "train", "abc123")
    val_path = _cache_path(str(tmp_path), "scalar_decay", "val", "abc123")

    assert train_path == str(tmp_path / "scalar_decay" / "evaluation" / "trainset" / "abc123.npz")
    assert val_path == str(tmp_path / "scalar_decay" / "evaluation" / "valset" / "abc123.npz")
    assert train_path != val_path


def test_log10_rel_error_matches_hand_computed_value():
    from steppo.training.error_dist import log10_rel_error

    y = np.array([[4.0, 3.0]])  # norm 5
    y_ref = np.array([[0.0, 0.0]])  # norm 0, atol floors the denominator

    result = log10_rel_error(y, y_ref, atol=1.0)

    assert np.allclose(result, [np.log10(5.0)])


def test_load_pid_batch_persists_dense_reference_trajectory(tmp_path, monkeypatch):
    """load_pid_batch's tight-tolerance reference solve must now save the
    full (ref_ts, ref_ys) trajectory, not just its final state — so a
    trajectory-level comparison (steppo.training.trajectory_compare) has a
    ground truth to interpolate against."""
    from steppo.training.error_dist import load_pid_batch
    from steppo.training.trajectory_compare import final_state_from_trajectory

    cfg = ODEEnvConfig(
        system="scalar_decay",
        lam_min=1.0,
        lam_max=10.0,
        immediate_dt_action=True,
    )
    monkeypatch.chdir(tmp_path)

    data = load_pid_batch(
        cfg,
        num_envs=4,
        seed=0,
        max_steps=20,
        bins=[(1.0, 10.0)],
        split="train",
        timing_n=4,
        timing_repeats=1,
        oracle_max_iters=1,
    )

    assert data["ref_ys"].shape[-1] == data["ref_final_y"].shape[-1]
    assert np.allclose(
        final_state_from_trajectory(data["ref_ts"], data["ref_ys"]),
        data["ref_final_y"],
        atol=1e-4,
    )

    reloaded = load_pid_batch(
        cfg,
        num_envs=4,
        seed=0,
        max_steps=20,
        bins=[(1.0, 10.0)],
        split="train",
        timing_n=4,
        timing_repeats=1,
        oracle_max_iters=1,
    )
    assert np.array_equal(reloaded["ref_ts"], data["ref_ts"])
    assert np.array_equal(reloaded["ref_ys"], data["ref_ys"])


def test_load_pid_batch_persists_pid_local_error_against_reference(tmp_path, monkeypatch):
    """load_pid_batch's PID-baseline solve must now be dense too, and persist
    per-step (pid_ts, pid_local_err) — the PID baseline's own local relative-
    L2 error against the reference at each of its own solve times — so
    downstream consumers (steppo.training.steps_vs_mu.collect_pid_errors)
    never need to re-solve PID to get last-step/time-integrated error."""
    from steppo.training.error_dist import load_pid_batch
    from steppo.training.trajectory_compare import last_valid_value, relative_l2_error

    cfg = ODEEnvConfig(
        system="scalar_decay",
        lam_min=1.0,
        lam_max=10.0,
        immediate_dt_action=True,
    )
    monkeypatch.chdir(tmp_path)

    data = load_pid_batch(
        cfg,
        num_envs=4,
        seed=0,
        max_steps=100,
        bins=[(1.0, 10.0)],
        split="train",
        timing_n=4,
        timing_repeats=1,
        oracle_max_iters=1,
    )

    assert data["pid_ts"].shape == data["pid_local_err"].shape

    # The local error at the PID baseline's own last valid time must match
    # relative_l2_error computed directly from pid_final_y/ref_final_y (the
    # two quantities the existing "err" field already compares).
    last_err = last_valid_value(data["pid_ts"], data["pid_local_err"])
    direct_err = relative_l2_error(data["pid_final_y"], data["ref_final_y"], cfg.atol)
    assert np.allclose(last_err, direct_err, atol=1e-3)

    reloaded = load_pid_batch(
        cfg,
        num_envs=4,
        seed=0,
        max_steps=100,
        bins=[(1.0, 10.0)],
        split="train",
        timing_n=4,
        timing_repeats=1,
        oracle_max_iters=1,
    )
    assert np.array_equal(reloaded["pid_ts"], data["pid_ts"])
    assert np.array_equal(reloaded["pid_local_err"], data["pid_local_err"])


def test_load_pid_batch_regenerates_when_cache_format_version_is_stale(tmp_path, monkeypatch):
    """A cache entry written under an older _CACHE_FORMAT_VERSION (e.g. one
    from before pid_ts/pid_local_err existed) must be treated as a miss and
    regenerated — never read back with those fields silently missing."""
    import steppo.training.pid_solve as pid_solve_module
    from steppo.training.error_dist import load_pid_batch

    cfg = ODEEnvConfig(
        system="scalar_decay",
        lam_min=1.0,
        lam_max=10.0,
        immediate_dt_action=True,
    )
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(pid_solve_module, "_CACHE_FORMAT_VERSION", 3)
    stale = load_pid_batch(
        cfg,
        num_envs=4,
        seed=0,
        max_steps=20,
        bins=[(1.0, 10.0)],
        split="train",
        timing_n=4,
        timing_repeats=1,
        oracle_max_iters=1,
    )

    monkeypatch.setattr(pid_solve_module, "_CACHE_FORMAT_VERSION", 4)
    fresh = load_pid_batch(
        cfg,
        num_envs=4,
        seed=0,
        max_steps=20,
        bins=[(1.0, 10.0)],
        split="train",
        timing_n=4,
        timing_repeats=1,
        oracle_max_iters=1,
    )

    # Different version -> different fingerprint -> a distinct cache file,
    # not the stale one reused; the fresh entry has the new fields.
    assert fresh["fingerprint"] != stale["fingerprint"]
    assert "pid_ts" in fresh and "pid_local_err" in fresh


def test_pid_eval_sampling_is_deterministic_but_split_specific():
    from steppo.training.error_dist import sample_eval_tasks

    config = ODEEnvConfig(
        system="scalar_decay",
        task_sample_scheme="binned",
    )
    bins = [(1.0, 10.0)]

    train = sample_eval_tasks(config, num_envs=16, seed=0, bins=bins, split="train")
    val = sample_eval_tasks(config, num_envs=16, seed=0, bins=bins, split="val")
    val_again = sample_eval_tasks(config, num_envs=16, seed=0, bins=bins, split="val")

    assert not np.array_equal(train["task_params"], val["task_params"])
    assert not np.array_equal(train["episode_keys"], val["episode_keys"])
    assert np.array_equal(val["task_params"], val_again["task_params"])
    assert np.array_equal(val["episode_keys"], val_again["episode_keys"])


def test_pid_training_figure_path_uses_training_namespace():
    from pathlib import Path

    from steppo.configs.base_config import ODEEnvConfig, TrainConfig
    from steppo.envs.ode.generate_pid_training_dataset import (
        pid_generation_fingerprint,
        pid_training_figure_path,
    )

    config = TrainConfig(env=ODEEnvConfig(system="scalar_decay"))
    fingerprint = pid_generation_fingerprint(config, 1.0, 10.0)
    output = pid_training_figure_path(config, 1.0, 10.0, output_root="outputs/datasets")

    assert (
        output == Path("outputs/datasets/scalar_decay/training") / fingerprint / "steps_vs_mu.png"
    )


def _oracle_env_config():
    return ODEEnvConfig(
        system="scalar_decay",
        t_end=1.0,
        dt0=0.1,
        rtol=1e-5,
        atol=1e-8,
        dt_min=1e-8,
        dt_max=100.0,
        immediate_dt_action=True,
        lam_min=1.0,
        lam_max=50.0,
        sample_lam=True,
        train_bins=((1.0, 50.0),),
    )


def test_load_oracle_training_batch_raises_when_nothing_cached(tmp_path):
    """load_oracle_training_batch must never generate — a cache miss is a
    hard error pointing at generate_pid_training_dataset.py, not a fallback
    live generation (the oracle search is too expensive to trigger
    implicitly from a training run)."""
    from steppo.training.oracle_dataset import load_oracle_training_batch

    with pytest.raises(FileNotFoundError, match="generate_pid_training_dataset"):
        load_oracle_training_batch(
            _oracle_env_config(),
            rollout_steps=3,
            oracle_max_iters=2,
            cache_dir=str(tmp_path),
        )


def test_oracle_training_batch_loads_from_cache_after_generation(tmp_path, monkeypatch):
    """load_oracle_training_batch, after generate_and_cache_oracle_training_batch
    has populated the cache, must be a pure load — it never calls the
    generator itself."""
    from steppo.training.oracle_dataset import (
        generate_and_cache_oracle_training_batch,
        load_oracle_training_batch,
    )

    env_config = _oracle_env_config()
    cache_dir = str(tmp_path)

    generated = generate_and_cache_oracle_training_batch(
        env_config,
        num_envs=2,
        rollout_steps=3,
        oracle_max_iters=2,
        cache_dir=cache_dir,
    )
    assert generated["states"].shape[:2] == (3, 2)

    def fail_if_regenerated(*args, **kwargs):
        raise AssertionError("oracle cache miss: generator ran instead of loading the cached npz")

    monkeypatch.setattr(
        "steppo.training.oracle_dataset.generate_oracle_batch",
        fail_if_regenerated,
    )

    loaded = load_oracle_training_batch(
        env_config,
        rollout_steps=3,
        oracle_max_iters=2,
        cache_dir=cache_dir,
    )
    assert np.array_equal(loaded["states"], generated["states"])
    assert np.array_equal(loaded["actions"], generated["actions"])


def test_oracle_training_fingerprint_ignores_dataset_size():
    """Dataset size (num_envs) is a generation-time-only parameter, not part
    of the cache identity — two different sizes for the same env/oracle
    knobs must resolve to the same fingerprint (see oracle_dataset.py's
    module docstring for why)."""
    import inspect

    from steppo.training.oracle_dataset import (
        generate_and_cache_oracle_training_batch,
        oracle_training_fingerprint,
    )

    assert "num_envs" not in inspect.signature(oracle_training_fingerprint).parameters
    assert "num_envs" in inspect.signature(generate_and_cache_oracle_training_batch).parameters


def test_oracle_training_fingerprint_changes_with_oracle_knobs():
    from steppo.training.oracle_dataset import oracle_training_fingerprint

    env_config = ODEEnvConfig(system="scalar_decay")
    base_kwargs = dict(
        rollout_steps=50,
        episodes_per_trial=0,
        seed=0,
        oracle_max_iters=20,
    )
    base_fp = oracle_training_fingerprint(env_config, **base_kwargs)

    assert base_fp != oracle_training_fingerprint(
        env_config, **{**base_kwargs, "oracle_max_iters": 10}
    )
    assert base_fp != oracle_training_fingerprint(
        env_config, **{**base_kwargs, "rollout_steps": 25}
    )
    assert base_fp == oracle_training_fingerprint(env_config, **base_kwargs)


def test_load_pid_batch_persists_completion_for_every_cached_series(tmp_path, monkeypatch):
    """Every cached step count has a companion completion mask. Without it a
    solve that stopped at dt_min contributes its aborted step count to the
    steps-vs-mu mean and reads as a cheap solve (chua_smooth m>=1.2), and an
    error graded against a reference that never reached t_end reads as an
    accurate one."""
    from steppo.training.error_dist import load_pid_batch

    cfg = ODEEnvConfig(
        system="scalar_decay",
        lam_min=1.0,
        lam_max=10.0,
        immediate_dt_action=True,
    )
    monkeypatch.chdir(tmp_path)

    # scalar_decay's t_end is 500 — a budget of 100 genuinely truncates, so
    # give this batch room to finish and assert on a well-posed one.
    kwargs = dict(
        num_envs=4,
        seed=0,
        max_steps=400,
        bins=[(1.0, 10.0)],
        split="train",
        timing_n=4,
        timing_repeats=1,
        oracle_max_iters=1,
    )
    data = load_pid_batch(cfg, **kwargs)

    for key, steps_key in (
        ("pid_completed", "pid_steps"),
        ("pid_completed_unlimited", "pid_steps_unlimited"),
        ("oracle_completed", "oracle_steps"),
        ("ref_completed", None),
    ):
        assert data[key].dtype == bool
        assert data[key].shape == data["task_params"].shape
        if steps_key is not None:
            assert data[key].shape == data[steps_key].shape
        assert data[key].all(), f"{key} should be all-true on a well-posed scalar_decay batch"

    reloaded = load_pid_batch(cfg, **kwargs)
    for key in ("pid_completed", "pid_completed_unlimited", "oracle_completed", "ref_completed"):
        assert np.array_equal(reloaded[key], data[key])


def test_load_pid_batch_marks_budget_truncated_pid_solves_incomplete(tmp_path, monkeypatch):
    """A step budget too small for the task must show up as completed=False,
    not as a small step count."""
    from steppo.training.error_dist import load_pid_batch

    cfg = ODEEnvConfig(
        system="scalar_decay",
        lam_min=1.0,
        lam_max=10.0,
        immediate_dt_action=True,
    )
    monkeypatch.chdir(tmp_path)

    data = load_pid_batch(
        cfg,
        num_envs=4,
        seed=0,
        max_steps=100,
        bins=[(1.0, 10.0)],
        split="train",
        timing_n=4,
        timing_repeats=1,
        oracle_max_iters=1,
    )

    assert not data["pid_completed"].any()
    # The x5 budget is enough, so the unlimited series still completes —
    # the two masks are independent.
    assert data["pid_completed_unlimited"].all()
