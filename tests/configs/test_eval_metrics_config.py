def test_eval_metrics_config_has_expected_fields_and_defaults():
    from steppo.configs.base_config import EvalMetricsConfig

    cfg = EvalMetricsConfig()

    assert cfg.enabled is True
    assert cfg.num_envs > 0
    assert cfg.max_steps == 0  # 0 = derive from rollout_steps (see base_trainer.py's usage)
    # Coarse cadence: rebuilds and re-jits the diffrax-native controller each
    # time it runs, same reasoning ErrorDistConfig's interval already used.
    assert cfg.interval == 100
