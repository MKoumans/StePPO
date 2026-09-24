"""Tests for algorithm-specific labels in shared post-run comparisons."""

from types import SimpleNamespace

from research.ode.post_run_analysis.compare import policy_method_label


def test_policy_method_label_names_steppo_runs():
    config = SimpleNamespace(algo="ppo", backbone="varibad")

    assert policy_method_label(config) == "StePPO (PPO)"
