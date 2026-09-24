import numpy as np


def test_plot_steps_vs_mu_writes_pid_only_png(tmp_path):
    from steppo.utils.figures.ode import plot_steps_vs_mu

    output = tmp_path / "steps_vs_mu.png"
    plot_steps_vs_mu(
        {"mu": np.array([1.0, 5.0, 10.0]), "steps": np.array([3.0, 8.0, 15.0])},
        None,
        1.0,
        10.0,
        1.0,
        10.0,
        1.0,
        8,
        False,
        str(output),
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_error_output_path_renames_conventional_steps_vs_mu_basename():
    from steppo.utils.figures.ode import error_output_path as _error_output_path

    assert (
        _error_output_path("outputs/run1/compare/steps_vs_mu.png", "error")
        == "outputs/run1/compare/error_vs_mu.png"
    )
    assert (
        _error_output_path("outputs/run1/compare/steps_vs_mu.png", "error_integrated")
        == "outputs/run1/compare/error_integrated_vs_mu.png"
    )


def test_error_output_path_suffixes_a_custom_basename():
    from steppo.utils.figures.ode import error_output_path as _error_output_path

    assert _error_output_path("my_custom_plot.png", "error") == "my_custom_plot_error.png"


def test_build_error_split_data_returns_none_without_pid_error():
    from steppo.utils.figures.ode import build_error_split_data as _build_error_split_data

    split_data = {
        "test": {"pid": {"mu": np.array([1.0]), "steps": np.array([3.0])}, "pid_error": None}
    }

    assert _build_error_split_data(split_data, "err") is None


def test_write_steps_vs_mu_error_outputs_writes_both_metrics(tmp_path):
    from steppo.utils.figures.ode import write_steps_vs_mu_error_outputs

    split_data = {
        "test": {
            "pid_error": {
                "mu": np.array([1.0, 5.0]),
                "err": np.array([-2.0, -3.0]),
                "err_integrated": np.array([-1.8, -2.9]),
            },
        },
    }
    steps_out = tmp_path / "steps_vs_mu.png"

    written = write_steps_vs_mu_error_outputs(split_data, 1.0, 10.0, str(steps_out))

    assert set(written) == {
        str(tmp_path / "error_vs_mu.txt"),
        str(tmp_path / "error_integrated_vs_mu.txt"),
    }
    assert (tmp_path / "error_vs_mu.png").is_file()
    assert (tmp_path / "error_integrated_vs_mu.png").is_file()


def test_write_steps_vs_mu_error_outputs_noop_without_error_data(tmp_path):
    from steppo.utils.figures.ode import write_steps_vs_mu_error_outputs

    split_data = {"test": {"pid": {"mu": np.array([1.0]), "steps": np.array([3.0])}}}
    steps_out = tmp_path / "steps_vs_mu.png"

    written = write_steps_vs_mu_error_outputs(split_data, 1.0, 10.0, str(steps_out))

    assert written == []
    assert not (tmp_path / "error_vs_mu.png").exists()


def test_build_error_split_data_remaps_pid_policy_oracle_error_series():
    from steppo.utils.figures.ode import build_error_split_data as _build_error_split_data

    split_data = {
        "test": {
            "pid_error": {
                "mu": np.array([1.0, 5.0]),
                "err": np.array([-2.0, -3.0]),
                "err_integrated": np.array([-1.8, -2.9]),
            },
            "policy_error": {
                "mu": np.array([1.0, 5.0]),
                "err": np.array([-2.5, -3.5]),
                "err_integrated": np.array([-2.2, -3.1]),
            },
            "oracle_error": {"mu": np.array([1.0, 5.0]), "err": np.array([-2.8, -3.8])},
        },
    }

    err = _build_error_split_data(split_data, "err")
    assert set(err["test"]) == {"pid", "policy", "oracle"}
    assert err["test"]["oracle"] is split_data["test"]["oracle_error"]

    integrated = _build_error_split_data(split_data, "err_integrated")
    assert integrated["test"]["oracle"] is None
    assert integrated["test"]["pid"] is split_data["test"]["pid_error"]


def test_eval_writer_creates_pid_oracle_steps_plot(tmp_path):
    from steppo.configs.base_config import ODEEnvConfig, TrainConfig
    from steppo.envs.ode.generate_pid_eval_dataset import save_eval_steps_vs_mu

    output = save_eval_steps_vs_mu(
        TrainConfig(env=ODEEnvConfig(system="scalar_decay")),
        {
            "task_params": np.array([0.1, 1.0, 10.0, 100.0, 1000.0]),
            "pid_steps": np.array([2.0, 3.0, 10.0, 31.0, 34.0]),
            "oracle_steps": np.array([2.0, 2.0, 7.0, 25.0, 29.0]),
            "pid_final_y": np.array([[0.1], [1.1], [10.1], [100.1], [1000.1]]),
            "ref_final_y": np.array([[0.1], [1.0], [10.0], [100.0], [1000.0]]),
            "pid_ts": np.tile(np.array([[0.0, 0.5, 1.0]]), (5, 1)),
            "pid_local_err": np.tile(np.array([[0.1, 0.2, 0.3]]), (5, 1)),
            "pid_ref_ts": np.tile(np.array([[0.0, 0.5, 1.0]]), (5, 1)),
            "pid_ref_ys": np.zeros((5, 3, 1)),
            "oracle_final_y": np.array([[0.1], [1.05], [10.05], [100.05], [1000.05]]),
            "pid_completed": np.ones(5, dtype=bool),
            "ref_completed": np.ones(5, dtype=bool),
            "oracle_completed": np.ones(5, dtype=bool),
            "fingerprint": "test-eval",
        },
        split="train",
        output_root=str(tmp_path),
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert (output.parent / "error_vs_mu.png").is_file()
    assert (output.parent / "error_integrated_vs_mu.png").is_file()

    assert (
        output
        == tmp_path / "scalar_decay" / "evaluation" / "trainset" / "test-eval" / "steps_vs_mu.png"
    )


def test_bin_stats_reads_an_arbitrary_value_key():
    from steppo.utils.figures.ode import bin_stats

    data = {"mu": np.array([1.0, 2.0, 9.0]), "err": np.array([-2.0, -3.0, -1.0])}
    bin_edges = np.array([0.0, 5.0, 10.0])

    means, stds, counts, _ = bin_stats(data, bin_edges, value_key="err")

    assert np.allclose(means, [-2.5, -1.0])
    assert counts == [2, 1]


def test_save_steps_vs_mu_splits_txt_writes_an_arbitrary_value_key(tmp_path):
    from steppo.utils.figures.ode import save_steps_vs_mu_splits_txt

    split_data = {"train": {"pid": {"mu": np.array([1.0, 5.0]), "err": np.array([-2.0, -3.0])}}}
    txt = tmp_path / "error_vs_mu.txt"

    save_steps_vs_mu_splits_txt(split_data, str(txt), 1.0, 10.0, value_key="err")

    content = txt.read_text()
    assert "pid_mean" in content


def test_plot_steps_vs_mu_splits_plots_an_arbitrary_value_key(tmp_path):
    from steppo.utils.figures.ode import plot_steps_vs_mu_splits

    split_data = {
        "test": {
            "pid": {"mu": np.array([1.0, 5.0, 10.0]), "err": np.array([-2.0, -3.0, -1.0])},
            "policy": {"mu": np.array([1.0, 5.0, 10.0]), "err": np.array([-2.5, -3.5, -1.5])},
        },
    }
    png = tmp_path / "error_vs_mu.png"

    plot_steps_vs_mu_splits(
        split_data,
        1.0,
        10.0,
        str(png),
        value_key="err",
        y_label="log10 relative L2 error",
    )

    assert png.is_file()
    assert png.stat().st_size > 0


def test_split_steps_vs_mu_writer_contains_all_requested_series(tmp_path):
    from steppo.utils.figures.ode import (
        plot_steps_vs_mu_splits,
        save_steps_vs_mu_splits_txt,
    )

    def series(offset):
        return {
            "pid": {"mu": np.array([1.0, 5.0, 10.0]), "steps": np.array([10.0, 20.0, 30.0])},
            "policy": {
                "mu": np.array([1.0, 5.0, 10.0]),
                "steps": np.array([8.0 + offset, 18.0, 28.0]),
            },
            "oracle": {"mu": np.array([1.0, 5.0, 10.0]), "steps": np.array([7.0, 17.0, 27.0])},
        }

    split_data = {"train": series(0.0), "val": series(1.0), "test": series(2.0)}
    txt = tmp_path / "steps_vs_mu.txt"
    png = tmp_path / "steps_vs_mu.png"
    save_steps_vs_mu_splits_txt(split_data, str(txt), 1.0, 10.0)
    plot_steps_vs_mu_splits(split_data, 1.0, 10.0, str(png))

    content = txt.read_text()
    assert all(f"# split: {split}" in content for split in ("train", "val", "test"))
    assert content.count("policy_mean") == 3
    assert content.count("oracle_mean") == 3
    assert png.is_file()
    assert png.stat().st_size > 0
