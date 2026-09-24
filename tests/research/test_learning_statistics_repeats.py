import json

import yaml


def test_aggregate_repeats_subcommand_preserves_summary_output(tmp_path):
    from research.ode.post_run_analysis.learning_statistics import main

    root = tmp_path / "runs"
    for seed, value in ((0, 1.0), (1, 3.0)):
        run = root / "20260902" / "ode" / "scalar_decay" / f"run{seed}"
        (run / "checkpoints").mkdir(parents=True)
        (run / "outputs").mkdir()
        (run / "checkpoints" / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "seed": seed,
                    "exp_name": "scalar_decay_e1",
                    "repeat_batch_uid": "batch-1",
                }
            )
        )
        (run / "outputs" / "scalar_decay_e1_metrics.json").write_text(
            json.dumps([{"eval/mean_return": value}])
        )

    out = tmp_path / "repeat-analysis"
    main(["aggregate-repeats", "--system", "scalar_decay", "--root", str(root), "--out", str(out)])

    summary = next(out.rglob("summary.txt")).read_text()
    assert "eval/mean_return: final=2.0000 +/- 1.0000" in summary
