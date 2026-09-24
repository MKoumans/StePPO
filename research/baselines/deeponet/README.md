# DeepONet baseline

DeepONet surrogates for `scalar_decay`, `van_der_pol` and `brusselator`, trained
with DeepXDE on PyTorch. PyTorch's CUDA libraries conflict with `jax[cuda12]`,
so training and DeepONet evaluation run in a separate container:

```bash
docker compose -f docker/docker-compose.deeponet.yml up -d
docker compose -f docker/docker-compose.deeponet.yml exec deeponet bash
```

## Train and evaluate (DeepONet container)

```bash
PYTHONPATH=. python research/baselines/deeponet/train_deeponet_unified.py \
  --profile retrain --system van_der_pol --domain complete
PYTHONPATH=. python research/baselines/deeponet/train_deeponet_unified.py \
  --profile retrain --system van_der_pol --domain binned
```

`--domain complete` trains on the system's `test_bins`; `--domain binned` trains
on the gapped subset `_BINNED_TRAIN_BINS`. Evaluation always uses the full
`test_bins`. Outputs go to `research/baselines/deeponet/output-deeponet/<system>/`.
`scripts/retrain_and_eval_van_der_pol.sh` runs both domains and
`deeponet_relerr_eval.py` for Van der Pol.

`deeponet_relerr_eval.py` re-evaluates a checkpoint in the same metric as the
PID/StePPO pipeline (`relerr_metrics.py`), and `wallclock_deeponet_vdp.py`
times inference per task parameter.

## Publish to Hugging Face (DeepONet container)

```bash
PYTHONPATH=. python research/baselines/deeponet/deeponet_hub.py \
  --checkpoint research/baselines/deeponet/output-deeponet/van_der_pol/models/deeponet_van_der_pol_retrain_complete_final-30000.pt \
  --config research/baselines/deeponet/output-deeponet/van_der_pol/results/deeponet_van_der_pol_retrain_complete_training_config.json
```

Commits the raw checkpoint and a `metadata.json` to `MKoumans/deeponet`, one
branch per (system, domain): `sd-`, `vdp-` or `bru-` plus `complete`/`binned`,
the layout `paper/utils.py` in the StePPO paper repo reads. Any other
`*.pt` on the branch is removed in the same commit, since readers expect
exactly one. The checkpoint must load into the net described by `--config`
before anything is uploaded. `--staging` appends `-staging` to the branch
name, to validate an upload before replacing a live branch. Needs `HF_TOKEN`
in the environment.

## Scripts that need the main (JAX) container

These import `steppo` and run in `docker/docker-compose.yml`:

| Script | Output |
| --- | --- |
| `pid_steppo_relerr_eval.py` | PID and StePPO error tables on the DeepONet test set and bins |
| `generate_pid_rl_trajectory_example.py` | one representative task per system for Fig. 2 |
| `generate_trajectory_examples.py` | a few representative tasks per system |

`plot_pid_rl_deeponet_trajectory.py` (Fig. 2: trajectory and local error of
PID, StePPO and DeepONet) runs the DeepONet model and so needs the DeepONet
container.

## Plots (either container, NumPy/Matplotlib only)

| Script | Figure |
| --- | --- |
| `plot_pid_steppo_deeponet_sweep.py` | error vs task parameter, PID / StePPO / DeepONet complete and binned |
| `plot_error_vs_param_with_deeponet.py` | last-step and time-integrated error vs task parameter |
| `compare_deeponet_binned_complete.py`, `compare_deeponet_relerr.py`, `plot_deeponet_relerr.py` | DeepONet-only comparisons |
| `merge_deeponet_wallclock.py`, `plot_wallclock_vs_param.py`, `plot_wallclock_all_systems.py` | wall-clock vs task parameter (Fig. 5) |
| `plot_all_systems.py`, `plot_trajectory_examples.py` | loss curves and trajectory examples |
