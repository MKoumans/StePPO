#!/usr/bin/env bash
# Execute mode, StePPO side: generate the PID datasets of one system's
# *_icassp2027 config, then train StePPO from scratch into
# outputs/paper/models/<system>/ (read by the figure scripts with
# --mode execute). Training takes hours; run it under nohup, tmux or Slurm.
# The DeepONets are trained separately in the DeepONet container (REPRODUCE.md).
#
#   bash paper/train.sh van_der_pol [--gpus 0] [extra research/ode/run.py args]
#   bash paper/train.sh van_der_pol --data-only [--gpus 0]   # datasets only, no training
set -euo pipefail

SYSTEM="${1:?usage: bash paper/train.sh <scalar_decay|van_der_pol|brusselator> [--gpus N] [--data-only] [run.py args]}"
shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
CONFIG="configs/envs/ode/${SYSTEM}/${SYSTEM}_icassp2027.yaml"
[ -f "${CONFIG}" ] || { echo "No config ${CONFIG}" >&2; exit 1; }

GPUS=0
DATA_ONLY=0
ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --gpus=*) GPUS="${1#*=}"; shift ;;
        --data-only) DATA_ONLY=1; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

echo "[*] PID datasets for ${CONFIG}"
CONFIG="${CONFIG}" _LAUNCH_DETACHED=1 bash scripts/generate-pid-dataset.sh --gpus "${GPUS}"
[ "${DATA_ONLY}" = 1 ] && exit 0

echo "[*] Training StePPO -> outputs/paper/models/${SYSTEM}"
PYTHONPATH=. python research/ode/run.py --config "${CONFIG}" --gpus "${GPUS}" \
    --run_uid "${SYSTEM}" --checkpoints_dir outputs/paper/models \
    --outputs_dir outputs/paper/training ${ARGS[@]+"${ARGS[@]}"}
