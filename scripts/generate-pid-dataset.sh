#!/bin/bash
#SBATCH --job-name=generate-pid-dataset
#SBATCH --output=./outputs/logs/slurm/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [OPTIONS]

Runs generate_pid_training_dataset.py then generate_pid_eval_dataset.py
against \$CONFIG; results are cached under data/<system>/training/ and data/<system>/evaluation/<split>set/. Figures are exported under outputs/datasets/<system>/training/ and outputs/datasets/<system>/evaluation/<split>set/.

Options:
  -g, --gpus DEVICES | --gpus=DEVICES   GPU(s) to expose to dataset generation, e.g. "0" or
                       "0,1". Defaults to GPU 0 when omitted.
  -h, --help           Show this help message and exit.

Env vars: CONFIG, FORCE, _LAUNCH_DETACHED.

Full docs: scripts/README.md

Examples:
  sbatch scripts/generate-pid-dataset.sh
  bash scripts/generate-pid-dataset.sh --gpus 5
  sbatch --export=ALL,CONFIG=configs/envs/ode/brusselator/brusselator_default.yaml scripts/generate-pid-dataset.sh
EOF
}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "${REPO_ROOT}/scripts/lib/device.sh" ] || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/lib/device.sh" || exit 1

GPU_ARG=""
SCRIPT_ARGS=()
device_extract_gpu_args GPU_ARG SCRIPT_ARGS "$@" || exit $?
if [ -n "${GPU_ARG}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ARG}"
fi
if [[ "${SCRIPT_ARGS[0]:-}" == "-h" || "${SCRIPT_ARGS[0]:-}" == "--help" ]]; then
    usage
    exit 0
fi

# CONFIG can be overridden without editing this file, e.g.:
#   sbatch --export=ALL,CONFIG=configs/envs/ode/brusselator/brusselator_default.yaml scripts/generate-pid-dataset.sh
# (scripts/generate-pid-datasets-all-ode.sh does this for every ODE system in one go)
CONFIG="${CONFIG:-configs/envs/ode/chemical_cascade/chemical_cascade_default.yaml}"

ENV_NAME="$(basename "$(dirname "$(dirname "${CONFIG}")")")"
SYSTEM_NAME="$(basename "$(dirname "${CONFIG}")")"
RUN_NAME="${ENV_NAME}_${SYSTEM_NAME}"

FORCE_ARGS=()
[ "${FORCE:-false}" = "true" ] && FORCE_ARGS=(--force)

if [ -z "${_LAUNCH_DETACHED:-}" ] && ! device_on_slurm; then
    LAUNCH_LOG_DIR="${REPO_ROOT}/outputs/logs/generate-pid-dataset/$(date +%Y%m%d)"
    LAUNCH_LOG="${LAUNCH_LOG_DIR}/${RUN_NAME}_$$.log"
    export _LAUNCH_DETACHED=1 CONFIG="${CONFIG}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" FORCE="${FORCE:-false}"
    device_detach "${LAUNCH_LOG}" "${SCRIPT_ARGS[@]}"
    exit 0
fi

echo "[*] System : ${SYSTEM_NAME}"
echo "[*] Config : ${CONFIG}"
echo "[*] Force  : ${FORCE:-false}"

echo "[*] Generating PID training cache ..."
device_run_py src/steppo/envs/ode/generate_pid_training_dataset.py --config "${CONFIG}" "${FORCE_ARGS[@]}"

echo "[*] Generating PID eval cache(s) ..."
mapfile -t EVAL_ARG_LINES < <(device_run_py src/steppo/envs/ode/check_oracle_dataset.py --config "${CONFIG}" --print-eval-args)
for EVAL_ARGS in "${EVAL_ARG_LINES[@]}"; do
    # shellcheck disable=SC2086
    device_run_py src/steppo/envs/ode/generate_pid_eval_dataset.py --config "${CONFIG}" ${EVAL_ARGS} "${FORCE_ARGS[@]}"
done
