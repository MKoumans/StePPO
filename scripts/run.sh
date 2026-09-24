#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=./outputs/logs/slurm/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00 # Estimated maximum runtime

# Run one ODE training job locally or through Slurm.
# Full docs: scripts/README.md

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [PYTHON_ARGS...]

Run one ODE training job; extra arguments go to research/ode/run.py.

Options:
  -g, --gpus DEVICES | --gpus=DEVICES  GPU(s) to expose.
  -h, --help                          Show help.

Full docs: scripts/README.md
EOF
}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "${REPO_ROOT}/scripts/lib/device.sh" ] || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/lib/device.sh" || exit 1

GPU_ARG=""
PYTHON_ARGS=()
device_extract_gpu_args GPU_ARG PYTHON_ARGS "$@" || exit $?
if [ -n "${GPU_ARG}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ARG}"
fi
if [[ "${PYTHON_ARGS[0]:-}" == "-h" || "${PYTHON_ARGS[0]:-}" == "--help" ]]; then
    usage
    exit 0
fi

# Select the run with CONFIG=configs/envs/ode/<system>/<system>_default.yaml.
CONFIG="${CONFIG:-configs/envs/ode/chemical_cascade/chemical_cascade_default.yaml}"

# Derive env and system from configs/envs/<env>/<system>/... for the log name and run ID.
ENV_NAME="$(basename "$(dirname "$(dirname "${CONFIG}")")")"
SYSTEM_NAME="$(basename "$(dirname "${CONFIG}")")"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_UID="${RUN_UID:-$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")}"
RUN_NAME="${ENV_NAME}_${SYSTEM_NAME}_${RUN_UID}"

# Without Slurm, detach from the terminal so training survives closing it.
if [ -z "${_LAUNCH_DETACHED:-}" ] && ! device_on_slurm; then
    LAUNCH_LOG_DIR="${REPO_ROOT}/outputs/logs/run/${RUN_DATE}"
    LAUNCH_LOG="${LAUNCH_LOG_DIR}/${RUN_NAME}.log"
    export _LAUNCH_DETACHED=1 CONFIG="${CONFIG}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" RUN_UID="${RUN_UID}" RUN_DATE="${RUN_DATE}"
    device_detach "${LAUNCH_LOG}" "${PYTHON_ARGS[@]}"
    exit 0
fi

device_run_py src/steppo/envs/ode/check_oracle_dataset.py --config "${CONFIG}" || exit $?

device_run_py research/ode/run.py --config "${CONFIG}" --run_uid "${RUN_UID}" --run_date "${RUN_DATE}" "${PYTHON_ARGS[@]}"
