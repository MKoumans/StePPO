#!/bin/bash
#SBATCH --job-name=ode-average-seeds
#SBATCH --output=./outputs/logs/slurm/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:10:00

# Average metrics across repeated seeds.
# Full docs: scripts/README.md

set -e

usage() {
    cat <<EOF
Usage:
  $(basename "$0") --runs DIR... --out DIR
  $(basename "$0") --discover-experiment ROOT --prefix PREFIX --out DIR

Average metrics across repeated seeds.

Options:
  -g, --gpus DEVICES | --gpus=DEVICES  GPU(s) to expose.
  -h, --help                           Show help.

Full docs: scripts/README.md
EOF
}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "${REPO_ROOT}/scripts/lib/device.sh" ] || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/lib/device.sh"
DEVICE_DEFAULT_GPU=6
GPU_ARG=""
SELF_SUBMIT_ARGS=("$@")
FORWARD_ARGS=()
device_extract_gpu_args GPU_ARG FORWARD_ARGS "$@" || exit $?
ORIG_ARGS=("${FORWARD_ARGS[@]}")
if [ -n "${GPU_ARG}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ARG}"
fi
if [[ "${FORWARD_ARGS[0]:-}" == "-h" || "${FORWARD_ARGS[0]:-}" == "--help" ]]; then
    usage
    exit 0
fi

# On an HPC login node (sbatch available, not yet inside a job), self-submit
# as a Slurm job — jax only exists inside the apptainer image on compute nodes.
if device_have_sbatch && ! device_on_slurm; then
    LOG_DIR="${REPO_ROOT}/outputs/logs/average-seeds"
    mkdir -p "${LOG_DIR}"
    JOB_ID=$(cd "${REPO_ROOT}" && sbatch --parsable --output="${LOG_DIR}/%x_%j.out" "${BASH_SOURCE[0]}" "${SELF_SUBMIT_ARGS[@]}")
    echo "[+] Submitted as Slurm job ${JOB_ID}"
    echo "    Follow with: tail -f ${LOG_DIR}/ode-average-seeds_${JOB_ID}.out"
    exit 0
fi

device_run_py "${REPO_ROOT}/research/ode/post_run_analysis/learning_statistics.py" average-seeds "${ORIG_ARGS[@]}"
