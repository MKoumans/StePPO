#!/bin/bash
#SBATCH --job-name=upload-to-huggingface
#SBATCH --output=./outputs/logs/slurm/%x_%j.out
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

set -e

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "${REPO_ROOT}/scripts/lib/device.sh" ] || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/lib/device.sh"

# Export a training checkpoint as a model artifact and upload it to the Hugging Face Hub.
# Usage: scripts/upload-model.sh <checkpoint_dir> <hf_user>/<repo_name>
if [ "$#" -ne 2 ]; then
    echo "Usage: $(basename "$0") <checkpoint_dir> <hf_user>/<repo_name>"
    exit 1
fi
CHECKPOINT="$1"
REPO_ID="$2"

if [ -z "${HF_TOKEN:-}" ] && [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi
: "${HF_TOKEN:?HF_TOKEN must be set, either exported before sbatch or defined in .env}"
export HF_TOKEN

device_run_py -m steppo.models.huggingface upload \
  --checkpoint "${CHECKPOINT}" \
  --repo-id "${REPO_ID}"
