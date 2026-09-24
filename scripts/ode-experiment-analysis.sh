#!/bin/bash
#SBATCH --job-name=ode-experiment-analysis
#SBATCH --output=./outputs/logs/slurm/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00

# Analyze all runs in one experiment batch and aggregate their results.
# Full docs: scripts/README.md

set -e

usage() {
    cat <<EOF
Usage: $(basename "$0") --experiment_uid UID [OPTIONS] [COMPARE_ARGS...]

Analyze and compare all runs in an experiment batch.

Options:
  --experiment_date YYYYMMDD           Date of the experiment batch.
  --skip-per-run                       Skip ode-post-run-analysis.sh for each run;
                                       go straight to the cross-run comparison.
  -g, --gpus DEVICES | --gpus=DEVICES  GPU(s) to expose.
  -h, --help                           Show help.

Full docs: scripts/README.md
EOF
}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "${REPO_ROOT}/scripts/lib/device.sh" ] || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/lib/device.sh"
DEVICE_DEFAULT_GPU=6

EXPERIMENT_UID=""
EXPERIMENT_DATE=""
SKIP_PER_RUN="false"
GPU_ARG=""
GPU_FILTERED_ARGS=()
device_extract_gpu_args GPU_ARG GPU_FILTERED_ARGS "$@" || exit $?
set -- "${GPU_FILTERED_ARGS[@]}"
if [ -n "${GPU_ARG}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ARG}"
fi
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --experiment_uid)  EXPERIMENT_UID="$2";  shift 2 ;;
        --experiment_date) EXPERIMENT_DATE="$2"; shift 2 ;;
        --skip-per-run)    SKIP_PER_RUN="true";  shift ;;
        -h|--help)         usage; exit 0 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$EXPERIMENT_UID" ]; then
    echo "Usage: $0 --experiment_uid <UID> [--experiment_date <YYYYMMDD>] [extra args...]"
    exit 1
fi

EXP_ID="${EXPERIMENT_UID#experiment_}"

if [ -z "$EXPERIMENT_DATE" ]; then
    mapfile -t DATE_MATCHES < <(find "${REPO_ROOT}/outputs/experiments" -mindepth 2 -maxdepth 2 -type d -name "${EXP_ID}" 2>/dev/null | sort)
    if [ "${#DATE_MATCHES[@]}" -eq 0 ]; then
        echo "[!] Could not find outputs/experiments/*/${EXP_ID}; pass --experiment_date explicitly"
        exit 1
    fi
    EXPERIMENT_ROOT="${DATE_MATCHES[-1]}"
    EXPERIMENT_DATE="$(basename "$(dirname "${EXPERIMENT_ROOT}")")"
else
    EXPERIMENT_ROOT="${REPO_ROOT}/outputs/experiments/${EXPERIMENT_DATE}/${EXP_ID}"
fi

echo "[*] Experiment UID  : ${EXPERIMENT_UID}"
echo "[*] Experiment date : ${EXPERIMENT_DATE}"
echo "[*] Extra args      : ${EXTRA_ARGS[*]:-<none>}"

CHECKPOINTS_ROOT="${EXPERIMENT_ROOT}/checkpoints"

# Run dirs are the deepest checkpoint-tree directories that contain a
# checkpoint_* subdir (run_uid is a bare hex id, not "run_"-prefixed).
mapfile -t RUN_DIRS < <(find "${CHECKPOINTS_ROOT}" -type d -name 'checkpoint_*' 2>/dev/null -printf '%h\n' | sort -u)

if [ "${SKIP_PER_RUN}" = "true" ]; then
    echo "[*] --skip-per-run: skipping ode-post-run-analysis.sh for all runs"
elif [ "${#RUN_DIRS[@]}" -eq 0 ]; then
    echo "[!] No run directories (containing checkpoint_*) found under ${CHECKPOINTS_ROOT}, skipping per-run analysis"
else
    FAILED_RUNS=()
    for run_dir in "${RUN_DIRS[@]}"; do
        latest_ckpt="$(find "${run_dir}" -maxdepth 1 -type d -name 'checkpoint_*' -printf '%f\n' | sort -V | tail -1)"
        if [ -z "$latest_ckpt" ]; then
            echo "[!] Skipping ${run_dir}: no checkpoint_* dirs"
            continue
        fi
        echo "[*] Running ode-post-run-analysis.sh for ${run_dir}/${latest_ckpt} ..."
        RUN_OUT_DIR="${run_dir/\/checkpoints\//\/outputs\/}"
        POST_RUN_GPU_ARGS=()
        [ -n "${GPU_ARG}" ] && POST_RUN_GPU_ARGS+=(--gpus "${GPU_ARG}")
        bash "${REPO_ROOT}/scripts/ode-post-run-analysis.sh" --checkpoint "${run_dir}/${latest_ckpt}" --out-dir "${RUN_OUT_DIR}" "${POST_RUN_GPU_ARGS[@]}" \
            || { echo "[!] Per-run analysis failed for ${run_dir}, continuing"; FAILED_RUNS+=("${run_dir}"); }
    done
    if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
        echo "[-] Per-run analysis failed for ${#FAILED_RUNS[@]} run(s): ${FAILED_RUNS[*]}"
    fi
fi

echo "[*] Running compare_experiment.py ..."
device_run_py research/ode/compare_experiment.py --experiment_uid "${EXPERIMENT_UID}" --experiment_date "${EXPERIMENT_DATE}" --workdir "${REPO_ROOT}" "${EXTRA_ARGS[@]}"

echo "[*] Averaging repeat groups (--repeat > 1) via learning_statistics.py average-seeds ..."
device_run_py research/ode/post_run_analysis/learning_statistics.py average-seeds \
    --discover-experiment "${EXPERIMENT_ROOT}/outputs" \
    --out "${EXPERIMENT_ROOT}/outputs/comparison"

echo "[+] Done."
