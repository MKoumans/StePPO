#!/bin/bash
#SBATCH --job-name=ode-post-run-analysis
#SBATCH --output=./outputs/logs/slurm/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00

# Run post-training analysis for one checkpoint, run directory, or config.
# Full docs: scripts/README.md

set -e

usage() {
    cat <<EOF
Usage:
  $(basename "$0") --checkpoint PATH [--config PATH]
  $(basename "$0") --config PATH
  $(basename "$0") --run-dir PATH

Run the ODE post-training analysis suite.

Options:
  --out-dir PATH                       Output directory.
  -g, --gpus DEVICES | --gpus=DEVICES  GPU(s) to expose.
  -h, --help                           Show help.

Full docs: scripts/README.md
EOF
}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "${REPO_ROOT}/scripts/lib/device.sh" ] || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/lib/device.sh"
DEVICE_DEFAULT_GPU=6

ORIG_ARGS=("$@")

CHECKPOINT=""
CONFIG=""
RUN_DIR_ARG=""
OUT_DIR=""
GPU_ARG=""
GPU_FILTERED_ARGS=()
device_extract_gpu_args GPU_ARG GPU_FILTERED_ARGS "$@" || exit $?
set -- "${GPU_FILTERED_ARGS[@]}"
if [ -n "${GPU_ARG}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ARG}"
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2";  shift 2 ;;
        --config)     CONFIG="$2";      shift 2 ;;
        --run-dir)    RUN_DIR_ARG="$2"; shift 2 ;;
        --out-dir)    OUT_DIR="$2";     shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -n "$CHECKPOINT" ] && [ -n "$RUN_DIR_ARG" ]; then
    echo "--checkpoint and --run-dir are mutually exclusive"
    exit 1
fi

if [ -z "$CHECKPOINT" ] && [ -z "$CONFIG" ] && [ -z "$RUN_DIR_ARG" ]; then
    echo "Usage: $0 --checkpoint <path> [--config <path>]"
    echo "  One of --checkpoint, --config, or --run-dir must be provided."
    exit 1
fi

# On an HPC login node (sbatch available, not yet inside a job), self-submit
# as a Slurm job — jax only exists inside the apptainer image on compute nodes.
if device_have_sbatch && ! device_on_slurm; then
    LOG_DIR="${REPO_ROOT}/outputs/logs/post-run-analysis"
    mkdir -p "${LOG_DIR}"
    JOB_ID=$(cd "${REPO_ROOT}" && sbatch --parsable --output="${LOG_DIR}/%x_%j.out" "${BASH_SOURCE[0]}" "${ORIG_ARGS[@]}")
    echo "[+] Submitted as Slurm job ${JOB_ID}"
    echo "    Follow with: tail -f ${LOG_DIR}/ode-post-run-analysis_${JOB_ID}.out"
    exit 0
fi

if [ -n "$RUN_DIR_ARG" ]; then
    # Single run from scripts/run.sh: outputs/runs/<DATE>/<UID>/checkpoints/checkpoint_*
    LATEST_CKPT="$(find "${RUN_DIR_ARG}/checkpoints" -maxdepth 1 -type d -name 'checkpoint_*' -printf '%f\n' 2>/dev/null | sort -V | tail -1)"
    if [ -z "$LATEST_CKPT" ]; then
        echo "[!] No checkpoint_* dirs found in ${RUN_DIR_ARG}/checkpoints"
        exit 1
    fi
    CHECKPOINT="${RUN_DIR_ARG}/checkpoints/${LATEST_CKPT}"
    RUN_ID="$(basename "$RUN_DIR_ARG")"
    RUN_DIR="$RUN_DIR_ARG"
elif [ -n "$CHECKPOINT" ]; then
    CKPT_PARENT="$(dirname "$CHECKPOINT")"
    RUN_ID="$(basename "$CKPT_PARENT")"
    # Checkpoints from scripts/run.sh live at outputs/runs/<DATE>/.../<UID>/checkpoints/
    # checkpoint_N; the run root is the checkpoints dir's parent, and the run
    # UID is its basename (not the literal "checkpoints" dir).
    if [ "$RUN_ID" = "checkpoints" ]; then
        RUN_DIR="$(dirname "$CKPT_PARENT")"
        RUN_ID="$(basename "$RUN_DIR")"
    fi
else
    RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
fi

[ -n "$OUT_DIR" ] || {
    if [ -n "$RUN_DIR" ]; then
        # Run root known (--run-dir, or a <run_root>/checkpoints/checkpoint_N
        # checkpoint): keep outputs alongside the run's checkpoints.
        OUT_DIR="${RUN_DIR}/outputs"
    elif [ -n "$CHECKPOINT" ] && [[ "$CHECKPOINT" == */checkpoints/* ]]; then
        # Experiment-batch layout (.../<UID>/checkpoints/<date>/<env>/<system>/
        # <run_uid>/checkpoint_N): mirror "checkpoints" to "outputs".
        OUT_DIR="$(dirname "${CHECKPOINT/\/checkpoints\//\/outputs\/}")"
    else
        OUT_DIR="outputs/experiment/${RUN_ID}"
    fi
}
mkdir -p "${OUT_DIR}"

CKPT_ARGS=()
if [ -n "$CHECKPOINT" ]; then
    CKPT_ARGS=(--checkpoint "$CHECKPOINT")
fi

CONFIG_ARGS=()
if [ -n "$CONFIG" ]; then
    CONFIG_ARGS=(--config "$CONFIG")
fi

echo "[*] Checkpoint : ${CHECKPOINT:-<none>}"
echo "[*] Config     : ${CONFIG:-<auto-detect from checkpoint>}"
echo "[*] Run ID     : ${RUN_ID}"
echo "[*] Output dir : ${OUT_DIR}"

FAILED=0

echo "[*] Running compare.py ..."
device_run_py research/ode/post_run_analysis/compare.py "${CONFIG_ARGS[@]}" "${CKPT_ARGS[@]}" --out "${OUT_DIR}/compare" \
    || { echo "[!] compare.py failed, continuing"; FAILED=1; }

echo "[*] Running steps_vs_mu.py (log-scale bins) ..."
device_run_py research/ode/post_run_analysis/steps_vs_mu.py "${CONFIG_ARGS[@]}" "${CKPT_ARGS[@]}" --log_bins -o "${OUT_DIR}/steps_vs_mu.png" \
    || { echo "[!] steps_vs_mu.py failed, continuing"; FAILED=1; }

echo "[*] Running analyze_rollout.py ..."
device_run_py research/ode/post_run_analysis/analyse_rollout.py "${CONFIG_ARGS[@]}" "${CKPT_ARGS[@]}" --out "${OUT_DIR}/analyze_rollout" \
    || { echo "[!] analyze_rollout.py failed, continuing"; FAILED=1; }

echo "[*] Running collapse_vs_task.py ..."
device_run_py research/ode/post_run_analysis/collapse_vs_task.py "${CONFIG_ARGS[@]}" "${CKPT_ARGS[@]}" --out "${OUT_DIR}/collapse_vs_task" \
    || { echo "[!] collapse_vs_task.py failed, continuing"; FAILED=1; }

echo "[*] Running z_sensitivity.py ..."
device_run_py research/ode/post_run_analysis/z_sensitivity.py "${CONFIG_ARGS[@]}" "${CKPT_ARGS[@]}" --out "${OUT_DIR}/z_sensitivity" \
    || { echo "[!] z_sensitivity.py failed, continuing"; FAILED=1; }

if [ "$FAILED" -eq 0 ]; then
    echo "[+] All experiment outputs saved to: ${OUT_DIR}"
else
    echo "[-] Some analysis steps failed; partial outputs saved to: ${OUT_DIR}"
fi
exit "$FAILED"
