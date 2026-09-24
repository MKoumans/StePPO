#!/bin/bash
#
# Pre-warm PID caches for one or more ODE systems.
# Full docs: scripts/README.md

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/device.sh"
REPO_ROOT="$(device_repo_root "${BASH_SOURCE[0]}")"
cd "${REPO_ROOT}"

ALL_SYSTEMS=(van_der_pol fitzhugh_nagumo brusselator chemical_cascade robertson scalar_decay fosm)

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [SYSTEM...] [OPTIONS]

Pre-warm PID training and evaluation caches for selected ODE systems.

Options:
  -g, --gpus DEVICES | --gpus=DEVICES  Round-robin GPU list.
  --force                              Rebuild existing caches.
  -h, --help                           Show help.

Full docs: scripts/README.md
EOF
}

# Preserved verbatim for the self-detach re-exec below, before the parsing
# loop consumes "$@" - keeps the detached re-exec in sync with whatever
# flags this script accepts, without needing to reconstruct them.
ORIGINAL_ARGS=("$@")

GPU_FILTERED_ARGS=()
device_extract_gpu_args GPUS GPU_FILTERED_ARGS "$@" || exit $?
set -- "${GPU_FILTERED_ARGS[@]}"

is_known_system() {
    local candidate
    for candidate in "${ALL_SYSTEMS[@]}"; do
        [[ "$1" == "${candidate}" ]] && return 0
    done
    return 1
}

SYSTEMS=()
FORCE="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --force)
            FORCE="true"
            shift
            ;;
        *)
            if is_known_system "$1"; then
                SYSTEMS+=("$1")
            else
                echo "[-] Unknown system '$1' (expected one of: ${ALL_SYSTEMS[*]})" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [ "${#SYSTEMS[@]}" -eq 0 ]; then
    SYSTEMS=("${ALL_SYSTEMS[@]}")
fi

IFS=',' read -ra GPU_LIST <<< "${GPUS}"

HAVE_SBATCH=1
device_have_sbatch || HAVE_SBATCH=0

# -- Direct mode (no Slurm): self-detach so the whole batch (including the  --
# -- per-GPU waiting below) survives the terminal/SSH session closing, the  --
# -- same way scripts/run-all-ode.sh detaches a batch of training runs.     --
# -- Skipped when already detached (_RUN_ALL_DETACHED guard).               --
if [ "${HAVE_SBATCH}" -eq 0 ] && [ -z "${_RUN_ALL_DETACHED:-}" ]; then
    LAUNCH_LOG_DIR="${REPO_ROOT}/outputs/logs/generate-pid-dataset"
    LAUNCH_LOG="${LAUNCH_LOG_DIR}/generate-pid-datasets-all-ode_$(date +%Y%m%d_%H%M%S)_$$.log"
    export _RUN_ALL_DETACHED=1
    device_detach "${LAUNCH_LOG}" "${ORIGINAL_ARGS[@]}"
    exit 0
fi

declare -A SLOT_PID

IDX=0
for SYSTEM in "${SYSTEMS[@]}"; do
    CONFIG="configs/envs/ode/${SYSTEM}/${SYSTEM}_default.yaml"
    if [ ! -f "${CONFIG}" ]; then
        echo "[-] Skipping ${SYSTEM}: no default config at ${CONFIG}"
        continue
    fi

    SYSTEM_GPU=""
    if [ "${#GPU_LIST[@]}" -gt 0 ]; then
        SYSTEM_GPU="${GPU_LIST[$(( IDX % ${#GPU_LIST[@]} ))]}"
    fi
    IDX=$(( IDX + 1 ))

    JOB_NAME="${SYSTEM}_pid_dataset"
    RUN_DATE="$(date +%Y%m%d)"

    if [ "${HAVE_SBATCH}" -eq 1 ]; then
        RUN_LOG_DIR="${REPO_ROOT}/outputs/logs/generate-pid-dataset/${RUN_DATE}"
        mkdir -p "${RUN_LOG_DIR}"
        echo "[+] Submitting ${JOB_NAME} (${CONFIG})${SYSTEM_GPU:+, GPU=${SYSTEM_GPU}}"
        JOB_ID=$(sbatch --parsable \
            --job-name="${JOB_NAME}" \
            --output="${RUN_LOG_DIR}/%x_%j.out" \
            --export="ALL,CONFIG=${CONFIG},FORCE=${FORCE}${SYSTEM_GPU:+,CUDA_VISIBLE_DEVICES=${SYSTEM_GPU}}" \
            scripts/generate-pid-dataset.sh)
        echo "[+] Submitted ${JOB_NAME} as Slurm job ${JOB_ID}"
        continue
    fi

    SLOT_KEY="${SYSTEM_GPU:-default}"
    if [ -n "${SLOT_PID[${SLOT_KEY}]:-}" ]; then
        echo "[*] Waiting for GPU ${SLOT_KEY} to free up (previous job PID ${SLOT_PID[${SLOT_KEY}]}) ..."
        wait "${SLOT_PID[${SLOT_KEY}]}" || true
    fi

    RUN_LOG_DIR="${REPO_ROOT}/outputs/logs/generate-pid-dataset/${RUN_DATE}"
    RUN_LOG="${RUN_LOG_DIR}/${JOB_NAME}.log"
    mkdir -p "${RUN_LOG_DIR}"
    echo "[+] Running ${JOB_NAME} (${CONFIG})${SYSTEM_GPU:+, GPU=${SYSTEM_GPU}} - log: ${RUN_LOG}"
    (
        CODE=0
        CONFIG="${CONFIG}" FORCE="${FORCE}" _LAUNCH_DETACHED=1 \
            device_run_with_gpu "${SYSTEM_GPU}" bash scripts/generate-pid-dataset.sh >"${RUN_LOG}" 2>&1 || CODE=$?
        if [ "$CODE" -ne 0 ]; then
            echo "[-] ${JOB_NAME} failed with exit code ${CODE} (see ${RUN_LOG})"
            exit "$CODE"
        fi
    ) &
    SLOT_PID[${SLOT_KEY}]=$!
done

if [ "${HAVE_SBATCH}" -eq 0 ]; then
    for PID in "${SLOT_PID[@]}"; do
        wait "${PID}" || true
    done
fi
