#!/bin/bash
#SBATCH --job-name=run-all-ode
#SBATCH --output=./outputs/logs/slurm/%x_%j.out
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#
# Launch one run per selected ODE system, with optional repeated seeds.
# Full docs: scripts/README.md

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPTS_DIR}/lib/device.sh" ] || SCRIPTS_DIR="${SLURM_SUBMIT_DIR:-}/scripts"
source "${SCRIPTS_DIR}/lib/device.sh"
REPO_ROOT="$(device_repo_root "${BASH_SOURCE[0]}")"
cd "${REPO_ROOT}"

ALL_SYSTEMS=(van_der_pol fitzhugh_nagumo brusselator chemical_cascade robertson scalar_decay fosm chua_smooth)

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [SYSTEM...] [OPTIONS]

Launch one run per selected ODE system.

Options:
  -g, --gpus DEVICES | --gpus=DEVICES  Round-robin GPU list.
  -r, --repeat N                       Seeded repeats (default: 1).
  --post-run-analysis BOOL             Enable or disable post-run analysis.
  --generate-datasets                  Generate missing training/PID evaluation caches first.
  --KEY VALUE                          Forward a TrainConfig override.
  -h, --help                           Show help.

Full docs: scripts/README.md
EOF
}

# Keep the original arguments for the self-detach re-exec below.
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
EXTRA_ARGS=()
REPEAT=1
POST_RUN_ANALYSIS="true"
GENERATE_DATASETS="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -r|--repeat)
            REPEAT="$2"
            shift 2
            ;;
        --repeat=*)
            REPEAT="${1#*=}"
            shift
            ;;
        --post-run-analysis)
            POST_RUN_ANALYSIS="$2"
            shift 2
            ;;
        --post-run-analysis=*)
            POST_RUN_ANALYSIS="${1#*=}"
            shift
            ;;
        --generate-datasets)
            GENERATE_DATASETS="true"
            shift
            ;;
        --*)
            # Arbitrary TrainConfig override forwarded to run.py, e.g.
            # --rollout_steps 400 or --env.progress_warp true.
            if [ $# -lt 2 ]; then
                echo "[-] Override '$1' is missing a value" >&2
                exit 1
            fi
            EXTRA_ARGS+=("$1" "$2")
            shift 2
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

case "$POST_RUN_ANALYSIS" in
    true|false) ;;
    *) echo "--post-run-analysis must be 'true' or 'false', got: ${POST_RUN_ANALYSIS}" >&2; exit 1 ;;
esac

if [ "${#SYSTEMS[@]}" -eq 0 ]; then
    SYSTEMS=("${ALL_SYSTEMS[@]}")
fi

if ! [[ "${REPEAT}" =~ ^[0-9]+$ ]] || [ "${REPEAT}" -lt 1 ]; then
    echo "[-] --repeat must be a positive integer, got: ${REPEAT}" >&2
    exit 1
fi

case "${GENERATE_DATASETS}" in
    true|false) ;;
    *) echo "--generate-datasets must be enabled as a flag" >&2; exit 1 ;;
esac

IFS=',' read -ra GPU_LIST <<< "${GPUS}"

HAVE_SBATCH=1
device_have_sbatch || HAVE_SBATCH=0

if [ "${HAVE_SBATCH}" -eq 1 ] && ! device_on_slurm; then
    LAUNCH_LOG_DIR="${REPO_ROOT}/outputs/logs/run-all-ode"
    mkdir -p "${LAUNCH_LOG_DIR}"
    JOB_ID=$(sbatch --parsable \
        --job-name="run-all-ode" \
        --output="${LAUNCH_LOG_DIR}/%j.out" \
        "${REPO_ROOT}/scripts/run-all-ode.sh" "${ORIGINAL_ARGS[@]}")
    echo "[*] On an HPC login node with no active Slurm job; resubmitting this launcher as Slurm job ${JOB_ID} (partition=gpu_mig) so Apptainer runs on a compute node."
    echo "[*] Follow with: tail -f ${LAUNCH_LOG_DIR}/${JOB_ID}.out"
    exit 0
fi

# Without Slurm, detach from the terminal so the whole batch survives closing it.
if [ "${HAVE_SBATCH}" -eq 0 ] && [ -z "${_RUN_ALL_DETACHED:-}" ]; then
    LAUNCH_LOG_DIR="${REPO_ROOT}/outputs/logs/run"
    LAUNCH_LOG="${LAUNCH_LOG_DIR}/run-all-ode_$(date +%Y%m%d_%H%M%S)_$$.log"
    export _RUN_ALL_DETACHED=1
    device_detach "${LAUNCH_LOG}" "${ORIGINAL_ARGS[@]}"
    exit 0
fi

# Check (and optionally generate) all caches serially before training, so runs
# do not race to create the same cache.
CONFIGS=()
for SYSTEM in "${SYSTEMS[@]}"; do
    CONFIG="configs/envs/ode/${SYSTEM}/${SYSTEM}_default.yaml"
    if [ -f "${CONFIG}" ]; then
        CONFIGS+=("${CONFIG}")
    fi
done

GENERATION_GPU="${GPU_LIST[0]:-}"
run_generation_py() {
    device_run_with_gpu "${GENERATION_GPU}" device_run_py "$@"
}

check_datasets() {
    local missing=0
    local config
    for config in "${CONFIGS[@]}"; do
        run_generation_py src/steppo/envs/ode/check_oracle_dataset.py --config "${config}" >&2 \
            || missing=$((missing + 1))
    done
    echo "${missing}"
}

echo "[*] Checking training and PID evaluation datasets for ${#CONFIGS[@]} config(s) ..."
MISSING="$(check_datasets)"

if [ "${MISSING}" -gt 0 ] && [ "${GENERATE_DATASETS}" = "true" ]; then
    echo ""
    echo "[*] --generate-datasets: generating missing training and PID evaluation datasets ..."
    for CONFIG in "${CONFIGS[@]}"; do
        echo "[*] Generating PID training cache for ${CONFIG} ..."
        run_generation_py src/steppo/envs/ode/generate_pid_training_dataset.py --config "${CONFIG}"
        echo "[*] Generating PID eval cache(s) for ${CONFIG} ..."
        mapfile -t EVAL_ARG_LINES < <(run_generation_py src/steppo/envs/ode/check_oracle_dataset.py \
            --config "${CONFIG}" --print-eval-args)
        for EVAL_ARGS in "${EVAL_ARG_LINES[@]}"; do
            # shellcheck disable=SC2086
            run_generation_py src/steppo/envs/ode/generate_pid_eval_dataset.py \
                --config "${CONFIG}" ${EVAL_ARGS}
        done
    done
    echo ""
    echo "[*] Re-checking datasets ..."
    MISSING="$(check_datasets)"
fi

if [ "${MISSING}" -gt 0 ]; then
    echo ""
    echo "[-] ${MISSING}/${#CONFIGS[@]} config(s) are missing required training or PID evaluation datasets." >&2
    if [ "${GENERATE_DATASETS}" = "true" ]; then
        echo "    Dataset generation was attempted but datasets are still missing; check the errors above." >&2
    else
        echo "    Re-run with --generate-datasets, or run the commands printed above." >&2
    fi
    exit 1
fi

declare -A SLOT_PID
AVG_PIDS=()

IDX=0
for SYSTEM in "${SYSTEMS[@]}"; do
    CONFIG="configs/envs/ode/${SYSTEM}/${SYSTEM}_default.yaml"
    if [ ! -f "${CONFIG}" ]; then
        echo "[-] Skipping ${SYSTEM}: no default config at ${CONFIG}"
        continue
    fi

    # Jobs and run dirs of this system's repeats, for the seed-averaging step.
    SYSTEM_RUN_DIRS=()
    SYSTEM_JOB_IDS=()
    SYSTEM_PIDS=()
    SYSTEM_RUN_DATE=""

    for (( R=0; R<REPEAT; R++ )); do
        SYSTEM_GPU=""
        if [ "${#GPU_LIST[@]}" -gt 0 ]; then
            SYSTEM_GPU="${GPU_LIST[$(( IDX % ${#GPU_LIST[@]} ))]}"
        fi
        IDX=$(( IDX + 1 ))

        RUN_ARGS=("${EXTRA_ARGS[@]}")
        JOB_NAME="${SYSTEM}"
        if [ "${REPEAT}" -gt 1 ]; then
            RUN_ARGS+=(--seed "${R}")
            JOB_NAME="${SYSTEM}_r${R}"
        fi

        # Fix RUN_UID/RUN_DATE up front so the post-run analysis knows the run dir.
        RUN_DATE="$(date +%Y%m%d)"
        RUN_UID="$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")"
        RUN_DIR="${REPO_ROOT}/outputs/runs/${RUN_DATE}/ode/${SYSTEM}/${RUN_UID}"
        SYSTEM_RUN_DIRS+=("${RUN_DIR}")
        [ -n "${SYSTEM_RUN_DATE}" ] || SYSTEM_RUN_DATE="${RUN_DATE}"

        if [ "${HAVE_SBATCH}" -eq 1 ]; then
            RUN_LOG_DIR="${REPO_ROOT}/outputs/logs/run/${RUN_DATE}"
            mkdir -p "${RUN_LOG_DIR}"
            echo "[+] Submitting ${JOB_NAME} (${CONFIG})${SYSTEM_GPU:+, GPU=${SYSTEM_GPU}}"
            JOB_ID=$(sbatch --parsable \
                --job-name="${JOB_NAME}" \
                --output="${RUN_LOG_DIR}/%x_%j.out" \
                --export="ALL,CONFIG=${CONFIG},RUN_UID=${RUN_UID},RUN_DATE=${RUN_DATE}${SYSTEM_GPU:+,CUDA_VISIBLE_DEVICES=${SYSTEM_GPU}}" \
                scripts/run.sh "${RUN_ARGS[@]}")
            echo "[+] Submitted ${JOB_NAME} as Slurm job ${JOB_ID} (run dir: ${RUN_DIR})"
            if [ "${POST_RUN_ANALYSIS}" = "true" ]; then
                ANALYSIS_LOG_DIR="${REPO_ROOT}/outputs/logs/post-run-analysis/${RUN_DATE}"
                mkdir -p "${ANALYSIS_LOG_DIR}"
                ANALYSIS_JOB_ID=$(sbatch --parsable \
                    --job-name="${JOB_NAME}_analysis" \
                    --output="${ANALYSIS_LOG_DIR}/%x_%j.out" \
                    --dependency="afterok:${JOB_ID}" \
                    --kill-on-invalid-dep=yes \
                    scripts/ode-post-run-analysis.sh --run-dir "${RUN_DIR}")
                echo "[+] Submitted ${JOB_NAME} post-run analysis as Slurm job ${ANALYSIS_JOB_ID} (after: ${JOB_ID})"
                SYSTEM_JOB_IDS+=("${ANALYSIS_JOB_ID}")
            else
                SYSTEM_JOB_IDS+=("${JOB_ID}")
            fi
            continue
        fi

        SLOT_KEY="${SYSTEM_GPU:-default}"
        if [ -n "${SLOT_PID[${SLOT_KEY}]:-}" ]; then
            echo "[*] Waiting for GPU ${SLOT_KEY} to free up (previous job PID ${SLOT_PID[${SLOT_KEY}]}) ..."
            wait "${SLOT_PID[${SLOT_KEY}]}" || true
        fi

        RUN_LOG_DIR="${REPO_ROOT}/outputs/logs/run/${RUN_DATE}"
        RUN_LOG="${RUN_LOG_DIR}/ode_${JOB_NAME}_${RUN_UID}.log"
        mkdir -p "${RUN_LOG_DIR}"
        echo "[+] Running ${JOB_NAME} (${CONFIG})${SYSTEM_GPU:+, GPU=${SYSTEM_GPU}} - log: ${RUN_LOG}"
        (
            CODE=0
            CONFIG="${CONFIG}" RUN_UID="${RUN_UID}" RUN_DATE="${RUN_DATE}" _LAUNCH_DETACHED=1 \
                device_run_with_gpu "${SYSTEM_GPU}" bash scripts/run.sh "${RUN_ARGS[@]}" >"${RUN_LOG}" 2>&1 || CODE=$?
            if [ "$CODE" -ne 0 ]; then
                echo "[-] ${JOB_NAME} failed with exit code ${CODE} (see ${RUN_LOG})"
                exit "$CODE"
            fi
            if [ "${POST_RUN_ANALYSIS}" = "true" ]; then
                ANALYSIS_LOG="${RUN_LOG_DIR}/ode_${JOB_NAME}_${RUN_UID}_analysis.log"
                echo "[+] Running post-run analysis for ${JOB_NAME} -> ${ANALYSIS_LOG}"
                CUDA_VISIBLE_DEVICES="${SYSTEM_GPU:-0}" \
                    bash scripts/ode-post-run-analysis.sh --run-dir "${RUN_DIR}" >"${ANALYSIS_LOG}" 2>&1 \
                    || echo "[-] Post-run analysis for ${JOB_NAME} failed (see ${ANALYSIS_LOG})"
            fi
        ) &
        SLOT_PID[${SLOT_KEY}]=$!
        SYSTEM_PIDS+=("$!")
    done

    if [ "${REPEAT}" -gt 1 ] && [ "${POST_RUN_ANALYSIS}" = "true" ]; then
        SEED_AVG_OUT="${REPO_ROOT}/outputs/runs/${SYSTEM_RUN_DATE}/ode/${SYSTEM}/seed_average"
        SEED_AVG_RUN_ARGS=()
        for RD in "${SYSTEM_RUN_DIRS[@]}"; do
            SEED_AVG_RUN_ARGS+=("${RD}/outputs")
        done

        if [ "${HAVE_SBATCH}" -eq 1 ]; then
            AVG_LOG_DIR="${REPO_ROOT}/outputs/logs/average-seeds/${SYSTEM_RUN_DATE}"
            mkdir -p "${AVG_LOG_DIR}"
            DEPENDENCY="afterok:$(IFS=:; echo "${SYSTEM_JOB_IDS[*]}")"
            AVG_JOB_ID=$(sbatch --parsable \
                --job-name="${SYSTEM}_seed_average" \
                --output="${AVG_LOG_DIR}/%x_%j.out" \
                --dependency="${DEPENDENCY}" \
                --kill-on-invalid-dep=yes \
                scripts/ode-average-seeds.sh --runs "${SEED_AVG_RUN_ARGS[@]}" --out "${SEED_AVG_OUT}")
            echo "[+] Submitted ${SYSTEM} seed-averaging as Slurm job ${AVG_JOB_ID} (after: ${SYSTEM_JOB_IDS[*]})"
        else
            AVG_LOG_DIR="${REPO_ROOT}/outputs/logs/run/${SYSTEM_RUN_DATE}"
            AVG_LOG="${AVG_LOG_DIR}/ode_${SYSTEM}_seed_average.log"
            mkdir -p "${AVG_LOG_DIR}"
            echo "[+] Will average ${SYSTEM}'s ${REPEAT} seeds once its jobs finish - log: ${AVG_LOG}"
            (
                for PID in "${SYSTEM_PIDS[@]}"; do
                    while kill -0 "${PID}" 2>/dev/null; do sleep 2; done
                done
                bash scripts/ode-average-seeds.sh --runs "${SEED_AVG_RUN_ARGS[@]}" --out "${SEED_AVG_OUT}" \
                    >"${AVG_LOG}" 2>&1 \
                    || echo "[-] Seed-averaging for ${SYSTEM} failed (see ${AVG_LOG})"
            ) &
            AVG_PIDS+=("$!")
        fi
    fi
done

if [ "${HAVE_SBATCH}" -eq 0 ]; then
    for PID in "${SLOT_PID[@]}"; do
        wait "${PID}" || true
    done
    for PID in "${AVG_PIDS[@]}"; do
        wait "${PID}" || true
    done
fi
