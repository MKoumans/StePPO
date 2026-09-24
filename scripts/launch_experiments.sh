#!/usr/bin/env bash
#SBATCH --job-name=launch_experiments
#SBATCH --output=./outputs/logs/slurm/%x_%j.out
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00

set -euo pipefail

ALL_SYSTEMS=(scalar_decay van_der_pol robertson brusselator fitzhugh_nagumo chemical_cascade fosm chua chua_smooth)

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") <system> [OPTIONS]

Launch a factorial experiment set for one ODE system.

Options:
  -g, --gpus DEVICES | --gpus=DEVICES  Local-mode GPU list.
  --base CONFIG                        Base config used with --sweep.
  --sweep SPEC                         Sweep spec used with --base.
  --post-run-analysis BOOL             Enable or disable post-run analysis.
  --generate-datasets                  Auto-generate missing training/PID evaluation
                                        datasets instead of aborting (default: on).
  --no-generate-datasets               Abort instead of generating missing datasets.
  --repeat N                           Seeded repeats (default: 1).
  --notes TEXT|FILE                    Attach notes to the experiment.
  -h, --help                           Show help.

Full docs: scripts/README.md
EOF
}

WORKDIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "${WORKDIR}/scripts/lib/device.sh" ] || WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKDIR}"

source "${WORKDIR}/scripts/lib/device.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SYSTEM="${1:?Usage: $0 <system> [gpu_ids_comma_separated] [--base BASE_CONFIG --sweep SWEEP_SPEC]}"
shift

GPU_ARG=""
GPU_FILTERED_ARGS=()
device_extract_gpu_args GPU_ARG GPU_FILTERED_ARGS "$@" || exit $?
set -- "${GPU_FILTERED_ARGS[@]}"
BASE_CONFIG=""
SWEEP_SPEC=""
NOTES=""
POST_RUN_ANALYSIS="true"
GENERATE_DATASETS="true"
REPEAT=1
while [ $# -gt 0 ]; do
    case "$1" in
        --base)                 BASE_CONFIG="$2";       shift 2 ;;
        --sweep)                SWEEP_SPEC="$2";        shift 2 ;;
        --notes)                NOTES="$2";             shift 2 ;;
        --post-run-analysis)    POST_RUN_ANALYSIS="$2"; shift 2 ;;
        --generate-datasets)    GENERATE_DATASETS="true"; shift ;;
        --no-generate-datasets) GENERATE_DATASETS="false"; shift ;;
        --repeat)               REPEAT="$2";            shift 2 ;;
        -h|--help)              usage; exit 0 ;;
        *)                      GPU_ARG="$1";           shift ;;
    esac
done

GENERATION_GPU="${GPU_ARG%%,*}"

case "$POST_RUN_ANALYSIS" in
    true|false) ;;
    *) echo "--post-run-analysis must be 'true' or 'false', got: ${POST_RUN_ANALYSIS}"; exit 1 ;;
esac

case "$GENERATE_DATASETS" in
    true|false) ;;
    *) echo "--generate-datasets must be 'true' or 'false', got: ${GENERATE_DATASETS}"; exit 1 ;;
esac

if ! [[ "${REPEAT}" =~ ^[0-9]+$ ]] || [ "${REPEAT}" -lt 1 ]; then
    echo "--repeat must be a positive integer, got: ${REPEAT}"
    exit 1
fi

if [ -n "$BASE_CONFIG" ] || [ -n "$SWEEP_SPEC" ]; then
    if [ -z "$BASE_CONFIG" ] || [ -z "$SWEEP_SPEC" ]; then
        echo "--base and --sweep must be given together"
        exit 1
    fi
fi

case "$SYSTEM" in
    scalar_decay|van_der_pol|robertson|brusselator|fitzhugh_nagumo|chemical_cascade|fosm|chua|chua_smooth)
        BASE_CONFIG_DIR="configs/envs/ode/${SYSTEM}"
        PREFIX="${SYSTEM}"
        ;;
    *)
        echo "Unknown system: $SYSTEM"
        exit 1
        ;;
esac



ON_HPC=0
device_have_sbatch && ON_HPC=1 || true

if [ "$ON_HPC" -eq 1 ] && ! device_on_slurm; then
    LAUNCH_LOG_DIR="${WORKDIR}/outputs/logs/launch_experiments"
    mkdir -p "$LAUNCH_LOG_DIR"
    ARGS=("$SYSTEM")
    [ -n "$GPU_ARG" ] && ARGS+=(--gpus "$GPU_ARG")
    [ -n "$BASE_CONFIG" ] && ARGS+=(--base "$BASE_CONFIG")
    [ -n "$SWEEP_SPEC" ] && ARGS+=(--sweep "$SWEEP_SPEC")
    [ -n "$NOTES" ] && ARGS+=(--notes "$NOTES")
    ARGS+=(--post-run-analysis "$POST_RUN_ANALYSIS")
    if [ "$GENERATE_DATASETS" = "true" ]; then
        ARGS+=(--generate-datasets)
    else
        ARGS+=(--no-generate-datasets)
    fi
    ARGS+=(--repeat "$REPEAT")
    JOB_ID=$(sbatch --parsable \
        --job-name="launch_${SYSTEM}" \
        --output="${LAUNCH_LOG_DIR}/${SYSTEM}_%j.out" \
        "${WORKDIR}/scripts/launch_experiments.sh" "${ARGS[@]}")
    echo "[*] On an HPC login node with no active Slurm job; resubmitting this launcher as Slurm job ${JOB_ID} (partition=gpu_mig) so Apptainer runs on a compute node."
    echo "[*] Follow with: tail -f ${LAUNCH_LOG_DIR}/${SYSTEM}_${JOB_ID}.out"
    exit 0
fi

run_py() {
    if [ "$ON_HPC" -eq 1 ]; then
        if [ -n "$GENERATION_GPU" ]; then
            apptainer exec --nv --pwd "${WORKDIR}" \
                --env "PYTHONPATH=${WORKDIR}" \
                --env "CUDA_VISIBLE_DEVICES=${GENERATION_GPU}" \
                "${WORKDIR}/steppo.sif" python "$@"
        else
            apptainer exec --nv --pwd "${WORKDIR}" --env "PYTHONPATH=${WORKDIR}" "${WORKDIR}/steppo.sif" python "$@"
        fi
    else
        if [ -n "$GENERATION_GPU" ]; then
            CUDA_VISIBLE_DEVICES="${GENERATION_GPU}" PYTHONPATH="${WORKDIR}" python "$@"
        else
            PYTHONPATH="${WORKDIR}" python "$@"
        fi
    fi
}

if [ "$ON_HPC" -eq 0 ] && [ -z "${_LAUNCH_DETACHED:-}" ] && ! device_on_slurm; then
    LAUNCH_LOG_DIR="${WORKDIR}/outputs/logs/launch_experiments"
    LAUNCH_LOG="${LAUNCH_LOG_DIR}/$(date +%Y%m%d_%H%M%S)_${SYSTEM}_$$.log"
    ARGS=("$SYSTEM")
    [ -n "$GPU_ARG" ] && ARGS+=(--gpus "$GPU_ARG")
    [ -n "$BASE_CONFIG" ] && ARGS+=(--base "$BASE_CONFIG")
    [ -n "$SWEEP_SPEC" ] && ARGS+=(--sweep "$SWEEP_SPEC")
    [ -n "$NOTES" ] && ARGS+=(--notes "$NOTES")
    ARGS+=(--post-run-analysis "$POST_RUN_ANALYSIS")
    if [ "$GENERATE_DATASETS" = "true" ]; then
        ARGS+=(--generate-datasets)
    else
        ARGS+=(--no-generate-datasets)
    fi
    ARGS+=(--repeat "$REPEAT")
    export _LAUNCH_DETACHED=1
    device_detach "${LAUNCH_LOG}" "${ARGS[@]}"
    exit 0
fi

if [ -z "$BASE_CONFIG" ] && [ -z "$SWEEP_SPEC" ]; then
    DEFAULT_BASE="${BASE_CONFIG_DIR}/${SYSTEM}_default.yaml"
    DEFAULT_SWEEP="${BASE_CONFIG_DIR}/${SYSTEM}_experiment.yml"
    if [ -f "$DEFAULT_BASE" ] && [ -f "$DEFAULT_SWEEP" ]; then
        BASE_CONFIG="$DEFAULT_BASE"
        SWEEP_SPEC="$DEFAULT_SWEEP"
        echo "[*] No --base/--sweep given; defaulting to ${BASE_CONFIG} + ${SWEEP_SPEC}"
    fi
fi

EXPERIMENT_DATE="$(date +%Y%m%d)"
EXPERIMENT_UID="$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")"
EXPERIMENT_DIR="${WORKDIR}/outputs/experiments/${EXPERIMENT_DATE}/${EXPERIMENT_UID}"
CHECKPOINTS_DIR="${EXPERIMENT_DIR}/checkpoints"
OUTPUTS_DIR="${EXPERIMENT_DIR}/outputs"
CONFIG_DIR="${EXPERIMENT_DIR}/configs"
LOG_DIR="${EXPERIMENT_DIR}/logs"

mkdir -p "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CONFIG_DIR"

echo "[*] Experiment UID: ${EXPERIMENT_UID}"
echo "[*] Checkpoints -> ${CHECKPOINTS_DIR}"
echo "[*] Outputs     -> ${OUTPUTS_DIR}"

if [ -n "$NOTES" ]; then
    if [ -f "$NOTES" ]; then
        cp "$NOTES" "${EXPERIMENT_DIR}/notes.txt"
    else
        echo "$NOTES" > "${EXPERIMENT_DIR}/notes.txt"
    fi
    echo "[*] Notes       -> ${EXPERIMENT_DIR}/notes.txt"
fi

if [ -n "$BASE_CONFIG" ]; then
    CONFIG_DIR="${EXPERIMENT_DIR}/configs"
    echo "[*] Generating experiment configs from ${BASE_CONFIG} + ${SWEEP_SPEC}"
    echo "[*] Configs     -> ${CONFIG_DIR}"
    run_py research/ode/generate_experiments.py \
        --base "${BASE_CONFIG}" \
        --sweep "${SWEEP_SPEC}" \
        --prefix "${PREFIX}" \
        --out-dir "${CONFIG_DIR}"
    echo ""
else
    CONFIG_DIR="$BASE_CONFIG_DIR"
fi

# -- Discover experiment configs (natural sort: e1, e2, ..., e10, not e1, e10, e2) --
mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -name "${PREFIX}_e*.yaml" -printf '%f\n' \
    | sort -V)

NUM_CONFIGS=${#CONFIGS[@]}
if [ "$NUM_CONFIGS" -eq 0 ]; then
    echo "No experiment configs found matching ${CONFIG_DIR}/${PREFIX}_e*.yaml"
    echo "Generate them first with research/ode/generate_experiments.py"
    exit 1
fi

# -- Populates MISSING_CFGS (not a subshell — array must survive the call). --
check_datasets() {
    MISSING_CFGS=()
    for CFG in "${CONFIGS[@]}"; do
        run_py src/steppo/envs/ode/check_oracle_dataset.py --config "${CONFIG_DIR}/${CFG}" >&2 || MISSING_CFGS+=("${CFG}")
    done
}

# Config -> Slurm job id of its dataset-generation job (HPC + --generate-datasets
# only). Populated below; training jobs for a config look themselves up here to
# add a --dependency=afterok on their own dataset job.
declare -A DATASET_JOB_ID

echo "[*] Checking training and PID evaluation datasets for ${NUM_CONFIGS} config(s) ..."
check_datasets
MISSING=${#MISSING_CFGS[@]}

if [ "$MISSING" -gt 0 ] && [ "$GENERATE_DATASETS" = "true" ]; then
    echo ""
    if [ "$ON_HPC" -eq 1 ]; then
        # One dataset-generation Slurm job per missing config.
        echo "[*] --generate-datasets: submitting one dataset-generation Slurm job per missing config (${MISSING}/${NUM_CONFIGS}) ..."
        for CFG in "${MISSING_CFGS[@]}"; do
            CFG_PATH="${CONFIG_DIR}/${CFG}"
            EID="${CFG#${PREFIX}_}"
            EID="${EID%.yaml}"
            DS_LOG="${LOG_DIR}/${PREFIX}_${EID}_dataset.log"
            JOB_ID=$(sbatch --parsable \
                --job-name="${PREFIX}_${EID}_dataset" \
                --output="${DS_LOG}" \
                --export=ALL,CONFIG="${CFG_PATH}" \
                "${WORKDIR}/scripts/generate-pid-dataset.sh")
            DATASET_JOB_ID["${CFG}"]="${JOB_ID}"
            echo "[+] Submitted dataset generation for ${CFG} as Slurm job ${JOB_ID} -> ${DS_LOG}"
        done
        echo ""
        echo "[*] Training jobs for these configs will wait on their dataset job to finish (Slurm --dependency=afterok)."
        MISSING=0
    else
        echo "[*] --generate-datasets: generating missing training and PID evaluation datasets for ${MISSING}/${NUM_CONFIGS} config(s) ..."
        for CFG in "${MISSING_CFGS[@]}"; do
            CFG_PATH="${CONFIG_DIR}/${CFG}"
            echo "[*] Generating PID training cache for ${CFG_PATH} ..."
            run_py src/steppo/envs/ode/generate_pid_training_dataset.py --config "${CFG_PATH}"
            echo "[*] Generating PID eval cache(s) for ${CFG_PATH} ..."
            mapfile -t EVAL_ARG_LINES < <(run_py src/steppo/envs/ode/check_oracle_dataset.py --config "${CFG_PATH}" --print-eval-args)
            for EVAL_ARGS in "${EVAL_ARG_LINES[@]}"; do
                # shellcheck disable=SC2086
                run_py src/steppo/envs/ode/generate_pid_eval_dataset.py --config "${CFG_PATH}" ${EVAL_ARGS}
            done
        done
        echo ""
        echo "[*] Re-checking datasets ..."
        check_datasets
        MISSING=${#MISSING_CFGS[@]}
    fi
fi

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "[-] ${MISSING}/${NUM_CONFIGS} config(s) are missing a required training or PID evaluation dataset (see above)."
    if [ "$GENERATE_DATASETS" = "true" ]; then
        echo "    Dataset generation was attempted but datasets are still missing; check the errors above."
    else
        echo "    Run the generation command(s) printed above, then re-run this launch, or pass --generate-datasets (now the default)."
    fi
    exit 1
fi

# -- Expand each config into REPEAT tasks (seeds 0..REPEAT-1); REPEAT=1 keeps plain "e<N>" ids. --
TASK_CFGS=()
TASK_EIDS=()
TASK_SEEDS=()
for CFG in "${CONFIGS[@]}"; do
    EID="${CFG#${PREFIX}_}"
    EID="${EID%.yaml}"
    for (( R=0; R<REPEAT; R++ )); do
        TASK_CFGS+=("${CFG}")
        if [ "${REPEAT}" -gt 1 ]; then
            TASK_EIDS+=("${EID}_r${R}")
            TASK_SEEDS+=("${R}")
        else
            TASK_EIDS+=("${EID}")
            TASK_SEEDS+=("")
        fi
    done
done
NUM_TASKS=${#TASK_CFGS[@]}

print_analysis_hint() {
    echo ""
    echo "Analyse results with:"
    echo "  python research/ode/analyze_experiment.py \\
      --discover ${OUTPUTS_DIR} --prefix ${PREFIX}_e \\
      --configs ${CONFIG_DIR} \\
      --out ${OUTPUTS_DIR}/${PREFIX}_comparison"
}

if [ "$ON_HPC" -eq 1 ]; then
    echo "[*] ${NUM_TASKS} experiment(s) for ${SYSTEM}; submitting one Slurm job each"
    JOB_IDS=()
    for TI in "${!TASK_CFGS[@]}"; do
        CFG="${TASK_CFGS[$TI]}"
        EID="${TASK_EIDS[$TI]}"
        SEED="${TASK_SEEDS[$TI]}"
        LOG="${LOG_DIR}/${PREFIX}_${EID}.log"
        RUN_ARGS=(--checkpoints_dir "${CHECKPOINTS_DIR}" --outputs_dir "${OUTPUTS_DIR}")
        [ -n "$SEED" ] && RUN_ARGS+=(--seed "$SEED")
        SBATCH_EXTRA=()
        DS_JOB="${DATASET_JOB_ID[$CFG]:-}"
        [ -n "$DS_JOB" ] && SBATCH_EXTRA+=(--dependency="afterok:${DS_JOB}")
        JOB_ID=$(sbatch --parsable \
            --job-name="${PREFIX}_${EID}" \
            --output="${LOG}" \
            --export=ALL,CONFIG="${CONFIG_DIR}/${CFG}" \
            "${SBATCH_EXTRA[@]}" \
            "${WORKDIR}/scripts/run.sh" \
            "${RUN_ARGS[@]}")
        JOB_IDS+=("$JOB_ID")
        echo "[+] Submitted ${PREFIX}_${EID} as Slurm job ${JOB_ID}${DS_JOB:+ (after dataset job ${DS_JOB})} -> ${LOG}"
    done

    echo ""
    echo "All experiments submitted as Slurm jobs: ${JOB_IDS[*]}"
    echo ""
    echo "  squeue -u \$USER                            # job status"
    echo "  tail -f ${LOG_DIR}/${PREFIX}_${TASK_EIDS[0]}.log      # follow one experiment"
    echo "  scancel ${JOB_IDS[*]}    # cancel all"

    if [ "$POST_RUN_ANALYSIS" = "true" ]; then
        DEPENDENCY="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
        ANALYSIS_LOG="${LOG_DIR}/${PREFIX}_post_run_analysis.log"
        # The analysis loops over every run, so scale its time limit with NUM_TASKS.
        ANALYSIS_MINUTES=$(( NUM_TASKS * 45 + 30 ))
        ANALYSIS_TIME=$(printf '%02d:%02d:00' $(( ANALYSIS_MINUTES / 60 )) $(( ANALYSIS_MINUTES % 60 )))
        ANALYSIS_JOB_ID=$(sbatch --parsable \
            --job-name="${PREFIX}_post_run_analysis" \
            --output="${ANALYSIS_LOG}" \
            --dependency="${DEPENDENCY}" \
            --time="${ANALYSIS_TIME}" \
            "${WORKDIR}/scripts/ode-experiment-analysis.sh" \
            --experiment_uid "${EXPERIMENT_UID}" --experiment_date "${EXPERIMENT_DATE}")
        echo ""
        echo "[+] Post-run analysis submitted as Slurm job ${ANALYSIS_JOB_ID} (after: ${JOB_IDS[*]}, time limit ${ANALYSIS_TIME}) -> ${ANALYSIS_LOG}"
    fi

    print_analysis_hint
    exit 0
fi

device_select_gpus "$GPU_ARG"
NUM_GPUS=${#GPUS[@]}

echo "[*] ${NUM_TASKS} experiment(s) for ${SYSTEM} across ${NUM_GPUS} GPU(s): ${GPUS[*]}"

# -- Split tasks into contiguous chunks, one per GPU --
CHUNK_SIZE=$(( (NUM_TASKS + NUM_GPUS - 1) / NUM_GPUS ))

PIDS=()

for g in "${!GPUS[@]}"; do
    GPU="${GPUS[$g]}"
    START=$(( g * CHUNK_SIZE ))
    if [ "$START" -ge "$NUM_TASKS" ]; then
        continue
    fi
    END=$(( START + CHUNK_SIZE ))
    if [ "$END" -gt "$NUM_TASKS" ]; then
        END=$NUM_TASKS
    fi

    (
        CHUNK_FAILED=0
        for (( TI=START; TI<END; TI++ )); do
            CFG="${TASK_CFGS[$TI]}"
            EID="${TASK_EIDS[$TI]}"
            SEED="${TASK_SEEDS[$TI]}"
            LOG="${LOG_DIR}/${PREFIX}_${EID}.log"
            echo "[+] GPU ${GPU}: running ${PREFIX}_${EID} -> ${LOG}"
            CODE=0
            RUN_ARGS=(--config "${CONFIG_DIR}/${CFG}" --checkpoints_dir "${CHECKPOINTS_DIR}" --outputs_dir "${OUTPUTS_DIR}")
            [ -n "$SEED" ] && RUN_ARGS+=(--seed "$SEED")
            CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${WORKDIR}" XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}" \
                XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
                python research/ode/run.py "${RUN_ARGS[@]}" \
                >> "${LOG}" 2>&1 || CODE=$?
            if [ "$CODE" -eq 0 ]; then
                echo "[+] GPU ${GPU}: ${PREFIX}_${EID} finished successfully"
            else
                echo "[-] GPU ${GPU}: ${PREFIX}_${EID} failed with exit code ${CODE}"
                CHUNK_FAILED=1
            fi
        done
        exit "$CHUNK_FAILED"
    ) &

    PIDS+=($!)
    echo "[+] Launched chunk on GPU ${GPU} (PID $!): ${TASK_EIDS[*]:START:CHUNK_SIZE}"
done

echo ""
echo "All chunks running in background."
echo ""
echo "  tail -f ${LOG_DIR}/${PREFIX}_${TASK_EIDS[0]}.log      # follow one experiment"
echo "  tail -f ${LOG_DIR}/${PREFIX}_e*.log       # follow all"
echo ""
echo "PIDs (one per GPU chunk): ${PIDS[*]}"
echo ""

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "[+] All GPU chunks completed successfully."
else
    echo "[-] ${FAILED}/${NUM_GPUS} GPU chunk(s) had at least one failed experiment. Check logs in ${LOG_DIR}/"
fi

if [ "$POST_RUN_ANALYSIS" = "true" ]; then
    echo ""
    echo "[*] Running post-run analysis (scripts/ode-experiment-analysis.sh) ..."
    bash "${WORKDIR}/scripts/ode-experiment-analysis.sh" --experiment_uid "${EXPERIMENT_UID}" --experiment_date "${EXPERIMENT_DATE}"
fi

print_analysis_hint
