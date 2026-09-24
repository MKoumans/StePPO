#!/usr/bin/env bash
#
# Shared GPU-selection and local/Slurm+apptainer dispatch helpers, sourced
# by the scripts under scripts/ that launch training or analysis jobs.
# Not meant to be run directly.
#
# Usage (from a script directly inside scripts/). Under sbatch the calling
# script runs from Slurm's spool dir, so fall back to SLURM_SUBMIT_DIR:
#   SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   [ -f "${SCRIPTS_DIR}/lib/device.sh" ] || SCRIPTS_DIR="${SLURM_SUBMIT_DIR:-}/scripts"
#   source "${SCRIPTS_DIR}/lib/device.sh"
#   REPO_ROOT="$(device_repo_root "${BASH_SOURCE[0]}")"

# Extract this script's GPU selector from an argument vector and return the
# remaining arguments in a caller-provided array. The caller can then parse its
# own options without duplicating the shared GPU syntax.
#
# Usage:
#   local gpu_arg remaining
#   device_extract_gpu_args gpu_arg remaining "$@" || exit $?
#   set -- remaining array contents
device_extract_gpu_args() {
    if [ "$#" -lt 2 ]; then
        echo "device_extract_gpu_args requires GPU and argument-array names" >&2
        return 2
    fi

    local gpu_var="$1"
    local args_var="$2"
    shift 2

    local gpu_arg=""
    local remaining=()

    while [ "$#" -gt 0 ]; do
        case "$1" in
            -g|--gpus)
                if [ "$#" -lt 2 ]; then
                    echo "[-] $1 requires a value" >&2
                    return 1
                fi
                gpu_arg="$2"
                shift 2
                ;;
            --gpus=*)
                gpu_arg="${1#*=}"
                shift
                ;;
            *)
                remaining+=("$1")
                shift
                ;;
        esac
    done

    local -n gpu_ref="$gpu_var"
    local -n args_ref="$args_var"
    gpu_ref="$gpu_arg"
    args_ref=("${remaining[@]}")
}

# Repo root: SLURM_SUBMIT_DIR under sbatch (Slurm changes cwd before the
# script runs), else resolved relative to the calling script's own path.
# Pass "${BASH_SOURCE[0]}" of the calling script.
device_repo_root() {
    local caller="$1"
    echo "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${caller}")/.." && pwd)}"
}

# True if currently running inside a Slurm job (sbatch/srun).
device_on_slurm() {
    [ -n "${SLURM_JOB_ID:-}" ]
}

# True if sbatch is available at all (i.e. we're on an HPC login node,
# whether or not a job is currently running).
device_have_sbatch() {
    command -v sbatch &>/dev/null
}

device_pick_gpu() {
    local min_free_mib="${1:-${DEVICE_MIN_FREE_MIB:-8192}}"
    local fallback="${DEVICE_DEFAULT_GPU:-0}"

    if ! command -v nvidia-smi &>/dev/null; then
        echo "${CUDA_VISIBLE_DEVICES:-$fallback}"
        return
    fi

    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        case "$CUDA_VISIBLE_DEVICES" in
            *,*) echo "$CUDA_VISIBLE_DEVICES"; return ;;
        esac
        local free_mib
        free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null)"
        if [ -n "$free_mib" ] && [ "$free_mib" -ge "$min_free_mib" ]; then
            echo "$CUDA_VISIBLE_DEVICES"
            return
        fi
        echo "[Device Setup] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} has only ${free_mib:-0}MiB free (<${min_free_mib}MiB); auto-selecting a freer GPU instead." >&2
    fi

    local best
    best="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t',' -k2 -rn | head -n1 | cut -d',' -f1 | tr -d ' ')"
    echo "${best:-$fallback}"
}

# Run a command with an optional explicit GPU binding.
device_run_with_gpu() {
    local gpu="$1"
    shift
    if [ -n "$gpu" ]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$@"
    else
        "$@"
    fi
}

# Run a Python module/script locally or, under Slurm, inside the Apptainer image.
# Preallocation is off by default: preallocating 90% of an 11 GiB card leaves too
# little memory for cuBLAS. Export XLA_PYTHON_CLIENT_PREALLOCATE=true to enable it.
device_run_py() {
    if device_on_slurm; then
        srun apptainer exec --nv --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" --env "PYTHONPATH=${REPO_ROOT}" --env "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}" --env "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" "${REPO_ROOT}/steppo.sif" \
            python "$@"
    else
        PYTHONPATH="${REPO_ROOT}" CUDA_VISIBLE_DEVICES="$(device_pick_gpu)" XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}" \
            XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
            python "$@"
    fi
}

# Like device_run_py for CPU-only scripts; runs the Apptainer image directly
# (no GPU, no Slurm job), e.g. on a login node.
device_run_py_cpu() {
    apptainer exec --bind "${REPO_ROOT}:${REPO_ROOT}" --pwd "${REPO_ROOT}" --env "PYTHONPATH=${REPO_ROOT}" "${REPO_ROOT}/steppo.sif" \
        python "$@"
}

# Populate the global GPUS array from an explicit comma-separated list (validated
# against nvidia-smi, since JAX silently falls back to CPU on an invalid id), else
# the GPU with the most free memory, else GPU 0.
device_select_gpus() {
    local gpu_arg="${1:-}"
    if [ -n "$gpu_arg" ]; then
        IFS=',' read -r -a GPUS <<< "$gpu_arg"
        if command -v nvidia-smi &>/dev/null; then
            local valid_ids
            valid_ids="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')"
            local bad=()
            for g in "${GPUS[@]}"; do
                grep -qx "$g" <<< "$valid_ids" || bad+=("$g")
            done
            if [ "${#bad[@]}" -gt 0 ]; then
                echo "[Device Setup] Invalid GPU id(s): ${bad[*]}. Available: $(tr '\n' ',' <<< "$valid_ids" | sed 's/,$//')" >&2
                exit 1
            fi
        fi
    elif command -v nvidia-smi &>/dev/null; then
        GPUS=("$(device_pick_gpu)")
    else
        GPUS=(0)
    fi
}

# Detaches the current script from its controlling terminal (setsid + nohup)
# so a long-running job survives the terminal, SSH connection, or
# screen/tmux pane closing. Re-execs "$0" with the given args, redirecting
# output to log_file. Any env vars the re-exec needs must be exported by the
# caller before calling this. Does not exit — the caller should exit 0 right
# after calling this.
device_detach() {
    local log_file="$1"
    shift
    mkdir -p "$(dirname "${log_file}")"
    echo "[*] Detaching from this terminal so the run survives it closing."
    echo "[*] Follow progress with: tail -f ${log_file}"
    setsid nohup "$0" "$@" </dev/null >"${log_file}" 2>&1 &
    disown
    echo "[*] Detached (PID $!)."
}
