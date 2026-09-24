#!/usr/bin/env bash
#
# Show what each running research/ode/run.py process is actually doing:
# GPU + memory (from nvidia-smi), system/experiment/seed (parsed from its
# command line), elapsed time, and the latest training-progress line pulled
# from its log file (resolved via /proc/<pid>/fd/1, so it works whether the
# run was launched directly, via run.sh, or via launch_experiments.sh).
#
# Full docs: scripts/README.md

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}")

List running ODE training processes with GPU, memory, system/experiment,
elapsed time, and latest progress line.

Full docs: scripts/README.md
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

PIDS=$(pgrep -f "research/ode/run.py" 2>/dev/null || true)

if [ -z "$PIDS" ]; then
    echo "No running experiments found."
    exit 0
fi

# -- PID -> "gpu_index memory_mib", from nvidia-smi, when available --
declare -A GPU_UUID_TO_INDEX
declare -A PID_GPU
declare -A PID_MEM

if command -v nvidia-smi >/dev/null 2>&1; then
    while IFS=', ' read -r IDX UUID; do
        [ -n "$IDX" ] && GPU_UUID_TO_INDEX["$UUID"]="$IDX"
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)

    while IFS=', ' read -r APID UUID MEM; do
        [ -z "$APID" ] && continue
        PID_GPU["$APID"]="${GPU_UUID_TO_INDEX[$UUID]:-?}"
        PID_MEM["$APID"]="${MEM}"
    done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null)
fi

ROWS=()

for PID in $PIDS; do
    CMD=$(ps -p "$PID" -o args= 2>/dev/null) || continue

    CONFIG=$(echo "$CMD" | grep -oP '(?<=--config )\S+' || true)
    SEED=$(echo "$CMD" | grep -oP '(?<=--seed )\S+' || echo "-")

    SYSTEM="-"
    EXP="-"
    if [[ "$CONFIG" =~ configs/envs/ode/([^/]+)/ ]]; then
        SYSTEM="${BASH_REMATCH[1]}"
    fi
    BASE=$(basename "${CONFIG:-}" .yaml)
    if [[ "$BASE" =~ ^([a-z_]+)_(e[0-9]+.*)$ ]]; then
        [ "$SYSTEM" = "-" ] && SYSTEM="${BASH_REMATCH[1]}"
        EXP="${BASH_REMATCH[2]}"
    elif [ "$SYSTEM" = "-" ] && [ -n "$BASE" ]; then
        SYSTEM="$BASE"
    fi

    ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ' || echo "?")

    LOG=$(readlink -f "/proc/${PID}/fd/1" 2>/dev/null || true)
    PROGRESS="-"
    FINGERPRINT=$(echo "$CMD" | grep -oP '(?<=--run_uid )\S+' || true)
    if [ -n "$LOG" ] && [ -r "$LOG" ]; then
        LINE=$(tail -n 500 "$LOG" 2>/dev/null | grep -a "^Iter " | tail -1 || true)
        if [ -n "$LINE" ]; then
            ITER=$(echo "$LINE" | grep -oP '^Iter \S+' || true)
            RET=$(echo "$LINE" | grep -oP 'R: \S+' || true)
            ETA=$(echo "$LINE" | grep -oP 'ETA: .*$' || true)
            PROGRESS="${ITER}  ${RET}  ${ETA}"
        fi
        if [ -z "$FINGERPRINT" ]; then
            # run_uid isn't printed at startup; recover it from the checkpoint
            # path (run_uid is always the directory segment right before
            # "checkpoints/checkpoint_N", whether or not --checkpoints_dir
            # was overridden — see TrainConfig.checkpoint_path).
            FINGERPRINT=$(grep -a "^\[+\] Saved model checkpoint to:" "$LOG" 2>/dev/null \
                | tail -1 | grep -oP '[^/]+(?=/checkpoint)' || true)
        fi
    fi
    [ -z "$FINGERPRINT" ] && FINGERPRINT="-"

    GPU="${PID_GPU[$PID]:--}"
    SORTKEY="$GPU"
    [[ "$SORTKEY" =~ ^[0-9]+$ ]] || SORTKEY=999
    ROW=$(printf "%-8s %-4s %-9s %-14s %-6s %-4s %-12s %-9s  %s" \
        "$PID" "$GPU" "${PID_MEM[$PID]:--}" "$SYSTEM" "$EXP" "$SEED" "$FINGERPRINT" "$ELAPSED" "$PROGRESS")
    ROWS+=("$(printf "%03d" "$SORTKEY")	${ROW}")
done

printf "%-8s %-4s %-9s %-14s %-6s %-4s %-12s %-9s  %s\n" \
    "PID" "GPU" "MEM(MiB)" "SYSTEM" "EXP" "SEED" "FINGERPRINT" "ELAPSED" "PROGRESS"
printf "%s\n" "${ROWS[@]}" | sort -t $'\t' -k1,1 | cut -f2-
