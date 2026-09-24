#!/usr/bin/env bash
#
# Stop running factorial experiments by killing research/ode/run.py
# processes (SIGTERM, then SIGKILL for stragglers).
# Full docs: scripts/README.md

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [SYSTEM]

Stop running ODE experiments, optionally filtered by system.

Full docs: scripts/README.md
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SYSTEM="${1:-}"

if [ -n "$SYSTEM" ]; then
    # Matches both hand-authored configs (configs/envs/ode/<system>/...) and
    # generated ones (configs/experiments/ode/<system>/experiment_.../...).
    PATTERN="research/ode/run.py --config configs/(envs|experiments)/ode/${SYSTEM}/"
else
    PATTERN="research/ode/run.py"
fi

PIDS=$(pgrep -f "$PATTERN" 2>/dev/null || true)

if [ -z "$PIDS" ]; then
    echo "No running experiments found."
    exit 0
fi

echo "Stopping experiments:"
for PID in $PIDS; do
    CMD=$(ps -p "$PID" -o args= 2>/dev/null || echo "unknown")
    echo "  kill ${PID}  (${CMD})"
    kill "$PID"
done

sleep 1

# Check for stragglers
REMAINING=$(pgrep -f "$PATTERN" 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo "Some processes still alive, sending SIGKILL..."
    for PID in $REMAINING; do
        kill -9 "$PID" 2>/dev/null || true
    done
fi

echo "Done."
