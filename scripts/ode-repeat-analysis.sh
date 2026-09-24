#!/bin/bash
#
# Aggregate repeated ODE runs into mean +/- std reports.
# Full docs: scripts/README.md

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") --system SYSTEM [OPTIONS]

Aggregate repeated ODE runs into mean +/- std reports.

Options:
  --root DIR       Runs root (default: outputs/runs).
  --date DATE      Restrict to one date.
  --out DIR        Output directory.
  --dry-run        Show grouping without writing.
  -h, --help       Show help.

Full docs: scripts/README.md
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPTS_DIR}/lib/device.sh"
REPO_ROOT="$(device_repo_root "${BASH_SOURCE[0]}")"
cd "${REPO_ROOT}"

device_run_py_cpu research/ode/post_run_analysis/learning_statistics.py aggregate-repeats "$@"
