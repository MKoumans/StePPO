#!/bin/sh
#SBATCH -p cbuild
#SBATCH --output=./outputs/logs/build/%x_%j.out
#SBATCH -t 02:00:00

# Pulls the project Docker image and converts it to an Apptainer/Singularity
# .sif image (steppo.sif) for Slurm clusters, where scripts/run.sh etc.
# dispatch into it via `apptainer exec` when running under sbatch/srun.
# Override the source image with STEPPO_IMAGE.
#
# Usage:
#   sbatch scripts/create-apptainer.sh
#   sh scripts/create-apptainer.sh    # direct, e.g. on a login/build node

case "${1:-}" in
    -h|--help)
        cat <<EOF
Usage: $(basename "$0")

Pulls docker://\${STEPPO_IMAGE} (default ghcr.io/mkoumans/steppo:latest)
and converts it to steppo.sif via 'apptainer pull'. Takes no arguments.

Options:
  -h, --help  Show this help message and exit.
EOF
        exit 0
        ;;
esac

export SINGULARITY_TMPDIR=$(mktemp -d /tmp/apptainer.XXXXXX)
export APPTAINER_TMPDIR=$SINGULARITY_TMPDIR

STEPPO_IMAGE="${STEPPO_IMAGE:-ghcr.io/mkoumans/steppo:latest}"
apptainer pull steppo.sif "docker://${STEPPO_IMAGE}"
