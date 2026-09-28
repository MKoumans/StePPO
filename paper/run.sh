#!/usr/bin/env bash
# Reproduce every figure of the ICASSP 2027 paper in one go (REPRODUCE.md).
# The Code Ocean capsule (code/run) runs import mode on CPU; execute mode is
# for the local containers (docker/Dockerfile and docker/Dockerfile.deeponet-gpu
# environments, or any setup with both Python environments below).
#
#   bash paper/run.sh                  # import: released models and datasets from Hugging Face
#   bash paper/run.sh --mode execute   # execute: generate the datasets, train StePPO and DeepONet, then plot (many GPU hours)
#
# Options:
#   --mode import|execute   see above; a bare "import" or "execute" also works
#   --gpus N|cpu            GPU to use (default 0), or cpu to run JAX and PyTorch on the CPU
#   --results DIR           copy figures, CSVs (and, in execute mode, the trained models) here;
#                           defaults to /results when that directory exists (Code Ocean)
#
# Env vars:
#   DEEPONET_PYTHON   Python of the PyTorch/DeepXDE environment (default /opt/deeponet/bin/python);
#                     without it, execute mode stops and import mode skips the DeepONet timing of Fig. 5.
set -euo pipefail

MODE=import
GPUS=0
RESULTS_DIR=""
[ -d /results ] && RESULTS_DIR=/results
while [ "$#" -gt 0 ]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --mode=*) MODE="${1#*=}"; shift ;;
        import|execute) MODE="$1"; shift ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --gpus=*) GPUS="${1#*=}"; shift ;;
        --results) RESULTS_DIR="$2"; shift 2 ;;
        --results=*) RESULTS_DIR="${1#*=}"; shift ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done
case "${MODE}" in import|execute) ;; *) echo "--mode must be import or execute, got '${MODE}'" >&2; exit 2 ;; esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The scripts write data/ and outputs/ inside the repository; work on a copy if it is read-only.
if [ ! -w "${REPO_ROOT}" ]; then
    WORK=/tmp/steppo
    echo "[*] ${REPO_ROOT} is read-only; working in ${WORK}"
    mkdir -p "${WORK}"
    cp -a "${REPO_ROOT}/." "${WORK}/"
    REPO_ROOT="${WORK}"
fi
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [ "${GPUS}" = cpu ]; then
    # GPUS=cpu makes steppo.utils.device.setup_devices keep JAX on the CPU.
    export GPUS=cpu JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
else
    export CUDA_VISIBLE_DEVICES="${GPUS}"
fi
export DDE_BACKEND=pytorch
DEEPONET_PYTHON="${DEEPONET_PYTHON:-/opt/deeponet/bin/python}"

# The Code Ocean image installs only the dependencies of steppo; make the package in src/ importable.
if ! python -c "import steppo" 2>/dev/null; then
    SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    echo "${REPO_ROOT}/src" > "${SITE}/steppo-src.pth"
fi

SYSTEMS=(scalar_decay van_der_pol brusselator)
step() { echo; echo "==== [$(date +%H:%M:%S)] $*"; }

python -c "import jax; print('[*] JAX', jax.__version__, 'devices:', jax.devices())"
HAVE_DEEPONET=0
if [ -x "${DEEPONET_PYTHON}" ]; then
    HAVE_DEEPONET=1
    "${DEEPONET_PYTHON}" -c "import torch; print('[*] PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
elif [ "${MODE}" = execute ]; then
    echo "No DeepONet environment at ${DEEPONET_PYTHON} (set DEEPONET_PYTHON); execute mode trains the DeepONets." >&2
    exit 1
fi

if [ "${MODE}" = import ]; then
    step "Datasets from Hugging Face (MKoumans/steppo-icassp2027-data)"
    python paper/download_data.py
else
    for system in "${SYSTEMS[@]}"; do
        step "StePPO ${system}: PID/Oracle datasets, then training (hours)"
        bash paper/train.sh "${system}" --gpus "${GPUS}"
    done
    for system in "${SYSTEMS[@]}"; do
        step "DeepONet ${system}, complete domain"
        "${DEEPONET_PYTHON}" research/baselines/deeponet/train_deeponet_unified.py \
            --profile retrain --system "${system}" --domain complete
    done
    step "DeepONet van_der_pol, binned domain"
    "${DEEPONET_PYTHON}" research/baselines/deeponet/train_deeponet_unified.py \
        --profile retrain --system van_der_pol --domain binned
fi

step "Fig. 2: trajectories"
python paper/fig2_trajectory.py --mode "${MODE}"
step "Fig. 3: sweep over the test split"
python paper/fig3_sweep.py --mode "${MODE}"
step "Fig. 4: robustness (Van der Pol)"
python paper/fig4_robustness.py --mode "${MODE}"
if [ "${HAVE_DEEPONET}" = 1 ]; then
    step "Fig. 5: DeepONet wall-clock (PyTorch)"
    "${DEEPONET_PYTHON}" paper/fig5_deeponet_timing.py --mode "${MODE}"
else
    echo "[!] No DeepONet environment at ${DEEPONET_PYTHON}: Fig. 5 without DeepONet"
fi
step "Fig. 5: wall-clock"
python paper/fig5_wallclock.py --mode "${MODE}"

if [ -n "${RESULTS_DIR}" ]; then
    step "Copying results to ${RESULTS_DIR}"
    mkdir -p "${RESULTS_DIR}"
    find outputs/paper -maxdepth 1 -type f \( -name '*.pdf' -o -name '*.csv' -o -name '*.png' \) \
        -exec cp {} "${RESULTS_DIR}/" \;
    if [ "${MODE}" = execute ]; then
        mkdir -p "${RESULTS_DIR}/models/deeponet"
        cp -a outputs/paper/models/. "${RESULTS_DIR}/models/"
        for system in "${SYSTEMS[@]}"; do
            cp -a "research/baselines/deeponet/output-deeponet/${system}/models" \
                "${RESULTS_DIR}/models/deeponet/${system}"
        done
    fi
    ls -la "${RESULTS_DIR}"B
fi
step "Done (${MODE} mode)"
