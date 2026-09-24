#!/usr/bin/env bash
# Retrain the van_der_pol DeepONet on the "complete" and "binned" domains for the
# same number of iterations, then compute the log10 relative-error metrics of both.
# Run inside the DeepONet container (docker/docker-compose.deeponet.yml).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SYSTEM=van_der_pol
ITERATIONS=15000
LOG_DIR="$REPO_ROOT/research/baselines/deeponet/output-deeponet/${SYSTEM}/logs"
mkdir -p "$LOG_DIR"

echo "[*] Training ${SYSTEM}/complete on GPU1 and ${SYSTEM}/binned on GPU0 in parallel..."

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python research/baselines/deeponet/train_deeponet_unified.py \
    --system "$SYSTEM" --domain complete --profile retrain --device gpu \
    --dataset-source cache --iterations "$ITERATIONS" \
    2>&1 | tee "$LOG_DIR/train_complete_${ITERATIONS}.log" &
pid_complete=$!

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python research/baselines/deeponet/train_deeponet_unified.py \
    --system "$SYSTEM" --domain binned --profile retrain --device gpu \
    --dataset-source cache --iterations "$ITERATIONS" \
    2>&1 | tee "$LOG_DIR/train_binned_${ITERATIONS}.log" &
pid_binned=$!

wait "$pid_complete"
wait "$pid_binned"

echo "[*] Training done. Locating final checkpoints..."

find_ckpt() {
    local domain="$1"
    local pattern="research/baselines/deeponet/output-deeponet/${SYSTEM}/models/deeponet_${SYSTEM}_retrain_${domain}_final-${ITERATIONS}.pt"
    if [[ ! -f "$pattern" ]]; then
        echo "ERROR: expected checkpoint not found: $pattern" >&2
        exit 1
    fi
    echo "$pattern"
}

ckpt_complete="$(find_ckpt complete)"
ckpt_binned="$(find_ckpt binned)"

echo "[*] complete checkpoint: $ckpt_complete"
echo "[*] binned checkpoint:   $ckpt_binned"

echo "[*] Computing rel_err metrics for complete..."
PYTHONPATH=. python research/baselines/deeponet/deeponet_relerr_eval.py \
    --system "$SYSTEM" --profile retrain \
    --checkpoint "$ckpt_complete" --label "retrain_complete_${ITERATIONS}" \
    2>&1 | tee "$LOG_DIR/relerr_complete_${ITERATIONS}.log"

echo "[*] Computing rel_err metrics for binned..."
PYTHONPATH=. python research/baselines/deeponet/deeponet_relerr_eval.py \
    --system "$SYSTEM" --profile retrain \
    --checkpoint "$ckpt_binned" --label "retrain_binned_${ITERATIONS}" \
    2>&1 | tee "$LOG_DIR/relerr_binned_${ITERATIONS}.log"

# The training config is named without the iteration suffix of --label, so pass it
# explicitly for plot_deeponet_relerr.py to shade the training domain.
RESULTS_DIR="research/baselines/deeponet/output-deeponet/${SYSTEM}/results"

echo "[*] Plotting rel_err vs parameter for complete..."
PYTHONPATH=. python research/baselines/deeponet/plot_deeponet_relerr.py \
    --system "$SYSTEM" --profile retrain --label "retrain_complete_${ITERATIONS}" \
    --training-config "$RESULTS_DIR/deeponet_${SYSTEM}_retrain_complete_training_config.json" \
    2>&1 | tee "$LOG_DIR/plot_complete_${ITERATIONS}.log"

echo "[*] Plotting rel_err vs parameter for binned..."
PYTHONPATH=. python research/baselines/deeponet/plot_deeponet_relerr.py \
    --system "$SYSTEM" --profile retrain --label "retrain_binned_${ITERATIONS}" \
    --training-config "$RESULTS_DIR/deeponet_${SYSTEM}_retrain_binned_training_config.json" \
    2>&1 | tee "$LOG_DIR/plot_binned_${ITERATIONS}.log"

echo
echo "======================================================================"
echo "Summary (log10 relative error)"
echo "======================================================================"
echo "-- complete --"
grep -E "^\[\+\] log10" "$LOG_DIR/relerr_complete_${ITERATIONS}.log"
echo "-- binned --"
grep -E "^\[\+\] log10" "$LOG_DIR/relerr_binned_${ITERATIONS}.log"
echo
echo "Plots:"
grep -E "^\[\+\] Plot" "$LOG_DIR/plot_complete_${ITERATIONS}.log"
grep -E "^\[\+\] Plot" "$LOG_DIR/plot_binned_${ITERATIONS}.log"
