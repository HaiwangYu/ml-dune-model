#!/usr/bin/env bash
# Run all MAE diagnostics for a given CONDOR_OUT run directory.
#
# Usage:
#   ./run_all_mae_diagnostics.sh <run_dir>
#
# Expects:
#   <run_dir>/debug/histories.json
#   <run_dir>/checkpoints/mae_epoch<N>.pt  (the latest is used to extract features)
#
# Outputs plots into <run_dir>/debug/ and <run_dir>/checkpoints/.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_dir>"
    exit 1
fi

RUN_DIR="$(cd "$1" && pwd)"
HISTORIES="${RUN_DIR}/debug/histories.json"
CKPT_DIR="${RUN_DIR}/checkpoints"

if [[ ! -f "$HISTORIES" ]]; then
    echo "Error: histories file not found: $HISTORIES"
    exit 1
fi

# Pick the highest-epoch checkpoint
CKPT=$(ls -1 "${CKPT_DIR}"/mae_epoch*.pt 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -z "$CKPT" ]]; then
    echo "Error: no mae_epoch*.pt checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

EPOCH=$(basename "${CKPT}" | sed 's/mae_epoch\([0-9]*\).pt/\1/')
FEATURES="${CKPT_DIR}/features_ep${EPOCH}.npz"

echo "=== MAE diagnostics for: ${RUN_DIR} ==="
echo

if [[ ! -f "$FEATURES" ]]; then
    echo "--- extract_features (ep${EPOCH}) ---"
    python -m mae.diagnostics.extract_features "${CKPT}"
    echo
fi

echo "--- plot_histories ---"
python -m mae.diagnostics.plot_histories "${HISTORIES}"
echo

echo "--- plot_knn (ep${EPOCH}) ---"
python -m mae.diagnostics.plot_knn "${FEATURES}"
echo

echo "=== Done. ==="
