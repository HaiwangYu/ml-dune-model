#!/bin/bash
# quant_larmamba.sh — worker: run larmamba.eval_quant for one checkpoint across
# precisions {none(bf16), int8, fp8}. Output goes to stdout (condor .out).
# Args: $1 polar_repo  $2 env_prefix  $3 larmamba_repo  $4 ckpt  $5 num_groups  $6 context_length
set -euo pipefail
polar_repo=$1; env_prefix=$2; larmamba_repo=$3; ckpt=$4; ng=$5; ctx=$6; enc=${7:-mamba}
export PATH="${env_prefix}/bin:${PATH}"
export PYTHONPATH="${larmamba_repo}:${polar_repo}:${PYTHONPATH:-}"
PY="${env_prefix}/bin/python"
echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)  ckpt=$(basename "$ckpt") ng=${ng}"
"$PY" -c "import torch;print('GPU',torch.cuda.get_device_name())"
for q in none int8 fp8; do
  echo "============================================================"
  "$PY" -m larmamba.eval_quant --encoder "$enc" --ckpt "$ckpt" --num_groups "$ng" --context_length "$ctx" --quant "$q" || echo "FAILED quant=$q"
done
echo "QUANT EVAL DONE"
