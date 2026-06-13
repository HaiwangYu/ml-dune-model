#!/bin/bash
# bench_sparse_mem.sh — worker: sparse-CNN backbone (mae/dino) inference memory.
# Runs in the ml-dune-model env (torch 2.10 + warpconvnet).
# Args: $1 codedir  $2 pyenv  $3 cache_dir
set -euo pipefail
codedir=${1:-/lbne/u/hyu/ml-dune-model}
pyenv=${2:-/gpfs01/lbne/users/fm/hyu/uvenv}
cache_dir=${3:-/gpfs01/lbne/users/fm/hyu/cache}
export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="${cache_dir}/warpconvnet"
mkdir -p "$WARPCONVNET_BENCHMARK_CACHE_DIR"
source "${pyenv}/bin/activate"
echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}" \
    python -u -m mae.diagnostics.bench_backbone_mem --N 2000,4000,6000,8000 --batch 1,8
echo "BENCH DONE"
