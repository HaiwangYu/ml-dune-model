#!/bin/bash
# nccl_test.sh — worker: run the standalone 2-GPU NCCL connectivity test with
# NCCL_DEBUG=INFO. Args: $1 env_prefix  $2 repodir  $3 n_gpus  [$4 extra NCCL env "K=V K=V"]
set -uo pipefail
env_prefix=$1; repodir=$2; n_gpus=${3:-2}; extra_env=${4:-}
export PATH="${env_prefix}/bin:${PATH}"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
for kv in $extra_env; do export "$kv"; echo "  set $kv"; done
echo "Running ${CLUSTER_ID:-?}.${JOB_ID:-?} on $(hostname)  n_gpus=${n_gpus}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name --format=csv,noheader || true
"${env_prefix}/bin/torchrun" --standalone --nproc_per_node="${n_gpus}" \
    "${repodir}/gridutils/nccl_test.py"
echo "EXIT_CODE=$?"
