#!/bin/bash
#
# MAE training script.
# Runs on the Condor worker; called by submit_mae.sh.
#
# Args (positional):
#   $1 codedir   -- path to ml-dune-model repo root
#   $2 pyenv     -- path to uv virtual environment to activate
#   $3 config    -- path to run_config.json
#   $4 outdir    -- path to output directory on GPFS (where outputs are rsynced back)
#   $5 cache_dir -- general cache base; ${cache_dir}/warpconvnet and ${cache_dir}/data are used
#   $6 run_name  -- run/training name
#
# I/O strategy: write everything to $_CONDOR_SCRATCH_DIR (fast local disk on
# the worker), rsync to GPFS at the end via an EXIT trap so partial outputs
# survive failures and preemption.

set -euo pipefail

codedir=$1
pyenv=$2
config=$3
outdir=$4
cache_dir=$5
run_name=$6

wp_cache="${cache_dir}/warpconvnet"
data_cache="${cache_dir}/data"
mkdir -p "$wp_cache" "$data_cache"

echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR}"
echo ""

echo "JOB CONFIGURATION:"
echo "  run_name=${run_name}"
echo "  codedir=${codedir}"
echo "  pyenv=${pyenv}"
echo "  config=${config}"
echo "  outdir=${outdir}"
echo "  cache_dir=${cache_dir}"
echo ""

echo "WarpConvNet overrides:"
export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"
echo "  WARPCONVNET_USE_FP16_ACCUM=${WARPCONVNET_USE_FP16_ACCUM}"
echo "  WARPCONVNET_BENCHMARK_CACHE_DIR=${WARPCONVNET_BENCHMARK_CACHE_DIR}"
echo ""

echo "Activating python environment..."
source "${pyenv}/bin/activate"

# Stage all outputs on local scratch.  main() will append /${run_name} under
# each base, so the actual write dirs are $scratch_ckpt/$run_name,
# $scratch_dbg/$run_name and $scratch_viz/$run_name.
scratch_ckpt=${_CONDOR_SCRATCH_DIR}/checkpoints
scratch_dbg=${_CONDOR_SCRATCH_DIR}/debug
scratch_viz=${_CONDOR_SCRATCH_DIR}/viz
mkdir -p "$scratch_ckpt" "$scratch_dbg" "$scratch_viz"

sync_back() {
  echo "Syncing ${_CONDOR_SCRATCH_DIR} -> ${outdir}"
  mkdir -p "${outdir}/checkpoints" "${outdir}/debug" "${outdir}/viz"
  # Trailing slash on source flattens the inner /${run_name} dir, so the GPFS
  # layout is ${outdir}/{checkpoints,debug,viz}/... without a redundant nest.
  rsync -a "${scratch_ckpt}/${run_name}/" "${outdir}/checkpoints/" || true
  rsync -a "${scratch_dbg}/${run_name}/"  "${outdir}/debug/"       || true
  rsync -a "${scratch_viz}/${run_name}/"  "${outdir}/viz/"         || true
}
cleanup() {
  if [[ -n "${nvsmi_pid:-}" ]]; then
    kill "${nvsmi_pid}" 2>/dev/null || true
  fi
  sync_back
}
trap cleanup EXIT
trap 'cleanup; exit 143' SIGTERM

# ── Background GPU usage logger ─────────────────────────────────────────────
# Writes a CSV line every 10 s to ${outdir}/gpu.log on GPFS so you can
# `tail -f` it live from the interactive node.
echo "Starting nvidia-smi polling loop -> ${outdir}/gpu.log"
nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv -l 10 > "${outdir}/gpu.log" 2>&1 &
nvsmi_pid=$!

echo "Executing mae.scripts.train_mae ..."

PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}" \
    python -u -m mae.scripts.train_mae from_config \
        --config_path="$config" \
        --checkpoints_dir="$scratch_ckpt" \
        --debug_dir="$scratch_dbg" \
        --viz_dir="$scratch_viz" \
        --cache_dir="$data_cache" \
        --device=cuda

echo "Training complete!"
