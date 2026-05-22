#!/bin/bash
#
# Worker-side script to run mae.diagnostics.extract_features on a checkpoint.
# Called by extract_features_mae.sub; uses the same env/cache layout as
# trainjob_mae.sh.
#
# Args (positional):
#   $1 codedir     -- path to ml-dune-model repo root
#   $2 pyenv       -- path to uv virtual environment to activate
#   $3 ckpt        -- path to mae checkpoint .pt
#   $4 cache_dir   -- general cache base; ${cache_dir}/warpconvnet and ${cache_dir}/data are used
#   $5 data_root   -- dataset root for feature extraction
#   $6 max_images  -- max images to extract features from
#   $7 pixel_truth -- "True" or "False" (default False); if True, saves per-voxel pid_labels

set -euo pipefail

codedir=$1
pyenv=$2
ckpt=$3
cache_dir=$4
data_root=$5
max_images=$6
pixel_truth=${7:-False}

wp_cache="${cache_dir}/warpconvnet"
data_cache="${cache_dir}/data"
mkdir -p "$wp_cache" "$data_cache"

echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
echo "  ckpt=${ckpt}"
echo "  data_root=${data_root}"
echo "  max_images=${max_images}"
echo ""

export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"

source "${pyenv}/bin/activate"

PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}" \
    python -u -m mae.diagnostics.extract_features \
        --checkpoint="$ckpt" \
        --data_root="$data_root" \
        --max_images="$max_images" \
        --cache_dir="$data_cache" \
        --pixel_truth="$pixel_truth" \
        --device=cuda

echo "Done."
