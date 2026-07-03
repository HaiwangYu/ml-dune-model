#!/bin/bash
#
# unified_probe_sf.sh -- worker: add sparseformer checkpoints to the unified
# PID-probe evaluation. Single env (uvenv), two stages:
#   1. export per-voxel features for sf_combo ep1, sf_w256 ep1, sf_w256 ep2
#   2. run ab_pid_probe on the three exports (external-only). Same dataset,
#      --n_events and seed as the main unified_probe run, so the event-level
#      split is identical and the numbers are directly comparable.
#
# Args (positional, NO quotes):
#   $1 repodir  $2 uvenv  $3 outdir  $4 cache_dir  $5 n_events

set -euo pipefail

repodir=$1; uvenv=$2; outdir=$3; cache_dir=$4; n_events=$5

datadir=/gpfs01/lbne/users/fm/cffm-data/prod-jay-100k-truth-2026-02-27
apa=0
view=W
sf_out=/gpfs01/lbne/users/fm/hyu/CONDOR_OUT

data_cache="${cache_dir}/data"
wp_cache="${cache_dir}/warpconvnet"
mkdir -p "$data_cache" "$wp_cache" "$outdir"

echo "Running ${CLUSTER_ID:-?}.${JOB_ID:-?} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"

source "${uvenv}/bin/activate"
export PYTHONPATH="${repodir}${PYTHONPATH:+:$PYTHONPATH}"

echo "=== stage 1: sparseformer feature exports ==="
python -u -m mae.diagnostics.export_pid_features \
    --ckpt "${sf_out}/sf_combo_260614/checkpoints/mae_epoch1.pt" \
    --config "${repodir}/sparseformer/configs/config_sf_combo.json" \
    --datadir "$datadir" --apa "$apa" --view "$view" --cache_dir "$data_cache" \
    --n_events "$n_events" --out "${outdir}/sf_combo_ep1_feats.npz"

python -u -m mae.diagnostics.export_pid_features \
    --ckpt "${sf_out}/sf_w256_260614/checkpoints/mae_epoch1.pt" \
    --config "${repodir}/sparseformer/configs/config_sf_w256.json" \
    --datadir "$datadir" --apa "$apa" --view "$view" --cache_dir "$data_cache" \
    --n_events "$n_events" --out "${outdir}/sf_w256_ep1_feats.npz"

python -u -m mae.diagnostics.export_pid_features \
    --ckpt "${sf_out}/sf_w256_260614/checkpoints/mae_epoch2.pt" \
    --config "${repodir}/sparseformer/configs/config_sf_w256.json" \
    --datadir "$datadir" --apa "$apa" --view "$view" --cache_dir "$data_cache" \
    --n_events "$n_events" --out "${outdir}/sf_w256_ep2_feats.npz"

echo "=== stage 2: unified probe (sparseformer externals) ==="
ext=$(printf '{"sf_combo_ep1": "%s", "sf_w256_ep1": "%s", "sf_w256_ep2": "%s"}' \
      "${outdir}/sf_combo_ep1_feats.npz" "${outdir}/sf_w256_ep1_feats.npz" \
      "${outdir}/sf_w256_ep2_feats.npz")

python -u -m dino.diagnostics.ab_pid_probe \
    --external "$ext" \
    --datadir "$datadir" --apa "$apa" --view "$view" --cache_dir "$data_cache" \
    --n_events "$n_events" \
    --out "${outdir}/pid_probe_unified_sf.json"

echo "UNIFIED PROBE DONE"
