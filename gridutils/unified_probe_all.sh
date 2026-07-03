#!/bin/bash
#
# unified_probe_all.sh -- worker: re-evaluate mae / dino / polarmae / larmamba
# with the unified leakage-free per-pixel PID probe (dino/diagnostics/ab_pid_probe.py).
#
# One GPU job, three sequential stages across the two envs:
#   1. uvenv (torch 2.10 + WarpConvNet):
#        - dump the probe event list (coords/charge/offsets) -> events.npz
#        - export mae per-voxel features (ep1 + ep5)          -> mae_ep{1,5}_feats.npz
#   2. uvenv-polar-mae (torch 2.5 + polarmae + larmamba):
#        - export polarmae + larmamba per-voxel features      -> {polarmae,larmamba}_feats.npz
#   3. uvenv: run ab_pid_probe -- dino live (teacher + student) + the four
#      external exports -- writing pid_probe_unified_{teacher,student}.json
#
# Args (positional, NO quotes -- condor arguments= cannot contain them):
#   $1 repodir  $2 uvenv  $3 polar_env  $4 polar_repo  $5 outdir  $6 cache_dir
#   $7 n_events  $8 dino_ckpt  $9 mae_ckpt_ep1  $10 mae_ckpt_ep5
#   $11 polarmae_ckpt  $12 larmamba_ckpt

set -euo pipefail

repodir=$1; uvenv=$2; polar_env=$3; polar_repo=$4; outdir=$5; cache_dir=$6
n_events=$7; dino_ckpt=$8; mae_ckpt_a=$9; mae_ckpt_b=${10}
polar_ckpt=${11}; mamba_ckpt=${12}

datadir=/gpfs01/lbne/users/fm/cffm-data/prod-jay-100k-truth-2026-02-27
apa=0
view=W

data_cache="${cache_dir}/data"
wp_cache="${cache_dir}/warpconvnet"
mkdir -p "$data_cache" "$wp_cache" "$outdir"

echo "Running ${CLUSTER_ID:-?}.${JOB_ID:-?} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "  outdir=${outdir}  n_events=${n_events}"

export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"

# ---------- stage 1: uvenv -- event dump + mae exports ----------
(
  source "${uvenv}/bin/activate"
  export PYTHONPATH="${repodir}${PYTHONPATH:+:$PYTHONPATH}"

  echo "=== stage 1a: dump probe events ==="
  python -u -m dino.diagnostics.export_probe_events \
      --datadir "$datadir" --apa "$apa" --view "$view" --cache_dir "$data_cache" \
      --n_events "$n_events" --out "${outdir}/events.npz"

  echo "=== stage 1b: mae feature exports ==="
  python -u -m mae.diagnostics.export_pid_features \
      --ckpt "$mae_ckpt_a" --datadir "$datadir" --apa "$apa" --view "$view" \
      --cache_dir "$data_cache" --n_events "$n_events" \
      --out "${outdir}/mae_ep1_feats.npz"
  python -u -m mae.diagnostics.export_pid_features \
      --ckpt "$mae_ckpt_b" --datadir "$datadir" --apa "$apa" --view "$view" \
      --cache_dir "$data_cache" --n_events "$n_events" \
      --out "${outdir}/mae_ep5_feats.npz"
)

# ---------- stage 2: uvenv-polar-mae -- polarmae + larmamba exports ----------
(
  export PATH="${polar_env}/bin:${PATH}"
  export PYTHONPATH="${repodir}:${polar_repo}${PYTHONPATH:+:$PYTHONPATH}"
  PY="${polar_env}/bin/python"

  echo "=== stage 2a: polarmae feature export ==="
  "$PY" -u -m larmamba.export_pid_features --encoder polarmae \
      --ckpt "$polar_ckpt" --events "${outdir}/events.npz" \
      --out "${outdir}/polarmae_feats.npz"

  echo "=== stage 2b: larmamba feature export ==="
  "$PY" -u -m larmamba.export_pid_features --encoder mamba \
      --ckpt "$mamba_ckpt" --events "${outdir}/events.npz" \
      --out "${outdir}/larmamba_feats.npz"
)

# ---------- stage 3: uvenv -- the unified probe ----------
(
  source "${uvenv}/bin/activate"
  export PYTHONPATH="${repodir}${PYTHONPATH:+:$PYTHONPATH}"

  ext=$(printf '{"mae_ep1": "%s", "mae_ep5": "%s", "polarmae": "%s", "larmamba": "%s"}' \
        "${outdir}/mae_ep1_feats.npz" "${outdir}/mae_ep5_feats.npz" \
        "${outdir}/polarmae_feats.npz" "${outdir}/larmamba_feats.npz")

  echo "=== stage 3a: unified probe (dino teacher + externals) ==="
  python -u -m dino.diagnostics.ab_pid_probe \
      --ckpts "dino_ep100=${dino_ckpt}" --backbone_view teacher \
      --external "$ext" \
      --datadir "$datadir" --apa "$apa" --view "$view" --cache_dir "$data_cache" \
      --n_events "$n_events" \
      --out "${outdir}/pid_probe_unified_teacher.json"

  echo "=== stage 3b: unified probe (dino student) ==="
  python -u -m dino.diagnostics.ab_pid_probe \
      --ckpts "dino_ep100=${dino_ckpt}" --backbone_view student \
      --datadir "$datadir" --apa "$apa" --view "$view" --cache_dir "$data_cache" \
      --n_events "$n_events" \
      --out "${outdir}/pid_probe_unified_student.json"
)

echo "UNIFIED PROBE DONE"
