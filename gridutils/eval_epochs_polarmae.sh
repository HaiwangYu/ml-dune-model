#!/bin/bash
# eval_epochs_polarmae.sh — trailing UNIFIED-probe eval for polarmae epoch
# checkpoints. Processes every epoch checkpoint in a run's checkpoint dir that
# doesn't yet have a probe JSON, then exits (re-submit as new epochs land).
#
# Reuses the prebuilt event dump (events.npz) so no WarpConvNet dump is needed
# per call; two stages per checkpoint:
#   A (uvenv-polar-mae): larmamba.export_pid_features --encoder polarmae -> npz
#   B (uvenv):           ab_pid_probe --external -> pid_probe_<ckpt>.json
#
# Args (positional):
#   $1 repodir  $2 uvenv  $3 polar_env  $4 polar_repo  $5 ckpt_dir
#   $6 events_npz  $7 eval_out_dir  $8 cache_dir  $9 n_events

set -euo pipefail

repodir=$1; uvenv=$2; polar_env=$3; polar_repo=$4; ckpt_dir=$5
events_npz=$6; eval_out=$7; cache_dir=$8; n_events=${9:-500}

datadir=/gpfs01/lbne/users/fm/cffm-data/prod-jay-100k-truth-2026-02-27
apa=0; view=W
data_cache="${cache_dir}/data"
wp_cache="${cache_dir}/warpconvnet"
scratch="${_CONDOR_SCRATCH_DIR:-/tmp}/pmae_eval_$$"
mkdir -p "$data_cache" "$wp_cache" "$eval_out" "$scratch"

echo "Running ${CLUSTER_ID:-?}.${JOB_ID:-?} on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "  ckpt_dir=${ckpt_dir}  eval_out=${eval_out}"

export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"

if [ ! -f "$events_npz" ]; then
  echo "FATAL: events dump missing: $events_npz" >&2; exit 1
fi

shopt -s nullglob
ckpts=( "${ckpt_dir}"/epoch*-step*.ckpt )
if [ ${#ckpts[@]} -eq 0 ]; then
  echo "No epoch*-step*.ckpt in ${ckpt_dir} yet; nothing to do."
  exit 0
fi

did_any=0
for ckpt in "${ckpts[@]}"; do
  base=$(basename "$ckpt" .ckpt)
  out_json="${eval_out}/pid_probe_${base}.json"
  if [ -f "$out_json" ]; then
    echo "skip ${base}: already evaluated"
    continue
  fi
  echo "============================================================"
  echo "EVAL ${base}"
  feat="${scratch}/feat_${base}.npz"

  # stage A: polarmae per-voxel feature export (torch 2.5 env)
  ( export PATH="${polar_env}/bin:${PATH}"
    export PYTHONPATH="${repodir}:${polar_repo}${PYTHONPATH:+:$PYTHONPATH}"
    "${polar_env}/bin/python" -u -m larmamba.export_pid_features --encoder polarmae \
        --ckpt "$ckpt" --events "$events_npz" --num_groups 256 --context_length 512 \
        --out "$feat" ) || { echo "FAILED export ${base}"; continue; }

  # stage B: unified probe on the export (uvenv / WarpConvNet for truth dataset)
  ( source "${uvenv}/bin/activate"
    export PYTHONPATH="${repodir}${PYTHONPATH:+:$PYTHONPATH}"
    ext=$(printf '{"polarmae_%s": "%s"}' "$base" "$feat")
    python -u -m dino.diagnostics.ab_pid_probe --external "$ext" \
        --datadir "$datadir" --apa "$apa" --view "$view" --cache_dir "$data_cache" \
        --n_events "$n_events" --out "$out_json" ) || { echo "FAILED probe ${base}"; continue; }

  rm -f "$feat"
  did_any=1
  echo "wrote ${out_json}"
done

echo "EVAL SWEEP DONE (did_any=${did_any})"
