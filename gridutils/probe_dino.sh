#!/bin/bash
#
# Worker-side script: run dino.diagnostics.run_probes on N checkpoints.
# All N runs happen in one Condor allocation to save scheduler overhead.
#
# Args (positional):
#   $1 codedir   -- path to ml-dune-model repo root
#   $2 pyenv     -- path to uv virtual environment to activate
#   $3 outdir    -- output dir on GPFS for probe JSONs (rsynced back)
#   $4 cache_dir -- general cache base; ${cache_dir}/warpconvnet and ${cache_dir}/data are used
#   $5..        -- one or more checkpoint paths
#
# Each checkpoint produces ${outdir}/probes_ep<N>.json (and copies the .out
# log alongside).  Outputs are staged on $_CONDOR_SCRATCH_DIR and rsynced.

set -euo pipefail

if [ $# -lt 5 ]; then
  echo "usage: $0 <codedir> <pyenv> <outdir> <cache_dir> <ckpt1> [ckpt2 ...]" >&2
  exit 2
fi

codedir=$1; shift
pyenv=$1; shift
outdir=$1; shift
cache_dir=$1; shift
ckpts=("$@")

wp_cache="${cache_dir}/warpconvnet"
data_cache="${cache_dir}/data"
mkdir -p "$wp_cache" "$data_cache" "$outdir"

scratch_out=${_CONDOR_SCRATCH_DIR}/probe_out
mkdir -p "$scratch_out"

sync_back() {
  echo "Syncing ${scratch_out} -> ${outdir} at $(date -Iseconds)"
  rsync -a "${scratch_out}/" "${outdir}/" || true
}
cleanup() { sync_back; }
trap cleanup EXIT
trap 'cleanup; exit 143' SIGTERM

echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  outdir=${outdir}"
echo "  cache_dir=${cache_dir}"
echo "  checkpoints (${#ckpts[@]}):"
for c in "${ckpts[@]}"; do echo "    - ${c}"; done
echo ""

export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"

source "${pyenv}/bin/activate"

for ckpt in "${ckpts[@]}"; do
  if [ ! -f "$ckpt" ]; then
    echo "WARNING: checkpoint missing, skipping: $ckpt" >&2
    continue
  fi
  base=$(basename "$ckpt" .pt)
  out_json="${scratch_out}/probes_${base}.json"
  log_file="${scratch_out}/probes_${base}.log"

  echo "============================================================"
  echo "PROBE: $ckpt"
  echo "  -> $out_json"
  echo "============================================================"

  PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}" \
      python -u -m dino.diagnostics.run_probes \
          --checkpoint="$ckpt" \
          --output="$out_json" \
          --batch_size=32 \
          --num_workers=4 \
          --cap_per_class=5000 \
          --val_frac=0.2 \
          --epochs=30 \
          --device=cuda 2>&1 | tee "$log_file"

  # Periodic rsync between checkpoints so partial results survive eviction
  sync_back
done

echo "All probes complete."
