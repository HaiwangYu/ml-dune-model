#!/bin/bash
#
# Submit the unified-probe re-evaluation (mae/dino/polarmae/larmamba) to the
# SDCC L40S pool. Single sequential GPU job; see unified_probe_all.sh.
#
# Usage:
#   bash gridutils/submit_unified_probe.sh [run_name] [n_events]

set -euo pipefail

CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
REPODIR="${REPODIR:-${HOME}/ml-dune-model}"
UVENV="${UVENV:-/gpfs01/lbne/users/fm/${USER}/uvenv}"
POLAR_ENV="${POLAR_ENV:-/gpfs01/lbne/users/fm/${USER}/uvenv-polar-mae}"
POLAR_REPO="${POLAR_REPO:-/direct/lbne+u/hyu/PoLAr-MAE}"
CACHE_DIR="${CACHE_DIR:-/gpfs01/lbne/users/fm/${USER}/cache}"

REQUEST_MEMORY="${REQUEST_MEMORY:-64000}"
REQUEST_GPUS="${REQUEST_GPUS:-1}"
REQUEST_CPUS="${REQUEST_CPUS:-4}"
GPU_REQUIREMENTS="${GPU_REQUIREMENTS:-(GPUs_DeviceName == \"NVIDIA L40S\") && (GPUs_Capability == 8.9)}"

DINO_CKPT="${DINO_CKPT:-/gpfs01/lbne/users/fm/mvicenzi/CONDOR_OUT/longer_contrast_slow/checkpoints/checkpoint_epoch100.pt}"
MAE_CKPT_EP1="${MAE_CKPT_EP1:-/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/mae_full_v3_260524/checkpoints/mae_epoch1.pt}"
MAE_CKPT_EP5="${MAE_CKPT_EP5:-/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/mae_full_v3_260524/checkpoints/mae_epoch5.pt}"
POLARMAE_CKPT="${POLARMAE_CKPT:-/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/polarmae_apa2d_full_260523_v3/lightning_logs/lightning_logs/version_0/checkpoints/epoch=3-step=20000.ckpt}"
LARMAMBA_CKPT="${LARMAMBA_CKPT:-/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/larmamba_full_260613/lightning_logs/lightning_logs/version_0/checkpoints/epoch=3-step=20000.ckpt}"

run_name="${1:-unified_probe_260702}"
n_events="${2:-500}"

out_dir="${CONDOR_OUT}/${run_name}"
if [ -d "$out_dir" ]; then
  echo "ERROR: ${out_dir} already exists; choose a new run_name." >&2
  exit 1
fi
mkdir -p "$out_dir"

subfile="${out_dir}/${run_name}.sub"

# condor arguments= must NOT contain quotes; all paths are space-free.
args="${REPODIR} ${UVENV} ${POLAR_ENV} ${POLAR_REPO} ${out_dir} ${CACHE_DIR} ${n_events} ${DINO_CKPT} ${MAE_CKPT_EP1} ${MAE_CKPT_EP5} ${POLARMAE_CKPT} ${LARMAMBA_CKPT}"

cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${REPODIR}/gridutils/unified_probe_all.sh
arguments               = ${args}
environment             = "CLUSTER_ID=\$(ClusterId) JOB_ID=\$(ProcId)"
output                  = ${out_dir}/\$(ClusterId).\$(ProcId).out
error                   = ${out_dir}/\$(ClusterId).\$(ProcId).err
log                     = ${out_dir}/\$(ClusterId).\$(ProcId).log
getenv                  = False
request_memory          = ${REQUEST_MEMORY}
request_cpus            = ${REQUEST_CPUS}
request_gpus            = ${REQUEST_GPUS}
Requirements            = ${GPU_REQUIREMENTS}
should_transfer_files   = NO
queue 1
EOF

echo "Submitting ${subfile}"
condor_submit "$subfile"
