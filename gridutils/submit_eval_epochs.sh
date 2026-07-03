#!/bin/bash
# submit_eval_epochs.sh — submit ONE trailing unified-eval sweep (1 GPU) over a
# polarmae run's current epoch checkpoints. Re-run as new epochs land; it skips
# checkpoints already evaluated.
#
# Usage:
#   bash gridutils/submit_eval_epochs.sh <train_run_name> [eval_run_suffix]
#
# Reads checkpoints from CONDOR_OUT/<train_run_name>/checkpoints, reuses the
# prebuilt events.npz, writes JSONs to CONDOR_OUT/<train_run_name>/unified_eval.

set -euo pipefail

CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
REPODIR="${REPODIR:-${HOME}/ml-dune-model}"
UVENV="${UVENV:-/gpfs01/lbne/users/fm/${USER}/uvenv}"
POLAR_ENV="${POLAR_ENV:-/gpfs01/lbne/users/fm/${USER}/uvenv-polar-mae}"
POLAR_REPO="${POLAR_REPO:-/direct/lbne+u/hyu/PoLAr-MAE}"
CACHE_DIR="${CACHE_DIR:-/gpfs01/lbne/users/fm/${USER}/cache}"
EVENTS_NPZ="${EVENTS_NPZ:-${CONDOR_OUT}/unified_probe_260702/events.npz}"
N_EVENTS="${N_EVENTS:-500}"

REQUEST_MEMORY="${REQUEST_MEMORY:-64000}"
REQUEST_CPUS="${REQUEST_CPUS:-4}"
GPU_REQUIREMENTS="${GPU_REQUIREMENTS:-(GPUs_DeviceName == \"NVIDIA L40S\") && (GPUs_Capability == 8.9)}"

if [ $# -lt 1 ]; then echo "usage: $0 <train_run_name> [eval_suffix]" >&2; exit 2; fi
train_run=$1
suffix=${2:-$(date +%H%M%S)}

ckpt_dir="${CONDOR_OUT}/${train_run}/checkpoints"
eval_out="${CONDOR_OUT}/${train_run}/unified_eval"
mkdir -p "$eval_out"
sub_dir="${eval_out}/sub"; mkdir -p "$sub_dir"
subfile="${sub_dir}/eval_${suffix}.sub"

args="${REPODIR} ${UVENV} ${POLAR_ENV} ${POLAR_REPO} ${ckpt_dir} ${EVENTS_NPZ} ${eval_out} ${CACHE_DIR} ${N_EVENTS}"

cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${REPODIR}/gridutils/eval_epochs_polarmae.sh
arguments               = ${args}
environment             = "CLUSTER_ID=\$(ClusterId) JOB_ID=\$(ProcId)"
output                  = ${eval_out}/eval_${suffix}.\$(ClusterId).out
error                   = ${eval_out}/eval_${suffix}.\$(ClusterId).err
log                     = ${eval_out}/eval_${suffix}.\$(ClusterId).log
getenv                  = False
request_memory          = ${REQUEST_MEMORY}
request_cpus            = ${REQUEST_CPUS}
request_gpus            = 1
Requirements            = ${GPU_REQUIREMENTS}
should_transfer_files   = NO
queue 1
EOF

echo "Submitting ${subfile}  (ckpts: ${ckpt_dir})"
condor_submit "$subfile"
