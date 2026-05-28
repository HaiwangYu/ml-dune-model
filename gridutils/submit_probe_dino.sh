#!/bin/bash
#
# Submit a dino-probe job to SDCC GPU pool.
#
# Default: probes mvicenzi's longer_contrast_slow checkpoints at ep10, ep50, ep100.
# Override with $1 (run-name suffix) + DINO_CKPTS env var (space-separated paths).
#
# Usage:
#   bash gridutils/submit_probe_dino.sh                       # use defaults
#   bash gridutils/submit_probe_dino.sh dino_probes_v2        # custom run_name suffix
#   DINO_CKPTS="/a.pt /b.pt" bash gridutils/submit_probe_dino.sh
#
# Outputs land in $CONDOR_OUT/<run_name>/probes_*.json

set -euo pipefail

CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
REPODIR="${REPODIR:-${HOME}/ml-dune-model}"
PYENV="${PYENV:-/gpfs01/lbne/users/fm/${USER}/uvenv}"
CACHE_DIR="${CACHE_DIR:-/gpfs01/lbne/users/fm/${USER}/cache}"

REQUEST_MEMORY="${REQUEST_MEMORY:-32000}"
REQUEST_GPUS="${REQUEST_GPUS:-1}"
REQUEST_CPUS="${REQUEST_CPUS:-4}"
GPU_REQUIREMENTS="${GPU_REQUIREMENTS:-(GPUs_DeviceName == \"NVIDIA L40S\") && (GPUs_Capability == 8.9)}"

DEFAULT_CKPT_ROOT="/gpfs01/lbne/users/fm/mvicenzi/CONDOR_OUT/longer_contrast_slow/checkpoints"
DEFAULT_CKPTS="${DEFAULT_CKPT_ROOT}/checkpoint_epoch10.pt ${DEFAULT_CKPT_ROOT}/checkpoint_epoch50.pt ${DEFAULT_CKPT_ROOT}/checkpoint_epoch100.pt"

run_name="${1:-dino_probes_longer_contrast_slow_260528}"
ckpts="${DINO_CKPTS:-${DEFAULT_CKPTS}}"

out_dir="${CONDOR_OUT}/${run_name}"

if [ -d "$out_dir" ]; then
  echo "ERROR: ${out_dir} already exists." >&2
  echo "       Choose a new run_name or delete the directory and retry." >&2
  exit 1
fi

echo "Creating run directory: ${out_dir}"
mkdir -p "$out_dir"

subfile="${out_dir}/${run_name}.sub"

# condor_submit doesn't accept unescaped double-quotes inside an arguments=
# value; checkpoint paths have no spaces so we just concatenate them.
args="${REPODIR} ${PYENV} ${out_dir} ${CACHE_DIR} ${ckpts}"

cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${REPODIR}/gridutils/probe_dino.sh
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
echo "  checkpoints: ${ckpts}"
condor_submit "$subfile"
