#!/bin/bash
#
# Submit the sparseformer addition to the unified-probe evaluation (L40S).
# See unified_probe_sf.sh. Usage:
#   bash gridutils/submit_unified_probe_sf.sh [run_name] [n_events]

set -euo pipefail

CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
REPODIR="${REPODIR:-${HOME}/ml-dune-model}"
UVENV="${UVENV:-/gpfs01/lbne/users/fm/${USER}/uvenv}"
CACHE_DIR="${CACHE_DIR:-/gpfs01/lbne/users/fm/${USER}/cache}"

REQUEST_MEMORY="${REQUEST_MEMORY:-64000}"
REQUEST_GPUS="${REQUEST_GPUS:-1}"
REQUEST_CPUS="${REQUEST_CPUS:-4}"
GPU_REQUIREMENTS="${GPU_REQUIREMENTS:-(GPUs_DeviceName == \"NVIDIA L40S\") && (GPUs_Capability == 8.9)}"

run_name="${1:-unified_probe_sf_260702}"
n_events="${2:-500}"

out_dir="${CONDOR_OUT}/${run_name}"
if [ -d "$out_dir" ]; then
  echo "ERROR: ${out_dir} already exists; choose a new run_name." >&2
  exit 1
fi
mkdir -p "$out_dir"

subfile="${out_dir}/${run_name}.sub"
args="${REPODIR} ${UVENV} ${out_dir} ${CACHE_DIR} ${n_events}"

cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${REPODIR}/gridutils/unified_probe_sf.sh
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
