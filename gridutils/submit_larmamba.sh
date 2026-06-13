#!/bin/bash
# submit_larmamba.sh — submit a larmamba (Mamba encoder) training run to SDCC.
#
# Usage:
#   bash gridutils/submit_larmamba.sh <run_name> <config.yml>
#
# Reuses the polarmae env + LightningCLI task; the config's encoder class_path
# is larmamba.MambaEncoder.  Outputs land under ${CONDOR_OUT}/${run_name}/.

set -euo pipefail

CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
POLAR_REPO="${POLAR_REPO:-/direct/lbne+u/hyu/PoLAr-MAE}"
LARMAMBA_REPO="${LARMAMBA_REPO:-/lbne/u/hyu/ml-dune-model}"
ENV_PREFIX="${ENV_PREFIX:-/gpfs01/lbne/users/fm/${USER}/uvenv-polar-mae}"
CACHE_DIR="${CACHE_DIR:-/gpfs01/lbne/users/fm/${USER}/cache}"

REQUEST_MEMORY="${REQUEST_MEMORY:-48000}"
REQUEST_GPUS="${REQUEST_GPUS:-1}"
REQUEST_CPUS="${REQUEST_CPUS:-4}"
GPU_REQUIREMENTS="${GPU_REQUIREMENTS:-(GPUs_DeviceName == \"NVIDIA L40S\") && (GPUs_Capability == 8.9)}"

if [ $# -lt 2 ]; then
    echo "usage: $0 <run_name> <config.yml>" >&2
    exit 2
fi
run_name=$1
config=$2
[ -f "$config" ] || { echo "ERROR: config not found: $config" >&2; exit 1; }
config=$(cd "$(dirname "$config")" && pwd)/$(basename "$config")

out_dir="${CONDOR_OUT}/${run_name}"
if [ -d "$out_dir" ]; then
    echo "ERROR: ${out_dir} already exists. Pick a fresh run_name or remove it." >&2
    exit 1
fi
mkdir -p "$out_dir"
echo "Created run directory: ${out_dir}"

subfile="${out_dir}/${run_name}.sub"
cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${LARMAMBA_REPO}/gridutils/trainjob_larmamba.sh
arguments               = ${POLAR_REPO} ${ENV_PREFIX} ${config} ${out_dir} ${CACHE_DIR} ${run_name} ${LARMAMBA_REPO}
environment             = "CLUSTER_ID=\$(ClusterId) JOB_ID=\$(ProcId)"
output                  = ${out_dir}/\$(ClusterId).\$(ProcId).out
error                   = ${out_dir}/\$(ClusterId).\$(ProcId).err
log                     = ${out_dir}/\$(ClusterId).\$(ProcId).log
getenv                  = True
request_memory          = ${REQUEST_MEMORY}
request_cpus            = ${REQUEST_CPUS}
request_gpus            = ${REQUEST_GPUS}
Requirements            = ${GPU_REQUIREMENTS}
should_transfer_files   = NO
stream_output           = True
stream_error            = True
queue 1
EOF

echo "Submitting ${subfile}  (config: ${config})"
condor_submit "$subfile"
