#!/bin/bash
# submit_larmamba2.sh — submit a larmamba2 (grid-tile MAE) training run (DDP-capable).
#
# Usage:
#   bash gridutils/submit_larmamba2.sh <run_name> <config.yml> [n_gpus] [extra fit args...]
#
# n_gpus (default 4) sets both request_gpus and --trainer.devices. Extra args
# are forwarded to `fit` (used by the smoke test to shrink the run).
#
# Examples:
#   # 2-GPU smoke: 1 epoch, tiny dataset
#   bash gridutils/submit_polarmae20.sh pmae20_smoke_260703 \
#       larmamba/configs/polarmae_apa2d_full20.yml 2 \
#       --trainer.max_epochs 1 --data.init_args.dataset_kwargs.maxlen 400 \
#       --data.init_args.test_dataset_kwargs.maxlen 200
#   # full 4-GPU 20-epoch run
#   bash gridutils/submit_polarmae20.sh pmae20_full_260703 \
#       larmamba/configs/polarmae_apa2d_full20.yml 4

set -euo pipefail

CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
POLAR_REPO="${POLAR_REPO:-/direct/lbne+u/hyu/PoLAr-MAE}"
LARMAMBA_REPO="${LARMAMBA_REPO:-/lbne/u/hyu/ml-dune-model}"
ENV_PREFIX="${ENV_PREFIX:-/gpfs01/lbne/users/fm/${USER}/uvenv-polar-mae}"
CACHE_DIR="${CACHE_DIR:-/gpfs01/lbne/users/fm/${USER}/cache}"

REQUEST_MEMORY="${REQUEST_MEMORY:-96000}"
REQUEST_CPUS="${REQUEST_CPUS:-8}"
GPU_REQUIREMENTS="${GPU_REQUIREMENTS:-(GPUs_DeviceName == \"NVIDIA L40S\") && (GPUs_Capability == 8.9)}"

if [ $# -lt 2 ]; then
    echo "usage: $0 <run_name> <config.yml> [n_gpus] [extra fit args...]" >&2
    exit 2
fi
run_name=$1; config=$2; n_gpus=${3:-4}
if [ $# -ge 3 ]; then shift 3; else shift $#; fi   # remaining = extra fit args
extra_args="$*"
[ -f "$config" ] || { echo "ERROR: config not found: $config" >&2; exit 1; }
config=$(cd "$(dirname "$config")" && pwd)/$(basename "$config")

out_dir="${CONDOR_OUT}/${run_name}"
if [ -d "$out_dir" ]; then
    echo "ERROR: ${out_dir} already exists. Pick a fresh run_name or remove it." >&2
    exit 1
fi
mkdir -p "$out_dir"
echo "Created run directory: ${out_dir}"

# --trainer.devices matches request_gpus; goes first in extra args so a
# user-supplied override (if any) would win by coming later.
fit_args="--trainer.devices ${n_gpus} ${extra_args}"

subfile="${out_dir}/${run_name}.sub"
cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${LARMAMBA_REPO}/gridutils/trainjob_larmamba2.sh
arguments               = ${POLAR_REPO} ${ENV_PREFIX} ${config} ${out_dir} ${CACHE_DIR} ${run_name} ${LARMAMBA_REPO} ${fit_args}
environment             = "CLUSTER_ID=\$(ClusterId) JOB_ID=\$(ProcId)"
output                  = ${out_dir}/\$(ClusterId).\$(ProcId).out
error                   = ${out_dir}/\$(ClusterId).\$(ProcId).err
log                     = ${out_dir}/\$(ClusterId).\$(ProcId).log
getenv                  = True
request_memory          = ${REQUEST_MEMORY}
request_cpus            = ${REQUEST_CPUS}
request_gpus            = ${n_gpus}
Requirements            = ${GPU_REQUIREMENTS}
should_transfer_files   = NO
stream_output           = True
stream_error            = True
queue 1
EOF

echo "Submitting ${subfile}"
echo "  n_gpus=${n_gpus}  extra_args='${extra_args}'"
condor_submit "$subfile"
