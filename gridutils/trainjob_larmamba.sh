#!/bin/bash
# trainjob_larmamba.sh — worker-side launcher for larmamba (Mamba encoder) runs.
#
# Reuses PoLAr-MAE's LightningCLI task (polarmae.tasks.polarmae) with a config
# whose encoder class_path is larmamba.MambaEncoder.  The only addition over
# trainjob_polarmae.sh is putting the ml-dune-model repo on PYTHONPATH so that
# class_path resolves.
#
# Args (positional):
#   $1 polar_repo    -- PoLAr-MAE repo root (provides polarmae.tasks + layers)
#   $2 env_prefix    -- conda/uv env prefix (uvenv-polar-mae)
#   $3 config        -- lightning yaml (larmamba_apa2d_*.yml)
#   $4 out_dir       -- GPFS path to rsync outputs to
#   $5 cache_dir     -- general cache base; ${cache_dir}/data for APA2D index
#   $6 run_name      -- run label
#   $7 larmamba_repo -- ml-dune-model repo root (provides larmamba package)

set -euo pipefail

polar_repo=$1
env_prefix=$2
config=$3
out_dir=$4
cache_dir=$5
run_name=$6
larmamba_repo=${7:-/lbne/u/hyu/ml-dune-model}

data_cache="${cache_dir}/data"
mkdir -p "$data_cache"

echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR}"
echo "JOB CONFIGURATION:"
echo "  run_name=${run_name}"
echo "  polar_repo=${polar_repo}"
echo "  larmamba_repo=${larmamba_repo}"
echo "  env_prefix=${env_prefix}"
echo "  config=${config}"
echo "  out_dir=${out_dir}"
echo

export PATH="${env_prefix}/bin:${PATH}"
export LD_LIBRARY_PATH="${env_prefix}/lib:${LD_LIBRARY_PATH:-}"
# Put both repos on PYTHONPATH: polarmae (editable-installed, but be explicit)
# and ml-dune-model so `larmamba.MambaEncoder` resolves.
export PYTHONPATH="${larmamba_repo}:${polar_repo}:${PYTHONPATH:-}"
PY="${env_prefix}/bin/python"
test -x "$PY" || { echo "FATAL: $PY missing" >&2; exit 1; }
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
"$PY" -c "import larmamba; print('larmamba OK', larmamba.__file__)"
"$PY" -c "from larmamba.ssm import kernel_selective_scan_fn; print('mamba CUDA kernel available:', kernel_selective_scan_fn() is not None)"

scratch_root=${_CONDOR_SCRATCH_DIR}/${run_name}
scratch_ckpt=${scratch_root}/checkpoints
scratch_probes=${scratch_root}/probes
scratch_logs=${scratch_root}/lightning_logs
mkdir -p "$scratch_ckpt" "$scratch_probes" "$scratch_logs"

sync_back() {
    echo "Syncing ${scratch_root} -> ${out_dir}"
    mkdir -p "${out_dir}"
    rsync -a "${scratch_root}/" "${out_dir}/" || true
}
cleanup() {
    if [[ -n "${nvsmi_pid:-}" ]]; then kill "${nvsmi_pid}" 2>/dev/null || true; fi
    if [[ -n "${rsync_pid:-}" ]]; then kill "${rsync_pid}" 2>/dev/null || true; fi
    sync_back
}
trap cleanup EXIT
trap 'cleanup; exit 143' SIGTERM

# periodic rsync every 5 min (kill-resilience)
( while true; do sleep 300; sync_back >/dev/null 2>&1 || true; done ) &
rsync_pid=$!

mkdir -p "${out_dir}"
echo "Starting nvidia-smi polling loop -> ${out_dir}/gpu.log"
nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv -l 10 > "${out_dir}/gpu.log" 2>&1 &
nvsmi_pid=$!

cd "$polar_repo"
echo "Launching polarmae.tasks.polarmae fit (larmamba encoder) ..."
"$PY" -u -m polarmae.tasks.polarmae fit \
    --config "$config" \
    --trainer.default_root_dir "$scratch_logs" \
    --trainer.callbacks.init_args.json_dir "$scratch_probes" \
    --data.init_args.dataset_kwargs.cache_dir "$data_cache" \
    --data.init_args.test_dataset_kwargs.cache_dir "$data_cache"

echo "Training complete!"
