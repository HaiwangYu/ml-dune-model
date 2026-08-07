#!/bin/bash
# trainjob_larmamba2.sh — worker-side launcher for larmamba2 (grid-tile MAE)
# polarmae run (multi-GPU DDP capable). Based on trainjob_larmamba.sh, but:
#   - launches via gridutils/run_polarmae_fit.py (applies the DDP-safe probe
#     patch before the LightningCLI),
#   - passes PROBE_JSON_DIR through the env (config reads it via oc.env),
#   - forwards any trailing args ($8..) to `fit` (e.g. --trainer.devices 2
#     --trainer.max_epochs 1 --data.init_args.dataset_kwargs.maxlen 300 for smoke).
#
# Args (positional):
#   $1 polar_repo  $2 env_prefix  $3 config  $4 out_dir  $5 cache_dir
#   $6 run_name  $7 larmamba_repo  $8.. extra fit args

set -euo pipefail

polar_repo=$1; env_prefix=$2; config=$3; out_dir=$4; cache_dir=$5
run_name=$6; larmamba_repo=${7:-/lbne/u/hyu/ml-dune-model}
shift 7 || true
extra_args=("$@")

data_cache="${cache_dir}/data"
mkdir -p "$data_cache"

echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  run_name=${run_name}  config=${config}"
echo "  extra_args=${extra_args[*]:-<none>}"
echo

export PATH="${env_prefix}/bin:${PATH}"
export LD_LIBRARY_PATH="${env_prefix}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${larmamba_repo}:${polar_repo}:${PYTHONPATH:-}"
# NCCL P2P/CUMEM peer transport is broken under Condor's cgroup GPU isolation on
# this pool (all_reduce silently moves no data, then the watchdog times out;
# verified with gridutils/nccl_test.py). Force NCCL onto shared-memory transport
# for intra-node multi-GPU DDP. Required for devices>1.
export NCCL_P2P_DISABLE=1
PY="${env_prefix}/bin/python"
test -x "$PY" || { echo "FATAL: $PY missing" >&2; exit 1; }
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

scratch_root=${_CONDOR_SCRATCH_DIR}/${run_name}
scratch_ckpt=${scratch_root}/checkpoints
scratch_probes=${scratch_root}/probes
scratch_logs=${scratch_root}/lightning_logs
mkdir -p "$scratch_ckpt" "$scratch_probes" "$scratch_logs"
export PROBE_JSON_DIR="$scratch_probes"
export CKPT_DIR="$scratch_ckpt"      # ModelCheckpoint.dirpath (rsync'd to out_dir/checkpoints)

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

# periodic rsync every 5 min so per-epoch checkpoints reach GPFS for the
# trailing unified-eval jobs while training continues.
( while true; do sleep 300; sync_back >/dev/null 2>&1 || true; done ) &
rsync_pid=$!

mkdir -p "${out_dir}"
nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 30 > "${out_dir}/gpu.log" 2>&1 &
nvsmi_pid=$!

cd "$larmamba_repo"
echo "Launching larmamba2.train fit (${run_name}) ..."
"$PY" -u -m larmamba2.train fit \
    --config "$config" \
    --trainer.default_root_dir "$scratch_logs" \
    --data.dataset_kwargs.cache_dir "$data_cache" \
    --data.test_dataset_kwargs.cache_dir "$data_cache" \
    "${extra_args[@]}"

echo "Training complete!"
