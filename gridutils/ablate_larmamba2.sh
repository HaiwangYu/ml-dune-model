#!/bin/bash
# ablate_larmamba2.sh — sequential larmamba2 ablation chain (phase 3, issue #1).
#
# For each "name|extra fit args" spec: wait for >=4 free L40S (GPU policy),
# submit a 4-GPU 20-epoch run, keep trailing unified-eval sweeps going until
# training is done AND all 20 epochs are evaluated, print the run's best probe,
# then advance. Prints one line per event (drive under Monitor).
#
# Usage: bash gridutils/ablate_larmamba2.sh   (specs hardcoded below)

set -uo pipefail
REPO=/lbne/u/hyu/ml-dune-model
CONDOR_OUT=/gpfs01/lbne/users/fm/hyu/CONDOR_OUT
CONFIG=$REPO/larmamba2/configs/larmamba2_full20.yml

SPECS=(
  "lm2a_mask50_260807|--model.mask_ratio 0.5"
  "lm2a_mask75_260807|--model.mask_ratio 0.75"
  "lm2a_tile7_260807|--model.tile_size 7"
  "lm2a_occw2_260807|--model.occ_weight 2.0"
  "lm2a_raster_260807|--model.order_kind raster_tick"
  "lm2a_linear_260807|--model.patch_embed linear"
)

free_l40s() {
  condor_status -af State Gpus GPUs_DeviceName 2>/dev/null \
    | awk '/Unclaimed/ && /L40S/ {s+=$2} END {print s+0}'
}

for spec in "${SPECS[@]}"; do
  name=${spec%%|*}; args=${spec#*|}
  d=$CONDOR_OUT/$name
  if ls "$d"/unified_eval/pid_probe_epoch=19-*.json >/dev/null 2>&1; then
    echo "[chain] $name already complete, skipping"; continue
  fi

  # GPU policy: need >=4 free before taking 4
  while true; do
    f=$(free_l40s)
    [ "$f" -ge 4 ] && break
    echo "[chain] $name waiting: only $f free L40S"; sleep 600
  done

  if [ ! -d "$d" ]; then
    echo "[chain] SUBMIT $name ($args)"
    (cd "$REPO" && bash gridutils/submit_larmamba2.sh "$name" "$CONFIG" 4 $args 2>&1 | tail -1)
  else
    echo "[chain] $name dir exists (resuming watch)"
  fi

  # wait: training done AND 20 evals present; keep sweeps flowing
  while true; do
    n=$(ls "$d"/unified_eval/pid_probe_epoch=*.json 2>/dev/null | wc -l)
    nck=$(ls "$d"/checkpoints/ 2>/dev/null | grep -c "^epoch=")
    train_q=$(condor_q -af Cmd 2>/dev/null | grep -c trainjob_larmamba2)
    evals_q=$(condor_q -af Cmd 2>/dev/null | grep -c eval_epochs_polarmae)
    if [ "$n" -ge 20 ]; then echo "[chain] $name COMPLETE (20/20 evals)"; break; fi
    if [ "${train_q:-0}" -eq 0 ] && [ "${nck:-0}" -eq 0 ]; then
      echo "[chain] $name ERROR: training gone with no checkpoints — check $d, aborting chain"; exit 1
    fi
    # submit an eval sweep if checkpoints are pending and none is running
    if [ "${evals_q:-0}" -eq 0 ] && [ "${nck:-0}" -gt "$n" ]; then
      ENCODER=larmamba2 bash "$REPO"/gridutils/submit_eval_epochs.sh "$name" "auto$(date +%H%M%S)" >/dev/null 2>&1 \
        && echo "[chain] $name eval sweep submitted ($n/${nck} evaluated)"
    fi
    sleep 300
  done

  best=$(python3 - "$d" <<'EOF'
import json, glob, re, sys
best=(0,-1,0)
for j in glob.glob(sys.argv[1]+'/unified_eval/pid_probe_epoch=*.json'):
    ep=int(re.search(r'epoch=(\d+)',j).group(1))
    e=next(iter(json.load(open(j)).values()))
    if e['voxel_svm_feat']>best[0]: best=(e['voxel_svm_feat'],ep,e['sft_feat'])
print(f"best svm={best[0]:.4f} (ep{best[1]}) sft={best[2]:.4f}")
EOF
)
  echo "[chain] $name RESULT: $best"
done
echo "[chain] ALL ABLATIONS DONE"
