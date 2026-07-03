#!/bin/bash
# watch_pmae20.sh — orchestrator for the 20-epoch polarmae run. Polls the run's
# checkpoints; auto-submits a trailing unified-eval sweep whenever an epoch
# checkpoint lacks its probe JSON (skips ones already done; won't double-submit
# while a sweep is in flight); reports per-epoch loss/val and finished eval
# macro-F1. Prints one line per event (each becomes a Monitor notification).
#
# Usage: bash watch_pmae20.sh <train_run_name> <cluster_id>
set -uo pipefail
run=${1:-pmae20_full_260703}
cid=${2:-832}
base=/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/$run
repo=/lbne/u/hyu/ml-dune-model
ff=$base/${cid}.0.out; fe=$base/${cid}.0.err
prevloss=""
while true; do
  loss=$(grep -hE "loss/val improved|Metric loss/val" "$fe" "$ff" 2>/dev/null | tail -1)
  if [ -n "$loss" ] && [ "$loss" != "$prevloss" ]; then echo "[train] $loss"; prevloss="$loss"; fi

  missing=0
  while IFS= read -r c; do
    [ -z "$c" ] && continue
    b=$(basename "$c" .ckpt)
    [ -f "$base/unified_eval/pid_probe_${b}.json" ] || missing=1
  done < <(ls "$base/checkpoints/" 2>/dev/null | grep -E '^epoch=.*\.ckpt$')
  if [ "$missing" = 1 ]; then
    inflight=$(condor_q -af Cmd 2>/dev/null | grep -c eval_epochs_polarmae)
    if [ "${inflight:-0}" -eq 0 ]; then
      echo "[eval] pending checkpoint(s) -> submitting eval sweep for $run"
      ( cd "$repo" && bash gridutils/submit_eval_epochs.sh "$run" "auto$(date +%H%M%S)" 2>&1 | tail -1 )
    fi
  fi

  while IFS= read -r j; do
    [ -z "$j" ] && continue
    tag=$(basename "$j"); marker="$base/unified_eval/.reported_${tag}"
    if [ ! -f "$marker" ]; then
      vals=$(python3 -c "import json;e=json.load(open('$j'));print('svm_feat=%.4f sft_feat=%.4f'%(e['voxel_svm_feat'],e['sft_feat']))" 2>/dev/null)
      echo "[eval-result] ${tag} -> ${vals}"; touch "$marker"
    fi
  done < <(ls "$base/unified_eval/" 2>/dev/null | grep -E '^pid_probe_epoch=.*\.json$' | sed "s|^|$base/unified_eval/|")

  grep -q "Training complete" "$ff" 2>/dev/null && echo ">>> TRAINING COMPLETE marker seen"
  st=$(condor_q "$cid" -af JobStatus 2>/dev/null | head -1)
  [ -z "$st" ] && echo ">>> FULL RUN $cid LEFT QUEUE (done or stopped)"
  sleep 300
done
