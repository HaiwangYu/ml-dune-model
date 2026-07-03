# PoLAr-MAE round 2: 20-epoch full-1M pretraining, per-epoch unified eval

**Date:** 2026-07-03   **Run:** `/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/pmae20_full_260703/`
**Config:** `larmamba/configs/polarmae_apa2d_full20.yml`   **Eval:** `dino/diagnostics/ab_pid_probe.py` (leakage-free, same 500 events / seed as `unified_probe_260702`)

## Motivation

Round 1 (`polarmae_apa2d_full_260523_v3`, 20k steps ≈ **0.36 epoch** over the 1M
set) scored **svm_feat 0.800 / sft_feat 0.783** under the leakage-free unified
probe (see `dino/docs/unified_probe_results.md`). Question: does training
polarmae for many more epochs on the full unlabeled 1M set lift the *honest*
per-pixel-PID representation above ~0.80, or was round 1 already saturated?

## Setup

- 4-GPU DDP, up to 20 epochs, epoch-based cosine schedule (1-ep warmup, LR 7e-5
  for effective batch 64), EarlyStopping on `loss/val` (patience 3, min_delta 0).
- Train `prod-jay-1M/00[1-8]`, val `009` held out. After the APA0/view-W/
  min_points≥256 filter this is **~80k events/epoch** (1260 steps × batch 64),
  so 20 epochs ≈ **1.6M event-views** vs round 1's ~0.32M — a ~5× increase in
  exposure. Wall time ~5.4 min/epoch, ~2 h total.
- Every epoch checkpoint scored by the **unified leakage-free probe** on the same
  500 truth events used for every other model in the comparison (match rate
  1.000 every epoch).

## Per-epoch unified macro-F1

| epoch | svm_feat | sft_feat |   | epoch | svm_feat | sft_feat |
|---|---|---|---|---|---|---|
| 0 | 0.790 | 0.787 | | 10 | 0.817 | 0.800 |
| 1 | 0.792 | 0.790 | | 11 | 0.816 | 0.804 |
| 2 | 0.797 | 0.787 | | 12 | 0.816 | 0.800 |
| 3 | 0.806 | 0.789 | | 13 | 0.817 | 0.802 |
| 4 | 0.816 | 0.793 | | 14 | 0.821 | 0.805 |
| 5 | 0.816 | 0.793 | | 15 | 0.821 | 0.808 |
| 6 | 0.818 | 0.796 | | 16 | 0.822 | 0.807 |
| 7 | 0.820 | 0.803 | | 17 | **0.823** | **0.809** |
| 8 | 0.818 | 0.801 | | 18 | 0.822 | 0.805 |
| 9 | 0.820 | 0.797 | | 19 | 0.822 | 0.808 |

Raw floors identical every epoch (svm_raw 0.402 / sft_raw 0.485), as expected —
same events, same split.

## Findings

1. **More epochs give a modest, real gain: +~0.02.** Best is epoch 17,
   **svm_feat 0.823 / sft_feat 0.809**, vs round-1 0.800 / 0.783. The whole 20
   epochs move svm_feat by +0.023 and sft_feat by +0.026 over round 1.
2. **Fast rise then plateau.** svm_feat climbs 0.790→0.820 over epochs 0–7, then
   is flat (0.816–0.823) for epochs 8–19 — a ±0.003 wobble around ~0.82.
   sft_feat lags ~5 epochs (crosses 0.80 at epoch 7, plateaus ~0.805 by epoch
   14). **The representation is saturated by ~epoch 7–8**; epochs 9–19 add
   nothing beyond noise.
3. **SSL loss decoupled from probe F1 — as flagged.** `loss/val` plateaued at
   ~0.21 by epoch ~8 and only crept (0.288→0.208 over 20 epochs), improving by
   ≤0.001/epoch from epoch 8 on. Because `min_delta=0`, EarlyStopping never
   fired (any non-negative change counts as improvement), so the run went the
   full 20 epochs. Had we monitored the probe instead, epoch ~8 was the natural
   stop. **For future runs: stop on the unified probe, not the recon loss.**
4. **Does NOT recover the old ~0.93.** Even with 5× the training, honest
   polarmae tops out at ~0.82 — nowhere near the 0.933/0.940 the *old
   pixel-split (leaky)* probe reported. This is direct confirmation that the
   ~0.93 was **protocol inflation, not undertraining**: you cannot train your
   way to it under an honest event-level split.

## Updated honest ranking (unified probe, best feature probe)

| model | svm_feat | sft_feat | note |
|---|---|---|---|
| **polarmae round 2 (ep17, 20 ep)** | **0.823** | **0.809** | this run |
| larmamba (round 1, 0.36 ep) | 0.805 | 0.782 | retrain for parity TBD |
| polarmae round 1 (0.36 ep) | 0.800 | 0.783 | |
| dino ep100 | 0.713 | 0.768 | |
| sparseformer w256 | 0.700 | 0.734 | |
| mae ep1 | 0.655 | 0.725 | |

The tokenizer+transformer lead over the sparse-CNN family widens slightly (to
~0.11 svm over dino) with full training, but the ceiling is ~0.82, not ~0.93.

## Caveats / follow-ups

- **Fair comparison to larmamba now needs a matched 20-epoch larmamba run** —
  larmamba's 0.805 is from a 0.36-epoch (20k-step) checkpoint, same as round-1
  polarmae. Round 2 only retrained polarmae; a 20-epoch larmamba would say
  whether the two stay tied at full training. (Recommended next step.)
- Single seed / 500 eval events; treat ±0.005 as noise (the epoch 8–19 plateau
  is within that band).
- Context-length warning persisted (a few dense events exceed 512 groups); match
  rate stayed 1.000, so no measurable harm, but tokenizer scaling could be
  revisited if pushing for more.

## Artifacts

- Checkpoints: `pmae20_full_260703/checkpoints/epoch=NN-step=M.ckpt` (all 20 + last)
- Per-epoch probe JSONs: `pmae20_full_260703/unified_eval/pid_probe_epoch=NN-*.json`
- Reproduce: `bash gridutils/submit_polarmae20.sh <run> larmamba/configs/polarmae_apa2d_full20.yml 4`
  then `bash gridutils/submit_eval_epochs.sh <run>` (auto-driven by
  `gridutils/watch_pmae20.sh`).
- DDP required `NCCL_P2P_DISABLE=1` (P2P/CUMEM broken under Condor cgroups) and a
  DDP-safe launcher dropping the rank-0-only in-training probe; see the commit
  message for `1866ff4` and `gridutils/run_polarmae_fit.py`.
