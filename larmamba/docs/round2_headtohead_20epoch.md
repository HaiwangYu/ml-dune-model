# Round 2 head-to-head: polarmae vs larmamba, matched 20-epoch full-1M

**Date:** 2026-07-03
**Runs:** `pmae20_full_260703` (polarmae) · `larmamba20_full_260703` (larmamba)
**Eval:** `dino/diagnostics/ab_pid_probe.py` (leakage-free, same 500 events / seed / split for both)

## Setup (identical for both)

4-GPU DDP, 20 epochs on `prod-jay-1M/00[1-8]` (~80k events/epoch after filter,
009 held out), epoch cosine schedule (1-ep warmup, LR 7e-5, eff-batch 64),
per-epoch checkpoint scored by the unified probe. Only the encoder differs:
polarmae = ViT-Small self-attention; larmamba = bidirectional Mamba (linear-time).
Loss curves were near-identical (both `loss/val` 0.29→0.207).

## Per-epoch unified macro-F1

| ep | polarmae svm / sft | larmamba svm / sft |   | ep | polarmae svm / sft | larmamba svm / sft |
|---|---|---|---|---|---|---|
| 0 | 0.790 / 0.787 | 0.780 / 0.781 | | 10 | 0.817 / 0.800 | 0.815 / 0.793 |
| 1 | 0.792 / 0.790 | 0.795 / 0.788 | | 11 | 0.816 / 0.804 | 0.815 / 0.795 |
| 2 | 0.797 / 0.787 | 0.801 / 0.790 | | 12 | 0.816 / 0.800 | 0.815 / 0.798 |
| 3 | 0.806 / 0.789 | 0.796 / 0.788 | | 13 | 0.817 / 0.802 | 0.818 / 0.791 |
| 4 | 0.816 / 0.793 | 0.804 / 0.784 | | 14 | 0.821 / 0.805 | 0.813 / 0.803 |
| 5 | 0.816 / 0.793 | 0.806 / 0.783 | | 15 | 0.821 / 0.808 | 0.816 / 0.800 |
| 6 | 0.818 / 0.796 | 0.807 / 0.786 | | 16 | 0.822 / 0.807 | 0.816 / 0.797 |
| 7 | 0.820 / 0.803 | 0.807 / 0.792 | | 17 | **0.823** / **0.809** | 0.817 / 0.795 |
| 8 | 0.818 / 0.801 | 0.811 / 0.789 | | 18 | 0.822 / 0.805 | 0.818 / 0.799 |
| 9 | 0.820 / 0.797 | 0.814 / 0.789 | | 19 | 0.822 / 0.808 | **0.818** / 0.797 |

## Summary

| | polarmae | larmamba | Δ (pm − lm) |
|---|---|---|---|
| best svm_feat | 0.823 (ep17) | 0.818 (ep19) | +0.005 |
| plateau svm_feat (ep7–19 mean) | 0.819 | 0.815 | **+0.005** |
| plateau sft_feat (ep7–19 mean) | 0.804 | 0.795 | **+0.009** |

## Findings

1. **At matched 20-epoch training the two are effectively tied, with polarmae a
   hair ahead.** Plateau svm_feat 0.819 vs 0.815 (+0.005) and sft_feat 0.804 vs
   0.795 (+0.009) — both differences at or just past the ±0.005 single-seed
   noise floor. Attention holds a small, consistent edge on the MLP probe (sft)
   in particular.
2. **Both saturate by epoch ~7–8** and wobble within ±0.005 thereafter; larmamba
   rises slightly more gradually early (behind through ep0–9) then converges
   onto polarmae's plateau (ep13–19 essentially overlapping on svm).
3. **The earlier "larmamba ≈ polarmae" conclusion holds under full training** —
   and is now fair: the previous comparison had larmamba at a 0.36-epoch
   checkpoint (0.805) against 0.36-epoch polarmae (0.800). With both at 20
   epochs it's 0.815 vs 0.819. Neither the extra epochs nor the encoder swap
   changes the standing: same accuracy class (~0.82), attention marginally
   better, Mamba's advantage remains its O(T) memory scaling (not needed at this
   task's ~256 tokens), not accuracy.
4. **Ceiling confirmed at ~0.82 for both.** Full-epoch training on the honest
   probe caps the tokenizer+transformer family at ~0.82 svm / ~0.80 sft,
   regardless of mixer. The old leaky ~0.93 is not reachable by either.

## Verdict

For this DUNE pixel-PID task: **polarmae and larmamba are equivalent in accuracy
(~0.82 svm) at matched 20-epoch training, with polarmae ~0.005–0.009 ahead —
within/at the noise floor, and larger on the sft probe.** Choose by the
non-accuracy axis: polarmae for slightly better accuracy + faster attention at
256 tokens; larmamba if a future task needs far more tokens than attention can
fit (the mixer-benchmark regime). Extra pretraining buys ~+0.02 for both over
the 0.36-epoch checkpoints and then saturates.

## Artifacts

- polarmae: `pmae20_full_260703/` (curve in `polarmae_round2_20epoch.md`)
- larmamba: `larmamba20_full_260703/` (config `larmamba/configs/larmamba_apa2d_full20.yml`)
- Per-epoch JSONs under each run's `unified_eval/`.
