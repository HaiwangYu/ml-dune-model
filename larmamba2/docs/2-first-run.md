# larmamba2 Phase 2 — first full runs: two findings and a 0.78 baseline

**Date:** 2026-08-07   **Tracker:** [issue #1](https://github.com/HaiwangYu/ml-dune-model/issues/1)
**Runs:** `lm2_full_260807` (L1, killed ep5) · `lm2v2_full_260807` (occ_l1, LR 7e-5, early-stopped ep8) · `lm2v3_full_260807` (occ_l1, LR 5e-5, full 20 ep)
**Eval:** unified leakage-free probe, same 500 events/seed as all baselines.

## Headline

First full training of the grid-tile Mamba MAE reaches **svm_feat 0.781 (ep16),
late-plateau 0.777** — decisively above every sparse-CNN (dino 0.713) and ~0.04
below the FPS-tokenizer transformers (larmamba R2 0.815 / polarmae R2 0.819
plateau) — with zero hyperparameter tuning yet. Two transferable findings came
out of getting there.

## Finding 1 — plain L1 on sparse patches is *provably* degenerate

Run 1 (plain L1 on all 25 pixels of masked tiles): probe frozen at random-init
level (0.70) for 5 epochs; the hit/empty loss split showed `l1_empty ≈ 0.008`,
`l1_hit ≈ 1.06` flat — the model predicts "empty" everywhere.

Not a tuning problem: **L1 is median-seeking, and each pixel of a masked tile
is hit with probability < 50%, so the L1-optimal prediction at every uncertain
pixel is exactly the empty value.** (`loss/val 0.414 ≈ occupancy 0.42 × mean
|q−(−1)|` — the degenerate optimum, numerically.)

Fix (plan §3.5's standby, now default `loss_type=occ_l1`): decoder head emits
2×25 per tile — **BCE on per-pixel occupancy + L1 on log-charge at hit pixels
only**. BCE's optimum is the hit *probability* (informative everywhere); the
charge loss stops being diluted by the empty majority. Immediately: `occ_bce`
0.645→0.442 by ep5 (below the 0.681 constant-base-rate floor), `l1_hit`
1.06→0.24, probe unstuck.

## Finding 2 — LR 7e-5 destabilizes; the recon loss again decouples from the probe

Run 2 (occ_l1, LR 7e-5 = the polarmae/larmamba round-2 value): healthy climb to
svm 0.762 @ ep6, then a genuine train-loss step (0.68→0.76 at ~step 9.8k) that
**damaged the features** (probe 0.762→0.754), and EarlyStopping(patience 3)
killed the run at ep8 — while the probe had been rising the whole time.
Response: LR → 5e-5, **EarlyStopping removed** (per-epoch probe curve is the
stop signal — the standing lesson from round 2).

## lm2v3 — the clean 20-epoch run

| ep | svm / sft | ep | svm / sft |
|---|---|---|---|
| 0 | 0.713 / 0.708 | 10 | 0.765 / 0.732 |
| 2 | 0.739 / 0.731 | 12 | 0.767 / 0.728 |
| 4 | 0.761 / 0.756 | 14 | 0.773 / 0.772 |
| 6 | 0.770 / 0.763 | **16** | **0.781** / 0.766 |
| 7 | 0.775 / 0.768 | 18 | 0.778 / 0.773 |
| 9 | 0.775 / 0.770 | 19 | 0.776 / 0.767 |

Fast climb to ~0.775 by ep7, a mid-schedule wobble (ep10–13, sft especially —
echoes of the LR sensitivity even at 5e-5), recovery to a late plateau
**0.777 mean (ep15–19), best 0.781 @ ep16**. No instability event like v2's.

## Where larmamba2 stands (unified probe, best / plateau svm)

| model | best svm | plateau svm | sft |
|---|---|---|---|
| polarmae R2 (20 ep) | 0.823 | 0.819 | 0.804 |
| larmamba R2 (20 ep) | 0.818 | 0.815 | 0.795 |
| **larmamba2 v3 (20 ep, untuned)** | **0.781** | **0.777** | ~0.77 |
| dino ep100 | 0.713 | — | 0.768 |
| sparseformer w256 | 0.700 | — | 0.734 |

Per-class (svm, best epochs): the gap to polarmae is **track 0.763 vs 0.802 and
shower 0.729 vs 0.806**; "other" is nearly closed (0.849 vs 0.861). The missing
~0.04 is extended-object identity — plausibly tile-boundary effects and/or the
5×5 receptive field of the tokenizer, both phase-3 levers.

Already-banked architectural wins: deterministic tokenizer (zero overflow
warnings), no pytorch3d / no z=0 hack, uncapped eval, exact voxel→token feature
assignment (match rate 1.000 every epoch), ~5.5 min/epoch on 4×L40S — same
cost as the FPS tokenizer models.

## Phase-3 priorities (to close 0.04)

1. **Mask ratio** {0.5, 0.75} — occupancy prediction may be too easy/hard at 0.6.
2. **Tile size 7 + overlap (stride < S)** — targets the track/shower gap
   (bigger receptive field, fewer boundary cuts).
3. **occ_weight** {0.5, 2} — balance BCE vs charge terms.
4. **Serialization** {hilbert, raster_tick} and **linear-embed control**.
5. LR schedule: the ep10–13 wobble suggests trying 3.5e-5 or longer warmup.

Runs are ~2 h each on 4 GPUs; batch respecting the ≤4-GPUs-when->4-free policy.
