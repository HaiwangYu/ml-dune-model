# larmamba2 Phase 3 — ablations

**Date:** 2026-08-07   **Tracker:** [issue #1](https://github.com/HaiwangYu/ml-dune-model/issues/1)
**Driver:** `gridutils/ablate_larmamba2.sh` (sequential, 4-GPU each, GPU-policy gated)
**Eval:** unified leakage-free probe, 500 events/seed. Baseline = lm2v3
(tile5, mask0.6, conv embed, morton, occ_weight1, LR 5e-5).

## Main effects (one knob off the baseline each, 20 epochs)

| run | change | best svm (ep) | plateau svm | verdict |
|---|---|---|---|---|
| lm2v3 | — baseline — | 0.781 (16) | 0.777 | — |
| lm2a_mask50 | mask 0.6→0.5 | 0.775 (14) | 0.774 | ✗ −0.005 |
| **lm2a_mask75** | **mask 0.6→0.75** | **0.792 (16)** | **0.791** | ✓ **+0.011** |
| **lm2a_tile7** | **tile 5→7** | **0.791 (12)** | 0.787 | ✓ **+0.010** |
| lm2a_occw2 | occ_weight 1→2 | 0.777 (10) | 0.765 | ✗ −0.004 |
| lm2a_raster | morton→raster_tick | 0.778 (10) | 0.760 | ✗ −0.003 |
| lm2a_linear | conv→linear embed | 0.781 (10) | 0.779 | = (simpler) |

## Findings

1. **Masking ratio is the strongest single lever, monotonic up.** 0.5→0.6→0.75
   gives plateau 0.774→0.777→0.791. Harder reconstruction ⇒ better
   representation (classic MAE), and larmamba2's optimum (0.75) is *higher* than
   the FPS-tokenizer models' 0.6 — the dense grid-tile target is easier to
   reconstruct, so it wants more masking. Worth probing 0.8+ later.
2. **tile7 targets exactly the diagnosed weakness.** +0.010 overall, but the
   per-class story is the point: **shower 0.729→0.750** (bigger 7×7 receptive
   field captures more EM-cluster context) while track is flat (0.763→0.765).
   Confirms the Phase-2 diagnosis that the gap is receptive-field / boundary
   limited, and motivates overlapping tiles (context without the token cost).
3. **The conv tokenizer earns nothing at this patch size.** linear ≈ conv
   (0.781/0.779 vs 0.781/0.777) — a 5×5 = 25-px patch is small enough that a
   single Linear captures it. Prefer **linear** (fewer params, no accuracy
   cost). VQ-VAE (option C) is not worth pursuing on this evidence.
4. **occ_weight 2 and raster both hurt**, the plateau most (0.765, 0.760).
   Keep occ_weight=1 (balanced BCE/charge) and morton (2D locality beats
   drift-time raster). These are settled — drop from future sweeps.

## Combined run — the effects do NOT stack

**lm2c_t7m75_260807** = tile7 + mask0.75 + linear embed: best svm **0.790**
(ep17) / plateau **0.787**. This is **tied with either knob alone** (mask0.75
0.792, tile7 0.791), not the ~0.80 an additive model predicted.

| config | best svm | plateau |
|---|---|---|
| baseline | 0.781 | 0.777 |
| mask0.75 | **0.792** | 0.791 |
| tile7 | 0.791 | 0.787 |
| tile7 + mask0.75 | 0.790 | 0.787 |

The combined run converges *faster* (ahead at every epoch ≤5) but hits the same
ceiling. Interpretation: the two winners are **redundant, not complementary** —
both work by giving the reconstruction more/harder context (larger patch;
fewer visible tiles), and they saturate at a common **~0.79 ceiling** for this
grid-tile tokenizer. That is ~0.025 below larmamba R2 (0.815) and ~0.03 below
polarmae R2 (0.819).

## Where this leaves larmamba2

**Best config: mask0.75 (tile5, linear embed, morton), plateau 0.791.** The
native-2D tokenizer gets within ~0.025 of the FPS-tokenizer transformers while
being deterministic (zero overflow), 2D-native (no z=0), pytorch3d-free, and
eval-uncapped — but a per-pixel-loss grid MAE appears to top out ~0.79 on this
task. Closing the last ~0.025 needs a *structural* change, not more knob-tuning.

## Deferred / next (structural — knob sweep is exhausted)

- **Overlapping tiles** (stride < S): reconstruct center region from a padded
  context patch — removes boundary cuts, the one receptive-field lever not yet
  tried. Strongest remaining candidate; needs a `tiler.py` change.
- mask 0.8 (ratio trend hadn't turned over at 0.75 — one cheap check).
- Contrastive/DINO-style objective instead of reconstruction (the sparse-CNN
  program showed DINO > MAE at fixed backbone; may lift the ceiling here too).
- All runs ~2.5 h on 4×L40S; chain driver restart-safe, GPU-policy gated.
