# sparseformer results — can we lift the sparse-CNN toward PoLAr-MAE?

**Date:** 2026-06-14   **Branch:** may15   **Env:** ml-dune-model uvenv (torch 2.10 + WarpConvNet, GPU-only)

## Question

mae/dino (sparse-CNN) plateau at ~0.67/0.72 feature-probe macro-F1 vs PoLAr-MAE's
0.93. Can architecture changes to the **sparse-CNN family** (keeping cheap
per-voxel inference, no FPS tokenizer) close the gap? We tested two ideas + combo,
holding the SSL pipeline fixed (standard MAE, batch 32, same data/masking) and
varying **only the backbone**:

- **opt2 — local-geometry stem**: replace the single 3×3 `conv0` with parallel
  conv stacks of receptive field 3/5/7 (richer per-voxel local descriptor).
- **opt1 — sparseformer**: replace the single bottleneck-attention block with a
  **stack of 6** (a deep sparse transformer at the coarsest scale).
- **combo**: both.
- **ctrl**: the current MinkUNet backbone (control, same pipeline).

## Result (5-epoch runs; probes plateau by epoch 1)

| Variant | sft_feat | voxel_svm_feat | sft_raw | svm_raw | params | peak GPU |
|---|---|---|---|---|---|---|
| ctrl (minkunet) | 0.663 | 0.626 | ~0.50 | ~0.43 | 0.48M | 9.1 GB |
| opt1 (deep attention) | 0.666 | 0.646 | ~0.50 | ~0.43 | 1.27M | 8.9 GB |
| opt2 (local stem) | 0.687 | 0.672 | ~0.50 | ~0.43 | 0.48M | 5.4 GB |
| **combo (stem + attention)** | **0.722** | **0.675** | ~0.50 | ~0.43 | 1.27M | 9.0 GB |
| — *reference* mae (true-MAE) | 0.67 | 0.65 | 0.46 | 0.43 | 0.48M | |
| — *reference* dino | 0.72 | 0.66 | 0.47 | 0.50 | 0.48M | |
| — *reference* PoLAr-MAE | **0.93** | **0.94** | 0.53 | 0.43 | ~22M | |

(Best over the run; values barely move epoch-to-epoch, e.g. combo
sft_feat = 0.722/0.709/0.718 — saturated.)

## What we learned

1. **The local-geometry stem helps** (+0.024 sft_feat, +0.046 svm over ctrl) at
   *no extra params* — enriching each voxel's local descriptor is a real, cheap win.
2. **Deep attention alone is ~neutral** (opt1 ≈ ctrl). Stacking 6 bottleneck-attention
   blocks at 64-dim did not help on its own — at this width the extra global mixing
   adds little (and is harder to optimise).
3. **The combo is best** (sft_feat **0.722**, +0.059 over ctrl) — the stem's richer
   local features give the attention something useful to mix. It matches DINO (0.72)
   and edges past the mae baseline.
4. **But all still plateau ~0.66–0.72, ~0.21 below PoLAr-MAE (0.93).** Architecture
   *shape* tweaks within the small sparse-CNN give a modest, real lift but do **not**
   close the gap. Raw-charge baselines match across all (svm_raw 0.43) → comparison is fair.

## Interpretation — the binding constraint is capacity (and maybe the objective)

These backbones are **0.48–1.27M params vs PoLAr-MAE's ~22M** — a 17–46× capacity
gap — and they run at only **5–9 GB** (batch 32), i.e. tons of headroom. The probes
saturate at epoch 1 regardless of architecture shape, which is the signature of a
representation that has hit its capacity/objective ceiling, not its optimisation or
data ceiling. So the shape of the mixing matters less than:
- **width/depth (capacity)**: 32/64-dim features can't encode what 384-dim tokens do;
- **the SSL objective**: standard charge-reconstruction MAE is a weak signal (DINO's
  contrastive objective already beat it by ~0.05 on the same backbone).

## Recommendation — next experiment

Take the **combo** (stem + attention, the best shape) and **scale it**, since memory
is abundant:
- widen the backbone (32/64 → 96/128 ch; the heads already expect 64 but we can lift
  the internal width and project to 64, or raise the head width),
- more bottleneck blocks (6 → 8–12) and larger encoding dim,
- optionally switch to the **true-MAE** objective (coordinate-removal masking) and/or
  add a DINO/feature-prediction objective,
- batch back to 64 (mem allows).

If a scaled combo still plateaus well below ~0.85, the evidence will strongly say the
sparse-CNN representation itself (not its size or training) is the ceiling, and the
tokenizer+transformer (polarmae/larmamba) is the necessary route — which is consistent
with everything measured so far.

## Round 2 — capacity × objective sweep

We scaled the combo (stem + deep attention) with a width-configurable core
(`WideComboCore` / `WideComboTrueMAECore`, base_ch-parameterised) in a 2×2:
width {128ch=4.9M, 256ch=19.5M} × objective {standard-MAE, true-MAE}. 256ch
≈ PoLAr-MAE's ~22M capacity, so this is a parity test.

Best feature-probe macro-F1 (all peak at **epoch 1**, then decline — see below):

| Variant | params | sft_feat | voxel_svm_feat | peak GPU |
|---|---|---|---|---|
| w128 standard-MAE | 4.9M | 0.706 | 0.683 | 11.8 GB |
| **w256 standard-MAE** | **19.5M** | **0.769** | **0.720** | 11.9 GB |
| w128 true-MAE | 4.9M | 0.674 | 0.656 | 11.0 GB |
| w256 true-MAE | 19.5M | 0.736 | 0.668 | 13.2 GB |

Full picture across both rounds:

| Model | sft_feat | svm_feat | params |
|---|---|---|---|
| mae (orig) | 0.67 | 0.65 | 0.48M |
| dino | 0.72 | 0.66 | 0.48M |
| sparseformer combo (R1) | 0.722 | 0.675 | 1.27M |
| **sparseformer w256 (R2, best)** | **0.769** | **0.720** | 19.5M |
| PoLAr-MAE | **0.93** | **0.94** | ~22M |
| larmamba (Mamba) | 0.94 | 0.93 | ~26M |

### Findings

1. **Capacity helps, with steep diminishing returns.** 1.27M→19.5M (15×) lifts
   sft_feat 0.722→0.769 (+0.047). The cumulative gain from the whole sparse-CNN
   program (orig 0.67 → 0.769) is **+0.10**, mostly from the stem + width.
2. **true-MAE did NOT help** — it is *below* standard-MAE at both widths
   (0.674 vs 0.706 at 128; 0.736 vs 0.769 at 256). The coordinate-removal +
   grid-patch objective gives no probe benefit here.
3. **Same "epoch-1 peak, then decline" pattern** as the original mae: every
   variant's best probe is at epoch 1 and drops thereafter (e.g. w256
   0.769→0.715 by epoch 2). The MAE charge-reconstruction objective saturates
   the probe-useful representation in one pass; more steps optimise
   reconstruction at the probe's expense. So longer training won't help.

### Verdict

**Even at PoLAr-MAE-matched capacity (19.5M params) and with true-MAE, the
per-voxel sparse-CNN plateaus at ~0.77 feature macro-F1 — ~0.16 below
PoLAr-MAE (0.93) and larmamba (0.94).** Capacity + the local stem closed only
about a third of the gap (0.67→0.77 of the 0.67→0.93 span) with steeply
diminishing returns, and the objective change backfired. This is strong
evidence the ceiling is the **per-voxel sparse-CNN representation itself**, not
its size or training: without tokenisation (aggregating local neighbourhoods
into geometry-aware tokens) + a global mixer over those tokens, the model
cannot reach the 0.9+ regime. **The tokenizer+transformer route
(PoLAr-MAE / larmamba) remains the necessary path to polarmae-level accuracy.**

The practical takeaway combined with the cost study: if you need the absolute
cheapest inference, the small sparse-CNN (~0.49M, ~49 MiB/event, GPU-only) at
~0.72 is the floor; for polarmae-level accuracy you must pay for
tokenizer+transformer. The sparse-CNN's accuracy ceiling (~0.77) does not move
with capacity.

## Reproduce

```bash
# Round 1 (backbone shape)
for v in ctrl opt1 opt2 combo; do
  bash gridutils/submit_mae.sh sparseformer/configs/config_sf_${v}.json
done
# Round 2 (capacity x objective)
for t in w128 w256 w128true w256true; do
  bash gridutils/submit_mae.sh sparseformer/configs/config_sf_${t}.json
done
```
Artifacts: `/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/sf_{ctrl,opt1,opt2,combo}_260614/`.
Backbones: `sparseformer/backbones.py`; configurable via `backbone_name` /
`backbone_kwargs` in the mae config (standard-MAE path, `true_mae=false`).
