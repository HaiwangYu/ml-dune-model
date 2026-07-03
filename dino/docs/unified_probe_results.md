# Unified-probe re-evaluation: mae / dino / sparseformer / polarmae / larmamba

**Date:** 2026-07-02   **Runs:** `/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/unified_probe_260702/`
(+ `unified_probe_sf_260702/` for the sparseformer rows — same dataset,
`--n_events 500`, seed 0, so the event-level split is identical)
**Probe:** `dino/diagnostics/ab_pid_probe.py` (leakage-free event-level split, one
probe for all models — see `unified_pid_probe.md`)

## Headline

**The prior ~0.93 polarmae/larmamba numbers were inflated ~0.13–0.15 by
pixel-level split leakage.** Under the unified leakage-free protocol all five
model families land between 0.64 and 0.80, the ranking is unchanged, but the
tokenizer-transformer advantage over the sparse-CNNs shrinks from ~0.2 to
**~0.03–0.09 macro-F1** — and the sparseformer capacity-scaling gain
(1.27M → 19.5M params) disappears entirely.

## Results (500 events, event-level 80/20 split, 5000 px/class/side, seed 0)

| run | voxel_svm_feat | sft_feat | voxel_svm_raw | sft_raw |
|---|---|---|---|---|
| mae ep1 (64-d, 0.48M)        | 0.655 | 0.725 | 0.402 | 0.485 |
| mae ep5 (64-d, 0.48M)        | 0.643 | 0.719 | 0.402 | 0.485 |
| sf_combo ep1 (64-d, 1.27M)   | 0.698 | 0.749 | 0.402 | 0.485 |
| sf_w256 ep1 (64-d, 19.5M)    | 0.700 | 0.734 | 0.402 | 0.485 |
| sf_w256 ep2 (64-d, 19.5M)    | 0.697 | 0.738 | 0.402 | 0.485 |
| dino ep100 teacher (64-d, 0.48M) | 0.713 | 0.768 | 0.402 | 0.485 |
| dino ep100 student (64-d, 0.48M) | 0.712 | 0.768 | 0.402 | 0.485 |
| **polarmae (384-d, ~22M)**  | **0.800** | 0.783 | 0.402 | 0.485 |
| **larmamba (384-d, ~26M)**  | **0.805** | 0.782 | 0.402 | 0.485 |

The `_raw` floors are *identical* across rows because every run scores the same
events under the same split — the unified-probe property. (In the old table the
floors scattered 0.43–0.53 across pipelines.)

Per-class `voxel_svm_feat` F1 (track / shower / other):

| run | track | shower | other |
|---|---|---|---|
| mae ep1  | 0.664 | 0.506 | 0.793 |
| sf_combo ep1 | 0.681 | 0.595 | 0.819 |
| sf_w256 ep1  | 0.687 | 0.591 | 0.821 |
| dino ep100 (t) | 0.713 | 0.645 | 0.780 |
| polarmae | 0.774 | 0.772 | 0.856 |
| larmamba | 0.774 | 0.777 | 0.863 |

## Old (heterogeneous) vs new (unified) numbers

| model | old sft_feat / svm_feat | unified sft_feat / svm_feat | Δ best |
|---|---|---|---|
| mae (ep1)   | 0.674 / 0.651 | 0.725 / 0.655 | ≈ / slightly up |
| sf_combo (ep1) | 0.722 / 0.675 | 0.749 / 0.698 | up ~0.03 |
| sf_w256 (ep1)  | 0.769 / 0.720 | 0.734 / 0.700 | **down ~0.03** |
| dino (ep100 teacher) | 0.719 / 0.655 | 0.768 / 0.713 | up ~0.05 |
| polarmae    | 0.933 / 0.940 | 0.783 / 0.800 | **down ~0.14** |
| larmamba    | 0.940 / 0.929 | 0.782 / 0.805 | **down ~0.13** |

Why the asymmetry:

- **polarmae/larmamba's own probe leaked.** `APA2DProbeCallback` pools per-voxel
  features across events and then `_split` random-permutes the *pooled pixel
  array* (`polarmae/eval/probes.py:464`) — pixels from the same event land on
  both sides of the split. Their per-voxel features are KNN-interpolations of
  ≤5 shared token vectors, i.e. extremely smooth within an event, so a probe
  can partially memorize each event's token layout. That protocol artifact is
  worth ~0.14 macro-F1 here — right in the ~15-pt band WC_FM_DINO docs/28
  measured for spatially smooth features.
- **mae/dino's old probe (`run_probes.py`) was already image-level**, so their
  numbers didn't fall; they moved *up* ~0.05 for benign protocol reasons
  (different MLP head, 500-event pool, per-run scaler — consistent across all
  rows of the new table, so within-table comparisons are clean).
- **sparseformer's old numbers came from the training-time offline-pool probe,
  which also split at event level** (train_mae.py splits the SFT *dataset*
  into disjoint event subsets before pooling) — so no leakage there either.
  Its w256 moved down ~0.035 and combo up ~0.027: ordinary protocol variance,
  which is telling — the old +0.047 "capacity win" of w256 over combo was the
  same size as this variance.

## Findings

1. **Honest ranking (gaps compressed):**
   polarmae ≈ larmamba (0.78–0.80) > dino (0.71–0.77) ≈ sparseformer
   (0.70–0.75) > mae (0.64–0.72).
2. **larmamba = polarmae survives** the protocol fix (0.805 vs 0.800 svm_feat,
   0.782 vs 0.783 sft_feat — well within probe noise). Every conclusion about
   the two being interchangeable stands.
3. **The tokenizer+transformer edge is real but modest: +0.03–0.09 over dino,
   not +0.2.** Its largest component is shower identification (0.77 vs 0.65
   dino / 0.51 mae per-class F1); track and other are much closer.
4. **The "sparse-CNN caps at ~0.77 / tokenizer needed for 0.9+" conclusion is
   retracted.** Nothing measured reaches 0.9 honestly; ~0.80 is the best
   observed on this task, and dino at 0.768 sft_feat is ~0.015 behind
   polarmae's 0.783. The efficiency story flips accordingly: the ~49 MiB/event
   sparse-CNN is far more competitive than the old table suggested.
5. **The sparseformer capacity-scaling story does not survive the unified
   protocol.** The 19.5M w256 is statistically tied with the 1.27M combo
   (svm 0.700 vs 0.698; sft 0.734 vs 0.749 — combo *ahead* on the MLP probe)
   and neither beats the 0.48M dino. What the sparseformer round-2 sweep
   measured as a +0.047 width gain is within cross-protocol variance; scaling
   the sparse-CNN 15× bought essentially nothing on honest numbers. The
   "epoch-1 peak then decline" pattern also flattens (w256 ep1 ≈ ep2). The
   remaining ~0.03–0.06 architecture gap to the token models is per-class
   concentrated in shower (0.59 vs 0.77).
6. **dino teacher ≈ student** (Δ < 0.001), as before.
7. Head-type note: for the 64-d sparse-CNN features the MLP (`sft_feat`) beats
   the linear SVM by ~0.05–0.07; for the 384-d token features the SVM is the
   stronger probe. Compare models on their best feature probe (or both), not
   on one head alone.

## Mechanics (how each model got into one probe)

- `dino`: native `--ckpts` path (live backbone, teacher and student views).
- `mae`: `mae/diagnostics/export_pid_features.py` — frozen backbone over the
  probe dataset (`log1p` norm), per-voxel 64-d features → `--external` npz.
- `sparseformer`: same exporter with `--config <run config JSON>`, which
  rebuilds the backbone from the config's `backbone_name`/`backbone_kwargs`
  (checkpoints store only the state dict). Job: `gridutils/unified_probe_sf.sh`.
- `polarmae`/`larmamba`: `dino/diagnostics/export_probe_events.py` dumps the
  probe's event list (coords + raw ADC, CSR) as the cross-env bridge;
  `larmamba/export_pid_features.py` (uvenv-polar-mae) rebuilds each encoder via
  `larmamba/eval_quant.py` helpers, replicates `polarmae.datasets.APA2D` input
  construction exactly (charge > 1 threshold, 8000-pt cap, log-transform,
  center+scale), and KNN-upsamples token features (K=5 inverse-distance — the
  same scheme as polarmae's own probe) at **all** dumped voxels.
- Every run had `align_match_rate = 1.0000` and identical `pixels_kept`
  (1,552,587 of 2,198,690 voxels carry PDG truth).

Checkpoints: mae `mae_full_v3_260524` ep1/ep5; sparseformer `sf_combo_260614`
ep1 + `sf_w256_260614` ep1/ep2; dino mvicenzi `longer_contrast_slow` ep100;
polarmae `polarmae_apa2d_full_260523_v3` 20k; larmamba `larmamba_full_260613`
(g256) 20k.

## Caveats

- Single seed, 500 events, 5000 px/class pools; treat ±0.01 as noise (matches
  the spread seen across quant re-runs).
- Absolute values are protocol-dependent (pool size, head capacity); only
  within-table comparisons are meaningful — which is the point of the probe.
- polarmae/larmamba features here are bf16-extracted (as in their training-time
  probes); mae/dino fp32. This did not matter for the quant study (Δ ≤ 0.01).

## Reproduce

```bash
bash gridutils/submit_unified_probe.sh unified_probe_<date> 500
# one L40S job: event dump -> mae exports (uvenv) -> polarmae/larmamba exports
# (uvenv-polar-mae) -> ab_pid_probe teacher+student runs (uvenv)
bash gridutils/submit_unified_probe_sf.sh unified_probe_sf_<date> 500
# sparseformer rows (uvenv only); same n_events/seed -> identical split
```
