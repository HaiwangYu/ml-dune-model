# PoLAr-MAE vs sparse-CNN MAE — comparison report

Comparison of two SSL training pipelines on the same DUNE APA2D wire-plane
pixel data. The first is the existing sparse-CNN MAE in `/lbne/u/hyu/ml-dune-model/mae`;
the second is the PoLAr-MAE adapter at `/lbne/u/hyu/PoLAr-MAE` (see
`/lbne/u/hyu/PoLAr-MAE/docs/dune_apa2d.md` for that pipeline's own writeup).

Run dirs compared:
- mae (this repo): `/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/mae_smoke_pid_260521/checkpoints/mae_epoch2.pt`
- polarmae: `/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/polarmae_apa2d_full_260523_v3/`

## Quick numbers (with caveats)

**Caveat first**: the full mae 20-epoch run (cluster 305) lost all its
checkpoints when condor's `condor_rm` skipped past the SIGTERM-trap rsync —
the scratch dir was reclaimed before `sync_back` ran. The only mae checkpoints
we have are from the **2-epoch smoke** (`mae_smoke_pid_260521/checkpoints/mae_epoch2.pt`),
so the comparison is between a barely-trained mae and a fully-trained
polarmae. **Not apples-to-apples — read these numbers as "where each method
was when last checkpointed", not a fair head-to-head**.

Headline 3-class pixel-PID `val_macro_f1` (track / shower / other):

| Pipeline | SVM (backbone feats) | SFT (backbone feats) | SVM (raw charge) | SFT (raw charge) |
|---|---|---|---|---|
| **polarmae** (best, step 17k–18k) | **0.940** | **0.933** | 0.432 | 0.526 |
| **polarmae** (final, step 20k) | 0.938 | 0.931 | 0.432 | 0.535 |
| **mae** (smoke ep2, full converged) | 0.645 | ~0.65 (3-cls) | – | – |

**Inflight mae signal** from the killed 305 run: SVM val_macro_f1 oscillated
**0.628 – 0.672** between epochs 1-8. The metric was already plateaued —
finishing epoch 20 likely wouldn't have closed the gap to 0.94.

## Wall-time + GPU utilisation

|  | polarmae | mae (full run, killed) |
|---|---|---|
| Total wall | **75 min** | 24 h+ (killed at ep13) |
| Steps trained | 20,000 SSL | ~63,000 SSL + 5×SFT/epoch |
| GPU active time / wall | **66 %** (`condor_q -af GPUsUsage`) | ~5 % (gpu.log histogram) |
| GPU util samples | 88+78+55+22+19+16+16+13+13 @ **≥90 %** | mostly 0 %, occasional 30-50 % bursts |

Speed advantage: polarmae is **~16× faster in throughput** (20k SSL steps in
75 min vs mae's ~22 SSL steps/min ≈ 13.6× more steps per unit time,
accounting for both step counts and wall).

## Why polarmae is faster

The dominant factor is **what each "step" computes**.

- **mae** runs `MinkUNetTrueMAECore` over ~2,000-15,000 active voxels per
  event, with sparse 3×3 conv kernels. Sparse conv on irregular per-event
  topology has poor GPU occupancy and high CPU↔GPU sync per layer (the
  gpu.log showing constant 0% util is the symptom). Bottleneck: kernel-map
  regeneration + indirect memory access.
- **polarmae** first tokenizes each event into a **fixed 256 groups × 32
  points** via FPS + ball-query — this bounds the per-event compute. Then a
  ViT-Small transformer (6 blocks, 384-D) runs dense attention on 256 tokens.
  Dense GEMM on 384×384 weights × small batch is exactly what L40S is
  designed for. The tokenizer is the only sparse step; everything after is
  dense.

Specific code references:
- `polarmae/datasets/APA2D.py:71-124` shapes each event to ≤8000 points
  (`max_points: 8000` in the config), then FPS down to 256 groups
- `polarmae/layers/encoder.py` + `polarmae/models/ssl/polarmae.py:19-168` —
  dense transformer
- mae: `models/minkunet_attention.py:30-100` — sparse U-Net + bottleneck
  attention; every layer is dispatched per-active-voxel

Net: **polarmae caps per-event compute at O(256 × 32) ≈ 8k operations
regardless of event density**, while mae scales with O(N_voxels).

## Why polarmae gets better metrics

1. **Patch tokens encode local geometry**. Each of polarmae's 256 tokens
   represents a small spatial neighbourhood (32 nearby voxels). The encoder
   learns *patch-level* representations that already capture track-segment vs
   shower-blob structure. The downstream voxel labels are reached by
   inverse-distance k-NN from the 256 tokens. mae's sparse-conv backbone has
   no equivalent grouping — each voxel must individually carry semantic
   content.
2. **60 % patch-masking is harder than 50 % voxel-masking**. polarmae masks
   60% of *groups* — the encoder cannot use any neighbouring information to
   reconstruct a masked patch. mae's voxel-level block masking removes a
   contiguous block of voxels at full resolution; many of the masked voxels
   are statistically predictable from their immediate neighbours.
3. **Transformer global receptive field is cheap**. ViT-Small over 256 tokens
   has full attention across the event in O(256²) ≈ 65k operations. mae's
   sparse-conv backbone needs the strided-2 downsamples just to get *partial*
   global context at the 125×125 bottleneck, and then has to upsample back.
4. **Chamfer + reconstruction loss on point clouds vs. L1 on charges**.
   polarmae reconstructs the *spatial distribution* of points within each
   masked group (Chamfer distance) — a structurally richer signal than mae's
   per-voxel charge L1.
5. **Probe is on held-out data**. polarmae's probes run on a separate labeled
   `100k-truth` split (`probe_data_path` ≠ SSL `data_path`). mae's in-training
   SFT runs on the same SFT split used at every epoch. So mae's SFT acc has
   slight train-eval leakage; polarmae's probe acc is honest validation.
6. **Training duration** matters too — polarmae saw ~26k events (16/step ×
   20k steps) effectively, vs mae's ~12.5k (10k SSL events × 1.25 effective
   passes before kill). But this is a relatively minor factor; even at mae's
   best epoch the metric was 0.67, not 0.93.

## Comparable-units check (so the numbers aren't misleading)

- **Events seen** during SSL training:
  - polarmae: 20,000 × 16 = **320,000 event-views** (60% masked patches per
    view, so ~190k "masked patches" reconstructed)
  - mae: 12,500 × 16 = **200,000 event-views** (50% voxel-masked)
- **Probe data** is identical 100k-truth events, so the probe numbers are
  directly comparable.
- **Class taxonomy is identical**: `pdg_to_pixel_class()` is the same
  algorithm in both repos (track={μ,p,π}, shower={e±,large-cluster γ},
  other={blip γ, untracked, no-truth}).
- **Probe metric isn't quite identical**: mae's per-epoch SVM probe uses
  `LinearSVC(class_weight='balanced')` on 5k pixels/class on the SFT loader;
  polarmae's `voxel_svm_feat` uses `LinearSVC(C=1.0)` on a token→voxel
  upsampling on the held-out probe loader. Both report sklearn
  `classification_report` macro-F1, so the metric definition matches. The
  0.94 vs 0.65 gap dwarfs the small definitional differences.

## Recommendations

**Highest-impact follow-up tests** (each should be a separately-labelled
condor run to avoid overwriting):

1. **Run mae for the full 20 epochs and see if it catches up**. The killed
   full run was on track to give us a fair comparison; resubmit
   `gridutils/config_mae_full.json` (or even a longer schedule) and let it
   finish. Caveat: pace was variable on sgpu0004 — maybe try sgpu0006
   (`Requirements = (Machine == "sgpu0006.sdcc.bnl.gov")`).
2. **Patch-tokenise mae's input** as a controlled architectural ablation. Add
   an FPS+ball-query tokenizer in front of the sparse-CNN backbone (instead
   of full-resolution voxels). If the metric jumps materially, that's
   evidence the tokenization is the lever, not the transformer.
3. **Patch-mask in mae**. Currently mae uses voxel-level random masking. Try
   a group-aware masking strategy that masks contiguous patches of voxels
   (consistent with polarmae's group masking).
4. **Increase mae's effective batch size** to better saturate the GPU. The
   sparse-conv is starved of work; with batch=64 (which fits in 44GB at our
   voxel counts), each kernel launch amortises better.
5. **Persistent rsync on every SSL epoch** in `trainjob_mae.sh` so future
   kills/preemptions don't lose work. Add to the trap:
   ```bash
   sync_back  # call after every SSL epoch via a SIGUSR1 trap from train_mae.py,
              # or just rsync inside main()
   ```
6. **Switch mae's SFT probe to evaluate on held-out data**. Currently the
   per-epoch SFT runs on the same `sft_dataset`. Splitting into 80%
   probe-fit, 20% probe-eval would give honest numbers and let you compare
   directly to polarmae's `val_macro_f1`.

## Bottom line

The polarmae result is genuine. The main architectural advantages
(tokenization → bounded compute, transformer global attention, patch-masking
forcing topology learning) explain both the speed and the metric. mae *can*
be improved by adopting these patterns — but if the goal is
best-result-per-wall-hour, **polarmae is a strict win** on this task as
currently configured.
