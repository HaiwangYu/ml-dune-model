# mae session summary — May 21-24, 2026

Landing page / context index for follow-up work on the sparse-CNN MAE
pipeline.  All deeper detail in the three companion docs in this folder:

- `polarmae_vs_mae_comparison.md` — first head-to-head against PoLAr-MAE.
- `v2_post_mortem_and_phase_plan.md` — diagnosis after the v2 run plus
  the A/B/C action plan.
- `v3_final_comparison.md` — final post-Phase-ABC numbers and
  interpretation.

## What got built

A working SSL → probe-eval pipeline for 2-D DUNE APA wire-plane data,
plus a clean apples-to-apples comparison against PoLAr-MAE.

```
ml-dune-model/
├── mae/
│   ├── config.py            # MAEConfig dataclass (run_name, data paths,
│   │                        # sft_mode, sft_pool_*, mask_mode, …)
│   ├── debug.py             # MAEDebugger (loss curves + sft_history.json)
│   ├── docs/                # all reports
│   ├── scripts/
│   │   └── train_mae.py     # main loop + from_config entry
│   └── diagnostics/         # extract_features + 4 k-NN scripts + log parser
├── gridutils/
│   ├── submit_mae.sh        # HTCondor submit wrapper
│   ├── trainjob_mae.sh      # worker entry, periodic-rsync trap, nvidia-smi log
│   ├── config_mae_full_v3.json   # canonical config (v3 = current best)
│   ├── config_mae_smoke_v3.json  # 2-epoch smoke
│   └── extract_features_mae.sh   # standalone feature-extract job
├── loader/
│   ├── apa_sparse_dataset.py        # SSL dataset, h5py LRU cache, CPU log1p
│   ├── apa_sparse_meta_dataset.py   # adds event labels + pid_labels
│   ├── sft_pixel_pid_dataset.py     # wrapper: pid_labels → 3-class
│   └── collate.py
└── models/
    ├── mae_model.py         # SparseMAEModel, SparseTrueMAEModel,
    │                        # SparsePixelHead, DensePixelHead
    └── …
```

The v3 train loop runs Phase-A dataloader (workers + h5py LRU + CPU
log1p + max_points) + Phase-B offline-pool SFT (extract-once → dense
MLP head + LinearSVC) + Phase-C probe parity (sft × {feat, raw} + svm ×
{feat, raw}, all with macro_f1).  Sparse-CNN backbone (the sole training
target) is the legacy `MinkUNetSparseAttentionCore` /
`MinkUNetTrueMAECore`.

## What was learned

| Probe | mae v3 (5 ep, ep1=best) | dino (teacher, ep10–100 best) | polarmae (20k steps, best) |
|---|---|---|---|
| sft_feat val_macro_f1 | **0.67** | **0.72** | **0.93** |
| voxel_svm_feat val_macro_f1 | **0.65** | **0.66** | **0.94** |
| sft_raw val_macro_f1 | 0.46 | 0.47 | 0.53 |
| voxel_svm_raw val_macro_f1 | 0.43 | 0.50 | 0.43 |

Raw-charge probes agree across pipelines (within ~0.06) → class taxonomy
and probe definitions are aligned.  Feature probes for both sparse-CNN
models (mae 0.67/0.65, dino 0.72/0.66) plateau ~0.22-0.27 below
polarmae's 0.93/0.94 — the gap is architectural (sparse CNN vs.
FPS-tokens + transformer), not training/training-loop.  Both mae and
dino saturate early (mae at ep1, dino by ep10) and barely move with
further SSL training.

Wall-time: v3 = ~2 h 10 m for 5 epochs (vs v2 = ~2 h **per epoch**); ~5×
end-to-end from Phase A+B+C.

## Recoverable artifacts (GPFS)

| Run | Path | What's there |
|---|---|---|
| v3 full (canonical, 5 ep) | `/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/mae_full_v3_260524/` | ep1–5 checkpoints, `sft_history.json` with all 4 probes per epoch, `histories.json`, viz PNGs, gpu.log, 15 diagnostic PNGs |
| v3 smoke | `…/mae_smoke_v3_260524/` | 2-epoch dry run, schema check |
| v2 full (killed at ep6) | `…/mae_full_v2_260523/` | ep1–6 checkpoints + sft_history (now retained thanks to periodic rsync) |
| polarmae v3 (reference) | `…/polarmae_apa2d_full_260523_v3/` | 20k-step polarmae run, `probes/probes_step*.json` |
| mae smoke (pixel-PID) | `…/mae_smoke_pid_260521/` | 2-epoch smoke + 9 pixel-level diag plots |

The persistent dataset/index caches live at
`/gpfs01/lbne/users/fm/hyu/cache/data/`.

## Pipeline knobs worth knowing

Config fields that matter the most:

- **`sft_mode`** = `"offline_pool"` (default, polarmae-style) or
  `"online"` (legacy per-batch SFT loop).  Offline is what should
  normally be used.
- **`sft_pool_max_pixels`** = 5000 → per-class cap when collecting
  features into the in-memory pool.
- **`mask_mode`** = `"grid_patch"` (non-overlapping `win_ch × win_tick`
  cells, drop `masking_frac` fraction) or `"block"` (legacy
  seed+window).
- **`num_workers`** ≥ 2 + `pin_memory=True` are wired automatically when
  you set `num_workers > 0`.  Workers are fork-safe (h5py handles open
  lazily per worker).
- **`apply_log_transform`** on the dataset = `True` by default — log1p
  is now done on CPU at `__getitem__` time.  The training code no longer
  calls `log1p_voxels()`.

## Open follow-ups (in order of expected payoff)

1. **Architectural shift toward polarmae's design**: FPS tokeniser +
   transformer + Chamfer reconstruction loss.  This is the only path to
   close the 0.30 macro_f1 gap; training tweaks alone won't.  Could be
   done as either:
   - a fork of polarmae adapted to mae's outputs/configs (cheap),
   - a tokeniser stub bolted in front of `MinkUNetSparseAttentionCore`
     (expensive, exploratory).
2. **Investigate the "ep1 = best, declines after" pattern** with a
   longer run that uses a much smaller learning rate after the first
   epoch (e.g. lr × 0.1 from ep2 onward).  If decline is from the
   backbone's reconstruction-task overfit, this would catch it.
3. **Add a "feature-norm" stat to MAEDebugger** to monitor representation
   collapse without needing a full probe.  Tells you epoch-by-epoch
   whether the backbone is still learning useful features.
4. **Sparse-conv kernel-map regen** is the remaining wall-time stall in
   Phase A (GPU still idle 84% of samples).  Eliminating that requires
   sharing the kernel map across batches with the same coordinate set
   — needs a code change in warpconvnet or a pre-sorted batch sampler.
5. **MultiGPU**: pipeline currently single-L40S.  Lightning-style DDP
   wrap of `train_mae.py` would let us scale, but only matters if we
   pursue (1) above first.

## Quick how-to

```bash
# Submit the canonical v3 config (5 epochs, ~2 h on L40S).
cd /lbne/u/hyu/ml-dune-model
bash gridutils/submit_mae.sh gridutils/config_mae_full_v3.json

# Smoke first (2 epochs, ~10 min):
bash gridutils/submit_mae.sh gridutils/config_mae_smoke_v3.json

# Diagnose any checkpoint:
condor_submit /gpfs01/lbne/users/fm/hyu/CONDOR_OUT/<run>/diagnostics_logs/extract_features_pid.sub
# (after extract finishes, run these locally:)
python -m mae.diagnostics.plot_knn_pid_merged <path>/features_ep<N>.npz
python -m mae.diagnostics.plot_knn_pid        <path>/features_ep<N>.npz
python -m mae.diagnostics.plot_knn_vertex     <path>/features_ep<N>.npz
python -m mae.diagnostics.plot_histories      <path>/debug/histories.json
```

## Git state

Latest commits on `may15`:

```
e7f4a3a  mae: Phase A+B+C — dataloader fixes, offline-pool SFT, full probe parity
a4fe4d3  mae: apply polarmae-comparison recs (probes, masking, rsync, v2 configs)
…
```

All Claude-authored commits include the
`Co-Authored-By: Claude Opus 4.7 (1M context)` trailer.
