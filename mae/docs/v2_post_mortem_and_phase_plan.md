# v2 post-mortem + Phase A/B/C plan toward parity with polarmae

This is the synthesis report after the `mae_full_v2_260523` run was killed at
ep6 (SVM val_acc still 0.63, no upward trend). Companion to
`polarmae_vs_mae_comparison.md`.

## TL;DR

- 307.0 killed cleanly. **Periodic rsync rescued ep1–ep6** (6 checkpoints +
  complete `sft_history.json`).
- v2's six-epoch data confirms what `polarmae_vs_mae_comparison.md`
  predicted: **the v2 changes (grid_patch masking, batch_size 32, held-out
  probe) did not move the metric**. SFT val_acc stays at 0.71–0.72 across 6
  epochs; SVM val_acc oscillates 0.63–0.68. Polarmae's 0.94 ceiling is out of
  reach for the sparse-CNN backbone regardless of training tweaks.
- Two concrete, implementable wins exist **before any backbone redesign**:
  1. **Dataloader fix**: `num_workers=2 + persistent_workers + h5py LRU
     cache + CPU-side log1p` → estimated 2–3× wall-time per SSL epoch.
  2. **SFT extract-once redesign**: pool backbone features once per probe,
     train an in-memory MLP on the pool → estimated 5× faster SFT block per
     SSL epoch, and brings us to full metric parity with polarmae.
- These are **dataloader + eval-loop changes only**, not backbone changes.
  They make the comparison cheaper and more honest, but don't change the
  architectural ceiling.

## v2 trajectory (the data the kill rescued)

| Epoch | SSL train L1 | SSL val L1 | SFT train acc | SFT val acc | SVM val acc |
|---|---|---|---|---|---|
| 1 | 1.247 | 0.746 | 0.723 | 0.724 | 0.627 |
| 2 | 0.767 | 0.738 | 0.719 | 0.719 | 0.663 |
| 3 |  –   |  –   | 0.716 | 0.716 | 0.661 |
| 4 |  –   |  –   | 0.713 | 0.712 | **0.684** |
| 5 |  –   |  –   | 0.714 | 0.716 | 0.638 |
| 6 |  –   |  –   | 0.710 | 0.712 | 0.630 |

SFT val_acc is flat-to-declining; SVM oscillates around 0.65. No statistical
improvement from grid_patch + bs=32 + held-out probe.

## 1. Why mae is slower per step than polarmae

### Dataloader side (mae's main bottleneck)

| Knob | mae (slow) | polarmae (fast) | What it costs |
|---|---|---|---|
| `num_workers` | 0 | 2 | mae blocks main thread on every `h5py.File()` open |
| `persistent_workers` | False | True | mae rebuilds the worker (and h5py chunk cache) every epoch |
| `pin_memory` | False | False | minor on both |
| `prefetch_factor` | n/a | (default 2 with workers) | mae has no prefetch queue at all |
| h5py file handle | reopened every `__getitem__` | LRU cache (cap 128) | repeated kernel `open()` syscalls + chunk-cache rebuild |
| `log1p` location | GPU, per-batch in train loop | CPU, per-sample in `__getitem__` | mae blocks `forward()` waiting for the log1p kernel; polarmae overlaps it with the previous GPU batch |
| `max_points` cap | none (variable, 2k–25k) | 8000 | mae's sparse-conv kernel-map regen scales with N |
| batch_size | 32 | 16 | bigger mae batches help GPU utilisation but don't fix the upstream stall |

**Diagnosis** for the "GPU = 0 % for 49/50 samples" pattern we kept seeing:
with `num_workers=0`, the main process serialises **disk read → h5py open →
numpy → torch → CPU coord rebase → CPU collate → GPU upload → GPU log1p →
forward**. The GPU spends most of its time *waiting* for the upstream
pipeline. polarmae avoids this by running 2 worker subprocesses with
persistent state.

### SFT side (mae's bigger waste)

The frozen-backbone forward is the real cost driver in mae's SFT block:

- Per SSL epoch, mae's SFT block does `n_sft_epochs_per_ssl_epoch=5 × ~156
  batches × 2 heads = ~1560 frozen-backbone forwards + 1560 sparse-head
  forward+backward pairs`. Each frozen forward is ~10–50 ms of sparse-conv
  kernel launches even though there are no grads — the kernel-map regen
  still happens. Roughly **30–80 s of pure waste per SSL epoch**, just
  re-running the backbone over the same images.
- polarmae's `APA2DProbeCallback._collect_pool` does the frozen forward
  **ONCE per probe call** (~31 batches), pools features into a numpy array,
  then trains the MLP head and fits the SVM purely on that in-memory tensor.
  Total backbone forwards per probe ≈ 31, vs. mae's ~1560.

That's a **~50× backbone-forward differential**.

## 2. Memory headroom — bigger batch?

condor_q reports `MemoryUsage=3 GB` (host) and `GpusMemory=8 GB` (GPU) for
the v2 run, vs. polarmae's `42 GB GpusMemory` at peak. **Yes there's a lot
of headroom** — could comfortably go to bs=64 or bs=128 *if* the dataloader
stops being the bottleneck. With current `num_workers=0`, bigger batches
just mean each batch has more work that the GPU finishes faster, while
still waiting equally long for the next batch. So **fix the dataloader
first**; the batch size knob amplifies the savings.

## 3. mae's SFT loop is wasteful by design

mae's `_train_sft_epoch` is "online" (every batch: forward backbone, forward
head, loss, backward). polarmae's probe is "offline" (extract features
once, train head separately on the pool). The redesign analysis estimates
**~5× speedup** for mae's SFT block by switching to the offline pattern,
plus a side benefit: the SFT head can become a tiny **dense** MLP instead of
a 3-layer **sparse-conv** head, which removes the kernel-map regen for the
head as well.

Borrowable changes (concrete file edits):

| Item | What to change | File:line |
|---|---|---|
| Replace `SparsePixelHead` with a dense MLP | New `class DensePixelHead(nn.Module)` operating on `[N, 64]` tensors | `models/mae_model.py` ~L155 |
| Pool features once per probe | New `_collect_sft_pool()` mirroring polarmae's `_collect_pool` | `mae/scripts/train_mae.py` (replacing `_train_sft_epoch` + `_val_sft_epoch`) |
| Train MLP on pool tensor | `_fit_mlp_head(pool, epochs=5, batch_size=256)` | `mae/scripts/train_mae.py` new function |
| Add SVM-on-raw probe | Extend `svm_probe()` to also fit a SVM on `(x, y, log_charge)` raw features | `mae/diagnostics/svm_probe.py` |

## 4. Metric parity with polarmae

Both pipelines report **{SVM, SFT} × {feat, raw} = 4 probes**. mae's
current state:

| polarmae key | mae equivalent | status |
|---|---|---|
| `voxel_svm_feat` (LinearSVC on backbone features) | `svm_probe` in sft_history.json | ✅ present, with val_macro_f1 + per-class F1 |
| `voxel_svm_raw` (LinearSVC on raw `(x, y, z, log_q)`) | — | ❌ **missing** |
| `sft_feat` (small MLP on backbone features) | `pixel_pid_head` trained per-SFT-subepoch | ✅ present, but uses sparse-conv head + on-line training; reports `acc`, not `macro_f1` |
| `sft_raw` (small MLP on raw) | `ref_pixel_pid_head` trained per-SFT-subepoch | ✅ present, same caveats |

Also: polarmae reports `train_macro_f1` *and* `val_macro_f1` plus per-class
eff/purity; mae reports `train_acc` + `val_acc` + per-class confusion. The
confusion matrix is enough to derive macro_f1, so this is a 5-line code
change in `svm_probe.py` + `MAEDebugger`.

**Verdict**: 3/4 probes wired, missing 1. Once `voxel_svm_raw` is added and
macro_f1 is reported alongside acc, the JSON schema matches polarmae 1-to-1.

## 5. Data fractions used by each pipeline

Important: the two pipelines use **different data slices** of the same
on-disk datasets.

| Slice | polarmae | mae v2 | factor |
|---|---|---|---|
| SSL training events | 8067 files (subdirs `13825/1/00[1-8]`) | ~10k files (10 % of 100,956 across all campaigns, random) | ~similar |
| SSL val events      | 998 files (subdir `13825/1/009`) | ~2k files (20 % val_frac of the SSL subset) | similar |
| Probe train events  | **8037 files** (subdirs `13874/1/00[1-8]`) | ~800 files (10 % of 9987, then 80 % train) | **mae uses ~10× less** |
| Probe val events    | 1011 files (subdir `13874/1/009`) | ~200 files (20 % held-out) | **mae uses ~5× less** |

So polarmae's probe quality benefits from **~10× more labeled training
events** than mae v2 (capped per-class at 5000 pixels). For Phase B we
should bump mae's `sft_subset_frac` to 1.0 so we match polarmae's labeled
budget.

Also note: polarmae uses an **explicit subdir-level split** (013825/1/001-008
train, 013825/1/009 val). mae currently does random per-file splits. Matching
the slice exactly would make the comparison even cleaner.

## Plan: Phase A → B → C → v3 smoke

### Phase A — close the dataloader gap (cheap, high-impact)

1. Add `num_workers: 2, persistent_workers: True, pin_memory: True` to the
   four DataLoader constructors in `train_mae.py`.
2. Add an h5py LRU cache to `APASparseDataset` (copy of polarmae's `_h5`
   pattern).
3. Move `log1p` into `__getitem__` of `APASparseDataset`/
   `APASparseMetaDataset`; remove the GPU-side `log1p_voxels` calls in
   `train_mae.py`.
4. Add a `max_points: 8000` cap to `APASparseDataset.__getitem__`
   (apply to **both** SSL and SFT datasets).

**Expected gain**: 2-3× SSL-epoch throughput, no metric change.

### Phase B — port polarmae's offline-probe pattern

5. Replace `_train_sft_epoch + _val_sft_epoch` with `_collect_sft_pool +
   _fit_dense_head + _eval_dense_head`.
6. Add `DensePixelHead` (tiny dense MLP) to `models/mae_model.py`.
7. Track per-SSL-epoch SFT acc + macro_f1 only — losing the per-sub-epoch CE
   curve is acceptable (user confirmed).
8. Bump `sft_subset_frac` to 1.0 in the v3 config to use the full ~10k
   labeled events polarmae uses.

**Expected gain**: another 4–5× speedup on the SFT block; SFT loop ~10 s
instead of ~5–10 min per SSL epoch.

### Phase C — close the metric-parity gap (small)

9. Add SVM-on-raw to `mae/diagnostics/svm_probe.py`: same machinery, just
   feed `(positions, log_charge)` instead of backbone features.
10. Add `val_macro_f1` (derivable from existing confusion matrices) to all
    four probe outputs in sft_history.json.

**Expected gain**: clean apples-to-apples comparison; no performance change.

### Phase D — submit a v3 smoke + a short full run

After A+B+C, submit `gridutils/config_mae_full_v3.json` with **5 epochs**
(user-requested cut from 20) at the larger batch size + workers. If
metrics still plateau at ~0.7, that's the final word on the sparse-CNN MAE
ceiling — and the case for the polarmae architectural direction is closed.

### What I'd defer / not do

- **Recommendation #2 from the original report** (FPS-tokenizer in front of
  the sparse backbone) — that's effectively building polarmae inside mae.
  If we conclude the sparse-CNN ceiling is real, this isn't a productive
  direction here.
