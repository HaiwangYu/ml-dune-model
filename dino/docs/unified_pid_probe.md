# Unified, leakage-free per-pixel PID probe

`dino/diagnostics/ab_pid_probe.py`

A per-pixel PID probe (track / shower / other, macro-F1) that is **leakage-free**
(whole-event train/val split) and **unified** (one probe scores a native DINO
checkpoint and a foreign model's precomputed features through identical code, so
the numbers are directly comparable).

Ported from WC_FM_DINO's `dino/diagnostics/ab_pid_probe.py`; see that repo's
`docs/28_polarmae_intrinsic_eval_leakage.md` for the full motivation. This page
covers how to use the version in *this* repo.

---

## Why this probe exists

There are two independent properties, both of which matter:

### 1. Leakage-free event-level split

The probe splits **whole events** 80/20 *first*, and only then pools per-pixel
examples within each side (`_fit_and_report`):

```python
uniq_events = np.unique(keys, axis=0)          # keys = (file_idx, event_idx) per pixel
perm = rng.permutation(len(uniq_events))
n_train = int(0.8 * len(uniq_events))
train_set = set(map(tuple, uniq_events[perm[:n_train]]))
is_train = np.array([tuple(k) in train_set for k in keys])
```

No event's pixels can land on both sides. A **pixel-level** split (permuting the
pooled voxel array directly) lets a probe partially memorize per-event structure
— geometry, track topology, local charge pattern — which inflates macro-F1 by
~15 pts on spatially-smooth backbone features. That inflation is a property of
the *protocol*, not of the representation, so it is not a fair thing to report or
chase (WC_FM_DINO docs/28 §2–§3).

> Note: this repo's existing `dino/diagnostics/run_probes.py` already splits at
> the image level, so it is also leakage-free for a *native* checkpoint. What
> `ab_pid_probe.py` adds on top is the unified native-vs-external comparison
> below plus explicit coordinate-based alignment.

### 2. One probe for native *and* foreign features

A live checkpoint's backbone features (`--ckpts`) and a foreign model's exported
per-voxel features (`--external`) both flow through the **same**
`_match_align_label` (coordinate matching) → `pdg_to_pixel_class` (labeling) →
raw floor → event-level split → `StandardScaler` → `LinearSVC` / MLP heads. The
only thing that differs per run is where the `(feature, coord)` pairs come from.
That is what makes cross-architecture numbers comparable rather than apples to
oranges.

### The four numbers

Each run reports four held-out macro-F1 values:

| key              | features                          | head       |
|------------------|-----------------------------------|------------|
| `voxel_svm_feat` | backbone features                 | LinearSVC  |
| `sft_feat`       | backbone features                 | small MLP  |
| `voxel_svm_raw`  | raw floor `(channel, tick, log_q)`| LinearSVC  |
| `sft_raw`        | raw floor `(channel, tick, log_q)`| small MLP  |

The `_raw` numbers bypass the backbone entirely and give the honest floor for
the task; the lift of `_feat` over `_raw` is the representation's real signal.

---

## Requirements

Run inside the project venv (it provides `torch`, `warpconvnet`, `sklearn`,
`scipy`):

```bash
source setup.sh        # or: source /nfs/data/1/mvicenzi/ml-venv/bin/activate
```

Truth comes from `APASparseMetaDataset(return_pixel_truth=True)`, i.e. the same
per-pixel PDG labels (`meta["pid_labels"]`) that `run_probes.py` and the MAE SFT
use, mapped to `{track=0, shower=1, other=2, -1=no-truth}` by
`models.mae_model.pdg_to_pixel_class`.

---

## Usage

### Native DINO checkpoint (student features, the default)

```bash
python -m dino.diagnostics.ab_pid_probe \
    --ckpts base=/path/to/checkpoint_ep100.pt \
    --n_events 500 \
    --out CONDOR_OUT/pid_probe_unified.json
```

Dataset config (`datadir`, `apa`, `view`, `cache_dir`) is read automatically from
the checkpoint's saved `cfg`. Use `--backbone_view teacher` to probe the teacher
backbone instead of the student.

### Several checkpoints at once

```bash
python -m dino.diagnostics.ab_pid_probe \
    --ckpts base=/path/ep100.pt,longer=/path/ep300.pt \
    --n_events 500 --out CONDOR_OUT/pid_probe_unified.json
```

### Head-to-head with a foreign feature export

```bash
python -m dino.diagnostics.ab_pid_probe \
    --ckpts base=/path/to/checkpoint_ep100.pt \
    --external '{"polarmae": "/path/to/polarmae_feats.npz"}' \
    --datadir /path/to/data --apa 0 --view W \
    --n_events 500 --out CONDOR_OUT/pid_probe_unified.json
```

All runs (native and external) land in the same JSON keyed by their label, so a
single file gives you the full comparison table.

> If you pass **only** `--external` (no `--ckpts`), there is no checkpoint to
> derive the dataset from, so you must give `--datadir`, `--apa`, and `--view`
> explicitly. When `--ckpts` is present these default from its cfg but can still
> be overridden on the CLI.

---

## The `--external` npz schema

Each foreign model exports its per-voxel features for the **same events in the
same order** as `Subset(dataset, range(n_events))`, as a single `.npz`:

| array     | shape                     | dtype   | meaning                                             |
|-----------|---------------------------|---------|-----------------------------------------------------|
| `coords`  | `[N, 2]`                  | int32   | `(channel, tick)` per exported voxel                |
| `feat`    | `[N, D]`                  | float32 | per-voxel backbone feature                          |
| `offsets` | `[n_events_avail + 1]`    | int64   | event `i` = rows `offsets[i] : offsets[i+1]` (CSR)  |

The probe joins these to the truth by `(channel, tick)` coordinate per event, so
the export does **not** need to preserve voxel order or count — voxels the model
dropped are simply unmatched. If fewer events were exported than `--n_events`,
the probe uses what is available and prints a warning.

---

## Alignment gate

A sparse-conv U-Net is not guaranteed to preserve input voxel order/count, so
output rows are matched to truth rows by coordinate when coordinates are
available (`_match_align_label`). Every run prints its match rate and **asserts
it is ≥ `--min_match_rate` (default 0.95)** before fitting — a low match rate
means the features and labels are misaligned and the F1 numbers would be
meaningless, so the run aborts rather than reporting a corrupt number. Lower the
threshold only if you understand why an export legitimately drops voxels.

---

## Key options

| flag                | default                   | meaning                                            |
|---------------------|---------------------------|----------------------------------------------------|
| `--ckpts`           | —                         | `label=/path/to/ckpt.pt[,label2=...]` live runs    |
| `--external`        | —                         | JSON `{label: npz_path}` precomputed runs          |
| `--backbone_view`   | `student`                 | `student` or `teacher` for `--ckpts` runs          |
| `--n_events`        | `500`                     | events scanned (dataset order)                     |
| `--batch`           | `16`                      | backbone forward batch size (live path)            |
| `--pool_per_class`  | `5000`                    | class-balanced pool cap per split                  |
| `--min_match_rate`  | `0.95`                    | abort below this coord match rate                  |
| `--seed`            | `0`                       | split + probe seed                                 |
| `--datadir/--apa/--view/--cache_dir` | from ckpt cfg | dataset config (required for external-only) |
| `--out`             | `pid_probe_unified.json`  | output JSON path                                   |

At least one of `--ckpts` / `--external` is required.

---

## Output

A JSON keyed by run label. Each entry:

```json
{
  "base": {
    "n_train": 15000, "n_val": 15000, "n_events": 500,
    "align_match_rate": 1.0,
    "voxel_svm_feat": 0.71, "sft_feat": 0.74,
    "voxel_svm_raw": 0.41,  "sft_raw": 0.47,
    "class_counts": {"train": {...}, "val": {...}},
    "per_class_f1": {"sft_feat": {"track": ..., "shower": ..., "other": ...}, ...}
  }
}
```

The headline comparison is `sft_feat` / `voxel_svm_feat` across labels, read
against each run's own `_raw` floor.
