#!/usr/bin/env python3
"""Unified, leakage-free per-pixel PID truth probe (track/shower/other, macro-F1).

Ported from WC_FM_DINO's `dino/diagnostics/ab_pid_probe.py` (the "unified probe"
described in WC_FM_DINO `docs/28_polarmae_intrinsic_eval_leakage.md`) and rewired
onto *this* repo's dataset/backbone conventions (APASparseMetaDataset, the
student/teacher DINO backbone loader, `models.mae_model.pdg_to_pixel_class`).

Two properties make it the "unified" probe:

1. Leakage-free EVENT-LEVEL split. Whole events are split 80/20 FIRST, and only
   then are per-pixel examples pooled within each side (`_fit_and_report`). No
   event's pixels can appear on both train and val. This is the honest protocol;
   a pixel-level split (permuting the pooled voxel array directly) lets a probe
   partially memorize per-event structure and inflates macro-F1 by ~15 pts on
   spatially-smooth backbone features (WC_FM_DINO docs/28 SS2-SS3).

2. ONE probe for native AND foreign features. A live checkpoint's backbone
   features (`--ckpts`) and a foreign model's precomputed per-voxel feature export
   (`--external`, e.g. PoLAr-MAE or ml-dune B2) both go through the exact same
   coordinate-matching (`_match_align_label`), PDG->3-class labeling, raw floor,
   event-level split, and SVM/MLP heads. The only thing that differs per run is
   where the `(feature, coord)` pairs come from -- so cross-architecture numbers
   are directly comparable.

Truth source: the per-pixel PDG labels exposed by
`APASparseMetaDataset(return_pixel_truth=True)` as `meta["pid_labels"]`, mapped to
{track=0, shower=1, other=2, -1=no-truth} by `models.mae_model.pdg_to_pixel_class`
(identical taxonomy to `run_probes.py` / the MAE offline-pool SFT).

Alignment gate: backbone-output rows are matched to truth rows by (channel,tick)
COORDINATE when the output carries coordinates, since a sparse-conv U-Net is not
guaranteed to preserve input voxel order/count; a per-run `--min_match_rate`
assertion catches silent misalignment before it can corrupt the F1 numbers.

Usage
-----
  # Native DINO checkpoint(s), student features (default):
  python -m dino.diagnostics.ab_pid_probe \
     --ckpts base=/path/to/checkpoint_ep100.pt \
     --n_events 500 --out CONDOR_OUT/pid_probe_unified.json

  # Head-to-head with a foreign feature export, same protocol:
  python -m dino.diagnostics.ab_pid_probe \
     --ckpts base=/path/to/checkpoint_ep100.pt \
     --external '{"polarmae": "/path/to/polarmae_feats.npz"}' \
     --datadir /path/to/data --apa 0 --view W \
     --n_events 500 --out CONDOR_OUT/pid_probe_unified.json

Dataset config (datadir/apa/view/cache_dir) defaults to the first `--ckpts`
checkpoint's saved cfg; supply it on the CLI for `--external`-only runs.

See `dino/docs/unified_pid_probe.md` for the full description and the npz schema.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader, Subset
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loader.apa_sparse_meta_dataset import APASparseMetaDataset
from loader.collate import voxels_meta_collate_fn
from models.mae_model import (
    pdg_to_pixel_class,
    PIXEL_PID_CLASS_NAMES as CLASS_NAMES,
    PIXEL_PID_N_CLASSES as N_CLASSES,
)
from dino.config import DINOConfig
from dino.transforms import FeatureLogTransform
from dino.diagnostics.run_probes import _load_backbone

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_q(charge, xmax=1.0e5, eps=1.0):
    """Log-charge transform matching APA2D.log_transform: [eps, xmax] -> [-1, 1].

    This is the "raw floor" feature (channel, tick, log_q) -- a probe on these
    bypasses the backbone entirely and reports the honest floor for the task. Its
    exact formula only needs to be consistent across runs of THIS script, which
    it is, so `_raw` numbers are comparable native-vs-external within one JSON."""
    y0 = np.log10(eps)
    y1 = np.log10(eps + xmax)
    return 2 * (np.log10(np.clip(charge, 0, None) + eps) - y0) / (y1 - y0) - 1


def _match_align_label(ci, qi, pid_i, fo, co, has_coord):
    """Match model-output rows to truth rows for ONE event, map PDG->3-class
    label, and package the kept (feat, raw, label) rows.

    Shared by the live-backbone path (`extract_pixel_examples`) and the
    precomputed-feature path (`extract_pixel_examples_precomputed`) so a foreign
    model's numbers are produced by the exact same matching/labeling/raw-floor
    code as a native checkpoint's -- only the source of `(fo, co)` differs.

    ci, qi, pid_i: this event's TRUTH-side (channel,tick) [N_in,2], raw ADC charge
      [N_in], raw PDG [N_in].
    fo, co: this event's MODEL-side feature [N_out,D] and (channel,tick)-or-None
      [N_out,2].
    has_coord: whether `co` is available (coordinate join) or None (positional
      fallback -- only trustworthy if the model preserves input voxel order/count).

    Returns ((feat, raw, label) or None if nothing survives `keep`, n_in, n_matched).
    """
    n_in = ci.shape[0]
    if has_coord and co.shape[0] == 0:
        # Empty model-output for this event (e.g. a foreign export that dropped a
        # near-empty event). Nothing can match; treat as 0/n_in rather than let
        # `sorted_keys[-1]` on an empty array raise IndexError.
        return None, n_in, 0
    if has_coord:
        key_in = ci[:, 0].astype(np.int64) * 1_000_000 + ci[:, 1].astype(np.int64)
        key_out = co[:, 0].astype(np.int64) * 1_000_000 + co[:, 1].astype(np.int64)
        order = np.argsort(key_out)
        sorted_keys = key_out[order]
        pos = np.searchsorted(sorted_keys, key_in)
        pos_c = np.clip(pos, 0, len(sorted_keys) - 1)
        valid = (pos < len(sorted_keys)) & (sorted_keys[pos_c] == key_in)
        matched_rows = order[pos_c[valid]]
        fo_m, pid_m, qi_m, ci_m = fo[matched_rows], pid_i[valid], qi[valid], ci[valid]
    else:
        n = min(ci.shape[0], fo.shape[0])
        fo_m, pid_m, qi_m, ci_m = fo[:n], pid_i[:n], qi[:n], ci[:n]
        valid = np.ones(n, dtype=bool)
    n_matched = int(valid.sum())

    cls = pdg_to_pixel_class(pid_m, ci_m)   # per-event: gamma CC must not cross events
    keep = cls >= 0
    if keep.sum() == 0:
        return None, n_in, n_matched
    feat = fo_m[keep]
    label = cls[keep]
    raw = np.stack([ci_m[keep, 0], ci_m[keep, 1], _log_q(qi_m[keep])], axis=1).astype(np.float32)
    return (feat, raw, label), n_in, n_matched


@torch.no_grad()
def extract_pixel_examples(backbone, norm, ds, n_events, batch):
    """Run `backbone` over the first `n_events` of `ds` and return per-pixel
    (feature, raw, label, event_idx) examples, no-truth pixels dropped.

    Output rows are matched to truth by (channel,tick) coordinate when the
    backbone output carries coordinates, else by row order. Returns an
    `align_info` dict so the caller can gate on the match rate."""
    sub = Subset(ds, list(range(n_events)))
    dl = DataLoader(sub, batch_size=batch, collate_fn=voxels_meta_collate_fn, num_workers=2)

    feats, raws, labels, ev_idx = [], [], [], []
    n_in_total, n_matched_total = 0, 0
    has_coord = None
    base = 0
    for xs, meta in dl:
        xs = xs.to(dev)
        input_charge = xs.feature_tensor.detach().clone().float()[:, 0].cpu().numpy()  # raw ADC, pre-norm
        if norm is not None:
            xs = norm(xs)
        in_coords = xs.coordinate_tensor.detach().cpu().numpy()
        in_offs = xs.offsets.cpu().tolist()

        out = backbone(xs)
        out_feats = out.feature_tensor.detach().float().cpu().numpy()
        if has_coord is None:
            has_coord = hasattr(out, "coordinate_tensor")
        out_coords = out.coordinate_tensor.detach().cpu().numpy() if has_coord else None
        out_offs = out.offsets.cpu().tolist() if hasattr(out, "offsets") else in_offs

        pid_list = meta.get("pid_labels", None)
        if pid_list is None:
            raise RuntimeError("dataset must be constructed with return_pixel_truth=True")

        B = len(in_offs) - 1
        for i in range(B):
            gi = base + i
            ia, ib = in_offs[i], in_offs[i + 1]
            ci = in_coords[ia:ib]
            qi = input_charge[ia:ib]
            pid_i = (pid_list[i].numpy() if hasattr(pid_list[i], "numpy")
                     else np.asarray(pid_list[i]))
            assert pid_i.shape[0] == ci.shape[0], f"event {gi}: pixel_pid/coords length mismatch"

            oa, ob = out_offs[i], out_offs[i + 1]
            fo = out_feats[oa:ob]
            co = out_coords[oa:ob] if has_coord else None
            n_in_total += ci.shape[0]

            result, _n_in, n_matched = _match_align_label(ci, qi, pid_i, fo, co, has_coord)
            n_matched_total += n_matched
            if result is None:
                continue
            feat, raw, label = result
            feats.append(feat)
            labels.append(label)
            raws.append(raw)
            ev_idx.append(np.full(len(label), gi, dtype=np.int64))
        base += B

    return _pack_examples(feats, raws, labels, ev_idx, n_in_total, n_matched_total, bool(has_coord))


def extract_pixel_examples_precomputed(npz_path, ds, n_events):
    """Like `extract_pixel_examples`, but sources per-event (feature, coord) from
    a precomputed CSR `.npz` -- an EXTERNAL model's per-voxel feature export --
    instead of a live backbone forward pass. Uses the exact same
    `_match_align_label` helper as the live path, so the raw floor, PDG labeling,
    and match-rate gate are byte-identical in formula to any live-backbone run in
    the same table/JSON; only the source of `(feat, coord)` differs.

    Truth-side (channel,tick,charge,PDG) come from iterating `ds` itself, so the
    same APASparseMetaDataset is the single source of truth for every run.

    Expected npz schema (written by the foreign repo's own exporter):
        coords  [N,2] int32    (channel,tick), one row per exported voxel
        feat    [N,D] float32  per-voxel backbone feature
        offsets [n_events_avail+1] int64  event i = rows offsets[i]:offsets[i+1],
            events indexed 0..n_events_avail-1 in the SAME order as
            `Subset(ds, range(n_events))` on the identical dataset.
    """
    npz = np.load(npz_path)
    out_coords_all, out_feat_all, out_offs = npz["coords"], npz["feat"], npz["offsets"]
    n_avail = len(out_offs) - 1
    if n_avail < n_events:
        print(f"[warn] {npz_path}: only {n_avail} events exported, requested {n_events}; "
              f"using {n_avail}", flush=True)
    n_use = min(n_events, n_avail, len(ds))

    feats, raws, labels, ev_idx = [], [], [], []
    n_in_total, n_matched_total = 0, 0
    for gi in range(n_use):
        voxels, meta = ds[gi]
        ci = voxels.coordinate_tensor.cpu().numpy()
        qi = voxels.feature_tensor[:, 0].cpu().numpy()   # raw ADC (ds.apply_log_transform=False)
        pid_i = np.asarray(meta["pid_labels"])
        assert pid_i.shape[0] == ci.shape[0], f"event {gi}: pixel_pid/coords length mismatch"

        oa, ob = int(out_offs[gi]), int(out_offs[gi + 1])
        fo = out_feat_all[oa:ob]
        co = out_coords_all[oa:ob]
        n_in_total += ci.shape[0]

        result, _n_in, n_matched = _match_align_label(ci, qi, pid_i, fo, co, True)
        n_matched_total += n_matched
        if result is None:
            continue
        feat, raw, label = result
        feats.append(feat)
        labels.append(label)
        raws.append(raw)
        ev_idx.append(np.full(len(label), gi, dtype=np.int64))

    return _pack_examples(feats, raws, labels, ev_idx, n_in_total, n_matched_total, True)


def _pack_examples(feats, raws, labels, ev_idx, n_in_total, n_matched_total, has_coord):
    align_info = {
        "match_rate": (n_matched_total / n_in_total) if n_in_total else 1.0,
        "has_coord": has_coord,
        "n_pixels_in": n_in_total,
        "n_matched": n_matched_total,
    }
    if not feats:
        empty = {"feat": np.zeros((0, 1), np.float32), "raw": np.zeros((0, 3), np.float32),
                 "label": np.zeros((0,), np.int64), "event_idx": np.zeros((0,), np.int64)}
        return empty, align_info
    examples = {
        "feat": np.concatenate(feats, axis=0).astype(np.float32),
        "raw": np.concatenate(raws, axis=0).astype(np.float32),
        "label": np.concatenate(labels, axis=0).astype(np.int64),
        "event_idx": np.concatenate(ev_idx, axis=0),
    }
    return examples, align_info


class _MLPHead(nn.Module):
    def __init__(self, in_dim, n_classes=N_CLASSES, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def fit_mlp_predict(Xtr, ytr, Xva, seed, epochs=30, lr=5e-3, batch=256):
    """sft head: small MLP trained on truth, matching the MAE sft_epochs/lr/batch."""
    torch.manual_seed(seed)
    model = _MLPHead(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t = torch.from_numpy(Xtr).float().to(dev)
    ytr_t = torch.from_numpy(ytr).long().to(dev)
    n = len(Xtr_t)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = Fnn.cross_entropy(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xva).float().to(dev)).argmax(1).cpu().numpy()
    return pred


def _f1_by_class(y_true, y_pred):
    labels = list(range(N_CLASSES))
    vals = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {CLASS_NAMES[c]: float(vals[c]) for c in range(N_CLASSES)}


def _class_balanced_pool(idx, labels, per_class, seed):
    rng = np.random.RandomState(seed)
    picked = []
    for c in range(N_CLASSES):
        ci = idx[labels[idx] == c].copy()
        rng.shuffle(ci)
        picked.append(ci[:per_class])
    return np.concatenate(picked)


def _gather_examples(run, extract_fn, ds, n_events, min_match_rate):
    """Drive `extract_fn(ds, n_events) -> (ex, align)` and return the
    (feat, raw, lab, keys, match_rate) tuple the probe-fitting code expects.
    `keys` is (M, 2) = [file_idx, event_idx]; file_idx is always 0 here (single
    dataset) but the 2-col shape keeps the event-level split code identical to
    the multi-file original."""
    ex, align = extract_fn(ds, n_events)
    print(f"[{run}] events={n_events} pixels_kept={len(ex['label'])} "
          f"align_match_rate={align['match_rate']:.4f} has_coord={align['has_coord']}", flush=True)
    assert align["match_rate"] >= min_match_rate, (
        f"{run}: model-output/input coordinate match rate {align['match_rate']:.3f} "
        f"< {min_match_rate} -- feature/label alignment cannot be trusted, aborting"
    )
    feat, raw, lab = ex["feat"], ex["raw"], ex["label"]
    keys = np.stack([np.zeros(len(ex["event_idx"]), dtype=np.int64), ex["event_idx"]], axis=1)
    return feat, raw, lab, keys, align["match_rate"]


def _fit_and_report(run, feat, raw, lab, keys, match_rate, seed, pool_per_class, t0):
    """Event-level split -> class-balanced pool -> StandardScaler -> SVM/MLP ->
    macro-F1. Identical for a live-backbone run and a precomputed-external run --
    this is the leakage-free, unified half of the probe."""
    uniq_events = np.unique(keys, axis=0)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq_events))
    n_train = int(0.8 * len(uniq_events))
    train_set = set(map(tuple, uniq_events[perm[:n_train]]))
    is_train = np.array([tuple(k) in train_set for k in keys])

    tr_idx = _class_balanced_pool(np.where(is_train)[0], lab, pool_per_class, seed)
    va_idx = _class_balanced_pool(np.where(~is_train)[0], lab, pool_per_class, seed + 1)

    class_counts = {
        "train": {CLASS_NAMES[c]: int((lab[tr_idx] == c).sum()) for c in range(N_CLASSES)},
        "val":   {CLASS_NAMES[c]: int((lab[va_idx] == c).sum()) for c in range(N_CLASSES)},
    }
    for split_name, cnts in class_counts.items():
        for cname, n in cnts.items():
            if n < pool_per_class:
                print(f"[{run}] WARNING: {split_name}/{cname} pool has only {n} "
                      f"pixels (< target {pool_per_class})", flush=True)

    feat_scaler = StandardScaler().fit(feat[tr_idx])
    raw_scaler = StandardScaler().fit(raw[tr_idx])
    Xf_tr, Xf_va = feat_scaler.transform(feat[tr_idx]), feat_scaler.transform(feat[va_idx])
    Xr_tr, Xr_va = raw_scaler.transform(raw[tr_idx]), raw_scaler.transform(raw[va_idx])
    y_tr, y_va = lab[tr_idx], lab[va_idx]

    svm_feat = LinearSVC(C=1.0, max_iter=10000, random_state=seed).fit(Xf_tr, y_tr)
    svm_raw = LinearSVC(C=1.0, max_iter=10000, random_state=seed).fit(Xr_tr, y_tr)
    pred_svm_feat = svm_feat.predict(Xf_va)
    pred_svm_raw = svm_raw.predict(Xr_va)
    pred_sft_feat = fit_mlp_predict(Xf_tr, y_tr, Xf_va, seed)
    pred_sft_raw = fit_mlp_predict(Xr_tr, y_tr, Xr_va, seed)

    f1_svm_feat = f1_score(y_va, pred_svm_feat, average="macro")
    f1_svm_raw = f1_score(y_va, pred_svm_raw, average="macro")
    f1_sft_feat = f1_score(y_va, pred_sft_feat, average="macro")
    f1_sft_raw = f1_score(y_va, pred_sft_raw, average="macro")

    entry = {
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
        "n_events": int(len(uniq_events)),
        "class_counts": class_counts,
        "align_match_rate": match_rate,
        "voxel_svm_feat": float(f1_svm_feat), "voxel_svm_raw": float(f1_svm_raw),
        "sft_feat": float(f1_sft_feat), "sft_raw": float(f1_sft_raw),
        "per_class_f1": {
            "voxel_svm_feat": _f1_by_class(y_va, pred_svm_feat),
            "voxel_svm_raw": _f1_by_class(y_va, pred_svm_raw),
            "sft_feat": _f1_by_class(y_va, pred_sft_feat),
            "sft_raw": _f1_by_class(y_va, pred_sft_raw),
        },
    }
    print(f"[{run}] N_train={len(tr_idx)} N_val={len(va_idx)}  "
          f"voxel_svm_feat={f1_svm_feat:.4f}  voxel_svm_raw={f1_svm_raw:.4f}  "
          f"sft_feat={f1_sft_feat:.4f}  sft_raw={f1_sft_raw:.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return entry


def _parse_ckpts(spec):
    """'label=/path/a.pt,label2=/path/b.pt' -> {label: path}."""
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--ckpts entry '{item}' must be label=/path/to/checkpoint.pt")
        label, path = item.split("=", 1)
        out[label.strip()] = path.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", default="",
                    help="comma-sep label=/path/to/checkpoint.pt for live DINO checkpoints")
    ap.add_argument("--external", default="",
                    help='JSON dict {label: npz_path} of precomputed foreign-model '
                         'feature exports (see extract_pixel_examples_precomputed for schema)')
    ap.add_argument("--backbone_view", default="student", choices=["student", "teacher"],
                    help="which DINO backbone to probe for --ckpts runs")
    ap.add_argument("--n_events", type=int, default=500, help="events to scan (dataset order)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pool_per_class", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_match_rate", type=float, default=0.95,
                    help="abort if output/input coordinate match rate falls below this")
    # Dataset config -- defaults to the first --ckpts cfg; required for --external-only runs.
    ap.add_argument("--datadir", default="")
    ap.add_argument("--apa", type=int, default=None)
    ap.add_argument("--view", default="")
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--out", default="pid_probe_unified.json")
    args = ap.parse_args()

    ckpts = _parse_ckpts(args.ckpts) if args.ckpts else {}
    external = json.loads(args.external) if args.external else {}
    if not ckpts and not external:
        ap.error("at least one of --ckpts / --external is required")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ---- Resolve dataset config: CLI overrides, else the first checkpoint's cfg.
    first_cfg = None
    if ckpts:
        first_path = next(iter(ckpts.values()))
        with torch.serialization.safe_globals([DINOConfig]):
            first_cfg = torch.load(first_path, map_location="cpu")["cfg"]
    datadir = args.datadir or (first_cfg.datadir if first_cfg else "")
    apa = args.apa if args.apa is not None else (first_cfg.apa if first_cfg else None)
    view = args.view or (first_cfg.view if first_cfg else "")
    cache_dir = args.cache_dir or (getattr(first_cfg, "cache_dir", "") if first_cfg else "")
    if not datadir or apa is None or not view:
        ap.error("dataset config incomplete: pass --datadir/--apa/--view "
                 "(auto-derived from --ckpts cfg only when checkpoints are given)")

    print(f"[setup] dataset datadir={datadir} apa={apa} view={view}", flush=True)
    ds = APASparseMetaDataset(
        datadir=datadir, apa=apa, view=view, use_cache=True,
        cache_dir=(cache_dir or None),
        return_full_metadata=True, return_pixel_truth=True,
    )
    ds.apply_log_transform = False   # raw ADC at __getitem__; norm applied per-backbone below
    print(f"[setup] {len(ds)} events available", flush=True)

    res = {}

    for run, path in ckpts.items():
        if not os.path.exists(path):
            print(f"[skip] {run}: no {path}", flush=True)
            continue
        with torch.serialization.safe_globals([DINOConfig]):
            ck = torch.load(path, map_location=dev)
        cfg = ck["cfg"]
        bb = _load_backbone(ck, args.backbone_view, dev)
        norm = (FeatureLogTransform(cfg.feat_min_val, cfg.feat_max_val)
                if cfg.use_log_transform else None)
        t0 = time.time()
        extract_fn = lambda d, n, _bb=bb, _norm=norm: extract_pixel_examples(
            _bb, _norm, d, n, args.batch)
        feat, raw, lab, keys, mr = _gather_examples(run, extract_fn, ds, args.n_events,
                                                    args.min_match_rate)
        res[run] = _fit_and_report(run, feat, raw, lab, keys, mr,
                                   args.seed, args.pool_per_class, t0)

    for run, npz_path in external.items():
        if not os.path.exists(npz_path):
            print(f"[skip] {run}: missing export {npz_path}", flush=True)
            continue
        t0 = time.time()
        extract_fn = lambda d, n, _p=npz_path: extract_pixel_examples_precomputed(_p, d, n)
        feat, raw, lab, keys, mr = _gather_examples(run, extract_fn, ds, args.n_events,
                                                    args.min_match_rate)
        res[run] = _fit_and_report(run, feat, raw, lab, keys, mr,
                                   args.seed, args.pool_per_class, t0)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
