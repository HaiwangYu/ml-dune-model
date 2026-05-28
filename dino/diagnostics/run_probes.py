"""
Pixel-level PID probe suite for trained DINO checkpoints.

Runs the same 4 probes the mae offline-pool SFT uses
(`sft_feat`, `sft_raw`, `voxel_svm_feat`, `voxel_svm_raw`) on dino's
student AND teacher backbone features.  Output JSON matches mae's
`sft_history.json` per-epoch entry schema so the two pipelines can be
compared in a single table.

Splits images into train/val (val_frac=0.2 by default, deterministic seed)
to mirror mae's `sft_val_frac`.  Per-class pool cap = 5000 voxels matches
mae's `sft_pool_max_pixels`.  Pixel-class taxonomy and γ-blip
connected-component split are imported from `models.mae_model` so the
class definitions are identical.

Usage
-----
  python -m dino.diagnostics.run_probes <ckpt.pt>
  python -m dino.diagnostics.run_probes <ckpt.pt> --output=probes_ep100.json
  python -m dino.diagnostics.run_probes <ckpt.pt> --max_images=10000
"""

import inspect
import json
import sys
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loader.apa_sparse_meta_dataset import APASparseMetaDataset
from loader.collate import voxels_meta_collate_fn
from models import BACKBONE_REGISTRY
from models.minkunet_attention import MinkUNetSparseAttentionCore
from models.mae_model import (
    DensePixelHead,
    PIXEL_PID_CLASS_NAMES, PIXEL_PID_N_CLASSES,
    pdg_to_pixel_class,
)
from dino.config import DINOConfig
from dino.transforms import FeatureLogTransform


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_backbone(ckpt: dict, key: str, device: torch.device):
    """Load a dino backbone.  Older dino checkpoints saved the *Core* state
    (keys like `conv0.0.weight`); newer registry entries wrap that core in a
    dense input/output adapter (keys like `core.conv0.0.weight`).  We detect
    the layout by sniffing the first key and load into whichever module the
    keys belong to.  The model is run on Voxels regardless, so for the wrapper
    case we return only `.core`."""
    cfg = ckpt["cfg"]
    state = ckpt[key]
    sample_key = next(iter(state.keys()))
    old_format = not sample_key.startswith("core.") and not sample_key.startswith("input.")

    if old_format:
        # Flat checkpoint: load directly into the Core (Voxels -> Voxels).
        backbone_kwargs = {}
        if "encoding_range" in inspect.signature(MinkUNetSparseAttentionCore.__init__).parameters:
            backbone_kwargs["encoding_range"] = cfg.encoding_range
        model = MinkUNetSparseAttentionCore(**backbone_kwargs).to(device)
        model.load_state_dict(state)
    else:
        # Wrapper-format: instantiate the registered class, load fully, then
        # return only the `.core` submodule so the caller can pass Voxels.
        backbone_cls = BACKBONE_REGISTRY[cfg.backbone_name]
        backbone_kwargs = {}
        if "encoding_range" in inspect.signature(backbone_cls.__init__).parameters:
            backbone_kwargs["encoding_range"] = cfg.encoding_range
        wrapper = backbone_cls(**backbone_kwargs).to(device)
        wrapper.load_state_dict(state)
        model = wrapper.core
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Streaming pool builder.
#
# Memory-safe: instead of accumulating per-voxel features for the full dataset
# (~60 GB per view at 80k images × 3k voxels × 64-d float32), we drain the
# loader and pool *capped per-class* on the fly.  The pool tops out at
# `cap_per_class × n_classes` voxels per view (≈ 4 MB at 5000 cap × 3 cls ×
# 64 dim × 4 B), regardless of dataset size.  Mirrors mae's
# `_collect_sft_pool` semantics, extended to dual student+teacher features.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_pool_stream(student, teacher, loader, device, normalizer,
                         n_classes, cap_per_class, seed):
    """Iterate the loader once, applying `pdg_to_pixel_class` per image and
    pooling per-class up to `cap_per_class` voxels for each of (student_feat,
    teacher_feat, raw=(channel, tick, log1p(charge))).  Returns
        X_s [M, D] float32   teacher_feat: X_t [M, D]
        R   [M, 3] float32   y [M] int64
    where M ≤ cap_per_class * n_classes."""
    rng = np.random.default_rng(seed)
    pool_s = [[] for _ in range(n_classes)]
    pool_t = [[] for _ in range(n_classes)]
    pool_r = [[] for _ in range(n_classes)]
    counts = np.zeros(n_classes, dtype=np.int64)
    seen   = np.zeros(n_classes, dtype=np.int64)
    cap    = int(cap_per_class)

    for xs, meta in loader:
        if int(counts.min()) >= cap:
            break
        xs = xs.to(device)
        raw_charge = xs.feature_tensor.float().clone().cpu().numpy()   # [N, 1]
        coords     = xs.coordinate_tensor.cpu().numpy()                # [N, 2]
        if normalizer is not None:
            xs = normalizer(xs)
        s_out = student(xs).feature_tensor.float().cpu().numpy()       # [N, D]
        t_out = teacher(xs).feature_tensor.float().cpu().numpy()       # [N, D]
        img_offs = xs.offsets.cpu().numpy()                            # [B+1]
        pid_list = meta.get("pid_labels", None)
        if pid_list is None:
            raise RuntimeError("dataset must be constructed with return_pixel_truth=True")

        B = img_offs.shape[0] - 1
        for b in range(B):
            a, c = int(img_offs[b]), int(img_offs[b + 1])
            if c == a:
                continue
            pid_img = (pid_list[b].numpy() if hasattr(pid_list[b], "numpy")
                       else np.asarray(pid_list[b]))
            pos_img = coords[a:c]
            pix_cls = pdg_to_pixel_class(pid_img, pos_img)             # [n_img] int64; -1 = no truth

            s_img = s_out[a:c]
            t_img = t_out[a:c]
            raw_img = np.concatenate([
                pos_img.astype(np.float32),
                np.log1p(raw_charge[a:c].astype(np.float32)),
            ], axis=1)

            for k in range(n_classes):
                if counts[k] >= cap:
                    continue
                mask = pix_cls == k
                n_avail = int(mask.sum())
                if n_avail == 0:
                    continue
                seen[k] += n_avail
                need = cap - int(counts[k])
                if n_avail <= need:
                    pool_s[k].append(s_img[mask])
                    pool_t[k].append(t_img[mask])
                    pool_r[k].append(raw_img[mask])
                    counts[k] += n_avail
                else:
                    idx_local = np.where(mask)[0]
                    pick = rng.choice(idx_local, size=need, replace=False)
                    pool_s[k].append(s_img[pick])
                    pool_t[k].append(t_img[pick])
                    pool_r[k].append(raw_img[pick])
                    counts[k] += need

    F_s, F_t, F_r, F_y = [], [], [], []
    for k in range(n_classes):
        if pool_s[k]:
            f_s = np.concatenate(pool_s[k])
            f_t = np.concatenate(pool_t[k])
            f_r = np.concatenate(pool_r[k])
            F_s.append(f_s); F_t.append(f_t); F_r.append(f_r)
            F_y.append(np.full(len(f_s), k, dtype=np.int64))
    if not F_s:
        D = 64
        return (np.zeros((0, D), dtype=np.float32),
                np.zeros((0, D), dtype=np.float32),
                np.zeros((0, 3),  dtype=np.float32),
                np.zeros((0,),    dtype=np.int64),
                counts, seen)
    return (np.concatenate(F_s).astype(np.float32),
            np.concatenate(F_t).astype(np.float32),
            np.concatenate(F_r).astype(np.float32),
            np.concatenate(F_y).astype(np.int64),
            counts, seen)


# ---------------------------------------------------------------------------
# Probes: dense MLP + LinearSVC (mirror of mae helpers).
# ---------------------------------------------------------------------------

def _confusion_and_metrics(y_true, y_pred, n_classes):
    valid = y_true >= 0
    yt = y_true[valid]; yp = y_pred[valid]
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (yt, yp), 1)
    diag = cm.diagonal().astype(np.float64)
    row  = cm.sum(axis=1).astype(np.float64)
    col  = cm.sum(axis=0).astype(np.float64)
    eff = np.where(row > 0, diag / np.maximum(row, 1), np.nan)
    pur = np.where(col > 0, diag / np.maximum(col, 1), np.nan)
    f1  = np.where((eff + pur) > 0, 2 * eff * pur / (eff + pur + 1e-12), np.nan)
    macro_f1 = float(np.nanmean(f1))
    acc = float(diag.sum() / max(cm.sum(), 1))
    return cm.tolist(), eff.tolist(), pur.tolist(), acc, macro_f1


def _fit_dense_head(X, y, n_classes, device, *,
                    epochs=30, batch_size=256, lr=5e-3, seed=0):
    torch.manual_seed(seed)
    head = DensePixelHead(in_ch=X.shape[1], n_classes=n_classes).to(device)
    opt  = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)
    N = X_t.shape[0]
    perm_rng = np.random.default_rng(seed)
    for _ in range(epochs):
        head.train()
        order = torch.from_numpy(perm_rng.permutation(N)).to(device)
        for s in range(0, N, batch_size):
            idx = order[s:s + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return head


@torch.no_grad()
def _predict_dense_head(head, X, device, batch_size=2048):
    head.eval()
    X_t = torch.from_numpy(X).to(device)
    out = torch.empty(X_t.shape[0], dtype=torch.long, device=device)
    for s in range(0, X_t.shape[0], batch_size):
        out[s:s + batch_size] = head(X_t[s:s + batch_size]).argmax(dim=1)
    return out.cpu().numpy()


def _fit_svm(X_tr, y_tr, X_va, y_va, n_classes, *, C=1.0, seed=0):
    from sklearn.svm import LinearSVC
    svm = LinearSVC(C=C, class_weight="balanced", random_state=seed, max_iter=2000)
    svm.fit(X_tr, y_tr)
    pred_tr = svm.predict(X_tr)
    pred_va = svm.predict(X_va)
    cm_tr, eff_tr, pur_tr, acc_tr, f1_tr = _confusion_and_metrics(y_tr, pred_tr, n_classes)
    cm_va, eff_va, pur_va, acc_va, f1_va = _confusion_and_metrics(y_va, pred_va, n_classes)
    return dict(
        train_acc=acc_tr, val_acc=acc_va,
        train_macro_f1=f1_tr, val_macro_f1=f1_va,
        train_cm=cm_tr, val_cm=cm_va,
        train_eff=eff_tr, val_eff=eff_va,
        train_pur=pur_tr, val_pur=pur_va,
    )


def _probe_view(X_tr, R_tr, y_tr, X_va, R_va, y_va, n_classes, device, *,
                epochs=30, batch_size=256, lr=5e-3, svm_C=1.0, seed=0):
    """Run all 4 probes (sft_feat, sft_raw, voxel_svm_feat, voxel_svm_raw)
    on one (X, R, y) pool.  `sft_raw` and `voxel_svm_raw` reuse the same raw
    pool — they only depend on (channel, tick, log_charge), not on the view."""
    out = {"n_train": int(len(y_tr)), "n_val": int(len(y_va))}

    head = _fit_dense_head(X_tr, y_tr, n_classes, device,
                           epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    pred_tr = _predict_dense_head(head, X_tr, device)
    pred_va = _predict_dense_head(head, X_va, device)
    cm_tr, eff_tr, pur_tr, acc_tr, f1_tr = _confusion_and_metrics(y_tr, pred_tr, n_classes)
    cm_va, eff_va, pur_va, acc_va, f1_va = _confusion_and_metrics(y_va, pred_va, n_classes)
    out["sft_feat"] = dict(train_acc=acc_tr, val_acc=acc_va,
                           train_macro_f1=f1_tr, val_macro_f1=f1_va,
                           train_cm=cm_tr, val_cm=cm_va,
                           train_eff=eff_tr, val_eff=eff_va,
                           train_pur=pur_tr, val_pur=pur_va)

    head = _fit_dense_head(R_tr, y_tr, n_classes, device,
                           epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    rpred_tr = _predict_dense_head(head, R_tr, device)
    rpred_va = _predict_dense_head(head, R_va, device)
    rcm_tr, reff_tr, rpur_tr, racc_tr, rf1_tr = _confusion_and_metrics(y_tr, rpred_tr, n_classes)
    rcm_va, reff_va, rpur_va, racc_va, rf1_va = _confusion_and_metrics(y_va, rpred_va, n_classes)
    out["sft_raw"] = dict(train_acc=racc_tr, val_acc=racc_va,
                          train_macro_f1=rf1_tr, val_macro_f1=rf1_va,
                          train_cm=rcm_tr, val_cm=rcm_va,
                          train_eff=reff_tr, val_eff=reff_va,
                          train_pur=rpur_tr, val_pur=rpur_va)

    out["voxel_svm_feat"] = _fit_svm(X_tr, y_tr, X_va, y_va, n_classes, C=svm_C, seed=seed)
    out["voxel_svm_raw"]  = _fit_svm(R_tr, y_tr, R_va, y_va, n_classes, C=svm_C, seed=seed)
    return out


def main(
    checkpoint: str,
    output: str = "",
    max_images: int = -1,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    val_frac: float = 0.2,
    cap_per_class: int = 5000,
    epochs: int = 30,
    sft_batch: int = 256,
    sft_lr: float = 5e-3,
    svm_C: float = 1.0,
    seed: int = 0,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    with torch.serialization.safe_globals([DINOConfig]):
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    epoch = ckpt.get("epoch", 0)
    print(f"  epoch={epoch}  backbone={cfg.backbone_name}  feature_dim={cfg.feature_dim}")
    print(f"  use_log_transform={cfg.use_log_transform}  feat_min={cfg.feat_min_val}  feat_max={cfg.feat_max_val}")

    if not output:
        output = str(ckpt_path.parent / f"probes_ep{epoch}.json")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Dataset.  apply_log_transform=False: model was trained with raw ADC
    # input to which FeatureLogTransform is then applied; we replicate that
    # exactly below, so the dataset itself must return raw charges.
    print(f"\nDataset: {cfg.datadir}")
    dataset = APASparseMetaDataset(
        datadir=cfg.datadir,
        apa=cfg.apa,
        view=cfg.view,
        use_cache=True,
        cache_dir=cfg.cache_dir,
        return_full_metadata=True,
        return_pixel_truth=True,
    )
    dataset.apply_log_transform = False   # raw ADC at __getitem__

    # ---- Train/val split at image level (matches mae's sft_val_frac=0.2)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    if 0 < max_images < len(perm):
        perm = perm[:max_images]
    n_val = max(1, int(len(perm) * val_frac))
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]
    print(f"  total images: {len(perm)}   train: {len(tr_idx)}   val: {len(val_idx)}")

    tr_loader = DataLoader(
        Subset(dataset, tr_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=voxels_meta_collate_fn,
    )
    va_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=voxels_meta_collate_fn,
    )

    # ---- Models + normalizer
    print("\nLoading student and teacher backbones ...")
    student = _load_backbone(ckpt, "student", device)
    teacher = _load_backbone(ckpt, "teacher", device)
    normalizer = (FeatureLogTransform(cfg.feat_min_val, cfg.feat_max_val)
                  if cfg.use_log_transform else None)

    # ---- Stream-pool features per class (memory-safe; see _collect_pool_stream)
    print("Pooling train split ...")
    Xs_tr, Xt_tr, R_tr, y_tr, cnt_tr, seen_tr = _collect_pool_stream(
        student, teacher, tr_loader, device, normalizer,
        n_classes=PIXEL_PID_N_CLASSES, cap_per_class=cap_per_class, seed=seed)
    print(f"  train pool: {len(y_tr)} voxels  (per-class counts={cnt_tr.tolist()}, seen={seen_tr.tolist()})")
    print("Pooling val split ...")
    Xs_va, Xt_va, R_va, y_va, cnt_va, seen_va = _collect_pool_stream(
        student, teacher, va_loader, device, normalizer,
        n_classes=PIXEL_PID_N_CLASSES, cap_per_class=cap_per_class, seed=seed + 1)
    print(f"  val   pool: {len(y_va)} voxels  (per-class counts={cnt_va.tolist()}, seen={seen_va.tolist()})")

    # ---- Probes per view (student, teacher) on the same pool
    out_json = {
        "checkpoint": str(ckpt_path),
        "epoch": int(epoch),
        "backbone_name": cfg.backbone_name,
        "feature_dim":   int(cfg.feature_dim),
        "n_train_images": len(tr_idx),
        "n_val_images":   len(val_idx),
        "cap_per_class":  int(cap_per_class),
        "val_frac":       float(val_frac),
        "seed":           int(seed),
        "class_names":    PIXEL_PID_CLASS_NAMES,
        "pool_counts": {
            "train": cnt_tr.tolist(),
            "val":   cnt_va.tolist(),
            "train_seen": seen_tr.tolist(),
            "val_seen":   seen_va.tolist(),
        },
        "views": {},
    }

    for view, X_tr_v, X_va_v in [("student", Xs_tr, Xs_va),
                                  ("teacher", Xt_tr, Xt_va)]:
        print(f"\n=== view: {view} ===")
        view_out = _probe_view(
            X_tr_v, R_tr, y_tr, X_va_v, R_va, y_va,
            PIXEL_PID_N_CLASSES, device,
            epochs=epochs, batch_size=sft_batch, lr=sft_lr, svm_C=svm_C, seed=seed)

        out_json["views"][view] = view_out

        print(f"  [{view}] sft_feat:       val_macro_f1={view_out['sft_feat']['val_macro_f1']:.3f}")
        print(f"  [{view}] sft_raw:        val_macro_f1={view_out['sft_raw']['val_macro_f1']:.3f}")
        print(f"  [{view}] voxel_svm_feat: val_macro_f1={view_out['voxel_svm_feat']['val_macro_f1']:.3f}")
        print(f"  [{view}] voxel_svm_raw:  val_macro_f1={view_out['voxel_svm_raw']['val_macro_f1']:.3f}")

    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nWrote: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    fire.Fire(main)
