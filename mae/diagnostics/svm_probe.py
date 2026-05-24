"""
Linear-SVM probe on MAE backbone features.

Implements a frozen-feature linear-probe diagnostic similar to PoLAr-MAE's
``SSLModel.validate``: collect a stratified sample of per-voxel backbone
features and their pixel-PID class labels, randomly split into train/val,
fit a ``LinearSVC(class_weight='balanced')`` on the train half, then report
overall accuracy + per-class F1 on both halves.

Intended to be called once per SSL epoch (after the SFT block, before the
checkpoint save) so the result becomes another entry in sft_history.json.

Cheap: at ~5 k pixels/class × 3 classes = 15 k samples × 64-D features, the
sklearn LinearSVC fit + score takes a few seconds on CPU.
"""

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC


@torch.no_grad()
def svm_probe(
    feature_fn,
    sft_loader,
    device,
    class_names,
    *,
    max_pixels_per_class: int = 5000,
    svm_C: float = 1.0,
    train_frac: float = 0.8,
    seed: int = 42,
    prepare_voxels=None,
):
    """Run a linear-SVM linear-probe over the SFT loader.

    Parameters
    ----------
    feature_fn : callable(Voxels) -> Voxels.  Wraps the model's backbone
                 forward (handles single-arg vs (vox_visible, vox_union)
                 signatures for SparseMAEModel vs SparseTrueMAEModel).
                 Returns Voxels with [N_total, D] feature_tensor.
    sft_loader : DataLoader yielding (Voxels, pixel_class LongTensor[N_total])
                 — i.e. what voxels_pixel_label_collate_fn produces.
    device   : torch.device
    class_names : list of strings, length n_classes
    max_pixels_per_class : per-class cap for the stratified sample
    svm_C, train_frac, seed : sklearn / sampling knobs
    prepare_voxels : optional callable that takes the CPU Voxels yielded
                     by the loader and returns the device-side Voxels ready
                     for the backbone (e.g. `log1p_voxels(voxels_to_device(v, device))`).
                     Defaults to no-op (caller is responsible for placement).

    Returns
    -------
    dict with keys
        train_acc, val_acc                       (floats)
        train_class_f1 / val_class_f1            (dict[class_name -> float])
        n_train, n_val                           (ints)
        train_macro_f1 / val_macro_f1            (floats; sklearn macro avg)
        per_class_counts                         (dict[class_name -> int])
    """
    rng = np.random.default_rng(seed)
    n_classes = len(class_names)
    pool_feats = [[] for _ in range(n_classes)]
    pool_counts = np.zeros(n_classes, dtype=np.int64)
    cap = max_pixels_per_class

    for vox_cpu, pix_labels in sft_loader:
        if int(pool_counts.min()) >= cap:
            break
        vox = prepare_voxels(vox_cpu) if prepare_voxels is not None else vox_cpu
        if vox.feature_tensor.shape[0] == 0:
            continue
        feats = feature_fn(vox).feature_tensor.float().cpu().numpy()
        labels = pix_labels.cpu().numpy() if hasattr(pix_labels, "cpu") else np.asarray(pix_labels)
        if feats.shape[0] != labels.shape[0]:
            # Should not happen for the pixel-class loader (collate preserves order).
            continue
        for c in range(n_classes):
            need = cap - int(pool_counts[c])
            if need <= 0:
                continue
            mask = labels == c
            n_avail = int(mask.sum())
            if n_avail == 0:
                continue
            if n_avail <= need:
                pool_feats[c].append(feats[mask])
                pool_counts[c] += n_avail
            else:
                idx = rng.choice(np.where(mask)[0], size=need, replace=False)
                pool_feats[c].append(feats[idx])
                pool_counts[c] += need

    X_parts, y_parts = [], []
    for c in range(n_classes):
        if pool_feats[c]:
            f = np.concatenate(pool_feats[c])
            X_parts.append(f)
            y_parts.append(np.full(len(f), c, dtype=np.int64))

    if not X_parts:
        return {
            "error": "no labelled pixels collected",
            "per_class_counts": {n: 0 for n in class_names},
        }

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)

    # Random train/val split
    perm = rng.permutation(len(y))
    n_train = int(len(y) * train_frac)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # LinearSVC with class_weight='balanced'.  Single-label here (pixels carry
    # exactly one class), so no OneVsRestClassifier wrapping is needed —
    # LinearSVC handles multi-class natively (one-vs-rest internally).
    svm = LinearSVC(C=svm_C, class_weight="balanced", random_state=seed)
    svm.fit(X_train, y_train)

    train_acc = float(svm.score(X_train, y_train))
    val_acc   = float(svm.score(X_val, y_val))

    train_pred = svm.predict(X_train)
    val_pred   = svm.predict(X_val)
    train_report = classification_report(
        y_train, train_pred, labels=list(range(n_classes)),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    val_report = classification_report(
        y_val, val_pred, labels=list(range(n_classes)),
        target_names=class_names, output_dict=True, zero_division=0,
    )

    return {
        "train_acc":      train_acc,
        "val_acc":        val_acc,
        "train_class_f1": {n: float(train_report[n]["f1-score"]) for n in class_names},
        "val_class_f1":   {n: float(val_report[n]["f1-score"])   for n in class_names},
        "train_macro_f1": float(train_report["macro avg"]["f1-score"]),
        "val_macro_f1":   float(val_report["macro avg"]["f1-score"]),
        "n_train":        int(len(y_train)),
        "n_val":          int(len(y_val)),
        "per_class_counts": {n: int(pool_counts[c]) for c, n in enumerate(class_names)},
    }
