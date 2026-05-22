"""
Vertex-focused k-NN analysis of MAE backbone features.

For each image the neutrino interaction vertex is estimated by:
  1. Restricting to a coarse **search region** around the approximate vertex.
  2. Within that region, applying a charge threshold and finding the earliest-row
     pixel that has at least `--min_neighbors` neighbours within radius
     `--neighbor_r` (Option A — earliest connected pixel).

An evaluation box of size `(2*box_h) × (2*box_w)` is centred on that refined
vertex.  Two analyses are run on the box pixels:

  Image-level (vertex pool):
    Mean-pool the box pixels into one vector per image, then run k-NN.
  Pixel-level (vertex pixels):
    Use each individual pixel as a sample, inheriting the parent image label.

Produces (in --out_dir):
  knn_vertex_image_purity.png     — image-level purity
  knn_vertex_image_confusion.png  — image-level confusion matrix
  knn_vertex_image_scatter.png    — image-level 2-D scatter
  knn_vertex_pixel_purity.png     — pixel-level purity
  knn_vertex_pixel_confusion.png  — pixel-level confusion matrix
  knn_vertex_pixel_scatter.png    — pixel-level 2-D scatter

Usage:
    python -m mae.diagnostics.plot_knn_vertex path/to/features_ep2.npz
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mae.diagnostics.plot_knn import CLASS_NAMES, _reduce_2d


# ---------------------------------------------------------------------------
# Vertex finding
# ---------------------------------------------------------------------------

def _find_vertex(
    pos: np.ndarray, chg: np.ndarray,
    search_row: int, search_col: int, search_h: int, search_w: int,
    charge_threshold: float, neighbor_r: float, min_neighbors: int,
) -> tuple:
    rows, cols = pos[:, 0], pos[:, 1]
    chg_vals   = chg[:, 0] if chg.ndim == 2 else chg

    mask = (
        (rows >= search_row) & (rows < search_row + search_h) &
        (cols >= search_col - search_w) & (cols < search_col + search_w) &
        (chg_vals >= charge_threshold)
    )
    candidates = pos[mask]

    if len(candidates) == 0:
        return float(search_row), float(search_col)

    diff   = candidates[:, None, :] - candidates[None, :, :]
    counts = (np.linalg.norm(diff, axis=-1) < neighbor_r).sum(axis=1) - 1

    connected = candidates[counts >= min_neighbors]
    if len(connected) == 0:
        connected = candidates[[counts.argmax()]]

    best = connected[connected[:, 0].argmin()]
    return float(best[0]), float(best[1])


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _filter_valid(labels, offsets, feats, positions, charges):
    """Drop images whose label is < 0 (e.g. nutau CC / missing metadata)."""
    valid = labels >= 0
    if valid.all():
        return labels, offsets, feats, positions, charges

    new_labels = labels[valid]
    new_offsets = [0]
    keep_feat_idx = []
    for i, ok in enumerate(valid):
        start = int(offsets[i]); end = int(offsets[i + 1])
        if ok:
            keep_feat_idx.append((start, end))
            new_offsets.append(new_offsets[-1] + (end - start))
    new_offsets = np.array(new_offsets, dtype=np.int64)

    if keep_feat_idx:
        mask = np.concatenate([np.arange(s, e) for s, e in keep_feat_idx])
        new_feats = feats[mask]
        new_pos   = positions[mask]
        new_chg   = charges[mask]
    else:
        new_feats = feats[:0]
        new_pos   = positions[:0]
        new_chg   = charges[:0]
    return new_labels, new_offsets, new_feats, new_pos, new_chg


def _collect(
    feats, positions, charges, offsets, labels, n_images,
    search_row, search_col, search_h, search_w,
    charge_threshold, neighbor_r, min_neighbors,
    box_h, box_w, seed=42,
):
    rng = np.random.default_rng(seed)

    if n_images >= len(labels):
        img_indices = np.arange(len(labels))
    else:
        classes = np.unique(labels)
        per_class = max(1, n_images // len(classes))
        chosen = []
        for c in classes:
            cls_idx = np.where(labels == c)[0]
            chosen.append(rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False))
        img_indices = np.concatenate(chosen)

    img_parts, img_lbl_parts = [], []
    pix_parts, pix_lbl_parts = [], []
    n_zero = 0

    for img_idx in img_indices:
        sl  = slice(int(offsets[img_idx]), int(offsets[img_idx + 1]))
        pos = positions[sl].astype(np.float32)
        chg = charges[sl]

        vr, vc = _find_vertex(
            pos, chg, search_row, search_col, search_h, search_w,
            charge_threshold, neighbor_r, min_neighbors,
        )

        rows, cols = pos[:, 0], pos[:, 1]
        in_box = (
            (rows >= vr - box_h) & (rows < vr + box_h) &
            (cols >= vc - box_w) & (cols < vc + box_w)
        )

        if in_box.sum() == 0:
            n_zero += 1
            continue

        lbl   = labels[img_idx]
        f_box = feats[sl][in_box]
        n     = in_box.sum()

        img_parts.append(f_box.mean(axis=0))
        img_lbl_parts.append(lbl)

        pix_parts.append(f_box)
        pix_lbl_parts.append(np.full(n, lbl, dtype=np.int64))

    f_img    = np.stack(img_parts, axis=0) if img_parts else np.zeros((0, feats.shape[1]), dtype=np.float32)
    img_lbls = np.array(img_lbl_parts, dtype=np.int64)
    f_pix    = np.concatenate(pix_parts, axis=0) if pix_parts else np.zeros((0, feats.shape[1]), dtype=np.float32)
    pix_lbls = np.concatenate(pix_lbl_parts, axis=0) if pix_lbl_parts else np.zeros(0, dtype=np.int64)

    return f_img, img_lbls, f_pix, pix_lbls, n_zero


# ---------------------------------------------------------------------------
# Batched k-NN
# ---------------------------------------------------------------------------

def _l2_normalise(X):
    return X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _knn_purity_batched(feats, labels, ks, device, batch_size):
    X = _l2_normalise(torch.from_numpy(feats.astype(np.float32)).to(device))
    N = X.shape[0]
    lbls = torch.from_numpy(labels.astype(np.int64)).to(device)
    max_k = min(max(ks), N - 1)

    nn_labels = torch.empty(N, max_k, dtype=torch.int64, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B   = end - start
        sim = X[start:end] @ X.T
        sim[torch.arange(B, device=device), torch.arange(start, end, device=device)] = -torch.inf
        _, idx = sim.topk(max_k, dim=1)
        nn_labels[start:end] = lbls[idx]

    results = {}
    for k in ks:
        k_eff   = min(k, N - 1)
        same    = (nn_labels[:, :k_eff] == lbls[:, None]).float().mean(dim=1)
        overall = float(same.mean())
        per_class = np.full(len(CLASS_NAMES), np.nan)
        for c in np.unique(labels):
            per_class[c] = float(same[lbls == c].mean())
        results[k] = (overall, per_class)
    return results


def _knn_predict_batched(feats, labels, k, device, batch_size):
    X = _l2_normalise(torch.from_numpy(feats.astype(np.float32)).to(device))
    N = X.shape[0]
    lbls = torch.from_numpy(labels.astype(np.int64)).to(device)
    n_cls = len(CLASS_NAMES)
    k_eff = min(k, N - 1)

    preds = torch.empty(N, dtype=torch.int64, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B   = end - start
        sim = X[start:end] @ X.T
        sim[torch.arange(B, device=device), torch.arange(start, end, device=device)] = -torch.inf
        _, idx = sim.topk(k_eff, dim=1)
        nn_lbls = lbls[idx]
        off = torch.arange(B, device=device).unsqueeze(1) * n_cls
        flat = (nn_lbls + off).reshape(-1)
        counts = torch.bincount(flat, minlength=B * n_cls).reshape(B, n_cls)
        preds[start:end] = counts.argmax(dim=1)
    return preds.cpu().numpy()


# ---------------------------------------------------------------------------
# Plots (single-feature)
# ---------------------------------------------------------------------------

def _plot_purity(purity_k, out_dir, tag, fname, title):
    ks = sorted(purity_k.keys())
    n_classes = len(CLASS_NAMES)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    x      = np.arange(n_classes + 1)
    width  = 0.8 / len(ks)
    xlbls  = list(CLASS_NAMES) + ["Overall"]

    for ki, k in enumerate(ks):
        overall, per_class = purity_k[k]
        values = list(per_class) + [overall]
        ax.bar(x + ki * width - 0.4 + width / 2, values, width=width, label=f"k={k}")
    chance = 1.0 / n_classes
    ax.axhline(chance, color="red", linestyle="--", linewidth=1.2,
               label=f"chance (1/{n_classes})")
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls)
    ax.set_ylabel("Label purity")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{title} k-NN purity  [{tag}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def _plot_confusion(preds, labels, k, out_dir, tag, fname, title):
    from sklearn.metrics import confusion_matrix
    n_classes = len(CLASS_NAMES)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)), normalize="true")
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > 0.5 else "black")
    fig.suptitle(f"{title} confusion (k={k})  [{tag}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def _plot_scatter(emb, labels, reducer_name, out_dir, tag, fname, title,
                  alpha=0.4, s=4.0):
    n_classes = len(CLASS_NAMES)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for c in range(n_classes):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=s, alpha=alpha,
                   color=colors[c], label=CLASS_NAMES[c], rasterized=True)
    ax.set_xlabel(f"{reducer_name} 1")
    ax.set_ylabel(f"{reducer_name} 2")
    ax.legend(markerscale=5, fontsize=8)
    fig.suptitle(f"{title} {reducer_name}  [{tag}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Shared k-NN driver
# ---------------------------------------------------------------------------

def _run_knn(feats, lbls, ks, knn_k, reducer, out_dir, tag, prefix, title,
             scatter_alpha, scatter_s, device, batch_size):
    print(f"\n[{title}]  n={len(lbls)}")
    print(f"  Class distribution: "
          f"{ {CLASS_NAMES[c]: int((lbls==c).sum()) for c in np.unique(lbls)} }")
    if len(lbls) == 0:
        print("  No samples — skipping.")
        return

    print(f"  Computing k-NN purity (device={device}, batch={batch_size}) ...")
    purity_k = _knn_purity_batched(feats, lbls, ks, device, batch_size)
    for k in ks:
        print(f"  k={k:2d}  purity={purity_k[k][0]:.3f}")
    _plot_purity(purity_k, out_dir, tag, f"{prefix}_purity.png", title)

    print(f"  Computing k-NN predictions (k={knn_k}) ...")
    preds = _knn_predict_batched(feats, lbls, knn_k, device, batch_size)
    _plot_confusion(preds, lbls, knn_k, out_dir, tag,
                    f"{prefix}_confusion.png", title)

    print("  Running dimensionality reduction ...")
    emb, rname = _reduce_2d(feats, method=reducer)
    _plot_scatter(emb, lbls, rname, out_dir, tag,
                  f"{prefix}_scatter.png", title,
                  alpha=scatter_alpha, s=scatter_s)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    npz_path, out_dir, tag,
    n_images,
    search_row, search_col, search_h, search_w,
    charge_threshold, neighbor_r, min_neighbors,
    box_h, box_w,
    ks, knn_k, reducer, device, batch_size, seed=42,
):
    print(f"Loading {npz_path}")
    data      = np.load(npz_path)
    feats     = data["backbone_features"]
    labels    = data["labels"].astype(int)
    offsets   = data["offsets"]
    positions = data["positions"]
    if "charges" not in data.files:
        print("Error: 'charges' not found in .npz — re-extract features with "
              "the updated extract_features.py.")
        sys.exit(1)
    charges   = data["charges"]

    n_before = len(labels)
    labels, offsets, feats, positions, charges = _filter_valid(
        labels, offsets, feats, positions, charges,
    )
    n_dropped = n_before - len(labels)

    print(f"  Valid pixels : {feats.shape[0]}   Feature dim: {feats.shape[1]}")
    print(f"  Images       : {len(labels)}  ({n_dropped} dropped, label<0)")
    print(f"  Class counts : { {CLASS_NAMES[c]: int((labels==c).sum()) for c in np.unique(labels)} }")
    print(f"\n  Search region : rows [{search_row}, {search_row+search_h})  "
          f"cols [{search_col-search_w}, {search_col+search_w})")
    print(f"  Charge threshold : {charge_threshold}  neighbor_r : {neighbor_r}  min_neighbors : {min_neighbors}")
    print(f"  Eval box      : ±{box_h} rows × ±{box_w} cols around refined vertex")
    print(f"  Images to use : {n_images}")
    print(f"  Device        : {device}")

    print("\nCollecting vertex pixels ...")
    f_img, img_lbls, f_pix, pix_lbls, n_zero = _collect(
        feats, positions, charges, offsets, labels,
        n_images=n_images,
        search_row=search_row, search_col=search_col,
        search_h=search_h, search_w=search_w,
        charge_threshold=charge_threshold,
        neighbor_r=neighbor_r,
        min_neighbors=min_neighbors,
        box_h=box_h, box_w=box_w,
        seed=seed,
    )
    print(f"  Images used  : {len(img_lbls)}  ({n_zero} skipped — empty search region)")
    print(f"  Total pixels : {len(pix_lbls)}")

    common = dict(ks=ks, knn_k=knn_k, reducer=reducer,
                  out_dir=out_dir, tag=tag, device=device, batch_size=batch_size)

    _run_knn(
        f_img, img_lbls,
        prefix="knn_vertex_image",
        title="Vertex region (image-level pool)",
        scatter_alpha=0.4, scatter_s=4.0,
        **common,
    )

    _run_knn(
        f_pix, pix_lbls,
        prefix="knn_vertex_pixel",
        title="Vertex region (pixel-level)",
        scatter_alpha=0.15, scatter_s=2.0,
        **common,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Vertex-focused k-NN analysis of MAE backbone features."
    )
    parser.add_argument("npz_path")
    parser.add_argument("--out_dir", default="")

    parser.add_argument("--search_row", type=int, default=0)
    parser.add_argument("--search_col", type=int, default=250)
    parser.add_argument("--search_h", type=int, default=100)
    parser.add_argument("--search_w", type=int, default=100)
    parser.add_argument("--charge_threshold", type=float, default=100.0)
    parser.add_argument("--neighbor_r", type=float, default=3.0)
    parser.add_argument("--min_neighbors", type=int, default=5)

    parser.add_argument("--box_h", type=int, default=50)
    parser.add_argument("--box_w", type=int, default=30)

    parser.add_argument("--ks", default="1,5,10,20")
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--reducer", default="auto", choices=["auto", "umap", "tsne"])

    parser.add_argument("--device", default="")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    run(
        npz_path=npz_path,
        out_dir=out_dir,
        tag=npz_path.stem,
        n_images=args.n_images,
        search_row=args.search_row,
        search_col=args.search_col,
        search_h=args.search_h,
        search_w=args.search_w,
        charge_threshold=args.charge_threshold,
        neighbor_r=args.neighbor_r,
        min_neighbors=args.min_neighbors,
        box_h=args.box_h,
        box_w=args.box_w,
        ks=[int(k) for k in args.ks.split(",")],
        knn_k=args.knn_k,
        reducer=args.reducer,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
