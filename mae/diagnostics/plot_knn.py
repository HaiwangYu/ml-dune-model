"""
k-NN analysis of MAE backbone features in 64-D cosine space.

Two analysis levels:
  1. Image-level (always): mean-pool voxel features per image, then k-NN
     purity, k-NN classifier accuracy, confusion matrix, and a 2-D
     UMAP / t-SNE scatter.
  2. Voxel-level (optional, --voxel_knn): a random subsample of voxels
     (each inheriting its parent image's label), same purity + scatter.

Images / voxels with label == -1 are excluded from all analyses.

Produces:
  knn_image_purity.png     — per-class k-NN label purity at k=1,5,10,20
  knn_image_confusion.png  — confusion matrix (k-NN majority vote, k=5)
  knn_image_scatter.png    — UMAP or t-SNE 2-D scatter (image-level)
  knn_voxel_purity.png     — (--voxel_knn) voxel-level purity
  knn_voxel_confusion.png  — (--voxel_knn) voxel-level confusion matrix
  knn_voxel_scatter.png    — (--voxel_knn) voxel-level scatter

Usage:
    python mae/diagnostics/plot_knn.py path/to/features_ep3.npz
    python mae/diagnostics/plot_knn.py path/to/features_ep3.npz --out_dir=./plots
    python mae/diagnostics/plot_knn.py path/to/features_ep3.npz --voxel_knn --n_voxel_samples=50000
    python mae/diagnostics/plot_knn.py path/to/features_ep3.npz --ks=1,5,10,20 --knn_k=10
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["numuCC", "nueCC", "NC"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_norm(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation — cosine similarity becomes a dot product."""
    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)
    return X / norms


def _mean_pool(feats: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Mean-pool voxel features per image → [N_images, D]."""
    n_images = len(offsets) - 1
    D = feats.shape[1]
    out = np.empty((n_images, D), dtype=np.float32)
    for i in range(n_images):
        sl = slice(offsets[i], offsets[i + 1])
        chunk = feats[sl]
        out[i] = chunk.mean(axis=0) if len(chunk) > 0 else np.zeros(D, dtype=np.float32)
    return out


def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for normalised rows → dot product."""
    Xn = _l2_norm(X)
    return Xn @ Xn.T   # [N, N]


def _knn_purity(sim: np.ndarray, labels: np.ndarray, k: int) -> tuple:
    """
    For each sample, find the k nearest neighbours (excluding self) and
    compute the fraction that share the same label.

    Returns:
        overall_purity  : float
        per_class_purity: ndarray [n_classes]  (NaN if class absent)
    """
    N = len(labels)
    S = sim.copy()
    np.fill_diagonal(S, -np.inf)
    nn_idx = np.argpartition(-S, kth=min(k, N - 1), axis=1)[:, :k]

    purities = np.empty(N, dtype=float)
    for i in range(N):
        purities[i] = (labels[nn_idx[i]] == labels[i]).mean()

    per_class = np.full(len(CLASS_NAMES), np.nan)
    for c in np.unique(labels):
        if 0 <= c < len(CLASS_NAMES):
            per_class[c] = purities[labels == c].mean()

    return float(purities.mean()), per_class


def _knn_predict(sim: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Majority-vote k-NN prediction (leave-one-out via -inf diagonal)."""
    N = len(labels)
    S = sim.copy()
    np.fill_diagonal(S, -np.inf)
    nn_idx = np.argpartition(-S, kth=min(k, N - 1), axis=1)[:, :k]

    preds = np.empty(N, dtype=int)
    for i in range(N):
        counts = np.bincount(labels[nn_idx[i]], minlength=len(CLASS_NAMES))
        preds[i] = int(counts.argmax())
    return preds


def _voxel_labels(labels: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Expand per-image labels to per-voxel labels."""
    n_voxels = offsets[-1]
    out = np.empty(n_voxels, dtype=np.int64)
    for i in range(len(labels)):
        out[offsets[i]:offsets[i + 1]] = labels[i]
    return out


def _reduce_2d(X: np.ndarray, method: str = "auto"):
    """Reduce [N, D] to [N, 2].  method: 'umap', 'tsne', or 'auto'."""
    if method in ("umap", "auto"):
        try:
            from umap import UMAP
            print("    Using UMAP for 2-D reduction.")
            return UMAP(n_components=2, metric="cosine",
                        random_state=42, verbose=False).fit_transform(X), "UMAP"
        except ImportError:
            if method == "umap":
                raise
            print("    umap-learn not found — falling back to t-SNE.")
    from sklearn.manifold import TSNE
    print("    Using t-SNE for 2-D reduction.")
    emb = TSNE(n_components=2, metric="cosine", init="pca",
               random_state=42, n_jobs=-1).fit_transform(X)
    return emb, "t-SNE"


# ---------------------------------------------------------------------------
# Plot: k-NN purity bar chart (single backbone)
# ---------------------------------------------------------------------------

def plot_purity(purity_k: dict, out_dir: Path, tag: str, fname: str, title_prefix: str):
    ks = sorted(purity_k.keys())
    n_classes = len(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(n_classes + 1)        # one bar per class + overall
    width = 0.8 / len(ks)
    tick_labels = CLASS_NAMES + ["Overall"]

    for ki, k in enumerate(ks):
        overall, per_class = purity_k[k]
        values = list(per_class) + [overall]
        ax.bar(
            x + ki * width - 0.4 + width / 2,
            values,
            width=width,
            label=f"k={k}",
        )

    chance = 1.0 / n_classes
    ax.axhline(chance, color="red", linestyle="--", linewidth=1.2,
               label=f"chance (1/{n_classes})")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("Label purity")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{title_prefix} k-NN label purity  [{tag}]")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot: confusion matrix (single backbone)
# ---------------------------------------------------------------------------

def plot_confusion(
    preds: np.ndarray, labels: np.ndarray, k: int,
    out_dir: Path, tag: str, fname: str, title_prefix: str,
):
    n_classes = len(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(labels, preds):
        if 0 <= true < n_classes and 0 <= pred < n_classes:
            cm[true, pred] += 1

    row_sums = cm.sum(axis=1, keepdims=True).clip(1)
    cm_norm  = cm / row_sums
    acc = (labels == preds).mean()

    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > 0.6 else "black")

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title_prefix} k-NN confusion (k={k}, acc={acc:.3f})  [{tag}]")

    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot: 2-D scatter (single backbone)
# ---------------------------------------------------------------------------

def plot_scatter(
    emb: np.ndarray, labels: np.ndarray, reducer_name: str,
    out_dir: Path, tag: str, fname: str, title_prefix: str,
    alpha: float = 0.4, s: float = 6.0,
):
    n_classes = len(CLASS_NAMES)
    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(n_classes):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   color=colors[c], label=CLASS_NAMES[c],
                   alpha=alpha, s=s, linewidths=0)
    ax.set_title(f"{title_prefix} {reducer_name} scatter  [{tag}]")
    ax.set_xlabel(f"{reducer_name} dim 1")
    ax.set_ylabel(f"{reducer_name} dim 2")
    ax.legend(markerscale=3, fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Image-level pipeline
# ---------------------------------------------------------------------------

def run_image_level(
    feats: np.ndarray, labels: np.ndarray, offsets: np.ndarray,
    out_dir: Path, tag: str, ks: list, knn_k: int, reducer: str,
):
    print("\n[Image-level analysis]")
    print("  Mean-pooling voxel features per image ...")
    img_feats = _mean_pool(feats, offsets)   # [N_images, D]

    # Filter out unlabelled images (label == -1)
    valid_mask = labels >= 0
    img_feats  = img_feats[valid_mask]
    img_labels = labels[valid_mask]
    print(f"  Image embeddings: {img_feats.shape}  |  classes: {np.unique(img_labels)}")
    print(f"  Class counts: { {CLASS_NAMES[c]: int((img_labels==c).sum()) for c in np.unique(img_labels)} }")

    print("  Computing pairwise cosine similarity matrix ...")
    sim = _cosine_sim_matrix(img_feats)

    purity_k = {}
    for k in ks:
        k_eff = min(k, len(img_labels) - 1)
        overall, pc = _knn_purity(sim, img_labels, k_eff)
        purity_k[k] = (overall, pc)
        print(f"  k={k:2d}  purity={overall:.3f}")

    plot_purity(purity_k, out_dir, tag, "knn_image_purity.png", "Image-level")

    k_eff = min(knn_k, len(img_labels) - 1)
    print(f"  Computing k-NN predictions (k={knn_k}) ...")
    preds = _knn_predict(sim, img_labels, k_eff)
    plot_confusion(preds, img_labels, knn_k, out_dir, tag,
                   "knn_image_confusion.png", "Image-level")

    print("  Running dimensionality reduction on image embeddings ...")
    emb, rname = _reduce_2d(img_feats, method=reducer)
    plot_scatter(emb, img_labels, rname, out_dir, tag,
                 "knn_image_scatter.png", "Image-level")


# ---------------------------------------------------------------------------
# Voxel-level pipeline
# ---------------------------------------------------------------------------

def run_voxel_level(
    feats: np.ndarray, labels: np.ndarray, offsets: np.ndarray,
    out_dir: Path, tag: str, ks: list, knn_k: int,
    n_samples: int, reducer: str, seed: int = 42,
):
    print("\n[Voxel-level analysis]")
    rng = np.random.default_rng(seed)

    # Expand per-image labels to per-voxel, then filter unlabelled
    vox_labels = _voxel_labels(labels, offsets)
    valid_mask = vox_labels >= 0
    vox_feats  = feats[valid_mask]
    vox_labels = vox_labels[valid_mask]

    # Subsample
    n = min(n_samples, len(vox_feats))
    idx = rng.choice(len(vox_feats), size=n, replace=False)
    vox_feats  = vox_feats[idx]
    vox_labels = vox_labels[idx]
    print(f"  Voxel subsample: {len(vox_feats)} voxels  (requested {n_samples})")

    print("  Computing pairwise cosine similarity matrix ...")
    sim = _cosine_sim_matrix(vox_feats)

    purity_k = {}
    for k in ks:
        k_eff = min(k, len(vox_labels) - 1)
        overall, pc = _knn_purity(sim, vox_labels, k_eff)
        purity_k[k] = (overall, pc)
        print(f"  k={k:2d}  purity={overall:.3f}")

    plot_purity(purity_k, out_dir, tag, "knn_voxel_purity.png", "Voxel-level")

    k_eff = min(knn_k, len(vox_labels) - 1)
    print(f"  Computing k-NN predictions (k={knn_k}) ...")
    preds = _knn_predict(sim, vox_labels, k_eff)
    plot_confusion(preds, vox_labels, knn_k, out_dir, tag,
                   "knn_voxel_confusion.png", "Voxel-level")

    print("  Running dimensionality reduction on voxel subsample ...")
    emb, rname = _reduce_2d(vox_feats, method=reducer)
    plot_scatter(emb, vox_labels, rname, out_dir, tag,
                 "knn_voxel_scatter.png", "Voxel-level",
                 alpha=0.15, s=2.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="k-NN analysis of MAE backbone features in cosine space."
    )
    parser.add_argument("npz_path", help="Path to features .npz produced by extract_features.py")
    parser.add_argument("--out_dir", default="",
                        help="Output directory (default: same dir as .npz)")
    parser.add_argument("--ks", default="1,5,10,20",
                        help="Comma-separated k values for purity (default: 1,5,10,20)")
    parser.add_argument("--knn_k", type=int, default=5,
                        help="k for confusion-matrix majority vote (default: 5)")
    parser.add_argument("--reducer", default="auto", choices=["auto", "umap", "tsne"],
                        help="Dimensionality reduction method (default: auto → UMAP then t-SNE)")
    parser.add_argument("--voxel_knn", action="store_true",
                        help="Also run the voxel-level k-NN analysis")
    parser.add_argument("--n_voxel_samples", type=int, default=30000,
                        help="Voxels to subsample for voxel-level analysis (default: 30000)")
    args = parser.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = [int(k) for k in args.ks.split(",")]

    print(f"Loading {npz_path}")
    data     = np.load(npz_path)
    feats    = data["backbone_features"]   # [N_voxels, D]
    labels   = data["labels"].astype(int)  # [N_images]
    offsets  = data["offsets"]             # [N_images+1]

    print(f"  Active voxels : {feats.shape[0]}   Feature dim: {feats.shape[1]}")
    print(f"  Images        : {len(labels)}")
    valid_labels = labels[labels >= 0]
    print(f"  Labelled images: {len(valid_labels)}")
    print(f"  Class counts  : { {CLASS_NAMES[c]: int((valid_labels==c).sum()) for c in np.unique(valid_labels) if 0 <= c < len(CLASS_NAMES)} }")
    print(f"  Output dir    : {out_dir}/")

    tag = npz_path.stem

    run_image_level(feats, labels, offsets,
                    out_dir, tag, ks=ks, knn_k=args.knn_k, reducer=args.reducer)

    if args.voxel_knn:
        run_voxel_level(feats, labels, offsets,
                        out_dir, tag, ks=ks, knn_k=args.knn_k,
                        n_samples=args.n_voxel_samples, reducer=args.reducer)

    print("\nDone.")


if __name__ == "__main__":
    main()
