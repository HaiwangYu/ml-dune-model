"""
Pixel-level PID k-NN analysis of MAE backbone features.

Pixels are labelled by the primary contributing particle (frame_pid_1st) and
grouped into different classes.

The features .npz must have been extracted with --pixel_truth so that the
`pid_labels` array is present.

Gamma pixels (PDG 22) are further split into two sub-classes:
  - blip  : isolated photon deposit — small connected cluster of gamma + e±
            pixels (cluster size <= --blip_max_pixels)
  - EM    : shower-producing photon — large cluster co-located with e±

Pixels are collected with **stratified sampling**: images are visited in
random order and pixels are drawn per-class until each class reaches
`--max_pixels_per_class`.

Produces (in --out_dir):
  knn_pid_purity.png     — per-class k-NN label purity at k=1,5,10,20
  knn_pid_confusion.png  — confusion matrix (majority-vote, --knn_k)
  knn_pid_scatter.png    — 2-D UMAP / t-SNE scatter (only if --plot_scatter)

Usage:
    python -m mae.diagnostics.plot_knn_pid path/to/features_ep10.npz
    python -m mae.diagnostics.plot_knn_pid path/to/features_ep10.npz \\
        --max_pixels_per_class=50000 --ks=1,5,10,20 --knn_k=5 --device=cuda
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mae.diagnostics.plot_knn import _reduce_2d


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

PID_CLASS_NAMES = ["mu±", "EM", "proton", "pion", "blip", "other"]

CLS_MU     = 0
CLS_EM     = 1
CLS_PROTON = 2
CLS_PION   = 3
CLS_BLIP   = 4
CLS_OTHER  = 5

_MU_PDGS     = {13, -13}
_EM_PDGS     = {11, -11}
_GAMMA_PDG   = 22
_PROTON_PDGS = {2212}
_PION        = {211, -211}

_BLIP_CONNECT_DIST = 5.0
_BLIP_MAX_PIXELS   = 30


def _classify_gammas(
    pid_labels: np.ndarray,
    positions: np.ndarray,
    offsets: np.ndarray,
    connect_dist: float = _BLIP_CONNECT_DIST,
    blip_max_pixels: int = _BLIP_MAX_PIXELS,
) -> np.ndarray:
    """
    Per-image connected-component over gamma+e± pixels.  Gamma pixels in a
    small cluster (<= blip_max_pixels including any co-located e±) are CLS_BLIP,
    otherwise CLS_EM.  Non-gamma pixels carry sentinel -2.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    result = np.full(len(pid_labels), -2, dtype=np.int32)
    n_images = len(offsets) - 1

    for img_idx in range(n_images):
        start = int(offsets[img_idx])
        end   = int(offsets[img_idx + 1])
        pdg = pid_labels[start:end]
        pos = positions[start:end]

        gamma_local = np.where(pdg == _GAMMA_PDG)[0]
        if len(gamma_local) == 0:
            continue
        elec_local = np.where(np.isin(pdg, list(_EM_PDGS)))[0]

        em_local = np.concatenate([gamma_local, elec_local])
        n_em     = len(em_local)
        em_pos   = pos[em_local].astype(float)

        if n_em == 1:
            result[start + gamma_local[0]] = CLS_BLIP
            continue

        tree  = cKDTree(em_pos)
        pairs = tree.query_pairs(connect_dist)

        if pairs:
            ra, ca = zip(*pairs)
            ra, ca = list(ra), list(ca)
            r = ra + ca
            c = ca + ra
            adj = csr_matrix(
                (np.ones(len(r), dtype=np.float32), (r, c)), shape=(n_em, n_em)
            )
        else:
            adj = csr_matrix((n_em, n_em), dtype=np.float32)

        _, comp_labels_em = connected_components(adj, directed=False)
        unique, counts    = np.unique(comp_labels_em, return_counts=True)
        comp_size         = dict(zip(unique.tolist(), counts.tolist()))

        n_gamma = len(gamma_local)
        for local_i, comp in zip(gamma_local, comp_labels_em[:n_gamma]):
            size = comp_size[comp]
            result[start + local_i] = CLS_BLIP if size <= blip_max_pixels else CLS_EM

    return result


def _pdg_to_class(pdg: np.ndarray, gamma_cls: np.ndarray | None = None) -> np.ndarray:
    out = np.full(len(pdg), CLS_OTHER, dtype=np.int32)
    out[np.isin(pdg, list(_MU_PDGS))]     = CLS_MU
    out[np.isin(pdg, list(_EM_PDGS))]     = CLS_EM
    out[np.isin(pdg, list(_PROTON_PDGS))] = CLS_PROTON
    out[np.isin(pdg, list(_PION))]        = CLS_PION

    gamma_mask = pdg == _GAMMA_PDG
    if gamma_cls is not None:
        out[gamma_mask] = gamma_cls[gamma_mask]
    else:
        out[gamma_mask] = CLS_EM

    out[pdg == 0] = -1
    return out


# ---------------------------------------------------------------------------
# Stratified pixel collection
# ---------------------------------------------------------------------------

def _collect(
    feats: np.ndarray,
    global_cls: np.ndarray,
    offsets: np.ndarray,
    max_pixels_per_class: int,
    seed: int = 42,
) -> tuple:
    """
    Visit images in random order; fill per-class pools until each reaches
    max_pixels_per_class.  Returns:
        pix_feats : [M, D]
        pix_cls   : [M]      class labels 0..n_classes-1
        counts    : [n_classes]
    """
    rng = np.random.default_rng(seed)
    n_images  = len(offsets) - 1
    n_classes = len(PID_CLASS_NAMES)

    pools  = [[] for _ in range(n_classes)]
    counts = np.zeros(n_classes, dtype=np.int64)

    for img_idx in rng.permutation(n_images):
        if counts.min() >= max_pixels_per_class:
            break

        sl  = slice(int(offsets[img_idx]), int(offsets[img_idx + 1]))
        cls = global_cls[sl]
        f   = feats[sl]

        for c in range(n_classes):
            need = max_pixels_per_class - int(counts[c])
            if need <= 0:
                continue
            mask    = cls == c
            n_avail = int(mask.sum())
            if n_avail == 0:
                continue
            if n_avail <= need:
                pools[c].append(f[mask])
                counts[c] += n_avail
            else:
                idx = rng.choice(np.where(mask)[0], size=need, replace=False)
                pools[c].append(f[idx])
                counts[c] += need

    parts, lbl_parts = [], []
    for c in range(n_classes):
        if pools[c]:
            f_c = np.concatenate(pools[c])
            parts.append(f_c)
            lbl_parts.append(np.full(len(f_c), c, dtype=np.int64))

    return np.concatenate(parts), np.concatenate(lbl_parts), counts


# ---------------------------------------------------------------------------
# Batched k-NN
# ---------------------------------------------------------------------------

def _l2_normalise(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _knn_purity_batched(
    feats: np.ndarray, labels: np.ndarray, ks: list,
    device: torch.device, batch_size: int,
) -> dict:
    """Returns {k: (overall_purity, per_class_array, per_sample_array)}."""
    n_classes = len(PID_CLASS_NAMES)
    X    = _l2_normalise(torch.from_numpy(feats.astype(np.float32)).to(device))
    N    = X.shape[0]
    lbls = torch.from_numpy(labels.astype(np.int64)).to(device)
    max_k = min(max(ks), N - 1)

    nn_labels = torch.empty(N, max_k, dtype=torch.int64, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B   = end - start
        sim = X[start:end] @ X.T
        sim[torch.arange(B, device=device),
            torch.arange(start, end, device=device)] = -torch.inf
        _, idx = sim.topk(max_k, dim=1)
        nn_labels[start:end] = lbls[idx]

    results = {}
    for k in ks:
        k_eff   = min(k, N - 1)
        same    = (nn_labels[:, :k_eff] == lbls[:, None]).float().mean(dim=1)
        overall = float(same.mean())
        per_class = np.full(n_classes, np.nan)
        for c in np.unique(labels):
            per_class[c] = float(same[lbls == c].mean())
        results[k] = (overall, per_class, same.cpu().numpy())

    return results


def _knn_predict_batched(
    feats: np.ndarray, labels: np.ndarray, k: int,
    device: torch.device, batch_size: int,
) -> np.ndarray:
    n_classes = len(PID_CLASS_NAMES)
    X     = _l2_normalise(torch.from_numpy(feats.astype(np.float32)).to(device))
    N     = X.shape[0]
    lbls  = torch.from_numpy(labels.astype(np.int64)).to(device)
    k_eff = min(k, N - 1)

    preds = torch.empty(N, dtype=torch.int64, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B   = end - start
        sim = X[start:end] @ X.T
        sim[torch.arange(B, device=device),
            torch.arange(start, end, device=device)] = -torch.inf
        _, idx  = sim.topk(k_eff, dim=1)
        nn_lbls = lbls[idx]
        off     = torch.arange(B, device=device).unsqueeze(1) * n_classes
        flat    = (nn_lbls + off).reshape(-1)
        counts  = torch.bincount(flat, minlength=B * n_classes).reshape(B, n_classes)
        preds[start:end] = counts.argmax(dim=1)

    return preds.cpu().numpy()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_purity(purity_k, out_dir, tag, fname, title_prefix):
    ks = sorted(purity_k.keys())
    n_classes = len(PID_CLASS_NAMES)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    x      = np.arange(n_classes + 1)
    width  = 0.8 / len(ks)
    xlbls  = PID_CLASS_NAMES + ["Overall"]

    for ki, k in enumerate(ks):
        overall, per_class, _ = purity_k[k]
        values = list(per_class) + [overall]
        ax.bar(x + ki * width - 0.4 + width / 2, values, width=width, label=f"k={k}")
    chance = 1.0 / n_classes
    ax.axhline(chance, color="red", linestyle="--", linewidth=1.2,
               label=f"chance (1/{n_classes})")
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=10)
    ax.set_ylabel("Label purity")
    ax.set_ylim(0, 1.05)
    ax.set_title("Backbone")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{title_prefix} k-NN label purity  [{tag}]", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def _plot_confusion(preds, labels, k, out_dir, tag, fname, title_prefix):
    from sklearn.metrics import confusion_matrix

    n_classes = len(PID_CLASS_NAMES)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)), normalize="true")
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(PID_CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(PID_CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Backbone")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > 0.5 else "black")

    fig.suptitle(f"{title_prefix} confusion (k={k})  [{tag}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def _plot_scatter(emb, labels, reducer_name, out_dir, tag, fname, title_prefix):
    n_classes = len(PID_CLASS_NAMES)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for c in range(n_classes):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=1.5, alpha=0.2,
                   color=colors[c], label=PID_CLASS_NAMES[c], rasterized=True)
    ax.set_title("Backbone")
    ax.set_xlabel(f"{reducer_name} 1")
    ax.set_ylabel(f"{reducer_name} 2")
    ax.legend(markerscale=5, fontsize=8)

    fig.suptitle(f"{title_prefix} {reducer_name} scatter  [{tag}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    npz_path: Path,
    out_dir: Path,
    tag: str,
    max_pixels_per_class: int,
    ks: list,
    knn_k: int,
    reducer: str,
    device: torch.device,
    batch_size: int,
    seed: int = 42,
    plot_scatter: bool = False,
    blip_connect_dist: float = _BLIP_CONNECT_DIST,
    blip_max_pixels: int = _BLIP_MAX_PIXELS,
):
    print(f"Loading {npz_path}")
    data = np.load(npz_path)

    if "pid_labels" not in data:
        print("Error: 'pid_labels' not found in .npz — re-run extract_features "
              "with --pixel_truth")
        sys.exit(1)

    feats      = data["backbone_features"]
    pid_labels = data["pid_labels"].astype(np.int32)
    offsets    = data["offsets"]
    positions  = data["positions"]

    n_images = len(offsets) - 1
    n_pixels = len(pid_labels)
    n_truth  = int((pid_labels != 0).sum())
    print(f"  Images        : {n_images}")
    print(f"  Total pixels  : {n_pixels}")
    print(f"  Pixels w/ pid1: {n_truth}  ({100*n_truth/max(n_pixels,1):.1f}%)")
    print(f"  Feature dim   : {feats.shape[1]}")
    print(f"  max_pixels_per_class : {max_pixels_per_class}")

    print(f"\nClassifying gamma pixels (connect_dist={blip_connect_dist}, "
          f"blip_max_pixels={blip_max_pixels}) ...")
    gamma_cls = _classify_gammas(pid_labels, positions, offsets,
                                  connect_dist=blip_connect_dist,
                                  blip_max_pixels=blip_max_pixels)
    n_gamma        = int((pid_labels == _GAMMA_PDG).sum())
    n_blips        = int((gamma_cls == CLS_BLIP).sum())
    n_shower_gamma = int((gamma_cls == CLS_EM).sum())
    print(f"  Total gamma pixels : {n_gamma:,}")
    print(f"  Blip gammas        : {n_blips:,}  ({100*n_blips/max(n_gamma, 1):.1f}%)")
    print(f"  Shower gammas (EM) : {n_shower_gamma:,}  ({100*n_shower_gamma/max(n_gamma, 1):.1f}%)")

    global_cls = _pdg_to_class(pid_labels, gamma_cls)

    print("\nCollecting pixels (stratified by class) ...")
    pix_feats, pix_cls, counts = _collect(
        feats, global_cls, offsets,
        max_pixels_per_class=max_pixels_per_class,
        seed=seed,
    )
    print("  Class counts after sampling:")
    for c, name in enumerate(PID_CLASS_NAMES):
        print(f"    {name:<8} {int(counts[c]):>8,}  (collected {int((pix_cls==c).sum()):,})")
    print(f"  Total sampled : {len(pix_cls):,}")

    print(f"\nComputing k-NN purity (device={device}, batch={batch_size}) ...")
    purity_k = _knn_purity_batched(pix_feats, pix_cls, ks, device, batch_size)
    for k in ks:
        print(f"  k={k:2d}  purity={purity_k[k][0]:.3f}")
    _plot_purity(purity_k, out_dir, tag, "knn_pid_purity.png", "Pixel PID")

    k_eff = min(knn_k, len(pix_cls) - 1)
    print(f"\nComputing k-NN predictions (k={knn_k}) ...")
    preds = _knn_predict_batched(pix_feats, pix_cls, k_eff, device, batch_size)
    _plot_confusion(preds, pix_cls, knn_k, out_dir, tag,
                    "knn_pid_confusion.png", "Pixel PID")

    if plot_scatter:
        print("\nRunning dimensionality reduction ...")
        emb, rname = _reduce_2d(pix_feats, method=reducer)
        _plot_scatter(emb, pix_cls, rname, out_dir, tag,
                      "knn_pid_scatter.png", "Pixel PID")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pixel-level PID k-NN analysis of MAE backbone features."
    )
    parser.add_argument("npz_path",
                        help="Path to features .npz from extract_features --pixel_truth")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--max_pixels_per_class", type=int, default=50_000)
    parser.add_argument("--ks", default="1,5,10,20")
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--reducer", default="auto", choices=["auto", "umap", "tsne"])
    parser.add_argument("--device", default="")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_scatter", action="store_true")
    parser.add_argument("--blip_connect_dist", type=float, default=_BLIP_CONNECT_DIST)
    parser.add_argument("--blip_max_pixels", type=int, default=_BLIP_MAX_PIXELS)
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
        max_pixels_per_class=args.max_pixels_per_class,
        ks=[int(k) for k in args.ks.split(",")],
        knn_k=args.knn_k,
        reducer=args.reducer,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
        plot_scatter=args.plot_scatter,
        blip_connect_dist=args.blip_connect_dist,
        blip_max_pixels=args.blip_max_pixels,
    )


if __name__ == "__main__":
    main()
