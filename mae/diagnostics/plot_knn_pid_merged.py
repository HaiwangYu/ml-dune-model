"""
Pixel-level PID k-NN analysis on the **merged 3-class** taxonomy used by the
SFT pixel-PID head (mae/scripts/train_mae.py):

  track  = μ±, proton, π±
  shower = e±, shower-γ
  other  = blip-γ, everything else

Reuses the blip detection from mae.diagnostics.plot_knn_pid; reuses the
extracted-features .npz (must have been extracted with --pixel_truth so that
pid_labels is present).

Outputs (in --out_dir, default = dir of the .npz):
  knn_pid_merged_eff.png        per-class **efficiency** (recall)    from majority-vote k-NN
  knn_pid_merged_pur.png        per-class **purity**     (precision) from majority-vote k-NN
  knn_pid_merged_confusion.png  row-normalised confusion matrix at k=--knn_k

Both plots are grouped bars across the configured k values, with an "Overall"
column (efficiency = accuracy; purity = micro-purity = accuracy).

Usage:
    python -m mae.diagnostics.plot_knn_pid_merged path/to/features_ep<N>.npz
    python -m mae.diagnostics.plot_knn_pid_merged path/to/features_ep<N>.npz \\
        --max_pixels_per_class=20000 --ks=1,5,10,20 --device=cuda
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mae.diagnostics.plot_knn_pid import (
    _classify_gammas,
    _knn_predict_batched,
    _BLIP_CONNECT_DIST, _BLIP_MAX_PIXELS,
    _MU_PDGS, _EM_PDGS, _GAMMA_PDG, _PROTON_PDGS, _PION,
    CLS_EM, CLS_BLIP,
)

# track = μ + proton + π±
_TRACK_PDGS = _MU_PDGS | _PROTON_PDGS | _PION
# shower (e±) is the only PDG-driven part of "shower"; γ are handled below
_ELEC_PDGS  = _EM_PDGS


# Merged class taxonomy
MERGED_CLASS_NAMES = ["track", "shower", "other"]
MERGED_N_CLASSES   = len(MERGED_CLASS_NAMES)
MERGED_TRACK  = 0
MERGED_SHOWER = 1
MERGED_OTHER  = 2


def _pdg_to_merged_class(pdg: np.ndarray, gamma_cls: np.ndarray) -> np.ndarray:
    """Map PDG codes (+ per-pixel gamma cluster classification from
    _classify_gammas) to merged class indices.  pdg == 0 → -1 (ignored)."""
    out = np.full(len(pdg), MERGED_OTHER, dtype=np.int32)
    out[np.isin(pdg, list(_TRACK_PDGS))] = MERGED_TRACK
    out[np.isin(pdg, list(_ELEC_PDGS))]  = MERGED_SHOWER
    gamma_mask = pdg == _GAMMA_PDG
    out[gamma_mask & (gamma_cls == CLS_EM)]   = MERGED_SHOWER
    out[gamma_mask & (gamma_cls == CLS_BLIP)] = MERGED_OTHER
    out[pdg == 0] = -1
    return out


def _collect(feats, merged_cls, offsets, max_per_class, seed):
    rng = np.random.default_rng(seed)
    n_images  = len(offsets) - 1
    n_classes = MERGED_N_CLASSES
    pools  = [[] for _ in range(n_classes)]
    counts = np.zeros(n_classes, dtype=np.int64)

    for img_idx in rng.permutation(n_images):
        if counts.min() >= max_per_class:
            break
        sl  = slice(int(offsets[img_idx]), int(offsets[img_idx + 1]))
        cls = merged_cls[sl]
        f   = feats[sl]
        for c in range(n_classes):
            need = max_per_class - int(counts[c])
            if need <= 0:
                continue
            mask = cls == c
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


def _confusion_eff_pur(true_lbls: np.ndarray, preds: np.ndarray, n_classes: int):
    """Return (eff_per_class, pur_per_class) np.float arrays of length n_classes,
    plus overall accuracy.  eff[c] = TP/row, pur[c] = TP/col; NaN where denom=0.
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (true_lbls, preds), 1)
    diag = cm.diagonal().astype(np.float64)
    row  = cm.sum(axis=1).astype(np.float64)
    col  = cm.sum(axis=0).astype(np.float64)
    eff  = np.where(row > 0, diag / np.maximum(row, 1), np.nan)
    pur  = np.where(col > 0, diag / np.maximum(col, 1), np.nan)
    acc  = float(diag.sum() / max(cm.sum(), 1))
    return eff, pur, acc


def _plot_confusion(true_lbls: np.ndarray, preds: np.ndarray, k: int,
                    out_dir: Path, tag: str, fname: str):
    """Row-normalised confusion matrix plot (matches the style of plot_knn_pid)."""
    from sklearn.metrics import confusion_matrix
    n_classes = MERGED_N_CLASSES
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    cm = confusion_matrix(true_lbls, preds,
                          labels=list(range(n_classes)),
                          normalize="true")
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(MERGED_CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(MERGED_CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if cm[i, j] > 0.5 else "black")
    fig.suptitle(f"Merged 3-class pixel-PID confusion (k={k})  [{tag}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def _plot_metric(metric_k: dict, kind: str, out_dir: Path, tag: str, fname: str):
    """metric_k: {k: (per_class_array[n_classes], overall_scalar)}.  kind: 'eff'|'pur'."""
    ks = sorted(metric_k.keys())
    n_classes = MERGED_N_CLASSES
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    x      = np.arange(n_classes + 1)
    width  = 0.8 / len(ks)
    xlbls  = list(MERGED_CLASS_NAMES) + ["Overall"]

    for ki, k in enumerate(ks):
        per_class, overall = metric_k[k]
        values = list(per_class) + [overall]
        ax.bar(x + ki * width - 0.4 + width / 2, values, width=width, label=f"k={k}")
    chance = 1.0 / n_classes
    ax.axhline(chance, color="red", linestyle="--", linewidth=1.2,
               label=f"chance (1/{n_classes})")
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=10)
    ylabel = "Efficiency (recall)" if kind == "eff" else "Purity (precision)"
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    title_metric = "efficiency" if kind == "eff" else "purity"
    fig.suptitle(f"Merged 3-class pixel-PID k-NN {title_metric}  [{tag}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def run(
    npz_path: Path, out_dir: Path, tag: str,
    max_pixels_per_class: int, ks: list, knn_k: int,
    device: torch.device, batch_size: int, seed: int = 42,
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

    merged_cls = _pdg_to_merged_class(pid_labels, gamma_cls)
    for c, name in enumerate(MERGED_CLASS_NAMES):
        n_c = int((merged_cls == c).sum())
        print(f"  {name:<7} pixels: {n_c:,}")

    print("\nCollecting pixels (stratified by merged class) ...")
    pix_feats, pix_cls, counts = _collect(
        feats, merged_cls, offsets,
        max_per_class=max_pixels_per_class, seed=seed,
    )
    for c, name in enumerate(MERGED_CLASS_NAMES):
        print(f"  {name:<7} sampled={int((pix_cls==c).sum()):,}  (avail={int(counts[c]):,})")
    print(f"  Total sampled : {len(pix_cls):,}")

    print(f"\nComputing k-NN predictions for ks={ks} (device={device}, batch={batch_size}) ...")
    eff_k, pur_k = {}, {}
    preds_for_cm = None
    for k in ks:
        preds = _knn_predict_batched(pix_feats, pix_cls, k, device, batch_size)
        if k == knn_k:
            preds_for_cm = preds
        eff, pur, acc = _confusion_eff_pur(pix_cls, preds, MERGED_N_CLASSES)
        eff_k[k] = (eff, acc)
        pur_k[k] = (pur, acc)
        eff_str = " ".join(f"{n}:{eff[c]:.3f}" for c, n in enumerate(MERGED_CLASS_NAMES))
        pur_str = " ".join(f"{n}:{pur[c]:.3f}" for c, n in enumerate(MERGED_CLASS_NAMES))
        print(f"  k={k:2d}  acc={acc:.3f}  eff: {eff_str}  pur: {pur_str}")

    if preds_for_cm is None:
        # knn_k wasn't in --ks; compute it once specifically for the confusion plot
        preds_for_cm = _knn_predict_batched(pix_feats, pix_cls, knn_k, device, batch_size)

    _plot_metric(eff_k, "eff", out_dir, tag, "knn_pid_merged_eff.png")
    _plot_metric(pur_k, "pur", out_dir, tag, "knn_pid_merged_pur.png")
    _plot_confusion(pix_cls, preds_for_cm, knn_k, out_dir, tag,
                    "knn_pid_merged_confusion.png")


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Merged 3-class pixel-PID k-NN evaluation (track/shower/other)."
    )
    p.add_argument("npz_path",
                   help="Path to features .npz from extract_features --pixel_truth")
    p.add_argument("--out_dir", default="")
    p.add_argument("--max_pixels_per_class", type=int, default=20_000)
    p.add_argument("--ks", default="1,5,10,20")
    p.add_argument("--knn_k", type=int, default=5,
                   help="k used for the confusion-matrix plot (default: 5)")
    p.add_argument("--device", default="")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--blip_connect_dist", type=float, default=_BLIP_CONNECT_DIST)
    p.add_argument("--blip_max_pixels", type=int, default=_BLIP_MAX_PIXELS)
    args = p.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    run(
        npz_path=npz_path, out_dir=out_dir, tag=npz_path.stem,
        max_pixels_per_class=args.max_pixels_per_class,
        ks=[int(k) for k in args.ks.split(",")],
        knn_k=args.knn_k,
        device=device, batch_size=args.batch_size, seed=args.seed,
        blip_connect_dist=args.blip_connect_dist,
        blip_max_pixels=args.blip_max_pixels,
    )


if __name__ == "__main__":
    main()
