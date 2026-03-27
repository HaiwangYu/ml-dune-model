"""
Reproduce all debug plots from a saved MAE histories.json.

Usage:
    python mae/diagnostics/plot_histories.py path/to/histories.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def compute_eigen(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues, eigenvectors) for a covariance matrix.
    Uses eigh (symmetric), so eigenvalues are real and in ascending order.
    """
    return np.linalg.eigh(mat)


def pr_from_eigen(vals: np.ndarray) -> float:
    """Participation ratio from eigenvalues: (sum λ)² / sum λ².
    Clips to zero to guard against tiny negative values from floating-point rounding.
    """
    v = vals.clip(0)
    denom = float((v ** 2).sum())
    return float(v.sum() ** 2 / denom) if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# Plot 1: loss_train + loss_val
# ---------------------------------------------------------------------------

def plot_loss(data: dict, out_dir: Path):
    loss_train = data.get("loss_train", [])
    loss_val   = data.get("loss_val", {})
    if not loss_train:
        print("  [skip] loss_train is empty")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_train, linewidth=1.0, alpha=0.8, label="Train (per batch)")
    if loss_val and loss_val.get("iter"):
        ax.plot(
            loss_val["iter"], loss_val["loss"],
            "o-", linewidth=2.0, markersize=5, label="Val (per epoch)",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Weighted L1 loss")
    ax.set_title("SSL Reconstruction Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  saved loss_curve.png")


# ---------------------------------------------------------------------------
# Plot 2: feature variance (scalar) over iterations
# ---------------------------------------------------------------------------

def plot_feat_var(data: dict, out_dir: Path):
    h = data.get("stats", {})
    iters    = h.get("iter", [])
    feat_var = h.get("feat_var", [])
    if not iters:
        print("  [skip] stats is empty")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iters, feat_var, linewidth=1.5, color="C0")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean per-feature variance")
    ax.set_title("Backbone Feature Variance  (low → dimensional collapse)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "feat_var.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  saved feat_var.png")


# ---------------------------------------------------------------------------
# Plot 3: covariance heatmap at first and last snapshots
# ---------------------------------------------------------------------------

def plot_cov_heatmap(data: dict, out_dir: Path):
    h     = data.get("stats", {})
    mats  = h.get("feat_cov", [])
    iters = h.get("iter", [])
    if not mats:
        print("  [skip] feat_cov is empty")
        return

    snapshots = [(0, "first")] if len(mats) == 1 else [(0, "first"), (-1, "last")]

    def to_corr(mat: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.diag(mat)).clip(1e-8)
        return mat / np.outer(std, std)

    for idx, label in snapshots:
        cov  = np.array(mats[idx])
        corr = to_corr(cov)
        title_suffix = f"iter {iters[idx]}" if iters else label

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        im0 = axes[0].imshow(cov, cmap="viridis", aspect="auto")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_title(f"Backbone feature covariance ({title_suffix})")
        axes[0].set_xlabel("Feature index")
        axes[0].set_ylabel("Feature index")

        im1 = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title(f"Backbone feature correlation ({title_suffix})")
        axes[1].set_xlabel("Feature index")
        axes[1].set_ylabel("Feature index")

        fig.tight_layout()
        fname = f"cov_heatmap_{label}.png"
        fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Plot 4: eigenvalue spectrum + covariance in eigenbasis
# ---------------------------------------------------------------------------

def plot_eigen(data: dict, out_dir: Path):
    h     = data.get("stats", {})
    mats  = h.get("feat_cov", [])
    iters = h.get("iter", [])
    if not mats:
        print("  [skip] feat_cov is empty (eigen)")
        return

    snapshots = [(0, "first")] if len(mats) == 1 else [(0, "first"), (-1, "last")]
    all_eigen = [compute_eigen(np.array(m)) for m in mats]

    # participation ratio over time
    pr = [pr_from_eigen(vals) for vals, _ in all_eigen]

    for idx, label in snapshots:
        cov          = np.array(mats[idx])
        vals, vecs   = all_eigen[idx]
        title_suffix = f"iter {iters[idx]}" if iters else label

        cov_eigen = (vecs.T @ cov @ vecs) / max(vals[-1], 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(np.arange(len(vals)), vals, width=1.0)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Eigenvalue index (ascending)")
        axes[0].set_ylabel("Eigenvalue")
        axes[0].set_title(f"Backbone covariance eigenvalue spectrum ({title_suffix})")
        axes[0].grid(True, alpha=0.3)

        vmax = np.abs(cov_eigen).max()
        im = axes[1].imshow(cov_eigen, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title(f"Covariance in eigenbasis ({title_suffix})")
        axes[1].set_xlabel("Eigenvector index")
        axes[1].set_ylabel("Eigenvector index")

        fig.tight_layout()
        fname = f"eigen_{label}.png"
        fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")

    # Participation ratio over training
    if len(pr) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(iters, pr, linewidth=1.5, color="C1")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Effective rank  [1, D]")
        ax.set_title("Backbone Participation Ratio  (low → few dominant channels)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "participation_ratio.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("  saved participation_ratio.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python mae/diagnostics/plot_histories.py path/to/histories.json")
        sys.exit(1)

    json_path = Path(sys.argv[1]).resolve()
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)

    out_dir = json_path.parent
    print(f"Reading {json_path}")
    print(f"Saving plots to {out_dir}/")

    with open(json_path) as f:
        data = json.load(f)

    plot_loss(data, out_dir)
    plot_feat_var(data, out_dir)
    plot_cov_heatmap(data, out_dir)
    plot_eigen(data, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
