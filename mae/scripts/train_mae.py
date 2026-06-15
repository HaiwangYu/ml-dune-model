"""
train_mae.py — Sparse Masked Auto-Encoder training: SSL epochs + SFT epochs.

Architecture
------------
  Backbone  : MinkUNetSparseAttentionCore  (Voxels[1ch] → Voxels[64ch])
  SSL head  : SparseConv2d(64, 1)          charge reconstruction
  SFT head  : SparseCNNHead(in_ch=64)      neutrino-flavour classification on backbone features
  Ref head  : SparseCNNHead(in_ch=1)       same architecture, applied to raw charge (no backbone)

Training loop
-------------
  For each SSL epoch:
    1. One full pass through ssl_train split (backbone + SSL head updated).
    2. Validation pass through ssl_val split (no grad, same loss).
    3. Both SFT heads are reset to random weights.
    4. n_sft_epochs_per_ssl_epoch full passes through sft_dataset,
       training both heads in parallel (backbone frozen).
  Both heads and their optimizers are recreated fresh each SSL epoch
  for an unbiased comparison of SSL features vs. raw charge.

  After each SSL epoch a PNG is saved comparing:
    original (unmasked) | masked input | reconstructed output
  for the first batch of that epoch.

SFT classes
-----------
  0  numuCC  (nu_pdg=14, nu_ccnc=0)
  1  nueCC   (nu_pdg=12, nu_ccnc=0)
  2  NC      (nu_ccnc=1)
 -1  skip

Usage
-----
  python mae/scripts/train_mae.py               # defaults
  python mae/scripts/train_mae.py --epochs=50 --batch_size=32
"""

import inspect
import json
import sys
import logging
from dataclasses import asdict
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import warp as wp
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR


# Suppress warpconvnet kernel-map cache-miss warnings.
# In true-MAE mode the visible coordinate set changes every batch (random
# masking), so the kernel map can never be reused — the warning fires once
# per conv layer per batch and is expected, not an error.
logging.getLogger("warpconvnet").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# GPU selection helper
# ---------------------------------------------------------------------------

def _least_occupied_cuda_device() -> torch.device:
    """Return the CUDA device with the most free VRAM, or CPU if none available."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n = torch.cuda.device_count()
    best_idx, best_free = 0, 0
    for i in range(n):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free, best_idx = free, i
    dev = torch.device(f"cuda:{best_idx}")
    print(f"Selected {dev}  ({best_free / 2**30:.1f} GB free"
          f" of {n} GPU{'s' if n > 1 else ''})")
    return dev


# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # mae/

from models.mae_model import (
    SparseMAEModel, SparseTrueMAEModel,
    voxels_to_device, log1p_voxels, expm1_voxels,
    DensePixelHead,
    PIXEL_PID_CLASS_NAMES, PIXEL_PID_N_CLASSES,
)
from models.sparse_masking import sparse_block_mask, sparse_block_mask_visible, sparse_patch_mask_visible, sparse_grid_patch_mask_visible
from loader.apa_sparse_dataset import APASparseDataset
from loader.apa_sparse_meta_dataset import APASparseMetaDataset
from loader.sft_pixel_pid_dataset import SFTPixelPIDDataset
from loader.collate import voxels_collate_fn, voxels_pixel_label_collate_fn
from mae.config import MAEConfig
from mae.debug import MAEDebugger
from mae.diagnostics.svm_probe import svm_probe


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss for multi-class classification.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Reduces to cross-entropy when gamma=0.  gamma=2 is a standard default
    that strongly down-weights easy (high-confidence correct) examples and
    concentrates training on rare / hard classes such as nueCC.
    """
    ce  = F.cross_entropy(logits, targets, reduction="none")   # (N,)
    pt  = torch.exp(-ce)                                        # confidence on correct class
    return ((1.0 - pt) ** gamma * ce).mean()


def vicreg_loss(
    features: torch.Tensor,
    lambda_v: float = 25.0,
    lambda_c: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    VICReg variance + covariance regularization to prevent dimensional collapse.

    Variance term:   penalizes dimensions whose std < gamma.
    Covariance term: penalizes off-diagonal entries of the feature covariance matrix.

    features : [N, D] backbone feature tensor for the current batch
    """
    if features.shape[0] < 2:
        return features.new_tensor(0.0)
    z = features - features.mean(dim=0)
    std = (z.var(dim=0) + eps).sqrt()
    loss_v = F.relu(gamma - std).mean()
    N, D = z.shape
    cov = (z.T @ z) / (N - 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    loss_c = off_diag / D
    return lambda_v * loss_v + lambda_c * loss_c


def patch_reconstruction_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    pred_charge: torch.Tensor,
    true_charge: torch.Tensor,
    patch_ids:   torch.Tensor,
    lambda_coord:  float = 1.0,
    lambda_charge: float = 1.0,
) -> torch.Tensor:
    """
    Per-patch Chamfer Distance on predicted voxel coordinates + L1 charge loss.

    For each patch, normalizes coordinates to patch-local [-1, 1] space before
    computing the symmetric Chamfer Distance, making the loss scale-invariant
    to patch dimensions.  Inspired by PoLAr-MAE (arxiv 2502.02558).

    pred_coords : [N_masked, 2]  predicted (channel, tick) — raw backbone output
    true_coords : [N_masked, 2]  ground-truth (channel, tick) — original voxel coords
    pred_charge : [N_masked]     predicted log1p charge
    true_charge : [N_masked]     ground-truth log1p charge
    patch_ids   : [N_masked]     patch index per voxel (from sparse_patch_mask_visible)
    """
    unique_patches = patch_ids.unique()
    chamfer_terms = []
    for pid in unique_patches:
        sel = patch_ids == pid
        tc = true_coords[sel].float()   # [K, 2]
        pc = pred_coords[sel].float()   # [K, 2]
        center = tc.mean(dim=0)
        scale  = (tc - center).abs().max().clamp(min=1.0)
        tc_n   = (tc - center) / scale
        pc_n   = (pc - center) / scale
        # Symmetric Chamfer Distance
        d1 = ((tc_n.unsqueeze(0) - pc_n.unsqueeze(1)) ** 2).sum(-1).min(dim=1).values
        d2 = ((pc_n.unsqueeze(0) - tc_n.unsqueeze(1)) ** 2).sum(-1).min(dim=0).values
        chamfer_terms.append((d1.mean() + d2.mean()) / 2)
    loss_coord  = torch.stack(chamfer_terms).mean() if chamfer_terms else pred_coords.new_tensor(0.0)
    loss_charge = F.l1_loss(pred_charge, true_charge)
    return lambda_coord * loss_coord + lambda_charge * loss_charge


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_confusion(cm: torch.Tensor, class_names: list[str]) -> None:
    n = len(class_names)
    width = max(len(n) for n in class_names) + 2
    header = " " * (width + 2) + "  ".join(f"{n:>{width}}" for n in class_names)
    print(f"\n  Confusion matrix  (rows = true, cols = predicted)")
    print(f"  {header}")
    for i, name in enumerate(class_names):
        row = f"  {name:>{width}}  " + "  ".join(f"{cm[i, j].item():>{width}d}" for j in range(n))
        print(row)


def _print_class_metrics(cm: torch.Tensor, class_names: list[str]) -> None:
    n = len(class_names)
    print(f"\n  Per-class metrics:")
    print(f"  {'class':>10}  {'efficiency':>12}  {'purity':>10}")
    for i in range(n):
        tp = cm[i, i].item()
        fn = cm[i, :].sum().item() - tp
        fp = cm[:, i].sum().item() - tp
        eff = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        pur = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        print(f"  {class_names[i]:>10}  {eff:>12.4f}  {pur:>10.4f}")


# ---------------------------------------------------------------------------
# SSL Visualization
# ---------------------------------------------------------------------------

def _sparse_to_dense(coords: torch.Tensor, feats: torch.Tensor) -> "np.ndarray | None":
    """
    Convert sparse (N, 2) int coords and (N, 1) float feats to a dense 2-D array.

    coords[:, 0] = channel,  coords[:, 1] = tick.
    Returns float32 numpy array of shape (H, W), or None if empty.
    """
    if len(coords) == 0:
        return None
    ch = coords[:, 0]
    tk = coords[:, 1]
    ch_min, ch_max = int(ch.min()), int(ch.max())
    tk_min, tk_max = int(tk.min()), int(tk.max())
    H = ch_max - ch_min + 1
    W = tk_max - tk_min + 1
    grid = torch.zeros(H, W)
    grid[ch - ch_min, tk - tk_min] = feats[:, 0]
    return grid.numpy()


def _visualize_ssl(original_vox, masked_vox, pred_vox, epoch: int, viz_dir: Path) -> None:
    """
    Save a 3-panel PNG: original (unmasked) | masked input | reconstructed output.

    Uses the first event in the batch.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [viz] matplotlib not available — skipping")
        return

    offsets = original_vox.offsets
    if len(offsets) < 2:
        return
    end = int(offsets[1].item())
    if end == 0:
        return

    # All three share the same coordinate structure (masking/conv preserve coords).
    coords = original_vox.coordinate_tensor[:end].cpu().int()
    orig_dense   = _sparse_to_dense(coords, original_vox.feature_tensor[:end].cpu())
    masked_dense = _sparse_to_dense(coords, masked_vox.feature_tensor[:end].cpu())
    pred_dense   = _sparse_to_dense(coords, pred_vox.feature_tensor[:end].cpu())

    if orig_dense is None:
        return

    vmax = float(orig_dense.max()) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Original (unmasked)", "Masked input", "Reconstructed output"]
    for ax, img, title in zip(axes, [orig_dense, masked_dense, pred_dense], titles):
        im = ax.imshow(img, aspect="auto", origin="lower", vmin=0.0, vmax=vmax, cmap="cubehelix_r")
        ax.set_title(title)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Channel")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"SSL Reconstruction — Epoch {epoch}", fontsize=13)
    plt.tight_layout()
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / f"ssl_viz_epoch{epoch:04d}.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [viz] Saved: {out_path}")


# ---------------------------------------------------------------------------
# One SSL train epoch
# ---------------------------------------------------------------------------

def _train_ssl_epoch(
    model, ssl_loader, opt_ssl,
    device, masking_frac, win_ch, win_tick,
    epoch, debugger: MAEDebugger, iteration_offset: int,
    viz_dir: Path, viz_batch: int = 0,
    true_mae: bool = False,
    vicreg_lambda_v: float = 0.0,
    vicreg_lambda_c: float = 0.0,
    use_patch_mae: bool = False,
    patch_ch: int = 15,
    patch_tick: int = 25,
    patch_mask_frac: float = 0.60,
    lambda_coord: float = 1.0,
    lambda_charge: float = 1.0,
    mask_mode: str = "block",
) -> tuple[list[float], int]:
    """
    Run one SSL training epoch.

    Returns
    -------
    ssl_losses       : per-batch loss values
    iteration_offset : updated global iteration counter after this epoch
    """
    model.train()
    ssl_losses = []
    viz_vox_cpu    = None   # fallback: batch 0
    viz_vox_target = None   # desired: batch viz_batch

    for batch_idx, vox_cpu in enumerate(ssl_loader):
        iteration = iteration_offset + batch_idx

        if batch_idx == 0 and vox_cpu is not None:
            viz_vox_cpu = vox_cpu
        if batch_idx == viz_batch and vox_cpu is not None:
            viz_vox_target = vox_cpu

        vox = voxels_to_device(vox_cpu, device)

        if vox.feature_tensor.shape[0] == 0:
            continue

        if use_patch_mae:
            vox_in, mask_bool, patch_ids = sparse_patch_mask_visible(
                vox, patch_ch, patch_tick, patch_mask_frac)
        elif true_mae and mask_mode == "grid_patch":
            # rec #3: non-overlapping grid-cell patches (mirrors polarmae groups)
            vox_in, mask_bool = sparse_grid_patch_mask_visible(
                vox, masking_frac, win_ch, win_tick)
        elif true_mae:
            vox_in, mask_bool = sparse_block_mask_visible(vox, masking_frac, win_ch, win_tick)
        else:
            vox_in, mask_bool = sparse_block_mask(vox, masking_frac, win_ch, win_tick)
        if batch_idx == 0 and epoch == 1:
            print(f"  [mask] effective masking rate: {mask_bool.float().mean():.1%}")

        # Backbone forward — always TrueMAE-style when use_patch_mae or true_mae.
        if use_patch_mae or true_mae:
            backbone_feats = model.backbone(vox_in, vox)  # Voxels [N_union, 64]
        else:
            backbone_feats = model.backbone(vox_in)        # Voxels [N_all, 64]

        if mask_bool.any():
            if use_patch_mae:
                # Patch-MAE: Chamfer coordinate loss + charge loss
                pred_charge_vox = model.charge_head(backbone_feats)
                pred_coord_vox  = model.coord_head(backbone_feats)
                masked_patch_ids = patch_ids[mask_bool]
                loss = patch_reconstruction_loss(
                    pred_coord_vox.feature_tensor[mask_bool],
                    vox.coordinate_tensor[mask_bool].float(),
                    pred_charge_vox.feature_tensor[mask_bool, 0],
                    vox.feature_tensor[mask_bool, 0],
                    masked_patch_ids,
                    lambda_coord=lambda_coord,
                    lambda_charge=lambda_charge,
                )
                clip_params = (list(model.backbone.parameters())
                               + list(model.charge_head.parameters())
                               + list(model.coord_head.parameters()))
            else:
                # Standard block-mask: charge reconstruction only
                pred = model.charge_head(backbone_feats)
                per_voxel = F.l1_loss(pred.feature_tensor, vox.feature_tensor,
                                       reduction="none")[:, 0]
                loss = per_voxel[mask_bool].mean()
                clip_params = (list(model.backbone.parameters())
                               + list(model.charge_head.parameters()))

            if vicreg_lambda_v > 0 or vicreg_lambda_c > 0:
                loss = loss + vicreg_loss(
                    backbone_feats.feature_tensor,
                    lambda_v=vicreg_lambda_v,
                    lambda_c=vicreg_lambda_c,
                )
            opt_ssl.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
            opt_ssl.step()
            ssl_losses.append(loss.item())
            debugger.log_batch(epoch, batch_idx, iteration, loss.item())
            debugger.log_feature_stats(iteration, backbone_feats.feature_tensor.detach())
            debugger.maybe_save_histories(iteration)

        if (batch_idx + 1) % 50 == 0:
            ssl_mean = sum(ssl_losses) / len(ssl_losses) if ssl_losses else float("nan")
            print(
                f"  [SSL] Epoch {epoch}  step [{batch_idx + 1}/{len(ssl_loader)}]"
                f"  loss={ssl_mean:.4f}"
            )

    # ── Visualization (end-of-epoch, model eval, no grad) ─────────────────
    if viz_vox_target is None and viz_batch != 0:
        print(f"  [viz] batch {viz_batch} not reached — falling back to batch 0")
    viz_vox_cpu = viz_vox_target if viz_vox_target is not None else viz_vox_cpu
    if viz_vox_cpu is not None:
        model.eval()
        with torch.no_grad():
            # Dataset now applies log1p in __getitem__, so what arrives is
            # already log-space.  Reconstruct raw-charge views via expm1 for
            # the saved PNG only.
            vox_viz_log = voxels_to_device(viz_vox_cpu, device)
            if vox_viz_log.feature_tensor.shape[0] > 0:
                if true_mae:
                    viz_in, _ = sparse_block_mask_visible(vox_viz_log, masking_frac, win_ch, win_tick)
                    pred_viz_log = model.charge_head(model.backbone(viz_in, vox_viz_log))
                    # For visualization show a zero-out version as "masked input"
                    masked_viz_log, _ = sparse_block_mask(vox_viz_log, masking_frac, win_ch, win_tick)
                else:
                    masked_viz_log, _ = sparse_block_mask(vox_viz_log, masking_frac, win_ch, win_tick)
                    pred_viz_log = model.charge_head(model.backbone(masked_viz_log))
                vox_viz_raw    = expm1_voxels(vox_viz_log)
                masked_viz_raw = expm1_voxels(masked_viz_log)
                pred_viz_raw   = expm1_voxels(pred_viz_log)
                _visualize_ssl(vox_viz_raw, masked_viz_raw, pred_viz_raw, epoch, viz_dir)
        model.train()

    return ssl_losses, iteration_offset + len(ssl_loader)


# ---------------------------------------------------------------------------
# One SSL validation epoch (no grad)
# ---------------------------------------------------------------------------

def _val_ssl_epoch(
    model, ssl_val_loader, device, masking_frac, win_ch, win_tick,
    true_mae: bool = False,
) -> list[float]:
    """Compute SSL reconstruction loss on the validation split (no backward pass)."""
    model.eval()
    val_losses = []
    with torch.no_grad():
        for vox_cpu in ssl_val_loader:
            vox = voxels_to_device(vox_cpu, device)
            if vox.feature_tensor.shape[0] == 0:
                continue
            if true_mae:
                vox_in, mask_bool = sparse_block_mask_visible(vox, masking_frac, win_ch, win_tick)
            else:
                vox_in, mask_bool = sparse_block_mask(vox, masking_frac, win_ch, win_tick)
            if not mask_bool.any():
                continue
            if true_mae:
                pred = model.charge_head(model.backbone(vox_in, vox))
            else:
                pred = model.charge_head(model.backbone(vox_in))
            per_voxel = F.l1_loss(pred.feature_tensor, vox.feature_tensor,
                                   reduction="none")[:, 0]
            loss = per_voxel[mask_bool].mean()
            val_losses.append(loss.item())
    model.train()
    return val_losses


# ---------------------------------------------------------------------------
# One SFT epoch (trains both SSL-feature head and raw-charge reference head)
# ---------------------------------------------------------------------------

def _train_sft_epoch(
    model, sft_loader, opt_sft, opt_ref,
    device, n_classes, epoch, sft_epoch,
    focal_gamma: float = 2.0,
):
    """Pixel-level PID SFT epoch.

    sft_loader is expected to yield (Voxels, pixel_labels: LongTensor[N_total])
    where pixel_labels carries the per-voxel class index (or -1 to ignore).
    """
    model.freeze_backbone()
    sft_losses, ref_losses = [], []
    confusion_sft = torch.zeros(n_classes, n_classes, dtype=torch.long)
    confusion_ref = torch.zeros(n_classes, n_classes, dtype=torch.long)

    for step, (vox_sft_cpu, pix_labels) in enumerate(sft_loader):
        vox_sft   = voxels_to_device(vox_sft_cpu, device)
        pix_labels = pix_labels.to(device)

        if vox_sft.feature_tensor.shape[0] == 0:
            continue
        valid = pix_labels >= 0
        if not valid.any():
            continue

        # ── SSL feature head (per-pixel logits) ───────────────────────────
        logits = model.forward_sft(vox_sft)              # [N_total, n_classes]
        if logits.shape[0] != pix_labels.shape[0]:
            continue
        sft_loss = focal_loss(logits[valid], pix_labels[valid], gamma=focal_gamma)
        opt_sft.zero_grad()
        sft_loss.backward()
        opt_sft.step()
        sft_losses.append(sft_loss.item())
        preds = logits[valid].argmax(dim=1)
        confusion_sft += torch.bincount(
            pix_labels[valid] * n_classes + preds,
            minlength=n_classes * n_classes,
        ).reshape(n_classes, n_classes).cpu()

        # ── Raw-charge reference head (per-pixel logits) ──────────────────
        logits_ref = model.forward_sft_ref(vox_sft)
        if logits_ref.shape[0] != pix_labels.shape[0]:
            continue
        ref_loss = focal_loss(logits_ref[valid], pix_labels[valid], gamma=focal_gamma)
        opt_ref.zero_grad()
        ref_loss.backward()
        opt_ref.step()
        ref_losses.append(ref_loss.item())
        preds_ref = logits_ref[valid].argmax(dim=1)
        confusion_ref += torch.bincount(
            pix_labels[valid] * n_classes + preds_ref,
            minlength=n_classes * n_classes,
        ).reshape(n_classes, n_classes).cpu()

        if (step + 1) % 50 == 0:
            sft_mean  = sum(sft_losses) / len(sft_losses) if sft_losses else float("nan")
            ref_mean  = sum(ref_losses)  / len(ref_losses)  if ref_losses  else float("nan")
            total_s   = int(confusion_sft.sum())
            acc_s     = 100.0 * int(confusion_sft.diagonal().sum()) / total_s if total_s > 0 else float("nan")
            total_r   = int(confusion_ref.sum())
            acc_r     = 100.0 * int(confusion_ref.diagonal().sum()) / total_r if total_r > 0 else float("nan")
            print(
                f"  [SFT] SSL-epoch {epoch}  SFT-epoch {sft_epoch}"
                f"  step [{step + 1}/{len(sft_loader)}]"
                f"  SSL-feat: loss={sft_mean:.4f} pixel-acc={acc_s:.1f}%"
                f"  | raw-charge: loss={ref_mean:.4f} pixel-acc={acc_r:.1f}%"
            )

    model.unfreeze_backbone()
    return sft_losses, confusion_sft, ref_losses, confusion_ref


@torch.no_grad()
def _val_sft_epoch(
    model, sft_val_loader,
    device, n_classes, focal_gamma: float = 2.0,
):
    """Evaluate the current SFT heads on a held-out probe-val split.

    Mirrors _train_sft_epoch's forward pass but never updates weights.  Returns
    (val_losses, confusion_sft, val_ref_losses, confusion_ref).
    """
    model.freeze_backbone()
    val_losses, ref_losses = [], []
    confusion_sft = torch.zeros(n_classes, n_classes, dtype=torch.long)
    confusion_ref = torch.zeros(n_classes, n_classes, dtype=torch.long)

    for vox_sft_cpu, pix_labels in sft_val_loader:
        vox_sft    = voxels_to_device(vox_sft_cpu, device)
        pix_labels = pix_labels.to(device)
        if vox_sft.feature_tensor.shape[0] == 0:
            continue
        valid = pix_labels >= 0
        if not valid.any():
            continue

        logits = model.forward_sft(vox_sft)
        if logits.shape[0] != pix_labels.shape[0]:
            continue
        val_losses.append(focal_loss(logits[valid], pix_labels[valid], gamma=focal_gamma).item())
        preds = logits[valid].argmax(dim=1)
        confusion_sft += torch.bincount(
            pix_labels[valid] * n_classes + preds,
            minlength=n_classes * n_classes,
        ).reshape(n_classes, n_classes).cpu()

        logits_ref = model.forward_sft_ref(vox_sft)
        if logits_ref.shape[0] != pix_labels.shape[0]:
            continue
        ref_losses.append(focal_loss(logits_ref[valid], pix_labels[valid], gamma=focal_gamma).item())
        preds_ref = logits_ref[valid].argmax(dim=1)
        confusion_ref += torch.bincount(
            pix_labels[valid] * n_classes + preds_ref,
            minlength=n_classes * n_classes,
        ).reshape(n_classes, n_classes).cpu()

    model.unfreeze_backbone()
    return val_losses, confusion_sft, ref_losses, confusion_ref


# ---------------------------------------------------------------------------
# Offline-pool SFT (rec #B from polarmae_vs_mae comparison)
# ---------------------------------------------------------------------------
#
# Mirrors polarmae's APA2DProbeCallback._collect_pool + _train_head pattern:
#   1. Extract backbone features ONCE over the SFT train+val splits (frozen).
#   2. Cap per-class to keep the pool balanced.
#   3. Train a tiny DensePixelHead on the in-memory pool for N epochs.
#   4. Evaluate on the val pool.
# Backbone is run num_sft_batches times per SSL epoch instead of num_sft_batches
# × n_sft_epochs × 2 heads in the legacy online path → ~10× less compute.

@torch.no_grad()
def _collect_sft_pool(
    model, sft_loader, device, n_classes,
    max_pixels_per_class: int = 5000,
    seed: int = 0,
):
    """Drain `sft_loader` once with the backbone frozen; pool per-voxel backbone
    features + raw charges + per-voxel pixel-PID class.  Stratified per-class
    sampling so the pool stays balanced even when one class dominates the data.

    Returns
    -------
    feats : np.float32 [M, D]   backbone features at pooled voxels
    raws  : np.float32 [M, 3]   raw (channel, tick, log_charge) at same voxels
    cls   : np.int64   [M]      pixel-PID class index 0..n_classes-1 (ignore -1 dropped)
    counts: np.int64   [n_classes]  total seen per class (before subsample cap)
    """
    rng = np.random.default_rng(seed)
    pool_feats = [[] for _ in range(n_classes)]
    pool_raws  = [[] for _ in range(n_classes)]
    pool_counts = np.zeros(n_classes, dtype=np.int64)
    cap = int(max_pixels_per_class)

    model.eval()
    for vox_cpu, pix_labels in sft_loader:
        if int(pool_counts.min()) >= cap:
            break
        vox = voxels_to_device(vox_cpu, device)
        if vox.feature_tensor.shape[0] == 0:
            continue
        # Use the same forward_sft path the online SFT uses — it handles both
        # SparseMAEModel and SparseTrueMAEModel signatures internally.  But
        # we want raw backbone features (before the head), so call backbone
        # directly (forward_sft would route through pixel_pid_head).
        if hasattr(model.backbone, "forward") and "vox_union" in model.backbone.forward.__code__.co_varnames:
            backbone_out = model.backbone(vox, vox)
        else:
            backbone_out = model.backbone(vox)
        feats = backbone_out.feature_tensor.detach().float().cpu().numpy()       # [N_total, D]
        coords = backbone_out.coordinate_tensor.detach().cpu().numpy()           # [N_total, 2]
        # Raw charge per voxel (vox_cpu carries log1p'd, undo it).
        raw_chg = torch.expm1(vox_cpu.feature_tensor.float()).cpu().numpy()       # [N_total, 1]
        raw = np.concatenate([coords.astype(np.float32),
                              np.log1p(raw_chg.astype(np.float32))], axis=1)     # [N_total, 3]

        labels = pix_labels.cpu().numpy() if hasattr(pix_labels, "cpu") else np.asarray(pix_labels)
        if labels.shape[0] != feats.shape[0]:
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
                pool_raws[c].append(raw[mask])
                pool_counts[c] += n_avail
            else:
                idx = rng.choice(np.where(mask)[0], size=need, replace=False)
                pool_feats[c].append(feats[idx])
                pool_raws[c].append(raw[idx])
                pool_counts[c] += need

    F_parts, R_parts, L_parts = [], [], []
    for c in range(n_classes):
        if pool_feats[c]:
            f_c = np.concatenate(pool_feats[c])
            r_c = np.concatenate(pool_raws[c])
            F_parts.append(f_c)
            R_parts.append(r_c)
            L_parts.append(np.full(len(f_c), c, dtype=np.int64))
    if not F_parts:
        return (np.zeros((0, 64), dtype=np.float32),
                np.zeros((0, 3),  dtype=np.float32),
                np.zeros((0,),    dtype=np.int64),
                pool_counts)
    return (np.concatenate(F_parts).astype(np.float32),
            np.concatenate(R_parts).astype(np.float32),
            np.concatenate(L_parts).astype(np.int64),
            pool_counts)


def _fit_dense_head(
    X: np.ndarray, y: np.ndarray, n_classes: int, device,
    *, epochs: int = 30, batch_size: int = 256, lr: float = 5e-3, seed: int = 0,
):
    """Train a fresh DensePixelHead on (X, y) for `epochs` epochs.  Returns
    the trained head + per-epoch train accuracy."""
    torch.manual_seed(seed)
    head = DensePixelHead(in_ch=X.shape[1], n_classes=n_classes).to(device)
    opt  = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)
    N = X_t.shape[0]
    perm_rng = np.random.default_rng(seed)
    accs = []
    for ep in range(epochs):
        head.train()
        order = torch.from_numpy(perm_rng.permutation(N)).to(device)
        correct = 0
        for s in range(0, N, batch_size):
            idx = order[s:s + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                correct += int((logits.argmax(dim=1) == yb).sum())
        accs.append(correct / max(N, 1))
    return head, accs


@torch.no_grad()
def _predict_dense_head(head, X: np.ndarray, device, batch_size: int = 2048) -> np.ndarray:
    """Run a trained DensePixelHead on X (chunked) and return argmax preds."""
    head.eval()
    X_t = torch.from_numpy(X).to(device)
    out = torch.empty(X_t.shape[0], dtype=torch.long, device=device)
    for s in range(0, X_t.shape[0], batch_size):
        out[s:s + batch_size] = head(X_t[s:s + batch_size]).argmax(dim=1)
    return out.cpu().numpy()


def _confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    """Return (confusion[n_classes, n_classes], per_class_eff, per_class_pur,
    overall_acc, macro_f1).  All ignore -1 labels."""
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


def _fit_svm_on_pool(X_tr, y_tr, X_va, y_va, n_classes, *, C: float = 1.0, seed: int = 0):
    """sklearn LinearSVC fit + score on a feature pool.  Returns the same
    metric dict structure as the dense-head probes."""
    from sklearn.svm import LinearSVC
    svm = LinearSVC(C=C, class_weight="balanced", random_state=seed)
    svm.fit(X_tr, y_tr)
    pred_tr = svm.predict(X_tr)
    pred_va = svm.predict(X_va)
    cm_tr, eff_tr, pur_tr, acc_tr, f1_tr = _confusion_and_metrics(y_tr, pred_tr, n_classes)
    cm_va, eff_va, pur_va, acc_va, f1_va = _confusion_and_metrics(y_va, pred_va, n_classes)
    return {
        "train_acc": acc_tr, "val_acc": acc_va,
        "train_macro_f1": f1_tr, "val_macro_f1": f1_va,
        "train_cm": cm_tr, "val_cm": cm_va,
        "train_eff": eff_tr, "val_eff": eff_va,
        "train_pur": pur_tr, "val_pur": pur_va,
    }


def _run_offline_sft(
    model, sft_train_loader, sft_val_loader, device, class_names,
    *, epochs: int = 30, batch_size: int = 256, lr: float = 5e-3,
    max_pixels_per_class: int = 5000, seed: int = 0,
    svm_C: float = 1.0,
):
    """End-to-end offline SFT + SVM for one SSL epoch.  Pools features once
    on train+val, runs **four** probes from the same pool:
        sft_feat, sft_raw, voxel_svm_feat, voxel_svm_raw
    matching polarmae's APA2DProbeCallback.  Returns a dict ready to drop
    into sft_history.json under the 'offline_sft' key.
    """
    n_classes = len(class_names)
    print(f"  [SFT offline] collecting train pool (cap {max_pixels_per_class}/class) ...")
    X_tr, R_tr, y_tr, cnt_tr = _collect_sft_pool(
        model, sft_train_loader, device, n_classes,
        max_pixels_per_class=max_pixels_per_class, seed=seed,
    )
    print(f"  [SFT offline] collecting val pool ...")
    X_va, R_va, y_va, cnt_va = _collect_sft_pool(
        model, sft_val_loader, device, n_classes,
        max_pixels_per_class=max_pixels_per_class, seed=seed + 1,
    )

    out = {"n_train": int(len(y_tr)), "n_val": int(len(y_va))}

    # --- SFT (dense MLP) on backbone features
    head_feat, _ = _fit_dense_head(X_tr, y_tr, n_classes, device,
                                   epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    pred_tr = _predict_dense_head(head_feat, X_tr, device)
    pred_va = _predict_dense_head(head_feat, X_va, device)
    cm_tr, eff_tr, pur_tr, acc_tr, f1_tr = _confusion_and_metrics(y_tr, pred_tr, n_classes)
    cm_va, eff_va, pur_va, acc_va, f1_va = _confusion_and_metrics(y_va, pred_va, n_classes)
    out["sft_feat"] = {
        "train_acc": acc_tr, "val_acc": acc_va,
        "train_macro_f1": f1_tr, "val_macro_f1": f1_va,
        "train_cm": cm_tr, "val_cm": cm_va,
        "train_eff": eff_tr, "val_eff": eff_va,
        "train_pur": pur_tr, "val_pur": pur_va,
    }

    # --- SFT (dense MLP) on raw (channel, tick, log_charge)
    head_raw, _ = _fit_dense_head(R_tr, y_tr, n_classes, device,
                                  epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    rpred_tr = _predict_dense_head(head_raw, R_tr, device)
    rpred_va = _predict_dense_head(head_raw, R_va, device)
    rcm_tr, reff_tr, rpur_tr, racc_tr, rf1_tr = _confusion_and_metrics(y_tr, rpred_tr, n_classes)
    rcm_va, reff_va, rpur_va, racc_va, rf1_va = _confusion_and_metrics(y_va, rpred_va, n_classes)
    out["sft_raw"] = {
        "train_acc": racc_tr, "val_acc": racc_va,
        "train_macro_f1": rf1_tr, "val_macro_f1": rf1_va,
        "train_cm": rcm_tr, "val_cm": rcm_va,
        "train_eff": reff_tr, "val_eff": reff_va,
        "train_pur": rpur_tr, "val_pur": rpur_va,
    }

    # --- SVM on backbone features  (rec #C)
    out["voxel_svm_feat"] = _fit_svm_on_pool(X_tr, y_tr, X_va, y_va, n_classes, C=svm_C, seed=seed)
    # --- SVM on raw (channel, tick, log_charge)  (rec #C)
    out["voxel_svm_raw"]  = _fit_svm_on_pool(R_tr, y_tr, R_va, y_va, n_classes, C=svm_C, seed=seed)

    print(f"  [SFT offline]  sft_feat:       train_acc={acc_tr:.3f}  val_acc={acc_va:.3f}  val_macro_f1={f1_va:.3f}")
    print(f"  [SFT offline]  sft_raw:        train_acc={racc_tr:.3f}  val_acc={racc_va:.3f}  val_macro_f1={rf1_va:.3f}")
    print(f"  [SFT offline]  voxel_svm_feat: val_acc={out['voxel_svm_feat']['val_acc']:.3f}  val_macro_f1={out['voxel_svm_feat']['val_macro_f1']:.3f}")
    print(f"  [SFT offline]  voxel_svm_raw:  val_acc={out['voxel_svm_raw']['val_acc']:.3f}  val_macro_f1={out['voxel_svm_raw']['val_macro_f1']:.3f}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_root                  = "/nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27",
    sft_data_root              = "",     # if empty, SFT reuses data_root
    apa                        = 0,
    view                       = "W",
    batch_size                 = 16,
    epochs                     = 2,
    lr                         = 1e-3,
    scheduler_step             = 10,
    gamma                      = 0.7,
    n_sft_epochs_per_ssl_epoch = 3,    # full SFT epochs per SSL epoch
    masking_frac               = 0.5,
    win_ch                     = 30,
    win_tick                   = 50,
    n_classes                  = PIXEL_PID_N_CLASSES,   # pixel-PID classes (track, shower, other)
    focal_gamma                = 2.0,  # focal loss gamma; 0 = plain cross-entropy
    ssl_subset_frac            = 1.0,  # fraction of SSL dataset to use
    sft_subset_frac            = 1.0,  # fraction of SFT dataset to use
    val_frac                   = 0.2,  # fraction of SSL dataset held out for validation
    sft_val_frac               = 0.2,  # fraction of SFT dataset held out for probe-eval (rec #6)
    sft_mode                   = "offline_pool",  # "offline_pool" (rec #B) or "online" (legacy)
    sft_pool_max_pixels        = 5000, # per-class cap when sft_mode=offline_pool
    sft_pool_epochs            = 30,
    sft_pool_batch             = 256,
    sft_pool_lr                = 5e-3,
    num_workers                = 0,    # set >0 only if warp is initialised in workers
    device                     = "cuda",
    checkpoints_dir            = "./checkpoints",
    save_every                 = 5,
    viz_dir                    = "./viz",
    viz_batch                  = 0,      # which batch to visualize (0-indexed); 0 if out of range
    debug_dir                  = "./debug",
    debug_every                = 50,     # how often (in iterations) to log feature stats
    true_mae                   = True,   # True → remove masked voxels from encoder input
    mask_mode                  = "block", # "block" (legacy) or "grid_patch" (rec #3)
    vicreg_lambda_v            = 0.0,   # VICReg variance weight  (0 = disabled)
    vicreg_lambda_c            = 0.0,   # VICReg covariance weight (0 = disabled)
    use_patch_mae              = False,  # True → patch-level masking + coordinate reconstruction
    patch_ch                   = 15,    # patch height in channels
    patch_tick                 = 25,    # patch width  in ticks
    patch_mask_frac            = 0.60,  # fraction of non-empty patches to mask
    lambda_coord               = 1.0,   # weight for Chamfer coordinate loss
    lambda_charge              = 1.0,   # weight for charge reconstruction loss (patch-MAE mode)
    resume                     = None,   # path to checkpoint to resume from
    run_name                   = "",     # optional label; nests outputs under run_name/ if set
    cache_dir                  = "./data",  # dataset index .pt cache directory (persist across jobs)
    backbone_name              = "",     # "" → legacy default; else sparseformer.build_backbone name
    backbone_kwargs            = None,   # extra kwargs for build_backbone (e.g. n_bottleneck_blocks)
):
    """Sparse MAE training: one SSL epoch → n_sft_epochs_per_ssl_epoch SFT epochs, repeated."""
    # If a run name is given, nest outputs under <base>/<run_name>/
    if run_name:
        checkpoints_dir = f"{checkpoints_dir}/{run_name}"
        debug_dir       = f"{debug_dir}/{run_name}"
        viz_dir         = f"{viz_dir}/{run_name}"

    # Persist a MAEConfig snapshot for offline reproducibility / from_config reload.
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    cfg = MAEConfig(**{
        k: v for k, v in {
            "run_name": run_name, "data_root": data_root, "sft_data_root": sft_data_root,
            "apa": apa, "view": view,
            "batch_size": batch_size, "num_workers": num_workers,
            "ssl_subset_frac": ssl_subset_frac, "sft_subset_frac": sft_subset_frac,
            "val_frac": val_frac, "sft_val_frac": sft_val_frac,
            "sft_mode": sft_mode,
            "sft_pool_max_pixels": sft_pool_max_pixels,
            "sft_pool_epochs": sft_pool_epochs,
            "sft_pool_batch": sft_pool_batch,
            "sft_pool_lr": sft_pool_lr,
            "epochs": epochs, "lr": lr,
            "scheduler_step": scheduler_step, "gamma": gamma,
            "n_sft_epochs_per_ssl_epoch": n_sft_epochs_per_ssl_epoch,
            "save_every": save_every, "resume": resume or "",
            "true_mae": true_mae, "mask_mode": mask_mode, "masking_frac": masking_frac,
            "win_ch": win_ch, "win_tick": win_tick,
            "use_patch_mae": use_patch_mae, "patch_ch": patch_ch,
            "patch_tick": patch_tick, "patch_mask_frac": patch_mask_frac,
            "lambda_coord": lambda_coord, "lambda_charge": lambda_charge,
            "n_classes": n_classes, "focal_gamma": focal_gamma,
            "vicreg_lambda_v": vicreg_lambda_v, "vicreg_lambda_c": vicreg_lambda_c,
            "checkpoints_dir": checkpoints_dir, "debug_dir": debug_dir,
            "debug_every": debug_every, "viz_dir": viz_dir, "viz_batch": viz_batch,
            "cache_dir": cache_dir,
        }.items()
    })
    with open(Path(debug_dir) / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Resolve device BEFORE wp.init() so Warp/CuPy establish their CUDA context
    # on the correct GPU.  torch.cuda.set_device() must be called first so that
    # CuPy's raw-kernel compiler targets the same device as our tensors.
    if device == "cuda":
        device = _least_occupied_cuda_device()
    else:
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    wp.init()
    torch.manual_seed(42)

    print(f"Device: {device}")

    # ── Datasets & DataLoaders ────────────────────────────────────────────
    sft_root = sft_data_root or data_root
    print(f"SSL data_root: {data_root}")
    print(f"SFT data_root: {sft_root}")
    ssl_dataset_full = APASparseDataset(
        data_root, apa=apa, view=view, frame_name="frame_rebinned_reco",
        cache_dir=cache_dir,
    )
    sft_dataset = APASparseMetaDataset(
        sft_root, apa=apa, view=view, frame_name="frame_rebinned_reco",
        cache_dir=cache_dir,
        return_full_metadata=True,
        return_pixel_truth=True,
    )

    # Optional SSL subset before train/val split
    if ssl_subset_frac < 1.0:
        n_ssl_use        = max(1, int(len(ssl_dataset_full) * ssl_subset_frac))
        ssl_dataset_full = Subset(ssl_dataset_full, torch.randperm(len(ssl_dataset_full))[:n_ssl_use].tolist())
        print(f"ssl_subset_frac={ssl_subset_frac}: using {n_ssl_use} SSL samples")

    # Train / val split on the (possibly subsetted) SSL dataset
    n_ssl_total = len(ssl_dataset_full)
    n_ssl_val   = max(1, int(n_ssl_total * val_frac))
    n_ssl_train = n_ssl_total - n_ssl_val
    indices     = torch.randperm(n_ssl_total).tolist()
    ssl_train_dataset = Subset(ssl_dataset_full, indices[:n_ssl_train])
    ssl_val_dataset   = Subset(ssl_dataset_full, indices[n_ssl_train:])
    print(f"SSL  train={n_ssl_train}  val={n_ssl_val}")

    if sft_subset_frac < 1.0:
        n_sft_use   = max(1, int(len(sft_dataset) * sft_subset_frac))
        sft_dataset = Subset(sft_dataset, torch.randperm(len(sft_dataset))[:n_sft_use].tolist())
        print(f"sft_subset_frac={sft_subset_frac}: using {n_sft_use} SFT samples")

    # Wrap the SFT dataset so it returns pixel-level PID class labels per voxel
    # (computed from frame_pid_1st with on-the-fly blip detection; cached
    # per-index so repeat epochs are fast).
    sft_dataset = SFTPixelPIDDataset(sft_dataset)

    # Train / val split on the SFT dataset for the per-SSL-epoch probe (rec #6).
    n_sft_total = len(sft_dataset)
    n_sft_val   = max(1, int(n_sft_total * sft_val_frac))
    n_sft_train = n_sft_total - n_sft_val
    sft_idx     = torch.randperm(n_sft_total).tolist()
    sft_train_dataset = Subset(sft_dataset, sft_idx[:n_sft_train])
    sft_val_dataset   = Subset(sft_dataset, sft_idx[n_sft_train:])

    # rec #A.1: persistent workers + pin_memory to keep the GPU fed.  Only
    # turn on persistent_workers when num_workers > 0 (avoids harmless
    # PyTorch warning).
    _dl_kwargs = dict(
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )
    ssl_train_loader = DataLoader(
        ssl_train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=voxels_collate_fn, **_dl_kwargs,
    )
    ssl_val_loader = DataLoader(
        ssl_val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=voxels_collate_fn, **_dl_kwargs,
    )
    sft_loader = DataLoader(
        sft_train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=voxels_pixel_label_collate_fn, **_dl_kwargs,
    )
    sft_val_loader = DataLoader(
        sft_val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=voxels_pixel_label_collate_fn, **_dl_kwargs,
    )

    print(f"SFT dataset: total={n_sft_total}  train={n_sft_train}  val={n_sft_val}")
    print(f"n_sft_epochs_per_ssl_epoch={n_sft_epochs_per_ssl_epoch}")

    # ── Model ─────────────────────────────────────────────────────────────
    if true_mae:
        model = SparseTrueMAEModel(n_classes=n_classes).to(device)
        print("Using true MAE (coordinate-removal masking)")
    else:
        backbone = None
        if backbone_name:
            from sparseformer.backbones import build_backbone
            backbone = build_backbone(backbone_name, **(backbone_kwargs or {}))
            print(f"Backbone: {backbone_name}  kwargs={backbone_kwargs or {}}")
        else:
            print("Backbone: minkunet (legacy default)")
        model = SparseMAEModel(n_classes=n_classes, backbone=backbone).to(device)

    # ── Optimizers ────────────────────────────────────────────────────────
    ssl_params = list(model.backbone.parameters()) + list(model.charge_head.parameters())
    if use_patch_mae:
        ssl_params += list(model.coord_head.parameters())
    opt_ssl = optim.AdamW(ssl_params, lr=lr)
    sched_ssl = StepLR(opt_ssl, step_size=scheduler_step, gamma=gamma)

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_epoch = 1
    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "opt_ssl" in ckpt:
            opt_ssl.load_state_dict(ckpt["opt_ssl"])
        if "sched_ssl" in ckpt:
            sched_ssl.load_state_dict(ckpt["sched_ssl"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {resume}  (epoch {start_epoch - 1} → continuing from {start_epoch})")

    # ── Debugger ──────────────────────────────────────────────────────────
    debugger  = MAEDebugger(debug_dir=debug_dir, debug_every=debug_every)

    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(exist_ok=True)
    viz_dir = Path(viz_dir)

    # Global iteration counter (SSL train batches only)
    iteration = 0

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*60}")
        print(f"SSL Epoch {epoch}/{epochs}")

        # ── SSL train epoch ───────────────────────────────────────────────
        ssl_losses, iteration = _train_ssl_epoch(
            model, ssl_train_loader, opt_ssl,
            device, masking_frac, win_ch, win_tick,
            epoch, debugger, iteration,
            viz_dir=viz_dir, viz_batch=viz_batch,
            true_mae=true_mae,
            vicreg_lambda_v=vicreg_lambda_v,
            vicreg_lambda_c=vicreg_lambda_c,
            use_patch_mae=use_patch_mae,
            patch_ch=patch_ch,
            patch_tick=patch_tick,
            patch_mask_frac=patch_mask_frac,
            lambda_coord=lambda_coord,
            lambda_charge=lambda_charge,
            mask_mode=mask_mode,
        )
        ssl_mean = sum(ssl_losses) / len(ssl_losses) if ssl_losses else float("nan")
        print(f"  SSL train epoch {epoch} done  |  mean L1={ssl_mean:.4f}")
        sched_ssl.step()

        # ── SSL validation epoch ──────────────────────────────────────────
        val_losses = _val_ssl_epoch(
            model, ssl_val_loader, device, masking_frac, win_ch, win_tick,
            true_mae=true_mae,
        )
        val_mean = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        print(f"  SSL val   epoch {epoch} done  |  mean L1={val_mean:.4f}")
        debugger.log_val_epoch(epoch, iteration, val_mean)
        debugger.log_ssl_epoch_summary(epoch, ssl_mean, val_mean)
        debugger.save_histories()

        # ── SFT (either offline-pool [rec #B] or legacy online) ───────────
        offline_sft_result = None
        if sft_mode == "offline_pool":
            offline_sft_result = _run_offline_sft(
                model, sft_loader, sft_val_loader, device, PIXEL_PID_CLASS_NAMES,
                epochs=sft_pool_epochs, batch_size=sft_pool_batch, lr=sft_pool_lr,
                max_pixels_per_class=sft_pool_max_pixels, seed=42 + epoch,
            )
            # Persist directly into sft_history under the new schema; skip the
            # online per-sub-epoch loop and the legacy log_sft_aggregate call.
            ent = debugger._sft_epoch_entry(epoch)
            ent["offline_sft"] = offline_sft_result
            if not debugger.sft_history["class_names"]:
                debugger.sft_history["class_names"] = list(PIXEL_PID_CLASS_NAMES)

        # Legacy online path (kept for fallback / ablation).
        if sft_mode == "online":
            model.reset_sft_head()
            opt_sft = optim.AdamW(model.pixel_pid_head.parameters(),     lr=lr)
            opt_ref = optim.AdamW(model.ref_pixel_pid_head.parameters(), lr=lr)

            all_sft_losses, all_ref_losses = [], []
            confusion_sft = torch.zeros(n_classes, n_classes, dtype=torch.long)
            confusion_ref = torch.zeros(n_classes, n_classes, dtype=torch.long)

            for sft_epoch in range(1, n_sft_epochs_per_ssl_epoch + 1):
                sft_losses, conf_sft_ep, ref_losses, conf_ref_ep = _train_sft_epoch(
                    model, sft_loader, opt_sft, opt_ref,
                    device, n_classes, epoch, sft_epoch,
                    focal_gamma=focal_gamma,
                )
                all_sft_losses.extend(sft_losses)
                all_ref_losses.extend(ref_losses)
                confusion_sft += conf_sft_ep
                confusion_ref += conf_ref_ep

                sft_mean_ep = sum(sft_losses) / len(sft_losses) if sft_losses else float("nan")
                ref_mean_ep = sum(ref_losses)  / len(ref_losses)  if ref_losses  else float("nan")
                total_s     = int(conf_sft_ep.sum())
                acc_s       = 100.0 * int(conf_sft_ep.diagonal().sum()) / total_s if total_s > 0 else float("nan")
                total_r     = int(conf_ref_ep.sum())
                acc_r       = 100.0 * int(conf_ref_ep.diagonal().sum()) / total_r if total_r > 0 else float("nan")

                # Held-out probe-eval split (rec #6) — no grad, current frozen heads.
                val_sft_losses, val_conf_sft, val_ref_losses, val_conf_ref = _val_sft_epoch(
                    model, sft_val_loader, device, n_classes, focal_gamma=focal_gamma,
                )
                val_sft_mean_ep = sum(val_sft_losses) / len(val_sft_losses) if val_sft_losses else float("nan")
                val_ref_mean_ep = sum(val_ref_losses) / len(val_ref_losses) if val_ref_losses else float("nan")
                v_tot_s         = int(val_conf_sft.sum())
                v_acc_s         = 100.0 * int(val_conf_sft.diagonal().sum()) / v_tot_s if v_tot_s > 0 else float("nan")
                v_tot_r         = int(val_conf_ref.sum())
                v_acc_r         = 100.0 * int(val_conf_ref.diagonal().sum()) / v_tot_r if v_tot_r > 0 else float("nan")

                print(
                    f"  SFT epoch {sft_epoch}/{n_sft_epochs_per_ssl_epoch}"
                    f"  |  SSL-feat: train CE={sft_mean_ep:.4f} acc={acc_s:.1f}%  val CE={val_sft_mean_ep:.4f} acc={v_acc_s:.1f}%"
                    f"  |  raw-charge: train CE={ref_mean_ep:.4f} acc={acc_r:.1f}%  val CE={val_ref_mean_ep:.4f} acc={v_acc_r:.1f}%"
                )
                debugger.log_sft_subepoch(
                    epoch=epoch, sft_epoch=sft_epoch,
                    sft_ce=sft_mean_ep, sft_acc=acc_s / 100.0,
                    ref_ce=ref_mean_ep, ref_acc=acc_r / 100.0,
                    sft_val_ce=val_sft_mean_ep, sft_val_acc=v_acc_s / 100.0,
                    ref_val_ce=val_ref_mean_ep, ref_val_acc=v_acc_r / 100.0,
                )

            sft_mean  = sum(all_sft_losses) / len(all_sft_losses) if all_sft_losses else float("nan")
            ref_mean  = sum(all_ref_losses)  / len(all_ref_losses)  if all_ref_losses  else float("nan")
            total_sft = int(confusion_sft.sum())
            sft_acc   = 100.0 * int(confusion_sft.diagonal().sum()) / total_sft if total_sft > 0 else 0.0
            total_ref = int(confusion_ref.sum())
            ref_acc   = 100.0 * int(confusion_ref.diagonal().sum()) / total_ref if total_ref > 0 else 0.0

            print(f"\n{'='*60}")
            print(f"Epoch {epoch:3d}  |  SSL train L1={ssl_mean:.4f}  val L1={val_mean:.4f}")
            print(f"  SSL features  :  CE={sft_mean:.4f}  acc={sft_acc:.1f}%")
            print(f"  Raw charge ref:  CE={ref_mean:.4f}  acc={ref_acc:.1f}%")
            print(f"\n  [SSL features]")
            _print_confusion(confusion_sft, PIXEL_PID_CLASS_NAMES)
            _print_class_metrics(confusion_sft, PIXEL_PID_CLASS_NAMES)
            print(f"\n  [Raw charge reference]")
            _print_confusion(confusion_ref, PIXEL_PID_CLASS_NAMES)
            _print_class_metrics(confusion_ref, PIXEL_PID_CLASS_NAMES)
            print(f"{'='*60}\n")

            debugger.log_sft_aggregate(
                epoch=epoch, class_names=PIXEL_PID_CLASS_NAMES,
                sft_ce=sft_mean, sft_acc=sft_acc / 100.0,
                ref_ce=ref_mean, ref_acc=ref_acc / 100.0,
                confusion_sft=confusion_sft, confusion_ref=confusion_ref,
            )

        # ── SVM linear-probe on backbone features ─────────────────────────
        # In offline_pool mode the offline_sft block already fit voxel_svm_feat
        # and voxel_svm_raw from the same pool, so this standalone svm_probe
        # call is redundant.  Run it only for the legacy online mode.
        if sft_mode == "online":
            try:
                if true_mae:
                    # MinkUNetTrueMAECore.forward(vox_visible, vox_union); for the
                    # probe we pass the same Voxels for both (no masking at eval).
                    feature_fn = lambda v: model.backbone(v, v)
                else:
                    feature_fn = lambda v: model.backbone(v)
                svm_res = svm_probe(
                    feature_fn=feature_fn,
                    sft_loader=sft_loader,
                    device=device,
                    class_names=PIXEL_PID_CLASS_NAMES,
                    max_pixels_per_class=5000,
                    svm_C=1.0,
                    train_frac=0.8,
                    seed=42 + epoch,
                    prepare_voxels=lambda v: voxels_to_device(v, device),
                )
                tr_f1 = svm_res.get("train_class_f1", {})
                va_f1 = svm_res.get("val_class_f1", {})
                print(
                    f"  [SVM probe]  train_acc={svm_res.get('train_acc', float('nan')):.3f}  "
                    f"val_acc={svm_res.get('val_acc', float('nan')):.3f}  "
                    f"val macro-F1={svm_res.get('val_macro_f1', float('nan')):.3f}  "
                    f"per-class F1 (val): "
                    + " ".join(f"{n}:{va_f1.get(n, float('nan')):.2f}"
                               for n in PIXEL_PID_CLASS_NAMES)
                )
                debugger.log_svm_probe(epoch, svm_res)
            except Exception as e:
                print(f"  [SVM probe] failed: {e}")

        debugger.save_sft_history()

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = checkpoints_dir / f"mae_epoch{epoch}.pt"
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "opt_ssl":   opt_ssl.state_dict(),
                "sched_ssl": sched_ssl.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")


def from_config(
    config_path: str,
    run_name: str = "",
    device: str = "cuda",
    **overrides,
):
    """
    Start MAE training from a saved run_config.json file.

    Loads training parameters from a previously saved run_config.json (e.g. from
    ./debug/<run_name>/run_config.json).  Any JSON field that does not match a
    parameter of main() is silently ignored, so old configs with stale or
    missing keys still work — missing fields fall back to main()'s defaults.

    The `run_name`, `device`, and any **overrides** override the corresponding
    values from the config file.
    """
    with open(config_path) as f:
        raw = json.load(f)

    sig = inspect.signature(main)
    valid_params = set(sig.parameters)
    kwargs = {k: v for k, v in raw.items() if k in valid_params}

    # JSON stores nested paths (base/run_name).  main() will re-nest, so strip
    # the trailing /<run_name> to avoid double-nesting.
    orig_run_name = kwargs.get("run_name", "")
    for key in ("checkpoints_dir", "debug_dir", "viz_dir"):
        stored = kwargs.get(key, "")
        if orig_run_name and isinstance(stored, str) and stored.endswith("/" + orig_run_name):
            kwargs[key] = stored[: -len("/" + orig_run_name)]

    if run_name:
        kwargs["run_name"] = run_name
    kwargs["device"] = device
    for k, v in overrides.items():
        if k in valid_params:
            kwargs[k] = v

    main(**kwargs)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "from_config":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        fire.Fire(from_config)
    else:
        fire.Fire(main)
