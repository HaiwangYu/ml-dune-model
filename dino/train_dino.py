"""
DINO training script for DUNE sparse UNet backbone.

Usage:
    python dino/train_dino.py --epochs=100 --batch_size=16 --backbone_name=attn_default
    python dino/train_dino.py --epochs=2 --batch_size=4 --test_mode=True --debug=True
"""

import fire
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

from loader.dataset import DUNEImageDataset
from loader.splits import train_val_split, Subset

from .config import DINOConfig
from .masking import SparseVoxelMasker
from .loss import PixelDINOLoss
from .scheduler import CosineScheduler
from .model import DINODuneModel
from .debug import DINODebugger


def main(
    backbone_name: str = "attn_default",
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    mask_ratio: float = 0.5,
    loss_type: str = "cosine",
    momentum_start: float = 0.996,
    momentum_end: float = 0.9999,
    weight_decay: float = 0.04,
    weight_decay_end: float = 0.4,
    warmup_epochs: int = 5,
    output_dir: str = "./dino_checkpoints",
    save_every: int = 10,
    device: str = "cuda",
    debug: bool = False,
    debug_every: int = 10,
    debug_dir: str = "./dino_debug",
    test_mode: bool = False,
    num_workers: int = 4,
):
    """
    DINO training loop for DUNE detector.

    Args:
        backbone_name: Model architecture ("attn_default", "base", etc.)
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        lr: Base learning rate
        mask_ratio: Fraction of active pixels to mask
        loss_type: "cosine" or "mse"
        momentum_start: Initial EMA momentum
        momentum_end: Final EMA momentum
        weight_decay: L2 regularization
        weight_decay_end: Final weight decay (cosine annealed)
        warmup_epochs: Linear warmup duration
        output_dir: Where to save checkpoints
        save_every: Save checkpoint every N epochs
        device: "cuda" or "cpu"
        debug: Enable debugging and visualization
        debug_every: Save debug visuals every N batches
        debug_dir: Directory for debug outputs
        test_mode: Use small subset for quick smoke tests
        num_workers: Number of dataloader workers
    """
    # ============ Setup ============
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Build config
    cfg = DINOConfig(
        backbone_name=backbone_name,
        mask_ratio=mask_ratio,
        loss_type=loss_type,
        momentum_start=momentum_start,
        momentum_end=momentum_end,
        lr=lr,
        weight_decay=weight_decay,
        weight_decay_end=weight_decay_end,
        warmup_epochs=warmup_epochs,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        save_every=save_every,
        debug=debug,
        debug_every=debug_every,
        debug_dir=debug_dir,
        num_workers=num_workers,
    )

    print(f"Device: {device}")
    print(f"Config: backbone={cfg.backbone_name}, mask_ratio={cfg.mask_ratio}, "
          f"lr={cfg.lr}, epochs={cfg.epochs}, batch_size={cfg.batch_size}")

    # ============ Data ============
    print("\nLoading dataset...")
    dataset = DUNEImageDataset(
        rootdir=cfg.rootdir,
        class_names=["numu", "nue", "nutau", "NC"],
        view_index=cfg.view_index,
        use_cache=True,
    )

    if test_mode:
        n_subset = 1000
        print(f"TEST MODE: using {n_subset} samples")
        subset_indices = torch.randperm(len(dataset))[:n_subset]
        dataset = Subset(dataset, subset_indices)

    train_ds, val_ds, _, _ = train_val_split(dataset, val_fraction=0.2, use_cache=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    epoch_len = len(train_loader)
    total_iters = epochs * epoch_len
    print(f"Total training iterations: {total_iters} (epochs={epochs}, epoch_len={epoch_len})")

    # ============ Model, optimizer, loss ============
    print("\nBuilding model...")
    model = DINODuneModel(backbone_name=backbone_name).to(device)
    optimizer = optim.AdamW(model.student.parameters(), lr=lr, weight_decay=weight_decay)

    masker = SparseVoxelMasker(mask_ratio=mask_ratio)
    loss_fn = PixelDINOLoss(loss_type=loss_type).to(device)

    # ============ Schedulers ============
    warmup_iters = min(warmup_epochs * epoch_len, int(0.2 * total_iters))
    print(f"Schedules: total_iters={total_iters}, warmup_iters={warmup_iters}")

    lr_schedule = CosineScheduler(
        base_value=cfg.lr,
        final_value=cfg.min_lr,
        total_iters=total_iters,
        warmup_iters=warmup_iters,
    )
    wd_schedule = CosineScheduler(
        base_value=cfg.weight_decay,
        final_value=cfg.weight_decay_end,
        total_iters=total_iters,
    )
    momentum_schedule = CosineScheduler(
        base_value=cfg.momentum_start,
        final_value=cfg.momentum_end,
        total_iters=total_iters,
    )

    # ============ Checkpointing ============
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============ Debugging ============
    debugger = DINODebugger(cfg, enabled=True)
    debugger.log_config(cfg)

    # ============ Training loop ============
    print("\nStarting training...")
    first_batch = True

    for epoch in range(1, epochs + 1):
        model.train()

        for batch_idx, (data, _) in enumerate(train_loader):
            iteration = (epoch - 1) * epoch_len + batch_idx
            data = data.to(device)

            # Warn about empty images (all-zero pixels) — these can cause warpconvnet
            # to silently drop batch entries due to bincount trailing-zero truncation.
            empty = (data.view(data.shape[0], -1) == 0).all(dim=1)
            if empty.any():
                empty_idx = empty.nonzero(as_tuple=True)[0].tolist()
                print(f"WARNING: epoch {epoch}, batch {batch_idx}: "
                      f"{len(empty_idx)} empty image(s) at batch positions {empty_idx}")

            # Apply schedules
            lr_val = lr_schedule[iteration]
            wd_val = wd_schedule[iteration]
            mom_val = momentum_schedule[iteration]

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_val
                param_group["weight_decay"] = wd_val

            # Masking (for debug visualization)
            x_student, mask_applied = masker(data)

            # Forward + backward
            optimizer.zero_grad()
            loss_val, s_feats, t_feats, _ = model.forward_backward(data, masker, loss_fn)
            optimizer.step()

            # EMA teacher update
            model.update_teacher(mom_val)

            # Logging
            n_valid = (~mask_applied & (data != 0)).sum().item()
            debugger.log_batch(epoch, batch_idx, iteration, loss_val, n_valid, lr_val, mom_val)

            # Debug visualizations
            if first_batch:
                debugger.log_shapes(data, x_student, mask_applied, s_feats, t_feats)
                first_batch = False
            debugger.maybe_save_visuals(iteration, data, x_student, mask_applied, s_feats, t_feats)

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                print(f"[{epoch}/{epochs}] iter {iteration}: loss={loss_val:.6f}, "
                      f"lr={lr_val:.2e}, mom={mom_val:.6f}")

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt = {
                "epoch": epoch,
                "student": model.student.state_dict(),
                "teacher": model.teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
            }
            ckpt_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
