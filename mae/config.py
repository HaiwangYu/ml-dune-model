"""Configuration dataclass for MAE training."""

from dataclasses import dataclass


@dataclass
class MAEConfig:
    """Configuration for sparse MAE training (SSL epochs + SFT epochs)."""

    # ============ Run ============
    run_name: str = ""                # optional label; outputs nest under run_name/ if set

    # ============ Data ============
    data_root: str = "/gpfs01/lbne/users/fm/cffm-data/prod-jay-1M-2026-02-27"
    sft_data_root: str = ""               # if empty, SFT reuses data_root; otherwise SFT loads from this path
    apa: int = 0
    view: str = "W"
    batch_size: int = 16
    num_workers: int = 0                  # >0 only if warp is initialised in workers
    ssl_subset_frac: float = 1.0          # fraction of SSL dataset to use
    sft_subset_frac: float = 1.0          # fraction of SFT dataset to use
    val_frac: float = 0.2                 # fraction of SSL dataset held out for validation
    sft_val_frac: float = 0.2             # fraction of SFT dataset held out for probe-eval (rec #6)
    cache_dir: str = "./data"             # directory for cached dataset index .pt files

    # ============ Training ============
    epochs: int = 2
    lr: float = 1e-3
    scheduler_step: int = 10
    gamma: float = 0.7
    n_sft_epochs_per_ssl_epoch: int = 3
    save_every: int = 5
    resume: str = ""

    # ============ MAE / masking ============
    true_mae: bool = True                 # remove masked voxels from encoder input
    masking_frac: float = 0.5
    win_ch: int = 30
    win_tick: int = 50
    mask_mode: str = "block"              # "block" (legacy seed+window) or "grid_patch" (rec #3,
                                          # non-overlapping grid cells of size win_ch × win_tick)

    # ============ Patch-MAE (optional) ============
    use_patch_mae: bool = False
    patch_ch: int = 15
    patch_tick: int = 25
    patch_mask_frac: float = 0.60
    lambda_coord: float = 1.0
    lambda_charge: float = 1.0

    # ============ Loss ============
    n_classes: int = 3
    focal_gamma: float = 2.0              # 0 = plain cross-entropy
    vicreg_lambda_v: float = 0.0
    vicreg_lambda_c: float = 0.0

    # ============ SFT mode (rec #B from polarmae comparison) ============
    sft_mode: str = "offline_pool"        # "offline_pool" or "online" (legacy)
    sft_pool_max_pixels: int = 5000       # per-class cap in offline pool
    sft_pool_epochs: int = 30
    sft_pool_batch: int = 256
    sft_pool_lr: float = 5e-3

    # ============ Output / debug ============
    checkpoints_dir: str = "./checkpoints"
    debug_dir: str = "./debug"
    debug_every: int = 50
    viz_dir: str = "./viz"
    viz_batch: int = 0
