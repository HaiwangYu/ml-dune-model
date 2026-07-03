#!/usr/bin/env python3
"""DDP-safe launcher for the polarmae LightningCLI, with clean trainer defaults.

Two reasons this exists instead of `python -m polarmae.tasks.polarmae fit`:

1. DDP safety. APA2DProbeCallback runs its SVM/MLP probe on rank 0 only
   (`on_validation_epoch_end` early-returns on non-zero ranks) but then logs the
   result with `sync_dist=True`. Under DDP that all-reduce is issued on rank 0
   alone and DEADLOCKS validation-epoch-end. We patch its `_log_scalar` to log
   `rank_zero_only=True, sync_dist=False` (consistent with the rank-0-only body).

2. Checkpoint control. `polarmae.tasks.polarmae` bakes THREE ModelCheckpoints
   into `trainer_defaults`; LightningCLI *appends* config callbacks to those, and
   PL 2.6 derives ModelCheckpoint.state_key without `save_on_train_epoch_end`, so
   our per-epoch checkpoint collides with the default (monitor=None,
   every_n_epochs=1) and PL raises "more than one stateful ModelCheckpoint".
   We construct the CLI here with minimal trainer_defaults (LearningRateMonitor
   only), so every checkpoint/callback comes from the config where we control
   uniqueness.

Usage (drop-in): python run_polarmae_fit.py fit --config <...>.yml ...
"""
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.cli import LightningCLI

OmegaConf.register_new_resolver("eval", eval)

from polarmae.models.ssl import PoLArMAE


def _patch_probe_callback():
    from polarmae.eval.probes import APA2DProbeCallback

    def _log_scalar(pl_module, key, value):
        if not np.isfinite(value):
            return
        pl_module.log(key, float(value), sync_dist=False,
                      rank_zero_only=True, prog_bar=True)

    APA2DProbeCallback._log_scalar = staticmethod(_log_scalar)
    print("[run_polarmae_fit] patched APA2DProbeCallback._log_scalar "
          "(rank_zero_only, no sync_dist) for DDP safety", flush=True)


if __name__ == "__main__":
    _patch_probe_callback()
    LightningCLI(
        PoLArMAE,
        trainer_defaults={
            "default_root_dir": "artifacts",
            "accelerator": "gpu",
            "precision": "bf16-mixed",
            "callbacks": [LearningRateMonitor()],
        },
        parser_kwargs={"parser_mode": "omegaconf"},
        seed_everything_default=123,
        save_config_callback=None,
    )
