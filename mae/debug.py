"""MAE debug utilities: history tracking for offline analysis."""

import json
import logging
from pathlib import Path

import torch
from torch import Tensor


class MAEDebugger:
    """
    Tracks training histories for MAE and persists them to histories.json.

    History structure (histories.json):
      loss_train : [float, ...]                      per-batch SSL train loss
      loss_val   : {iter: [...], loss: [...]}         per-epoch validation loss
      stats      : {iter: [...], feat_var: [...],
                    feat_cov: [...]}                  per-batch backbone feature stats
                                                      (sampled every debug_every iters)
    """

    def __init__(self, debug_dir: str | Path, debug_every: int = 50):
        self.debug_dir   = Path(debug_dir)
        self.debug_every = debug_every
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.loss_train: list[float] = []
        self.loss_val   = {"iter": [], "loss": []}
        self.stats      = {"iter": [], "feat_var": [], "feat_cov": []}

        self.logger = logging.getLogger("mae_debug")
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate handlers when script is re-imported in interactive sessions
        if not self.logger.handlers:
            handler = logging.FileHandler(self.debug_dir / "training.log")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.logger.addHandler(handler)

        # Write empty / fresh file at startup to overwrite any stale run
        self.save_histories()

    # ------------------------------------------------------------------
    # Per-batch train loss
    # ------------------------------------------------------------------

    def log_batch(self, epoch: int, batch_idx: int, iteration: int, loss: float) -> None:
        """Record one training batch loss and write to training.log."""
        self.loss_train.append(loss)
        self.logger.info(
            f"[epoch {epoch:3d} batch {batch_idx:4d} iter {iteration:6d}] loss={loss:.6f}"
        )

    # ------------------------------------------------------------------
    # Feature statistics (backbone output, every debug_every iters)
    # ------------------------------------------------------------------

    def log_feature_stats(self, iteration: int, feats: Tensor) -> None:
        """
        Compute and record backbone feature statistics.

        feats : [N, D] feature tensor (all active voxels in the batch).

        Recorded every debug_every iterations:
          feat_var : mean per-dimension variance (scalar) — collapses toward 0
                     if the backbone ignores all but a few dimensions.
          feat_cov : full D×D covariance matrix (list-of-lists) — saved for
                     offline heatmap plotting.
        """
        if iteration % self.debug_every != 0:
            return

        with torch.no_grad():
            f = feats.detach().float()   # [N, D]
            if f.shape[0] < 2:
                return
            feat_var = f.var(dim=0).mean().item()   # mean over feature dims
            cov_mat  = torch.cov(f.T)               # [D, D]

        self.logger.info(
            f"[iter {iteration:6d}] FEAT_STATS: feat_var={feat_var:.6f}"
        )
        self.stats["iter"].append(iteration)
        self.stats["feat_var"].append(feat_var)
        self.stats["feat_cov"].append(cov_mat.cpu().tolist())

    # ------------------------------------------------------------------
    # Validation loss (per epoch)
    # ------------------------------------------------------------------

    def log_val_epoch(self, epoch: int, iteration: int, val_loss: float) -> None:
        """Record end-of-SSL-epoch validation loss."""
        self.logger.info(
            f"[epoch {epoch:3d} iter {iteration:6d}] VAL_LOSS: {val_loss:.6f}"
        )
        self.loss_val["iter"].append(iteration)
        self.loss_val["loss"].append(val_loss)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_histories(self) -> None:
        """Write all in-memory histories to histories.json."""
        data = {
            "loss_train": self.loss_train,
            "loss_val":   self.loss_val,
            "stats":      self.stats,
        }
        try:
            with open(self.debug_dir / "histories.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving histories: {e}")

    def maybe_save_histories(self, iteration: int) -> None:
        """Persist histories every debug_every iterations."""
        if iteration % self.debug_every == 0:
            self.save_histories()
