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

    A second file sft_history.json mirrors the schema produced by
    mae.diagnostics.parse_training_log:
      class_names : [str, ...]
      ssl_epochs  : [
        {epoch, ssl_train_l1, ssl_val_l1,
         sft_subepochs:[{sft_epoch, sft_ce, sft_acc, ref_ce, ref_acc}, ...],
         sft_aggregate:{sft_ce, sft_acc, ref_ce, ref_acc,
                        confusion_sft, confusion_ref,
                        per_class_sft, per_class_ref}}, ...]
    """

    def __init__(self, debug_dir: str | Path, debug_every: int = 50):
        self.debug_dir   = Path(debug_dir)
        self.debug_every = debug_every
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.loss_train: list[float] = []
        self.loss_val   = {"iter": [], "loss": []}
        self.stats      = {"iter": [], "feat_var": [], "feat_cov": []}

        self.sft_history = {"class_names": [], "ssl_epochs": []}

        self.logger = logging.getLogger("mae_debug")
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate handlers when script is re-imported in interactive sessions
        if not self.logger.handlers:
            handler = logging.FileHandler(self.debug_dir / "training.log")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.logger.addHandler(handler)

        # Write empty / fresh file at startup to overwrite any stale run
        self.save_histories()
        self.save_sft_history()

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
    # SFT history (per-SSL-epoch SFT/ref metrics and confusion)
    # ------------------------------------------------------------------

    def _sft_epoch_entry(self, epoch: int) -> dict:
        """Return (creating if needed) the dict for SSL epoch `epoch`."""
        for e in self.sft_history["ssl_epochs"]:
            if e["epoch"] == epoch:
                return e
        e = {"epoch": epoch, "sft_subepochs": []}
        self.sft_history["ssl_epochs"].append(e)
        return e

    def log_ssl_epoch_summary(self, epoch: int, ssl_train_l1: float, ssl_val_l1: float) -> None:
        """Record per-SSL-epoch L1 train + val losses (paired with the SFT block)."""
        ent = self._sft_epoch_entry(epoch)
        ent["ssl_train_l1"] = float(ssl_train_l1)
        ent["ssl_val_l1"]   = float(ssl_val_l1)

    def log_sft_subepoch(
        self, epoch: int, sft_epoch: int,
        sft_ce: float, sft_acc: float,
        ref_ce: float, ref_acc: float,
        sft_val_ce: float = None, sft_val_acc: float = None,
        ref_val_ce: float = None, ref_val_acc: float = None,
    ) -> None:
        """Append one SFT sub-epoch entry under SSL epoch `epoch`.

        Accuracies are expected as fractions in [0, 1] (NOT percentages).
        The *_val_* fields (rec #6) come from a held-out SFT split — if None
        (legacy callers) they're omitted from the JSON.
        """
        ent = self._sft_epoch_entry(epoch)
        rec = {
            "sft_epoch": int(sft_epoch),
            "sft_ce": float(sft_ce), "sft_acc": float(sft_acc),
            "ref_ce": float(ref_ce), "ref_acc": float(ref_acc),
        }
        if sft_val_ce is not None:
            rec["sft_val_ce"]  = float(sft_val_ce)
            rec["sft_val_acc"] = float(sft_val_acc)
        if ref_val_ce is not None:
            rec["ref_val_ce"]  = float(ref_val_ce)
            rec["ref_val_acc"] = float(ref_val_acc)
        ent["sft_subepochs"].append(rec)

    def log_svm_probe(self, epoch: int, result: dict) -> None:
        """Attach the svm_probe(...) result dict to SSL epoch `epoch`."""
        ent = self._sft_epoch_entry(epoch)
        ent["svm_probe"] = result

    def log_sft_aggregate(
        self, epoch: int, class_names,
        sft_ce: float, sft_acc: float,
        ref_ce: float, ref_acc: float,
        confusion_sft, confusion_ref,
    ) -> None:
        """Record the per-SSL-epoch SFT aggregate (overall + confusion + per-class)."""
        if not self.sft_history["class_names"]:
            self.sft_history["class_names"] = list(class_names)

        cm_sft = _to_int_2d(confusion_sft)
        cm_ref = _to_int_2d(confusion_ref)
        per_sft = _per_class_eff_pur(cm_sft, class_names)
        per_ref = _per_class_eff_pur(cm_ref, class_names)

        ent = self._sft_epoch_entry(epoch)
        ent["sft_aggregate"] = {
            "sft_ce":         float(sft_ce),
            "sft_acc":        float(sft_acc),
            "ref_ce":         float(ref_ce),
            "ref_acc":        float(ref_acc),
            "confusion_sft":  cm_sft,
            "confusion_ref":  cm_ref,
            "per_class_sft":  per_sft,
            "per_class_ref":  per_ref,
        }

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

    def save_sft_history(self) -> None:
        """Write the per-SSL-epoch SFT history to sft_history.json."""
        try:
            with open(self.debug_dir / "sft_history.json", "w") as f:
                json.dump(self.sft_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving sft_history: {e}")

    def maybe_save_histories(self, iteration: int) -> None:
        """Persist histories every debug_every iterations."""
        if iteration % self.debug_every == 0:
            self.save_histories()


# ----------------------------------------------------------------------
# Helpers (module-level)
# ----------------------------------------------------------------------

def _to_int_2d(mat):
    """Convert a confusion matrix (torch.Tensor or list of lists) to plain int list-of-lists."""
    if torch.is_tensor(mat):
        return mat.detach().cpu().long().tolist()
    return [[int(x) for x in row] for row in mat]


def _per_class_eff_pur(cm_2d, class_names):
    """Compute per-class efficiency (recall) and purity (precision) from a 2D
    confusion list with rows = true, cols = predicted.  Returns a list of dicts
    aligned with class_names; NaN-equivalent (None) when the denominator is 0.
    """
    out = []
    n = len(class_names)
    for c in range(n):
        row = sum(cm_2d[c]) if c < len(cm_2d) else 0
        col = sum(cm_2d[r][c] for r in range(len(cm_2d)) if c < len(cm_2d[r]))
        diag = cm_2d[c][c] if c < len(cm_2d) and c < len(cm_2d[c]) else 0
        eff = (diag / row) if row > 0 else None
        pur = (diag / col) if col > 0 else None
        out.append({
            "name":       class_names[c],
            "efficiency": eff,
            "purity":     pur,
        })
    return out
