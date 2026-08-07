"""larmamba2 SSL LightningModule: grid-tile masked autoencoder.

points -> tiler -> patch-embed + 2D pos -> random tile mask (MAE-style: encoder
sees visible tokens only, compacted) -> BiMamba encoder -> transformer decoder
with [MASK] tokens -> reconstruction head -> loss on masked tiles.

Loss (`loss_type`):
- "occ_l1" (default): BCE on per-pixel occupancy + L1 on log-charge at hit
  pixels. REQUIRED for sparse targets: plain L1 is median-seeking, and with
  per-pixel hit probability < 50% inside a masked tile its optimum is exactly
  "predict empty everywhere" — measured on the first full run (l1_hit flat at
  ~1.06, l1_empty ~0.008, probe stuck at random-init level).
- "l1": plain L1 over all pixels (kept as the documented-degenerate control).

No energy head either way: the patch target carries geometry (occupancy) and
charge in one object (plan §3.5).
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn

from polarmae.utils.scheduler import LinearWarmupCosineAnnealingLR

from larmamba2.decoder import PatchDecoder
from larmamba2.encoder import Larmamba2Encoder
from larmamba2.patch_embed import TilePosEncoder, build_patch_embed
from larmamba2.tiler import EMPTY_VAL, tile_events

HIT_THR = EMPTY_VAL + 1e-3


def _compact(x: torch.Tensor, keep: torch.Tensor):
    """Move kept tokens to the front (stable). Returns compacted x, its valid
    mask, and the original-slot index of each compacted position."""
    B, T = keep.shape
    n = keep.sum(1)
    Tv = max(int(n.max()), 1)
    idx = torch.argsort((~keep).long(), dim=1, stable=True)[:, :Tv]   # (B, Tv)
    xc = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    mc = torch.arange(Tv, device=x.device).unsqueeze(0) < n.unsqueeze(1)
    return xc, mc, idx


class Larmamba2MAE(pl.LightningModule):
    def __init__(
        self,
        tile_size: int = 5,
        t_max: int = 1024,
        dim: int = 384,
        depth: int = 12,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
        mask_ratio: float = 0.6,
        loss_type: str = "occ_l1",       # occ_l1 | l1
        occ_weight: float = 1.0,         # BCE weight in occ_l1
        patch_embed: str = "conv",       # conv | linear
        order_kind: str = "morton",      # morton | morton_t | raster_tick | raster_ch
        drop_path_rate: float = 0.25,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        learning_rate: float = 7.0e-5,
        weight_decay: float = 0.05,
        warmup_epochs: int = 1,
        lr_min: float = 3.0e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embed = build_patch_embed(patch_embed, tile_size, dim)
        self.pos = TilePosEncoder(dim, tile_size)
        self.encoder = Larmamba2Encoder(
            dim=dim, depth=depth, drop_path_rate=drop_path_rate,
            d_state=d_state, d_conv=d_conv, expand=expand, order_kind=order_kind,
        )
        self.decoder = PatchDecoder(dim=dim, depth=decoder_depth,
                                    num_heads=decoder_heads, tile_size=tile_size,
                                    out_channels=2 if loss_type == "occ_l1" else 1)

    # ---------------- core ----------------

    def _random_vis_mask(self, tile_mask: torch.Tensor) -> torch.Tensor:
        """Per-event exact-ratio random visible mask over valid tiles."""
        B, T = tile_mask.shape
        noise = torch.rand(B, T, device=tile_mask.device)
        noise[~tile_mask] = 2.0                          # invalid sorts last
        ranks = noise.argsort(dim=1).argsort(dim=1)      # (B, T) 0 = smallest
        n_valid = tile_mask.sum(1, keepdim=True)
        n_vis = ((1.0 - self.hparams.mask_ratio) * n_valid).round().long().clamp(min=1)
        return (ranks < n_vis) & tile_mask

    def forward_features(self, points: torch.Tensor, lengths: torch.Tensor,
                         t_max: int = -1):
        """Full (unmasked) encoding — used by probes/export. Returns tokens
        (B, T, D), tile_mask, vox_tile."""
        patches, tile_coords, tile_mask, vox_tile = tile_events(
            points, lengths, self.hparams.tile_size, t_max)
        x = self.embed(patches) + self.pos(tile_coords)
        tokens = self.encoder(x, tile_coords, tile_mask)
        return tokens, tile_mask, vox_tile

    def _step(self, batch, stage: str):
        points, lengths = batch["points"], batch["lengths"]
        patches, tile_coords, tile_mask, _ = tile_events(
            points, lengths, self.hparams.tile_size, self.hparams.t_max)

        vis = self._random_vis_mask(tile_mask)           # (B, T) visible tiles
        msk = tile_mask & ~vis                           # tiles to reconstruct

        x = self.embed(patches) + self.pos(tile_coords)
        xc, mc, _ = _compact(x, vis)
        cc, _, _ = _compact(tile_coords.float(), vis)
        latent_c = self.encoder(xc, cc.long(), mc)       # (B, Tv, D)

        # scatter compacted latents back to original slots via cumsum inverse
        pos_in_c = (vis.long().cumsum(1) - 1).clamp(min=0)        # (B, T)
        latent_full = torch.gather(
            latent_c, 1,
            pos_in_c.unsqueeze(-1).expand(-1, -1, latent_c.shape[-1]).clamp(
                max=latent_c.shape[1] - 1))
        pred = self.decoder(latent_full, self.pos(tile_coords), vis, tile_mask)

        S2 = patches.shape[-1]
        hit = patches > HIT_THR                          # (B, T, S^2)
        hit_w = (hit & msk.unsqueeze(-1)).float()
        bs = points.shape[0]

        if self.hparams.loss_type == "occ_l1":
            occ_logit, pred_q = pred[..., :S2], pred[..., S2:]
            mw = msk.unsqueeze(-1).float()
            bce = nn.functional.binary_cross_entropy_with_logits(
                occ_logit, hit.float(), reduction="none")
            occ_loss = (bce * mw).sum() / (mw.sum() * S2).clamp(min=1)
            err = (pred_q - patches).abs()
            l1_hit = (err * hit_w).sum() / hit_w.sum().clamp(min=1)
            loss = self.hparams.occ_weight * occ_loss + l1_hit
            self.log(f"occ_bce/{stage}", occ_loss, sync_dist=True,
                     on_step=False, on_epoch=True, batch_size=bs)
            self.log(f"l1_hit/{stage}", l1_hit, sync_dist=True,
                     on_step=False, on_epoch=True, batch_size=bs)
        else:                                            # plain L1 (degenerate control)
            err = (pred - patches).abs()
            mw = msk.unsqueeze(-1).float()
            loss = (err * mw).sum() / (mw.sum() * S2).clamp(min=1)
            emp_w = (~hit & msk.unsqueeze(-1)).float()
            self.log(f"l1_hit/{stage}", (err * hit_w).sum() / hit_w.sum().clamp(min=1),
                     sync_dist=True, on_step=False, on_epoch=True, batch_size=bs)
            self.log(f"l1_empty/{stage}", (err * emp_w).sum() / emp_w.sum().clamp(min=1),
                     sync_dist=True, on_step=False, on_epoch=True, batch_size=bs)

        self.log(f"loss/{stage}", loss, sync_dist=True, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)
        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=self.hparams.lr_min,
            eta_min=self.hparams.lr_min,
        )
        return [opt], [sched]     # epoch-stepped (round-2 recipe)
