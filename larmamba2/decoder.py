"""MAE decoder for larmamba2 (plan §3.4, option A): shallow pre-norm transformer
over the full tile sequence (visible latents + learnable [MASK] tokens, both
plus pos-embed), then a Linear(D -> S^2) reconstruction head.

O(T^2) lives only here (depth 4, T <= ~1k at train) — the encoder stays O(T).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class PatchDecoder(nn.Module):
    def __init__(self, dim: int = 384, depth: int = 4, num_heads: int = 6,
                 tile_size: int = 5, out_channels: int = 1):
        """out_channels=1: charge only (plain L1). out_channels=2: occupancy
        logit + charge (occ_l1 loss) — output is (B, T, out_channels * S^2)."""
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([_DecoderBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels * tile_size * tile_size)

    def forward(self, latent_full: torch.Tensor, pos: torch.Tensor,
                vis_mask: torch.Tensor, tile_mask: torch.Tensor) -> torch.Tensor:
        """latent_full (B, T, D): encoder output scattered to original slots
        (anything at non-visible slots is ignored); pos (B, T, D);
        vis_mask/tile_mask (B, T) bool. Returns (B, T, S^2) predictions."""
        x = torch.where(vis_mask.unsqueeze(-1), latent_full,
                        self.mask_token.to(latent_full.dtype).expand_as(latent_full))
        x = x + pos
        pad = ~tile_mask
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad)
        return self.head(self.norm(x))
