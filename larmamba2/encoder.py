"""larmamba2 encoder: bidirectional-Mamba blocks over serialized patch tokens.

Reuses larmamba's MambaBlock (norm -> BiMamba mixer -> norm -> MLP, with
per-block curve reordering); the serialization order is computed ONCE per
forward from the tile coordinates and shared across blocks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from larmamba.mamba_block import MambaBlock
from larmamba2.serialize import serialize_order


class Larmamba2Encoder(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.25,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        order_kind: str = "morton",
    ):
        super().__init__()
        self.dim = dim
        self.order_kind = order_kind
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            MambaBlock(dim, mlp_ratio=mlp_ratio, drop_path=dpr[i],
                       d_state=d_state, d_conv=d_conv, expand=expand)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, tile_coords: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """x (B, T, D), tile_coords (B, T, 2) long, mask (B, T) bool."""
        order = serialize_order(tile_coords, mask, self.order_kind)
        for blk in self.blocks:
            x = blk(x, x_mask=mask, order=order)
        x = self.norm(x)
        return x * mask.unsqueeze(-1)
