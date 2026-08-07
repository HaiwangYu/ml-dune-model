"""Patch tokenizers (plan §3.2) + 2D tile-center positional encoder.

ConvPatchEmbed (option A, primary): small conv stack over the SxS log-charge
patch, trained end-to-end through the MAE objective — with the pixel decoder,
the model is effectively a masked conv autoencoder.
LinearPatchEmbed (option D, control): flatten -> Linear.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvPatchEmbed(nn.Module):
    def __init__(self, tile_size: int, dim: int, width: int = 64):
        super().__init__()
        self.tile_size = tile_size
        self.conv = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(2 * width * tile_size * tile_size, dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, T, S*S) -> (B, T, dim)"""
        B, T, _ = patches.shape
        S = self.tile_size
        x = patches.reshape(B * T, 1, S, S)
        x = self.conv(x).reshape(B * T, -1)
        return self.proj(x).reshape(B, T, -1)


class LinearPatchEmbed(nn.Module):
    def __init__(self, tile_size: int, dim: int):
        super().__init__()
        self.tile_size = tile_size
        self.proj = nn.Linear(tile_size * tile_size, dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches)


def build_patch_embed(name: str, tile_size: int, dim: int) -> nn.Module:
    if name == "conv":
        return ConvPatchEmbed(tile_size, dim)
    if name == "linear":
        return LinearPatchEmbed(tile_size, dim)
    raise KeyError(f"unknown patch embed '{name}' (conv|linear)")


class TilePosEncoder(nn.Module):
    """MLP positional encoding of tile centers — native 2D (no z=0 hack).

    Centers are the tile-center pixel coordinates, normalized with the same
    center/scale convention as the rest of the project ((525, 562), 1/600).
    """

    def __init__(self, dim: int, tile_size: int,
                 center=(525.0, 562.0), scale: float = 1.0 / 600.0):
        super().__init__()
        self.tile_size = tile_size
        self.register_buffer("center", torch.tensor(center), persistent=False)
        self.scale = scale
        self.mlp = nn.Sequential(nn.Linear(2, 128), nn.GELU(), nn.Linear(128, dim))

    def forward(self, tile_coords: torch.Tensor) -> torch.Tensor:
        """(B, T, 2) long tile coords -> (B, T, dim)"""
        S = self.tile_size
        c = (tile_coords.float() * S + S / 2.0 - self.center) * self.scale
        return self.mlp(c)
