"""Token serialization orders for larmamba2 (plan §3.3).

Morton/Z-order is reused verbatim from larmamba (it already operates on integer
2D coordinates); raster orders are trivial physics-motivated alternatives
(tick-major = drift-time order). Invalid tokens always sort to the tail.
"""
from __future__ import annotations

import torch

from larmamba.serialize import serialize_order as _morton_order


@torch.no_grad()
def raster_order(tile_coords: torch.Tensor, mask: torch.Tensor,
                 major: str = "tick") -> torch.Tensor:
    """Row-major raster scan: 'tick' sorts by (tick_col, ch_row), 'ch' by
    (ch_row, tick_col)."""
    tr, tc = tile_coords[..., 0], tile_coords[..., 1]
    key = (tc * 100000 + tr) if major == "tick" else (tr * 100000 + tc)
    big = key.max() + 1 + torch.arange(key.shape[1], device=key.device).view(1, -1)
    key = torch.where(mask.bool(), key, big)
    return torch.argsort(key, dim=1)


@torch.no_grad()
def serialize_order(tile_coords: torch.Tensor, mask: torch.Tensor,
                    kind: str = "morton") -> torch.Tensor:
    """(B, T, 2) long tile coords + (B, T) bool mask -> (B, T) permutation."""
    if kind == "morton":
        return _morton_order(tile_coords.float(), mask)
    if kind == "morton_t":
        return _morton_order(tile_coords.float(), mask, transpose=True)
    if kind == "raster_tick":
        return raster_order(tile_coords, mask, major="tick")
    if kind == "raster_ch":
        return raster_order(tile_coords, mask, major="ch")
    raise KeyError(f"unknown serialization '{kind}' "
                   "(morton|morton_t|raster_tick|raster_ch)")
