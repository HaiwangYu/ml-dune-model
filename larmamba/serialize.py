"""
Space-filling-curve serialization of point-cloud tokens.

Mamba is a sequential recurrence, but FPS+ball-query tokens have no natural 1D
order.  Following PointMamba / Point Cloud Mamba, we order tokens along a
Z-order (Morton) curve over their 2D (channel, tick) centers so that
consecutive tokens in the scan are spatial neighbours.  Invalid / masked tokens
are pushed to the end of the order so they don't split spatial runs (the SSM
also makes them transparent via Δ=0, see mamba_block).

`serialize_order` returns a permutation `order` (B, T) such that
`x.gather(1, order)` lays tokens out along the curve.
"""

from __future__ import annotations

import torch


def _part1by1(n: torch.Tensor) -> torch.Tensor:
    """Spread the low 16 bits of `n` into even bit positions (Morton helper)."""
    n = n & 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def morton2d(xq: torch.Tensor, yq: torch.Tensor) -> torch.Tensor:
    """Interleave two 16-bit integer coordinate tensors into a Morton code."""
    return _part1by1(xq) | (_part1by1(yq) << 1)


@torch.no_grad()
def serialize_order(
    centers: torch.Tensor,        # (B, T, >=2)  token centers; dims 0,1 used
    mask: torch.Tensor,           # (B, T) bool — True = valid token
    bits: int = 16,
    transpose: bool = False,      # swap the two axes before interleaving
) -> torch.Tensor:
    """Return `order` (B, T) long: argsort of Morton code over valid tokens,
    with invalid tokens forced to the tail.

    Centers are min-max normalised per sample over valid tokens, then quantised
    to a `bits`-bit grid per axis.  `transpose` yields a second, distinct curve
    (axis-swapped) useful for multi-order scans.
    """
    B, T = centers.shape[:2]
    device = centers.device
    cx = centers[..., 1 if transpose else 0]
    cy = centers[..., 0 if transpose else 1]

    maskf = mask.bool()
    # Per-sample min/max over valid tokens (fallback to global if a row is empty).
    big = torch.finfo(centers.dtype).max
    cx_valid = torch.where(maskf, cx, torch.full_like(cx, big))
    cy_valid = torch.where(maskf, cy, torch.full_like(cy, big))
    cx_min = cx_valid.min(dim=1, keepdim=True).values
    cy_min = cy_valid.min(dim=1, keepdim=True).values
    cx_neg = torch.where(maskf, cx, torch.full_like(cx, -big))
    cy_neg = torch.where(maskf, cy, torch.full_like(cy, -big))
    cx_max = cx_neg.max(dim=1, keepdim=True).values
    cy_max = cy_neg.max(dim=1, keepdim=True).values

    scale = float((1 << bits) - 1)
    xq = ((cx - cx_min) / (cx_max - cx_min + 1e-6) * scale).clamp(0, scale).long()
    yq = ((cy - cy_min) / (cy_max - cy_min + 1e-6) * scale).clamp(0, scale).long()

    code = morton2d(xq, yq)                                  # (B, T) long
    # Force invalid tokens to sort last: give them the maximum possible code + their index.
    sentinel = (1 << (2 * bits)) + torch.arange(T, device=device).view(1, T)
    code = torch.where(maskf, code, sentinel)

    order = torch.argsort(code, dim=1)                       # (B, T)
    return order
