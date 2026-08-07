"""Grid tiler: sparse (channel, tick, logq) points -> dense SxS patches.

Native-2D replacement for the FPS+ball-query tokenizer (plan §3.1):
`tile_id = (ch // S, tick // S)`; empty tiles dropped; optional cap with random
subsample (train-time augmentation — eval runs uncapped). Deterministic, pure
torch, CPU-capable.

Empty pixels are filled with EMPTY_VAL = -1.0 == APA2D.log_transform(0), so a
patch is exactly the log-charge image crop of its tile.
"""
from __future__ import annotations

from typing import Optional

import torch

EMPTY_VAL = -1.0
_KEYW = 4096   # tile-column key stride; tick//S < 4096 for any sane S


def tile_events(
    points: torch.Tensor,        # (B, N, 4) padded: (ch, tick, 0, logq)
    lengths: torch.Tensor,       # (B,)
    tile_size: int = 5,
    t_max: int = -1,             # cap on tiles/event; <=0 disables (eval)
    generator: Optional[torch.Generator] = None,
):
    """Returns
        patches     (B, T, S*S) float — log-charge crops, EMPTY_VAL-filled
        tile_coords (B, T, 2)   long  — (tile_row, tile_col) = (ch//S, tick//S)
        tile_mask   (B, T)      bool  — True = real tile
        vox_tile    (B, N)      long  — tile slot of each voxel (-1 = pad/dropped)
    """
    B, N, _ = points.shape
    S = tile_size
    dev = points.device

    per_ev = []
    T_max_batch = 1
    for b in range(B):
        L = int(lengths[b])
        c = points[b, :L, :2].long()
        q = points[b, :L, 3]
        key = (c[:, 0] // S) * _KEYW + (c[:, 1] // S)
        uniq, inv = torch.unique(key, return_inverse=True)
        T_b = len(uniq)

        if t_max > 0 and T_b > t_max:
            perm = torch.randperm(T_b, device=dev, generator=generator)
            remap = torch.full((T_b,), -1, dtype=torch.long, device=dev)
            remap[perm[:t_max]] = torch.arange(t_max, device=dev)
            inv = remap[inv]                       # -1 for voxels in dropped tiles
            uniq = uniq[perm[:t_max]]
            T_b = t_max

        keep = inv >= 0
        pix = (c[:, 0] % S) * S + (c[:, 1] % S)    # (L,) within-tile pixel index
        patch = torch.full((T_b, S * S), EMPTY_VAL, dtype=points.dtype, device=dev)
        patch[inv[keep], pix[keep]] = q[keep]

        coords = torch.stack([uniq // _KEYW, uniq % _KEYW], dim=1)  # (T_b, 2)
        vt = torch.full((N,), -1, dtype=torch.long, device=dev)
        vt[:L] = inv
        per_ev.append((patch, coords, vt))
        T_max_batch = max(T_max_batch, T_b)

    patches = torch.full((B, T_max_batch, S * S), EMPTY_VAL, dtype=points.dtype, device=dev)
    tile_coords = torch.zeros(B, T_max_batch, 2, dtype=torch.long, device=dev)
    tile_mask = torch.zeros(B, T_max_batch, dtype=torch.bool, device=dev)
    vox_tile = torch.full((B, N), -1, dtype=torch.long, device=dev)
    for b, (patch, coords, vt) in enumerate(per_ev):
        T_b = patch.shape[0]
        patches[b, :T_b] = patch
        tile_coords[b, :T_b] = coords
        tile_mask[b, :T_b] = True
        vox_tile[b] = vt
    return patches, tile_coords, tile_mask, vox_tile


def untile(patches: torch.Tensor, tile_coords: torch.Tensor, tile_mask: torch.Tensor,
           tile_size: int):
    """Inverse of tile_events for testing: returns per-event (coords, q) of all
    non-EMPTY pixels."""
    S = tile_size
    out = []
    for b in range(patches.shape[0]):
        p = patches[b][tile_mask[b]]               # (T_b, S*S)
        tc = tile_coords[b][tile_mask[b]]          # (T_b, 2)
        hit = p > EMPTY_VAL + 1e-6
        t_idx, pix = hit.nonzero(as_tuple=True)
        ch = tc[t_idx, 0] * S + pix // S
        tk = tc[t_idx, 1] * S + pix % S
        out.append((torch.stack([ch, tk], dim=1), p[t_idx, pix]))
    return out
