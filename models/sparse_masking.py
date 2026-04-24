# models/sparse_masking.py

import math
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures


def sparse_block_mask(
    voxels: Voxels,
    masking_frac: float,
    win_ch: int,
    win_tick: int,
) -> tuple[Voxels, torch.Tensor]:
    """
    Block-mask a fraction of active voxels and their spatial neighbours.

    For each batch item independently:
      1. Randomly sample ceil(masking_frac × N_i) voxels as seeds.
      2. Mark all voxels within [±win_ch, ±win_tick] of any seed as masked.
      3. Zero the feature values at masked positions (coordinates unchanged).

    Parameters
    ----------
    voxels      : batched Voxels; coordinates are (channel, tick) int32
    masking_frac: fraction of active voxels used as seeds, in [0, 1]
    win_ch      : half-window radius in the channel direction
    win_tick    : half-window radius in the tick direction

    Returns
    -------
    masked_voxels : Voxels
        Same coordinate structure as input; features zeroed at masked positions.
    mask_bool : BoolTensor, shape (N_total,)
        True at every position that was masked.
    """
    coords  = voxels.coordinate_tensor   # (N_total, 2)  int32, on device
    feats   = voxels.feature_tensor      # (N_total, C)  float, on device
    offsets = voxels.offsets             # (B+1,)        int64, on CPU

    device    = feats.device
    N_total   = feats.shape[0]
    mask_bool = torch.zeros(N_total, dtype=torch.bool, device=device)

    B = len(offsets) - 1
    for i in range(B):
        start = int(offsets[i].item())
        end   = int(offsets[i + 1].item())
        N_i   = end - start
        if N_i == 0:
            continue

        n_seeds = math.ceil(masking_frac * N_i)
        if n_seeds == 0:
            continue

        coords_i  = coords[start:end]                        # (N_i, 2)
        seed_idx  = torch.randperm(N_i, device=device)[:n_seeds]
        seeds     = coords_i[seed_idx]                       # (n_seeds, 2)

        # Vectorised window check:
        #   diff[n, s, d] = coords_i[n, d] - seeds[s, d]
        diff   = coords_i.unsqueeze(1) - seeds.unsqueeze(0)  # (N_i, n_seeds, 2)
        in_win = (diff[:, :, 0].abs() <= win_ch) & \
                 (diff[:, :, 1].abs() <= win_tick)            # (N_i, n_seeds)
        mask_bool[start:end] = in_win.any(dim=1)             # (N_i,)

    # Clone features and zero out masked positions; keep coordinates intact.
    new_feats = feats.clone()
    new_feats[mask_bool] = 0.0

    masked_voxels = Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets),
        batched_features=CatFeatures(new_feats, offsets=offsets),
        offsets=offsets,
    )
    return masked_voxels, mask_bool


# ---------------------------------------------------------------------------
# True MAE masking utilities
# ---------------------------------------------------------------------------

# Coordinate key stride: must exceed the maximum tick (or channel) coordinate
# value present in the data.  DUNE ticks reach ~6000 at full resolution;
# 65536 (2^16) is safe for all intermediate downsampled resolutions too.
_COORD_KEY_STRIDE = 65536


def sparse_block_mask_visible(
    voxels: Voxels,
    masking_frac: float,
    win_ch: int,
    win_tick: int,
) -> tuple[Voxels, torch.Tensor]:
    """
    True MAE masking: compute the same block mask as sparse_block_mask but
    *remove* the masked voxels from the sparse tensor instead of zeroing them.

    Parameters
    ----------
    voxels      : batched Voxels (C_union)
    masking_frac: fraction of voxels used as seeds
    win_ch      : half-window radius in channel direction
    win_tick    : half-window radius in tick direction

    Returns
    -------
    vox_visible : Voxels with only C_visible voxels (C_masked removed)
    mask_bool   : BoolTensor [N_union]  True at C_masked positions
                  Indexes into the *original* voxels.feature_tensor.
    """
    coords  = voxels.coordinate_tensor   # [N_union, 2]
    feats   = voxels.feature_tensor      # [N_union, C]
    offsets = voxels.offsets             # [B+1], CPU

    device    = feats.device
    N_total   = feats.shape[0]
    mask_bool = torch.zeros(N_total, dtype=torch.bool, device=device)

    B = len(offsets) - 1
    for i in range(B):
        start = int(offsets[i].item())
        end   = int(offsets[i + 1].item())
        N_i   = end - start
        if N_i == 0:
            continue
        n_seeds = math.ceil(masking_frac * N_i)
        if n_seeds == 0:
            continue
        coords_i = coords[start:end]
        seed_idx = torch.randperm(N_i, device=device)[:n_seeds]
        seeds    = coords_i[seed_idx]
        diff     = coords_i.unsqueeze(1) - seeds.unsqueeze(0)   # [N_i, n_seeds, 2]
        in_win   = (diff[:, :, 0].abs() <= win_ch) & \
                   (diff[:, :, 1].abs() <= win_tick)             # [N_i, n_seeds]
        mask_bool[start:end] = in_win.any(dim=1)

    # Keep only visible (un-masked) voxels and recompute CSR offsets.
    vis = ~mask_bool

    # warpconvnet rebuilds offsets via torch.bincount(batch_index), which only
    # counts up to max(batch_index).  A batch item with 0 visible voxels never
    # appears in batch_index → it gets silently dropped from the offset array,
    # causing a batch-size mismatch in the decoder.  Force ≥1 visible voxel
    # per item so the batch structure is preserved through the encoder.
    for i in range(B):
        start = int(offsets[i].item())
        end   = int(offsets[i + 1].item())
        if end > start and not vis[start:end].any():
            vis[start] = True   # un-mask the first voxel of this item

    new_coords = coords[vis]
    new_feats  = feats[vis]

    new_off = [0]
    for i in range(B):
        start = int(offsets[i].item())
        end   = int(offsets[i + 1].item())
        new_off.append(new_off[-1] + int(vis[start:end].sum().item()))
    new_offsets = torch.tensor(new_off, dtype=offsets.dtype)

    vox_visible = Voxels(
        batched_coordinates=IntCoords(new_coords, offsets=new_offsets),
        batched_features=CatFeatures(new_feats, offsets=new_offsets),
        offsets=new_offsets,
    )
    return vox_visible, mask_bool


def downsample_union_coords(vox: Voxels, stride: int = 2) -> Voxels:
    """
    Create a coordinate-only reference Voxels at stride-downsampled resolution.

    Coordinates are floor-divided by *stride* and deduplicated per batch item.
    Features are a single zero channel — they are ignored by ConvTrBlock2D
    (which uses the reference purely for its coordinate structure).

    Used to provide C_union reference tensors at intermediate decoder resolutions
    so that transposed convolutions expand to all union coordinates.
    """
    coords  = vox.coordinate_tensor   # [N, 2]
    offsets = vox.offsets             # [B+1], CPU
    device  = coords.device

    new_coords_list: list[torch.Tensor] = []
    new_off = [0]
    B = len(offsets) - 1
    for i in range(B):
        start = int(offsets[i])
        end   = int(offsets[i + 1])
        if start == end:
            new_off.append(new_off[-1])
            continue
        c_down   = (coords[start:end] // stride).int()            # [N_i, 2]
        c_unique = torch.unique(c_down, dim=0)                    # [M_i, 2]
        new_coords_list.append(c_unique)
        new_off.append(new_off[-1] + len(c_unique))

    if new_coords_list:
        new_coords = torch.cat(new_coords_list, dim=0)
    else:
        new_coords = torch.zeros((0, 2), dtype=torch.int32, device=device)
    new_offsets = torch.tensor(new_off, dtype=offsets.dtype)
    new_feats   = torch.zeros(new_coords.shape[0], 1,
                              device=device, dtype=vox.feature_tensor.dtype)

    return Voxels(
        batched_coordinates=IntCoords(new_coords, offsets=new_offsets),
        batched_features=CatFeatures(new_feats, offsets=new_offsets),
        offsets=new_offsets,
        tensor_stride=stride,   # tells warpconvnet this is stride-downsampled coords
    )


def zero_fill_skip(encoder_skip: Voxels, union_ref: Voxels) -> Voxels:
    """
    Broadcast encoder skip features (at C_visible) into C_union coordinate space.

    For each batch item:
      - Coordinates present in encoder_skip  → copy encoder features.
      - Coordinates present only in union_ref → zero features.

    The result is a Voxels with the *same* coordinate structure as union_ref,
    ready to be concatenated with a decoder tensor that also has union_ref coords.

    Parameters
    ----------
    encoder_skip : Voxels at C_visible (encoder skip-connection output)
    union_ref    : Voxels at C_union   (defines the output coordinate structure)

    Returns
    -------
    Voxels at C_union with encoder features at visible positions, zeros elsewhere.
    """
    D      = encoder_skip.feature_tensor.shape[1]
    device = encoder_skip.feature_tensor.device
    dtype  = encoder_skip.feature_tensor.dtype

    union_coords  = union_ref.coordinate_tensor      # [M, 2]
    union_offsets = union_ref.offsets                # [B+1], CPU
    skip_coords   = encoder_skip.coordinate_tensor  # [N, 2]
    skip_feats    = encoder_skip.feature_tensor      # [N, D]
    skip_offsets  = encoder_skip.offsets             # [B+1], CPU

    filled = torch.zeros(union_coords.shape[0], D, device=device, dtype=dtype)

    B = len(union_offsets) - 1
    for i in range(B):
        u0, u1 = int(union_offsets[i]), int(union_offsets[i + 1])
        s0, s1 = int(skip_offsets[i]),  int(skip_offsets[i + 1])

        if s0 == s1 or u0 == u1:
            continue

        uc = union_coords[u0:u1]   # [M_i, 2]
        sc = skip_coords[s0:s1]    # [N_i, 2]
        sf = skip_feats[s0:s1]     # [N_i, D]

        # Encode 2-D integer coordinates as unique 1-D keys.
        u_keys = uc[:, 0].long() * _COORD_KEY_STRIDE + uc[:, 1].long()  # [M_i]
        s_keys = sc[:, 0].long() * _COORD_KEY_STRIDE + sc[:, 1].long()  # [N_i]

        # Sort union keys for binary search.
        sort_ord    = torch.argsort(u_keys)
        sorted_keys = u_keys[sort_ord]                                   # [M_i] sorted

        # For each skip coord, find its position in sorted union keys.
        pos   = torch.searchsorted(sorted_keys, s_keys)                  # [N_i]
        pos   = pos.clamp(0, sorted_keys.shape[0] - 1)
        valid = sorted_keys[pos] == s_keys                               # [N_i] bool

        union_local = sort_ord[pos[valid]]   # indices into uc  [N_valid]
        skip_local  = torch.where(valid)[0]  # indices into sc  [N_valid]

        filled[u0 + union_local] = sf[skip_local]

    return Voxels(
        batched_coordinates=IntCoords(union_coords, offsets=union_offsets),
        batched_features=CatFeatures(filled, offsets=union_offsets),
        offsets=union_offsets,
    )
