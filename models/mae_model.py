# models/mae_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.nn.modules.sparse_conv import SparseConv2d

from .minkunet_attention import MinkUNetSparseAttentionCore, MinkUNetTrueMAECore


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def voxels_to_device(vox: Voxels, device: torch.device) -> Voxels:
    """
    Move a Voxels object to the target device.

    offsets always stays on CPU (WarpConvNet CSR requirement);
    coordinate_tensor and feature_tensor are moved to device.
    """
    coords  = vox.coordinate_tensor.to(device)
    feats   = vox.feature_tensor.to(device)
    offsets = vox.offsets   # keep on CPU
    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


# ---------------------------------------------------------------------------
# Sparse global average pooling
# ---------------------------------------------------------------------------

def sparse_global_avg_pool(vox: Voxels) -> Tensor:
    """
    Compute the mean feature vector for each batch item.

    Returns
    -------
    Tensor of shape [B, C] on the same device as vox.feature_tensor.
    """
    feats   = vox.feature_tensor   # (N_total, C)
    offsets = vox.offsets          # (B+1,), CPU
    B       = len(offsets) - 1
    C       = feats.shape[1]
    device  = feats.device

    counts    = (offsets[1:] - offsets[:-1]).to(device=device, dtype=torch.float32)   # (B,)
    batch_idx = torch.repeat_interleave(
        torch.arange(B, device=device),
        counts.long(),
    )  # (N_total,)

    pooled = torch.zeros(B, C, device=device, dtype=feats.dtype)
    pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(feats), feats)
    pooled = pooled / counts.unsqueeze(1).clamp(min=1.0)
    return pooled   # [B, C]


# ---------------------------------------------------------------------------
# Feature replacement helper
# ---------------------------------------------------------------------------

def _replace_features(vox: Voxels, new_feats: Tensor) -> Voxels:
    """Return a new Voxels with the same coords/offsets but replaced features."""
    offsets = vox.offsets  # keep on CPU
    return Voxels(
        batched_coordinates=IntCoords(vox.coordinate_tensor, offsets=offsets),
        batched_features=CatFeatures(new_feats, offsets=offsets),
        offsets=offsets,
    )


# ---------------------------------------------------------------------------
# Charge normalization helpers
# ---------------------------------------------------------------------------

def log1p_voxels(vox: Voxels) -> Voxels:
    """Apply log1p to all feature values.  Maps raw ADC → log(ADC+1).

    Compresses the ~350× ADC dynamic range to ~10×, making reconstruction
    targets uniform across the charge spectrum.  Pair with expm1_voxels to
    recover raw ADC for visualization or downstream use.
    """
    return _replace_features(vox, torch.log1p(vox.feature_tensor))


def expm1_voxels(vox: Voxels) -> Voxels:
    """Apply expm1 to all feature values.  Inverts log1p_voxels: log(ADC+1) → ADC."""
    return _replace_features(vox, torch.expm1(vox.feature_tensor))


# ---------------------------------------------------------------------------
# Sparse CNN classification head
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pixel-level PID class mapping
# ---------------------------------------------------------------------------

PIXEL_PID_CLASS_NAMES = ["track", "shower", "other"]
PIXEL_PID_N_CLASSES = len(PIXEL_PID_CLASS_NAMES)
_PIXEL_PID_IGNORE = -1   # used as ignore_index in cross-entropy

_TRACK_PDGS = {13, -13, 2212, 211, -211}    # μ, p, π±
_ELEC_PDGS  = {11, -11}                       # e±
_GAMMA_PDG  = 22

_BLIP_CONNECT_DIST = 5.0
_BLIP_MAX_PIXELS   = 30


def pdg_to_pixel_class(
    pid_labels,            # np.ndarray[N] int32 raw PDG codes (0 = no truth)
    positions,             # np.ndarray[N, 2] int32 (channel, tick) — same image only
    connect_dist: float = _BLIP_CONNECT_DIST,
    blip_max_pixels: int = _BLIP_MAX_PIXELS,
):
    """
    Map per-voxel PDG codes to pixel-PID class indices for a SINGLE image.

    Classes:
      0 = track   (μ±, proton, π±)
      1 = shower  (e±, shower-γ)
      2 = other   (blip-γ, everything else)
     -1 = no truth (pdg == 0) — caller should pass to CE as ignore_index

    Gamma pixels (PDG 22) are split into shower vs blip via connected-component
    analysis over the (γ + e±) pixels in this image: small clusters
    (<= blip_max_pixels) → blip → other; larger clusters → shower.

    Returns
    -------
    np.ndarray[N] int64
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    pdg = pid_labels.astype(np.int32, copy=False)
    n = len(pdg)
    out = np.full(n, 2, dtype=np.int64)   # default → other (2)

    out[np.isin(pdg, list(_TRACK_PDGS))] = 0   # track
    out[np.isin(pdg, list(_ELEC_PDGS))]  = 1   # shower (e±)

    gamma_local = np.where(pdg == _GAMMA_PDG)[0]
    if len(gamma_local) > 0:
        elec_local = np.where(np.isin(pdg, list(_ELEC_PDGS)))[0]
        em_local   = np.concatenate([gamma_local, elec_local])
        em_pos     = positions[em_local].astype(float)
        n_em       = len(em_local)

        if n_em == 1:
            # single gamma pixel → blip → other
            out[gamma_local[0]] = 2
        else:
            tree  = cKDTree(em_pos)
            pairs = tree.query_pairs(connect_dist)
            if pairs:
                ra, ca = zip(*pairs)
                ra, ca = list(ra), list(ca)
                r = ra + ca
                c = ca + ra
                adj = csr_matrix(
                    (np.ones(len(r), dtype=np.float32), (r, c)),
                    shape=(n_em, n_em),
                )
            else:
                adj = csr_matrix((n_em, n_em), dtype=np.float32)
            _, comp = connected_components(adj, directed=False)
            unique, counts = np.unique(comp, return_counts=True)
            comp_size = dict(zip(unique.tolist(), counts.tolist()))
            n_gamma = len(gamma_local)
            for local_i, ci in zip(gamma_local, comp[:n_gamma]):
                if comp_size[ci] <= blip_max_pixels:
                    out[local_i] = 2   # blip → other
                else:
                    out[local_i] = 1   # shower γ

    out[pdg == 0] = _PIXEL_PID_IGNORE
    return out


# ---------------------------------------------------------------------------
# Sparse heads
# ---------------------------------------------------------------------------

class DensePixelHead(nn.Module):
    """
    Dense MLP version of SparsePixelHead.  Used by the offline "extract once,
    train head on pool" SFT pattern (rec #B from the v2 post-mortem): the
    backbone produces feature tensors per batch, those features are pooled
    into one numpy array, then this head is trained on that pool with no
    further backbone forwards.

    Operates on a plain [N, in_ch] feature tensor (no Voxels container).
    Forward returns [N, n_classes] logits.
    """

    def __init__(self, in_ch: int = 64, n_classes: int = 3, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class SparsePixelHead(nn.Module):
    """
    Pointwise sparse classification head.  No spatial mixing — three 1×1
    sparse convolutions (in_ch → 128 → 128 → n_classes), then return the
    raw feature tensor as per-voxel logits.

    Output is a Tensor of shape [N_total, n_classes] (one logit row per
    active voxel, in the same row order as the input Voxels).

    Parameters
    ----------
    in_ch     : input feature channels (64 for backbone output, 1 for raw charge)
    n_classes : number of output classes
    """

    def __init__(self, in_ch: int = 64, n_classes: int = 3):
        super().__init__()
        self.conv1 = SparseConv2d(in_ch, 128, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm1d(128)
        self.conv2 = SparseConv2d(128, 128, kernel_size=1, bias=False)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = SparseConv2d(128, n_classes, kernel_size=1, bias=True)

    def forward(self, vox: Voxels) -> Tensor:
        vox = self.conv1(vox)
        vox = _replace_features(vox, F.relu(self.bn1(vox.feature_tensor)))
        vox = self.conv2(vox)
        vox = _replace_features(vox, F.relu(self.bn2(vox.feature_tensor)))
        vox = self.conv3(vox)
        return vox.feature_tensor          # [N_total, n_classes]


class SparseCNNHead(nn.Module):
    """
    Sparse CNN classification head.

    Three stride-2 3×3 sparse conv layers with expanding channels (64→128→256),
    followed by a 1×1 conv at 256 ch, sparse global average pooling, and a
    linear classifier.  Mirrors the standard CNN pattern of doubling channels
    while halving spatial resolution at each stride-2 stage.

    Each stride-2 conv halves the coordinate grid, so after three stages the
    pooling operates over ~N/64 voxels — more spatially discriminative.

    The architecture is identical for both use cases; only in_ch differs:
      in_ch=64  — applied to backbone features
      in_ch=1   — applied to raw charge directly

    Parameters
    ----------
    in_ch     : input feature channels (64 for backbone output, 1 for raw charge)
    n_classes : number of output classes
    """

    def __init__(self, in_ch: int = 64, n_classes: int = 3):
        super().__init__()
        self.conv1 = SparseConv2d(in_ch,  64,  kernel_size=3, stride=2, bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = SparseConv2d(64,  128, kernel_size=3, stride=2, bias=False)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = SparseConv2d(128, 256, kernel_size=3, stride=2, bias=False)
        self.bn3   = nn.BatchNorm1d(256)
        self.conv4 = SparseConv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.bn4   = nn.BatchNorm1d(256)
        self.fc    = nn.Linear(256, n_classes)

    def forward(self, vox: Voxels) -> Tensor:
        vox    = self.conv1(vox)
        vox    = _replace_features(vox, F.relu(self.bn1(vox.feature_tensor)))
        vox    = self.conv2(vox)
        vox    = _replace_features(vox, F.relu(self.bn2(vox.feature_tensor)))
        vox    = self.conv3(vox)
        vox    = _replace_features(vox, F.relu(self.bn3(vox.feature_tensor)))
        vox    = self.conv4(vox)
        vox    = _replace_features(vox, F.relu(self.bn4(vox.feature_tensor)))
        pooled = sparse_global_avg_pool(vox)   # [B, 256]
        return self.fc(pooled)                  # [B, n_classes]


# ---------------------------------------------------------------------------
# MAE model
# ---------------------------------------------------------------------------

class SparseMAEModel(nn.Module):
    """
    Sparse Masked Auto-Encoder for DUNE wire-plane data.

    Components
    ----------
    backbone            : MinkUNetSparseAttentionCore  (Voxels[1 ch] → Voxels[64 ch])
    charge_head         : 1×1 SparseConv2d(64→1)       SSL reconstruction head
    pixel_pid_head      : SparsePixelHead(in_ch=64)    SFT pixel-PID head on backbone features
    ref_pixel_pid_head  : SparsePixelHead(in_ch=1)     SFT reference on raw charge

    Usage
    -----
    SSL training:
        pred   = model.forward_ssl(masked_voxels)
        loss   = F.l1_loss(pred.feature_tensor[mask_bool],
                           original.feature_tensor[mask_bool])

    SFT training (backbone frozen):
        model.freeze_backbone()
        logits     = model.forward_sft(voxels)      # [N_total, n_classes]
        logits_ref = model.forward_sft_ref(voxels)  # [N_total, n_classes]
        loss = F.cross_entropy(logits, pixel_targets, ignore_index=-1)
        model.unfreeze_backbone()
    """

    def __init__(
        self,
        n_classes: int = PIXEL_PID_N_CLASSES,
        spatial_encoding: bool = True,
        flash_attention:  bool = True,
        encoding_dim:     int  = 32,
        encoding_range:   float = 300.0,
    ):
        super().__init__()

        self.backbone = MinkUNetSparseAttentionCore(
            spatial_encoding=spatial_encoding,
            flash_attention=flash_attention,
            encoding_dim=encoding_dim,
            encoding_range=encoding_range,
        )

        # SSL head: 64 → 1 feature channel (charge reconstruction)
        self.charge_head = SparseConv2d(64, 1, kernel_size=1, bias=True)

        # Coordinate reconstruction head: predicts (Δchannel, Δtick) per voxel
        # Used in patch-MAE mode to reconstruct local point geometry (PoLAr-MAE).
        self.coord_head = SparseConv2d(64, 2, kernel_size=1, bias=True)

        # SFT pixel-PID head on backbone features (64 ch)
        self.pixel_pid_head = SparsePixelHead(in_ch=64, n_classes=n_classes)

        # Reference SFT pixel-PID head on raw 1-ch charge (no backbone)
        self.ref_pixel_pid_head = SparsePixelHead(in_ch=1, n_classes=n_classes)

    # ------------------------------------------------------------------ #

    def forward_ssl(self, masked_voxels: Voxels) -> Voxels:
        """
        Forward pass for SSL (self-supervised) training.

        Input  : masked Voxels with 1 feature channel
        Output : Voxels with 1 feature channel (predicted charge amplitude)
        """
        feats = self.backbone(masked_voxels)    # Voxels [64 ch]
        return self.charge_head(feats)          # Voxels [1 ch]

    def forward_sft(self, voxels: Voxels) -> Tensor:
        """
        Forward pass for SFT using backbone features.  Backbone runs under
        torch.no_grad() for memory efficiency.

        Output : [N_total, n_classes] per-voxel logits
        """
        with torch.no_grad():
            feats = self.backbone(voxels)
        return self.pixel_pid_head(feats)

    def forward_sft_ref(self, voxels: Voxels) -> Tensor:
        """Reference SFT on raw 1-ch charge.  Returns [N_total, n_classes]."""
        return self.ref_pixel_pid_head(voxels)

    # ------------------------------------------------------------------ #

    def freeze_backbone(self):
        """Freeze backbone weights and BatchNorm running stats for SFT training.

        requires_grad_(False) stops weight updates, but BatchNorm1d still
        updates running_mean/running_var in train() mode regardless of grads.
        Calling .eval() on the backbone prevents that stat drift, so the
        saved checkpoint has running stats calibrated to SSL data, not SFT data.
        """
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def unfreeze_backbone(self):
        """Re-enable gradient flow and restore BatchNorm to training mode."""
        self.backbone.requires_grad_(True)
        self.backbone.train()

    def reset_sft_head(self):
        """Re-initialize both SFT heads to random weights.

        Call at the start of each SFT evaluation so both heads probe
        the current backbone features and raw charge from a clean slate,
        giving an unbiased comparison at each SSL checkpoint.
        """
        for m in self.pixel_pid_head.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for m in self.ref_pixel_pid_head.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()


# ---------------------------------------------------------------------------
# True MAE model
# ---------------------------------------------------------------------------

class SparseTrueMAEModel(nn.Module):
    """
    Sparse Masked Autoencoder with true coordinate-removal masking.

    The encoder receives only C_visible (unmasked voxels).  The decoder
    expands back to C_union (all original voxels) via union-guided transposed
    convolutions and zero-filled skip connections.  This prevents the network
    from using topology hints at masked positions.

    Components
    ----------
    backbone            : MinkUNetTrueMAECore  (C_visible, C_union → C_union [64ch])
    charge_head         : 1×1 SparseConv2d(64→1)   SSL reconstruction head
    pixel_pid_head      : SparsePixelHead(in_ch=64) SFT pixel-PID head on backbone features
    ref_pixel_pid_head  : SparsePixelHead(in_ch=1)  SFT reference on raw charge

    SSL usage:
        vox_visible, mask_bool = sparse_block_mask_visible(vox, ...)
        pred = model.forward_ssl(vox_visible, vox)
        loss = weighted_l1(pred.feature_tensor, vox.feature_tensor, mask_bool)

    SFT usage (no masking):
        model.freeze_backbone()
        logits = model.forward_sft(vox)   # [N_total, n_classes] per-voxel
    """

    def __init__(
        self,
        n_classes:        int   = PIXEL_PID_N_CLASSES,
        spatial_encoding: bool  = True,
        flash_attention:  bool  = True,
        encoding_dim:     int   = 32,
        encoding_range:   float = 300.0,
    ):
        super().__init__()
        self.backbone = MinkUNetTrueMAECore(
            spatial_encoding=spatial_encoding,
            flash_attention=flash_attention,
            encoding_dim=encoding_dim,
            encoding_range=encoding_range,
        )
        self.charge_head        = SparseConv2d(64, 1, kernel_size=1, bias=True)
        self.coord_head         = SparseConv2d(64, 2, kernel_size=1, bias=True)
        self.pixel_pid_head     = SparsePixelHead(in_ch=64, n_classes=n_classes)
        self.ref_pixel_pid_head = SparsePixelHead(in_ch=1,  n_classes=n_classes)

    def forward_ssl(self, vox_visible: Voxels, vox_union: Voxels) -> Voxels:
        """
        SSL forward pass.

        Parameters
        ----------
        vox_visible : Voxels — visible voxels only (output of sparse_block_mask_visible)
        vox_union   : Voxels — all original voxels (C_union); also the reconstruction target

        Returns
        -------
        Voxels at C_union with 1 feature channel (predicted log1p charge).
        pred.feature_tensor has the same length as vox_union.feature_tensor,
        so mask_bool (which indexes into C_union) applies directly.
        """
        feats = self.backbone(vox_visible, vox_union)   # [N_union, 64]
        return self.charge_head(feats)                   # [N_union, 1]

    def forward_sft(self, voxels: Voxels) -> Tensor:
        """SFT forward using backbone features (no masking; visible == union).

        Returns [N_total, n_classes] per-voxel logits.
        """
        with torch.no_grad():
            feats = self.backbone(voxels, voxels)
        return self.pixel_pid_head(feats)

    def forward_sft_ref(self, voxels: Voxels) -> Tensor:
        """SFT reference on raw 1-ch charge.  Returns [N_total, n_classes]."""
        return self.ref_pixel_pid_head(voxels)

    def freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def unfreeze_backbone(self):
        self.backbone.requires_grad_(True)
        self.backbone.train()

    def reset_sft_head(self):
        for m in self.pixel_pid_head.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for m in self.ref_pixel_pid_head.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
