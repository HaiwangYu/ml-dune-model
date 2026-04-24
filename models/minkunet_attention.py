# ----------------------------- U-Res + Attention -----------------------------
# Architecture overview:
#   Encoder:   Sparse residual blocks + sparse pooling (efficient on empty space)
#   Bottleneck: Self-attention at small resolution (global context)
#   Decoder:   Sparse upsampling (transposed convolutions) + skip connections + residual blocks
#   Head:      Dense global pooling and classification
#
# The backbone is split into three composable modules:
#
#   FromDense                       Dense Tensor  →  Voxels
#   MinkUNetSparseAttentionCore     Voxels        →  Voxels   (all learnable layers)
#   ToDense                         Voxels        →  Dense Tensor
#
# MinkUNetSparseAttention composes all three for the original Dense → Dense interface.
# Use MinkUNetSparseAttentionCore directly when input already arrives as Voxels
# (e.g. from APASparseDataset), avoiding the from_dense overhead entirely.

from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.transforms import cat
from warpconvnet.nn.modules.sparse_conv import SparseConv2d

from .blocks import (
    ConvBlock2D, ConvTrBlock2D,
    ResidualSparseBlock2D,
    BottleneckSparseAttention2D,
    FromDense, ToDense
)
from .sparse_masking import downsample_union_coords, zero_fill_skip


# ---------------------------------------------------------------------------
# Core backbone: Voxels → Voxels
# ---------------------------------------------------------------------------

class MinkUNetSparseAttentionCore(nn.Module):
    """
    Core U-ResNet backbone operating entirely in sparse Voxels space.

    Architecture:
    - Initial conv at full resolution                      [B, 1,  500×500]
    - Encoder stage 1: strided conv + residual block       [B, 32, 500×500 → 250×250]
    - Encoder stage 2: strided conv + residual block       [B, 32→64, 250×250 → 125×125]
    - Bottleneck: sparse self-attention at 125×125         [B, 64,  125×125]
    - Decoder stage 1: transposed conv + skip + residual   [B, 64,  250×250]
    - Decoder stage 2: transposed conv + skip + residual   [B, 64,  500×500]
    - Final 1×1 conv (feature refinement)                  [B, 64,  500×500]

    Interface: Voxels  →  Voxels
    """

    def __init__(self, *,
                 spatial_encoding: bool = True,
                 flash_attention:  bool = True,
                 encoding_dim:     int  = 32,
                 encoding_range:   float = 125.0):
        super().__init__()

        # ---- Initial convolution (full resolution feature extraction) ----
        self.conv0  = ConvBlock2D(1, 32, kernel_size=3, stride=1)   # [B,1,500,500] → [B,32,500,500]

        # ---- Encoder (2 stages) ----
        self.conv1  = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # 500×500 → 250×250
        self.block1 = ResidualSparseBlock2D(32, 32, kernel_size=3)

        self.conv2  = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # 250×250 → 125×125
        self.block2 = ResidualSparseBlock2D(32, 64, kernel_size=3)  # ch 32 → 64

        # ---- Bottleneck: sparse attention at 125×125 ----
        self.bottleneck = BottleneckSparseAttention2D(
            channels=64, attn_channels=128, heads=4,
            encoding=spatial_encoding, flash=flash_attention,
            encoding_range=encoding_range, encoding_channels=encoding_dim,
        )

        # ---- Decoder (2 stages) ----
        self.convtr5 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)    # 125×125 → 250×250
        self.block6  = ResidualSparseBlock2D(64 + 32, 64, kernel_size=3) # merge skip1

        self.convtr7 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)    # 250×250 → 500×500
        self.block8  = ResidualSparseBlock2D(64 + 32, 64, kernel_size=3) # merge skip0

        # ---- Final 1×1 conv (feature refinement) ----
        self.final = SparseConv2d(64, 64, kernel_size=1, bias=True)

    def forward(self, xs: Voxels) -> Voxels:
        """
        Input:  Voxels — sparse representation of [B, 1, 500, 500]
        Output: Voxels — sparse feature map        [B, 64, 500, 500]
        """
        # ---- Encoder ----
        out      = self.conv0(xs)           # [B, 1,  500×500] → [B, 32, 500×500]
        out_p1   = out                      # skip: [B, 32, 500×500]

        out      = self.conv1(out_p1)       # → [B, 32, 250×250]
        out      = self.block1(out)
        out_b1p2 = out                      # skip: [B, 32, 250×250]

        out      = self.conv2(out_b1p2)     # → [B, 32, 125×125]
        out      = self.block2(out)         # → [B, 64, 125×125]

        # ---- Bottleneck ----
        out = self.bottleneck(out)          # [B, 64, 125×125] (attention)

        # ---- Decoder ----
        out = self.convtr5(out, out_b1p2)   # → [B, 64, 250×250]
        out = cat(out, out_b1p2)            # → [B, 96, 250×250]
        out = self.block6(out)              # → [B, 64, 250×250]

        out = self.convtr7(out, out_p1)     # → [B, 64, 500×500]
        out = cat(out, out_p1)              # → [B, 96, 500×500]
        out = self.block8(out)              # → [B, 64, 500×500]

        return self.final(out)              # → [B, 64, 500×500]




# ---------------------------------------------------------------------------
# True MAE backbone: encoder sees only C_visible, decoder expands to C_union
# ---------------------------------------------------------------------------

class MinkUNetTrueMAECore(nn.Module):
    """
    U-ResNet backbone for true Masked Autoencoding.

    Differences from MinkUNetSparseAttentionCore:
      - forward(vox_visible, vox_union)  (two inputs instead of one)
      - Encoder processes only C_visible (no zero-feature topology hints)
      - Decoder uses C_union as output coordinate reference at each scale,
        so the final output covers all original voxels including masked ones
      - Skip connections are zero-filled at C_masked positions

    Architecture (identical layer definitions to MinkUNetSparseAttentionCore):
      conv0, conv1/block1, conv2/block2  — encoder
      bottleneck                          — sparse attention at 125×125
      convtr5/block6, convtr7/block8      — decoder
      final                               — 1×1 refinement

    Interface:
        forward(vox_visible: Voxels, vox_union: Voxels) -> Voxels [C_union, 64ch]
    """

    def __init__(self, *,
                 spatial_encoding: bool = True,
                 flash_attention:  bool = True,
                 encoding_dim:     int  = 32,
                 encoding_range:   float = 125.0):
        super().__init__()

        self.conv0  = ConvBlock2D(1, 32, kernel_size=3, stride=1)
        self.conv1  = ConvBlock2D(32, 32, kernel_size=2, stride=2)
        self.block1 = ResidualSparseBlock2D(32, 32, kernel_size=3)
        self.conv2  = ConvBlock2D(32, 32, kernel_size=2, stride=2)
        self.block2 = ResidualSparseBlock2D(32, 64, kernel_size=3)

        self.bottleneck = BottleneckSparseAttention2D(
            channels=64, attn_channels=128, heads=4,
            encoding=spatial_encoding, flash=flash_attention,
            encoding_range=encoding_range, encoding_channels=encoding_dim,
        )

        self.convtr5 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)
        self.block6  = ResidualSparseBlock2D(64 + 32, 64, kernel_size=3)
        self.convtr7 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)
        self.block8  = ResidualSparseBlock2D(64 + 32, 64, kernel_size=3)

        self.final = SparseConv2d(64, 64, kernel_size=1, bias=True)

    def forward(self, vox_visible: Voxels, vox_union: Voxels) -> Voxels:
        """
        Parameters
        ----------
        vox_visible : Voxels — only unmasked (visible) voxels, 1-ch charge
        vox_union   : Voxels — all original voxels (C_union), 1-ch charge
                      Used as coordinate reference for the decoder; its
                      feature values are not consumed by any learnable layer.

        Returns
        -------
        Voxels at C_union with 64 feature channels.
        """
        # Dataloader-constructed Voxels arrive with tensor_stride=None.
        # Set it explicitly so warpconvnet's transposed-conv stride assertions pass.
        vox_visible.set_tensor_stride(1)
        vox_union.set_tensor_stride(1)

        # ---- Encoder (C_visible only) ----
        out    = self.conv0(vox_visible)   # [N_vis, 32] @ 500×500
        out_p1 = out                       # skip₀: [N_vis, 32] @ 500×500

        out      = self.conv1(out_p1)      # [N_vis↓, 32] @ 250×250
        out      = self.block1(out)
        out_b1p2 = out                     # skip₁: [N_vis↓, 32] @ 250×250

        out = self.conv2(out_b1p2)         # [N_vis↓↓, 32] @ 125×125
        out = self.block2(out)             # → 64ch

        # ---- Bottleneck (C_visible @ 125×125) ----
        out = self.bottleneck(out)

        # ---- Decoder: expand to C_union at each resolution ----
        # Compute C_union reference at 250×250 by floor-dividing C_union@500 coords.
        union_250 = downsample_union_coords(vox_union, stride=2)

        # Stage 1: 125×125 → 250×250, guided by C_union@250
        out      = self.convtr5(out, union_250)          # [N_union↓, 64] @ 250×250
        skip_250 = zero_fill_skip(out_b1p2, out)         # [N_union↓, 32] @ 250×250
        out      = cat(out, skip_250)                    # → 96ch
        out      = self.block6(out)                      # → 64ch

        # Stage 2: 250×250 → 500×500, guided by C_union@500
        out      = self.convtr7(out, vox_union)          # [N_union, 64] @ 500×500
        skip_500 = zero_fill_skip(out_p1, out)           # [N_union, 32] @ 500×500
        out      = cat(out, skip_500)                    # → 96ch
        out      = self.block8(out)                      # → 64ch

        return self.final(out)                           # [N_union, 64] @ 500×500


# ---------------------------------------------------------------------------
# Full backbone wrapper (backward compatible): Dense Tensor → Dense Tensor
# ---------------------------------------------------------------------------

class MinkUNetSparseAttention(nn.Module):
    """
    Backbone: U-ResNet-style sparse model with attention bottleneck.
    Dense Tensor → Dense Tensor (original, backward-compatible interface).

    Composes:
        self.input  = FromDense   (Dense → Sparse)
        self.core   = MinkUNetSparseAttentionCore     (Sparse → Sparse)
        self.output = ToDense  (Sparse → Dense)

    For pipelines where data arrives as Voxels, use self.core directly and
    call self.output manually with the true batch_size.

    Returns: [B, 64, 500, 500] dense feature map (no classification head)
    """

    def __init__(self, *,
                 spatial_encoding: bool  = True,
                 flash_attention:  bool  = True,
                 encoding_dim:     int   = 32,
                 encoding_range:   float = 125.0,
                 **kwargs):
        super().__init__()
        self.input  = FromDense()
        self.core   = MinkUNetSparseAttentionCore(
            spatial_encoding=spatial_encoding,
            flash_attention=flash_attention,
            encoding_dim=encoding_dim,
            encoding_range=encoding_range,
        )
        self.output = ToDense()

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:  [B, 1, 500, 500] dense tensor
        Output: [B, 64, 500, 500] dense feature map
        """
        B  = x.shape[0]
        xs = self.input(x)
        xs = self.core(xs)
        return self.output(xs, B)


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

class MinkUNetSparseAttentionClassifier(nn.Module):
    """
    Supervised classification wrapper: MinkUNetSparseAttention backbone + head.
    Returns: [B, n_classes] log-probabilities
    """

    def __init__(self, n_classes: int = 4, **backbone_kwargs):
        super().__init__()
        self.backbone = MinkUNetSparseAttention(**backbone_kwargs)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)             # [B, 64, 500, 500]
        logits   = self.head(features)          # [B, n_classes]
        return F.log_softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Backbone variants (different attention settings)
# ---------------------------------------------------------------------------

class MinkUNetSparseAttentionNoEnc(MinkUNetSparseAttention):
    """Variant: spatial positional encoding disabled."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=False, flash_attention=True, **kwargs)


class MinkUNetSparseAttentionNoFlash(MinkUNetSparseAttention):
    """Variant: flash attention disabled."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=True, flash_attention=False, **kwargs)


class MinkUNetSparseAttentionNoFlashEnc(MinkUNetSparseAttention):
    """Variant: flash attention and spatial encoding both disabled."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=False, flash_attention=False, **kwargs)


# ---------------------------------------------------------------------------
# Classifier wrappers for the variants
# ---------------------------------------------------------------------------

class MinkUNetSparseAttentionNoEncClassifier(MinkUNetSparseAttentionClassifier):
    def __init__(self, n_classes: int = 4):
        super().__init__(n_classes=n_classes, spatial_encoding=False, flash_attention=True)


class MinkUNetSparseAttentionNoFlashClassifier(MinkUNetSparseAttentionClassifier):
    def __init__(self, n_classes: int = 4):
        super().__init__(n_classes=n_classes, spatial_encoding=True, flash_attention=False)


class MinkUNetSparseAttentionNoFlashEncClassifier(MinkUNetSparseAttentionClassifier):
    def __init__(self, n_classes: int = 4):
        super().__init__(n_classes=n_classes, spatial_encoding=False, flash_attention=False)
