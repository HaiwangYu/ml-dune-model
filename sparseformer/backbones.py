"""
Sparse-CNN-family backbones aimed at closing the gap to PoLAr-MAE while keeping
the cheap per-voxel sparse representation (no FPS tokenizer).

Two ideas (and their combo), each a Voxels[1ch] -> Voxels[64ch] core that drops
into the existing MAE pipeline (SparseMAEModel) in place of
MinkUNetSparseAttentionCore:

  Option 1 -- SparseFormerCore: the diagnosis is that the baseline has only ONE
    global-attention block (at the bottleneck) while PoLAr-MAE's transformer has
    12.  So we deepen the bottleneck into a STACK of attention blocks (a sparse
    transformer at the coarsest scale, where all active voxels attend globally),
    optionally adding attention at the mid scale too.  Reuses the repo's proven
    2D sparse attention (BottleneckSparseAttention2D) rather than WarpConvNet's
    3D-oriented PatchAttention.

  Option 2 -- StemUNetCore: replace the single 3x3 conv0 with a multi-scale
    LOCAL-GEOMETRY STEM (parallel conv stacks of increasing receptive field),
    so each voxel carries a richer local neighbourhood descriptor (geometry +
    charge context) before the U-Net -- importing PoLAr-MAE's "tokens carry
    local geometry" benefit without reducing voxel count.

  Combo -- StemSparseFormerCore: both.

All are GPU-only (WarpConvNet sparse ops require CUDA).
"""

from __future__ import annotations

import torch.nn as nn

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.nn.functional.transforms import cat
from warpconvnet.nn.modules.sparse_conv import SparseConv2d

from models.minkunet_attention import MinkUNetSparseAttentionCore
from models.blocks import (
    BottleneckSparseAttention2D, ConvBlock2D, ConvTrBlock2D, ResidualSparseBlock2D,
)
from models.sparse_masking import downsample_union_coords, zero_fill_skip


def _replace_feats(vox: Voxels, feats):
    off = vox.offsets
    return Voxels(
        batched_coordinates=IntCoords(vox.coordinate_tensor, offsets=off),
        batched_features=CatFeatures(feats, offsets=off),
        offsets=off,
    )


class _SConvBNAct(nn.Module):
    """SparseConv2d -> BatchNorm1d(on features) -> optional GELU."""
    def __init__(self, cin, cout, k=3, stride=1, act=True):
        super().__init__()
        self.conv = SparseConv2d(cin, cout, kernel_size=k, stride=stride, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.GELU() if act else None

    def forward(self, x: Voxels) -> Voxels:
        x = self.conv(x)
        f = self.bn(x.feature_tensor)
        if self.act is not None:
            f = self.act(f)
        return _replace_feats(x, f)


# ---------------------------------------------------------------------------
# Option 2: multi-scale local-geometry stem
# ---------------------------------------------------------------------------

class LocalGeometryStem(nn.Module):
    """Parallel 3x3-conv stacks of increasing depth (receptive field 3/5/7),
    concatenated and projected -- a richer per-voxel local descriptor than a
    single 3x3 conv.  1ch -> out_ch (default 32, matching conv0)."""
    def __init__(self, in_ch: int = 1, out_ch: int = 32):
        super().__init__()
        b = max(out_ch // 3, 8)
        self.s1 = nn.ModuleList([_SConvBNAct(in_ch, b, k=3)])                       # RF 3
        self.s2 = nn.ModuleList([_SConvBNAct(in_ch, b, k=3), _SConvBNAct(b, b, k=3)])           # RF 5
        self.s3 = nn.ModuleList([_SConvBNAct(in_ch, b, k=3), _SConvBNAct(b, b, k=3),
                                 _SConvBNAct(b, b, k=3)])                            # RF 7
        self.proj = _SConvBNAct(3 * b, out_ch, k=1)

    @staticmethod
    def _run(seq, x):
        for m in seq:
            x = m(x)
        return x

    def forward(self, x: Voxels) -> Voxels:
        a = self._run(self.s1, x)
        b = self._run(self.s2, x)
        c = self._run(self.s3, x)
        return self.proj(cat(a, b, c))


class StemUNetCore(MinkUNetSparseAttentionCore):
    """Option 2: MinkUNet with conv0 replaced by the multi-scale local stem."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.conv0 = LocalGeometryStem(1, 32)


# ---------------------------------------------------------------------------
# Option 1: deep sparse transformer (stacked global attention)
# ---------------------------------------------------------------------------

class _AttnStack(nn.Module):
    """A stack of BottleneckSparseAttention2D blocks (a sparse transformer)."""
    def __init__(self, channels, attn_channels, heads, n_blocks,
                 encoding=True, encoding_range=125.0, encoding_channels=32, flash=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            BottleneckSparseAttention2D(
                channels=channels, attn_channels=attn_channels, heads=heads,
                encoding=encoding, encoding_range=encoding_range,
                encoding_channels=encoding_channels, flash=flash)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class SparseFormerCore(MinkUNetSparseAttentionCore):
    """Option 1: MinkUNet skeleton, but the single bottleneck-attention block is
    replaced by a STACK of n_bottleneck_blocks (a deep sparse transformer at the
    coarsest scale, where all active voxels attend globally).  Optionally also
    stacks mid_attn_blocks of attention at the 250x250 (32ch) encoder stage."""
    def __init__(self, *, n_bottleneck_blocks: int = 6, mid_attn_blocks: int = 0, **kw):
        super().__init__(**kw)
        sp = kw.get("spatial_encoding", True)
        fl = kw.get("flash_attention", True)
        ed = kw.get("encoding_dim", 32)
        er = kw.get("encoding_range", 125.0)
        self.bottleneck = _AttnStack(64, 128, 4, n_bottleneck_blocks,
                                     encoding=sp, encoding_range=er,
                                     encoding_channels=ed, flash=fl)
        self.mid_attn = None
        if mid_attn_blocks > 0:
            # mid scale is 250x250 with 32ch (after block1); use a wider range
            self.mid_attn = _AttnStack(32, 64, 4, mid_attn_blocks,
                                       encoding=sp, encoding_range=er * 2,
                                       encoding_channels=ed, flash=fl)

    def forward(self, xs: Voxels) -> Voxels:
        # Mirrors MinkUNetSparseAttentionCore.forward, inserting optional mid attn.
        out = self.conv0(xs)
        out_p1 = out
        out = self.conv1(out_p1)
        out = self.block1(out)
        out_b1p2 = out
        if self.mid_attn is not None:
            out_b1p2 = self.mid_attn(out_b1p2)
        out = self.conv2(out_b1p2)
        out = self.block2(out)
        out = self.bottleneck(out)
        from warpconvnet.nn.functional.transforms import cat as _cat
        out = self.convtr5(out, out_b1p2)
        out = _cat(out, out_b1p2)
        out = self.block6(out)
        out = self.convtr7(out, out_p1)
        out = _cat(out, out_p1)
        out = self.block8(out)
        return self.final(out)


class StemSparseFormerCore(SparseFormerCore):
    """Combo: local-geometry stem (Opt 2) + deep sparse transformer (Opt 1)."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.conv0 = LocalGeometryStem(1, 32)


# ---------------------------------------------------------------------------
# Width-configurable combo cores (capacity sweep): stem + deep attention,
# base_ch scales the whole U-Net (orig is base_ch=64).  final projects to 64ch
# (the SSL/SFT heads require 64).  Two variants: standard MAE (single input) and
# true-MAE (dual input, coordinate-removal masking).
# ---------------------------------------------------------------------------

def _wide_layers(self, base_ch, n_bottleneck_blocks, use_stem,
                 spatial_encoding, flash_attention, encoding_dim, encoding_range):
    c0 = base_ch // 2
    c2 = base_ch
    self.conv0 = LocalGeometryStem(1, c0) if use_stem else ConvBlock2D(1, c0, kernel_size=3, stride=1)
    self.conv1 = ConvBlock2D(c0, c0, kernel_size=2, stride=2)
    self.block1 = ResidualSparseBlock2D(c0, c0, kernel_size=3)
    self.conv2 = ConvBlock2D(c0, c0, kernel_size=2, stride=2)
    self.block2 = ResidualSparseBlock2D(c0, c2, kernel_size=3)
    self.bottleneck = _AttnStack(c2, 2 * c2, 4, n_bottleneck_blocks,
                                 encoding=spatial_encoding, encoding_range=encoding_range,
                                 encoding_channels=encoding_dim, flash=flash_attention)
    self.convtr5 = ConvTrBlock2D(c2, c2, kernel_size=2, stride=2)
    self.block6 = ResidualSparseBlock2D(c2 + c0, c2, kernel_size=3)
    self.convtr7 = ConvTrBlock2D(c2, c2, kernel_size=2, stride=2)
    self.block8 = ResidualSparseBlock2D(c2 + c0, c2, kernel_size=3)
    self.final = SparseConv2d(c2, 64, kernel_size=1, bias=True)


class WideComboCore(nn.Module):
    """Width-configurable stem + deep-attention U-Net (standard MAE, 1 input)."""
    def __init__(self, *, base_ch=128, n_bottleneck_blocks=6, use_stem=True,
                 spatial_encoding=True, flash_attention=True,
                 encoding_dim=32, encoding_range=125.0):
        super().__init__()
        _wide_layers(self, base_ch, n_bottleneck_blocks, use_stem,
                     spatial_encoding, flash_attention, encoding_dim, encoding_range)

    def forward(self, xs: Voxels) -> Voxels:
        out = self.conv0(xs)
        out_p1 = out
        out = self.block1(self.conv1(out_p1))
        out_b1p2 = out
        out = self.block2(self.conv2(out_b1p2))
        out = self.bottleneck(out)
        out = self.convtr5(out, out_b1p2)
        out = cat(out, out_b1p2)
        out = self.block6(out)
        out = self.convtr7(out, out_p1)
        out = cat(out, out_p1)
        out = self.block8(out)
        return self.final(out)


class WideComboTrueMAECore(nn.Module):
    """Width-configurable stem + deep-attention U-Net for true MAE (dual input:
    encoder sees visible voxels, decoder expands to the union coords)."""
    def __init__(self, *, base_ch=128, n_bottleneck_blocks=6, use_stem=True,
                 spatial_encoding=True, flash_attention=True,
                 encoding_dim=32, encoding_range=125.0):
        super().__init__()
        _wide_layers(self, base_ch, n_bottleneck_blocks, use_stem,
                     spatial_encoding, flash_attention, encoding_dim, encoding_range)

    def forward(self, vox_visible: Voxels, vox_union: Voxels) -> Voxels:
        vox_visible.set_tensor_stride(1)
        vox_union.set_tensor_stride(1)
        out = self.conv0(vox_visible)
        out_p1 = out
        out = self.block1(self.conv1(out_p1))
        out_b1p2 = out
        out = self.block2(self.conv2(out_b1p2))
        out = self.bottleneck(out)
        union_250 = downsample_union_coords(vox_union, stride=2)
        out = self.convtr5(out, union_250)
        out = cat(out, zero_fill_skip(out_b1p2, out))
        out = self.block6(out)
        out = self.convtr7(out, vox_union)
        out = cat(out, zero_fill_skip(out_p1, out))
        out = self.block8(out)
        return self.final(out)


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

_BACKBONES = {
    "minkunet":          MinkUNetSparseAttentionCore,
    "stem_unet":         StemUNetCore,
    "sparseformer":      SparseFormerCore,
    "stem_sparseformer": StemSparseFormerCore,
    "wide_combo":        WideComboCore,           # standard-MAE, width-configurable
    "wide_combo_true":   WideComboTrueMAECore,    # true-MAE, width-configurable
}


def build_backbone(name: str,
                   spatial_encoding: bool = True,
                   flash_attention: bool = True,
                   encoding_dim: int = 32,
                   encoding_range: float = 125.0,
                   n_bottleneck_blocks: int = 6,
                   mid_attn_blocks: int = 0,
                   base_ch: int = 128,
                   use_stem: bool = True,
                   **extra) -> nn.Module:
    """Build a backbone by name. Standard cores are Voxels[1ch]->Voxels[64ch];
    *_true cores take (vox_visible, vox_union)->Voxels[64ch]. Extra kwargs apply
    only where relevant (n_bottleneck_blocks/mid_attn_blocks for sparseformer;
    base_ch/use_stem/n_bottleneck_blocks for the wide_* cores)."""
    if name not in _BACKBONES:
        raise KeyError(f"unknown backbone '{name}'; choices: {list(_BACKBONES)}")
    cls = _BACKBONES[name]
    kw = dict(spatial_encoding=spatial_encoding, flash_attention=flash_attention,
              encoding_dim=encoding_dim, encoding_range=encoding_range)
    if name in ("sparseformer", "stem_sparseformer"):
        kw["n_bottleneck_blocks"] = n_bottleneck_blocks
        kw["mid_attn_blocks"] = mid_attn_blocks
    elif name in ("wide_combo", "wide_combo_true"):
        kw["n_bottleneck_blocks"] = n_bottleneck_blocks
        kw["base_ch"] = base_ch
        kw["use_stem"] = use_stem
    return cls(**kw)
