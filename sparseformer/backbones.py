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
from models.blocks import BottleneckSparseAttention2D


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
# Registry / factory
# ---------------------------------------------------------------------------

_COMMON = ("spatial_encoding", "flash_attention", "encoding_dim", "encoding_range")
_BACKBONES = {
    "minkunet":          MinkUNetSparseAttentionCore,
    "stem_unet":         StemUNetCore,
    "sparseformer":      SparseFormerCore,
    "stem_sparseformer": StemSparseFormerCore,
}


def build_backbone(name: str,
                   spatial_encoding: bool = True,
                   flash_attention: bool = True,
                   encoding_dim: int = 32,
                   encoding_range: float = 125.0,
                   n_bottleneck_blocks: int = 6,
                   mid_attn_blocks: int = 0,
                   **extra) -> nn.Module:
    """Build a Voxels[1ch]->Voxels[64ch] backbone by name. Extra kwargs
    (n_bottleneck_blocks, mid_attn_blocks) apply only to the sparseformer cores."""
    if name not in _BACKBONES:
        raise KeyError(f"unknown backbone '{name}'; choices: {list(_BACKBONES)}")
    cls = _BACKBONES[name]
    kw = dict(spatial_encoding=spatial_encoding, flash_attention=flash_attention,
              encoding_dim=encoding_dim, encoding_range=encoding_range)
    if name in ("sparseformer", "stem_sparseformer"):
        kw["n_bottleneck_blocks"] = n_bottleneck_blocks
        kw["mid_attn_blocks"] = mid_attn_blocks
    return cls(**kw)
