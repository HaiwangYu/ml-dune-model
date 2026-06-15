"""
sparseformer — sparse-CNN-family backbones to close the gap to PoLAr-MAE while
keeping cheap per-voxel sparse inference (no FPS tokenizer).

Plugs into the existing mae pipeline (SparseMAEModel) via a configurable
backbone (config field `backbone_name` + `backbone_kwargs`). Variants:
  - minkunet           : baseline control (the current backbone)
  - stem_unet          : Option 2 — multi-scale local-geometry stem + U-Net
  - sparseformer       : Option 1 — deep stacked-attention (sparse transformer)
  - stem_sparseformer  : combo

Runs in the ml-dune-model env (torch 2.10 + WarpConvNet, GPU-only).
"""

from sparseformer.backbones import (
    build_backbone,
    LocalGeometryStem, StemUNetCore, SparseFormerCore, StemSparseFormerCore,
)

__all__ = [
    "build_backbone",
    "LocalGeometryStem", "StemUNetCore", "SparseFormerCore", "StemSparseFormerCore",
]
