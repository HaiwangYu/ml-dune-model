# models/__init__.py
"""
Model registry for DUNE neutrino detector classifiers and backbones.

Backbone naming convention:
- `MinkUNetSparseAttentionCore` / `MinkUNetTrueMAECore`: Voxels → Voxels sparse cores
  used directly by Voxels-native pipelines (e.g. MAE pretraining).
- `MinkUNetSparseAttention*`:                            Tensor → Tensor wrappers
  (Dense input/output via `FromDense` / `ToDense` boundary layers) for Dense
  dataloaders, DINO, and the supervised classifier head.

- Backbones: pure feature extractors, return [B, 64, H, W] dense features
  (or Voxels for the *Core variants).
- Classifiers: backbone + classification head, return [B, n_classes] class logits.
"""

# ============ Sparse cores (Voxels → Voxels) ============
from .minkunet_attention import (
    MinkUNetSparseAttentionCore,
    MinkUNetTrueMAECore,
)

# ============ Backbone classes (Dense Tensor → Dense Tensor) ============
from .minkunet import MinkUNetSparse
from .minkunet_attention import (
    MinkUNetSparseAttention,
    MinkUNetSparseAttentionNoEnc,
    MinkUNetSparseAttentionNoFlash,
    MinkUNetSparseAttentionNoFlashEnc,
)

# ============ Classifier wrapper classes (backbone + head for supervised training) ============
from .minkunet import MinkUNetSparseClassifier
from .minkunet_attention import (
    MinkUNetSparseAttentionClassifier,
    MinkUNetSparseAttentionNoEncClassifier,
    MinkUNetSparseAttentionNoFlashClassifier,
    MinkUNetSparseAttentionNoFlashEncClassifier,
)

# ============ MODEL_REGISTRY (classifiers for backward compatibility with training.py) ============
MODEL_REGISTRY = {
    # Backbone with sparse attention + classification head
    "attn_default":     MinkUNetSparseAttentionClassifier,

    # Variants of sparse attention module + classification head
    "attn_noenc":       MinkUNetSparseAttentionNoEncClassifier,
    "attn_noflash":     MinkUNetSparseAttentionNoFlashClassifier,
    "attn_noflashenc":  MinkUNetSparseAttentionNoFlashEncClassifier,

    # Backbone without attention + classification head
    "base":             MinkUNetSparseClassifier,
}

# ============ BACKBONE_REGISTRY (exposed for DINO and other self-supervised methods) ============
# Dense Tensor → Dense Tensor backbones (DINO consumes [B,1,H,W] images).
BACKBONE_REGISTRY = {
    # Backbone with sparse attention
    "attn_default":     MinkUNetSparseAttention,

    # Variants of sparse attention module
    "attn_noenc":       MinkUNetSparseAttentionNoEnc,
    "attn_noflash":     MinkUNetSparseAttentionNoFlash,
    "attn_noflashenc":  MinkUNetSparseAttentionNoFlashEnc,

    # Backbone without attention
    "base":             MinkUNetSparse,
}
