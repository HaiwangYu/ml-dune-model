# models/__init__.py
"""
Model registry for DUNE neutrino detector classifiers and backbones.

After refactoring:
- Backbones: pure feature extractors, return [B, 64, 500, 500] dense features
- Classifiers: backbone + classification head, return [B, 4] class logits
"""

# ============ Backbone classes (feature extractors only) ============
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
