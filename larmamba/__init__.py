"""
larmamba — a linear-time (Mamba/SSM) foundation model for 2D LArTPC data.

Reuses PoLAr-MAE's tokenizer, dataset, reconstruction loss, training loop, and
4-probe evaluation; swaps the ViT self-attention encoder for a stack of
bidirectional selective-scan (Mamba) blocks over space-filling-curve-serialized
tokens.  Goal: polarmae-level probe accuracy at bounded peak GPU memory.

Run inside the polarmae env (uvenv-polar-mae) with this repo on PYTHONPATH.
Point a polarmae lightning config at `larmamba.MambaEncoder`.
"""

from larmamba.mamba_encoder import MambaEncoder, MambaTransformer
from larmamba.mamba_block import MambaBlock, BiMambaMixer

__all__ = ["MambaEncoder", "MambaTransformer", "MambaBlock", "BiMambaMixer"]
