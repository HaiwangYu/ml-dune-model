"""
MambaTransformer + MambaEncoder.

`MambaEncoder` is a drop-in replacement for polarmae's `TransformerEncoder`:
same `__init__` signature and the same `prepare_tokens` / `transformer` /
`pos_embed` / `embed_dim` interface that `PoLArMAE` and `APA2DProbeCallback`
rely on.  Internally it swaps the attention transformer for a stack of
linear-time bidirectional Mamba blocks.

The only extra plumbing: `prepare_tokens` stashes the token `centers` on the
transformer so its forward can compute the space-filling-curve order (the
polarmae `transformer(x, pos, mask)` call signature carries no geometry).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from polarmae.layers.encoder import TransformerEncoder
from polarmae.layers.masking import MaskedLayerNorm
from polarmae.layers.transformer import TransformerOutput

from larmamba.mamba_block import MambaBlock
from larmamba.serialize import serialize_order


_ARCH = {
    "vit_tiny":  dict(embed_dim=192, depth=12, mlp_ratio=4.0),
    "vit_small": dict(embed_dim=384, depth=12, mlp_ratio=4.0),
    "vit_base":  dict(embed_dim=768, depth=12, mlp_ratio=4.0),
}


class _Identity(nn.Module):
    def forward(self, x, mask=None):
        return x


class MambaTransformer(nn.Module):
    """Bidirectional-Mamba stand-in for polarmae's `Transformer`.

    Matches the `forward(x, pos_x, x_mask, ...) -> TransformerOutput` contract
    so `PoLArMAE.compute_loss` and the probe callback work unchanged.  Token
    centers are read from `self._centers` (set by `MambaEncoder.prepare_tokens`).
    """

    def __init__(self, embed_dim=384, depth=12, mlp_ratio=4.0, drop_rate=0.0,
                 drop_path_rate=0.0, add_pos_at_every_layer=False, postnorm=True,
                 d_state=16, d_conv=4, expand=2, **_ignored):
        super().__init__()
        self.embed_dim = embed_dim
        self.add_pos_at_every_layer = add_pos_at_every_layer

        if isinstance(drop_path_rate, (list, tuple)):
            dpr = list(drop_path_rate)
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            MambaBlock(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                       drop_path=dpr[i], d_state=d_state, d_conv=d_conv, expand=expand)
            for i in range(depth)
        ])
        self.norm = MaskedLayerNorm(embed_dim) if postnorm else _Identity()
        self._centers: Optional[torch.Tensor] = None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def set_centers(self, centers: torch.Tensor):
        self._centers = centers

    def forward(
        self,
        x: torch.Tensor,
        pos_x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        pos_y: torch.Tensor | None = None,
        y_mask: torch.Tensor | None = None,
        rpb: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
        return_ffns: bool = False,
    ) -> TransformerOutput:
        assert y is None, "MambaTransformer does not support cross-attention (y must be None)"
        B_, T, C = x.shape
        if x_mask is None:
            x_mask = x.new_ones(B_, T, dtype=torch.bool)

        # Serialization order from stashed centers (fallback: identity order).
        if self._centers is not None and self._centers.shape[:2] == (B_, T):
            order = serialize_order(self._centers, x_mask.bool())
        else:
            order = torch.arange(T, device=x.device).unsqueeze(0).expand(B_, -1)

        hidden_states = [] if return_hidden_states else None

        if not self.add_pos_at_every_layer:
            x = x + pos_x
        for blk in self.blocks:
            if self.add_pos_at_every_layer:
                x = x + pos_x
            x = blk(x, x_mask, order)
            if return_hidden_states:
                hidden_states.append(x)

        x = self.norm(x, x_mask)
        return TransformerOutput(x, None, hidden_states, None, None)


class MambaEncoder(TransformerEncoder):
    """polarmae TransformerEncoder with the attention transformer replaced by a
    bidirectional-Mamba stack.  Same constructor args, plus `mamba_kwargs`."""

    def __init__(
        self,
        num_channels: int = 4,
        arch: str = "vit_small",
        masking_ratio: float = 0.6,
        masking_type: str = "rand",
        voxel_size: float = 5,
        tokenizer_kwargs: dict = {},
        transformer_kwargs: dict = {},
        apply_relative_position_bias: bool = False,
        mamba_kwargs: dict = {},
    ):
        # Build the parent (tokenizer, pos_embed, masking, + a throwaway attn transformer).
        super().__init__(
            num_channels=num_channels,
            arch=arch,
            masking_ratio=masking_ratio,
            masking_type=masking_type,
            voxel_size=voxel_size,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs,
            apply_relative_position_bias=False,   # Mamba has no attn heads for RPB
        )
        arch_cfg = dict(_ARCH[arch])
        depth = transformer_kwargs.get("depth", arch_cfg["depth"])
        self.transformer = MambaTransformer(
            embed_dim=arch_cfg["embed_dim"],
            depth=depth,
            mlp_ratio=transformer_kwargs.get("mlp_ratio", arch_cfg["mlp_ratio"]),
            drop_rate=transformer_kwargs.get("drop_rate", 0.0),
            drop_path_rate=transformer_kwargs.get("drop_path_rate", 0.0),
            add_pos_at_every_layer=transformer_kwargs.get("add_pos_at_every_layer", False),
            postnorm=transformer_kwargs.get("postnorm", True),
            **mamba_kwargs,
        )
        self.embed_dim = self.transformer.embed_dim

    def prepare_tokens(self, points, lengths, ids=None, endpoints=None):
        out = super().prepare_tokens(points, lengths, ids, endpoints)
        # Stash centers so MambaTransformer.forward can build the curve order.
        self.transformer.set_centers(out["centers"])
        return out
