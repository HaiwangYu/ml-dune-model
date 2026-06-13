"""
Bidirectional Mamba mixer + block for serialized point-cloud tokens.

The block is a drop-in replacement for polarmae's attention `Block`: same
residual structure (norm -> mixer -> droppath, norm -> mlp -> droppath),
reusing polarmae's `MaskedLayerNorm`, `MaskedDropPath`, `Mlp`.  The only change
is the token mixer: instead of O(T^2) self-attention, a linear-time bidirectional
selective scan over tokens laid out along a space-filling curve.

The block receives the serialization `order` (and `x_mask`) from
`MambaTransformer` — it does NOT match polarmae's Block.forward signature
because only `MambaTransformer` ever calls it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from polarmae.layers.masking import MaskedLayerNorm, MaskedDropPath
from polarmae.layers.mlp import Mlp

from larmamba.ssm import selective_scan


class Identity(nn.Module):
    def forward(self, x, mask=None):
        return x


class BiMambaMixer(nn.Module):
    """Bidirectional selective-SSM token mixer (Mamba-1 / Vim style).

    Operates on tokens already permuted into curve order; forward + backward
    scans share projections and are averaged, reducing scan-direction bias.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_inner = expand * dim
        self.dt_rank = dt_rank or max(dim // 16, 1)

        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        # Non-causal depthwise conv (bidirectional context); padding keeps T.
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv // 2, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is (d_inner, d_state), parameterised as -exp(A_log) so it stays < 0.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def _scan_one_direction(self, u, delta, A, B, C):
        return selective_scan(u, delta, A, B, C, self.D)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: (B, T, dim) in curve order.  valid: (B, T) bool in curve order."""
        B_, T, _ = x.shape
        xz = self.in_proj(x)                                  # (B,T,2*d_inner)
        u, z = xz.chunk(2, dim=-1)                            # each (B,T,d_inner)

        # zero invalid tokens before the conv so they don't leak into neighbours
        vf = valid.unsqueeze(-1).to(u.dtype)
        u = u * vf
        u = self.conv1d(u.transpose(1, 2))[..., :T].transpose(1, 2)
        u = F.silu(u)
        u = u * vf

        dbl = self.x_proj(u)                                  # (B,T,dt_rank+2N)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                  # (B,T,d_inner)
        # transparency: Δ=0 at invalid -> a=1,b=0 -> state passes through untouched
        delta = delta * vf
        A = -torch.exp(self.A_log.float())                    # (d_inner,d_state)

        y_fwd = self._scan_one_direction(u, delta, A, Bm, Cm)
        # backward: reverse the sequence, scan, reverse back
        rev = torch.flip(torch.arange(T, device=x.device), dims=[0])
        y_bwd = self._scan_one_direction(
            u[:, rev], delta[:, rev], A, Bm[:, rev], Cm[:, rev]
        )[:, rev]
        y = 0.5 * (y_fwd + y_bwd)

        y = y * F.silu(z)
        out = self.out_proj(y)
        return out * vf


class MambaBlock(nn.Module):
    """Residual block: MaskedLayerNorm -> BiMambaMixer -> MaskedLayerNorm -> Mlp.

    Reorders tokens into curve `order` for the mixer, then scatters back so the
    output token axis is aligned with the input (and with centers/groups).
    """

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0,
                 d_state=16, d_conv=4, expand=2, act_layer=nn.GELU,
                 norm_layer=MaskedLayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mixer = BiMambaMixer(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = MaskedDropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x, x_mask=None, order=None, inv_order=None):
        B_, T, C = x.shape
        if x_mask is None:
            x_mask = x.new_ones(B_, T, dtype=torch.bool)

        # --- token mixer (in curve order) ---
        xn = self.norm1(x, x_mask)
        if order is not None:
            idx = order.unsqueeze(-1).expand(-1, -1, C)
            xn_ord = torch.gather(xn, 1, idx)
            valid_ord = torch.gather(x_mask.to(torch.bool), 1, order)
            y_ord = self.mixer(xn_ord, valid_ord)
            y = torch.zeros_like(y_ord).scatter_(1, idx, y_ord)
        else:
            y = self.mixer(xn, x_mask.to(torch.bool))
        x = x + self.drop_path(y, x_mask)

        # --- MLP ---
        ffn = self.mlp(self.norm2(x, x_mask))
        if x_mask is not None:
            ffn = ffn * x_mask.unsqueeze(-1)
        x = x + self.drop_path(ffn, x_mask)
        return x
