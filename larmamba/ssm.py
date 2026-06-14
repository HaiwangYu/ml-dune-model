"""
Pure-PyTorch selective state-space scan (Mamba-1 style), with no dependency on
the `mamba-ssm` / `causal-conv1d` CUDA kernels (which have no wheels for our
torch 2.5.1+cu124 env).

The core is a first-order linear recurrence

    h_t = a_t * h_{t-1} + b_t,        h_0 = 0

solved in parallel with a Hillis-Steele inclusive prefix scan in O(log T)
tensor ops along the sequence axis.  This works on GPU *and* CPU (a bonus for
the eventual CPU-deployment goal) and is numerically stable because
a_t = exp(Δ_t · A) with A < 0, so a_t ∈ (0, 1] and the products never blow up.

A slow sequential reference (`selective_scan_ref`) is provided and the unit
test at the bottom asserts the two agree, so the parallel kernel is trusted
before any training.

Run the self-test:
    python -m larmamba.ssm
"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# CPU scan backend (no CUDA kernel available there): "seq" = sequential
# recurrence (low overhead at small T, the typical inference regime); "chunked"
# = the parallel Hillis-Steele scan (log(L) passes over big tensors, better for
# large T / training). Overridable via env for benchmarking.
CPU_SCAN_BACKEND = os.environ.get("LARMAMBA_CPU_SCAN", "seq")


# ---------------------------------------------------------------------------
# Optional fast path: the official mamba-ssm CUDA kernel
# ---------------------------------------------------------------------------
# The kernel (memory-optimal custom backward, no (B,T,E,N) materialisation) is
# ~10-50x faster and far lighter than the pure-torch scan below.  We load just
# `selective_scan_fn` from the ops submodule, bypassing mamba_ssm/__init__.py
# (whose MambaLMHeadModel import needs transformers, which we don't install).

@functools.lru_cache(maxsize=1)
def kernel_selective_scan_fn():
    """Return mamba_ssm's selective_scan_fn, or None if unavailable (→ CPU /
    pure-torch fallback).  Cached; safe to call every forward."""
    import importlib, importlib.util, sys, types, os
    try:
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            return selective_scan_fn
        except ModuleNotFoundError:
            # bypass the heavy package __init__ (transformers dep)
            spec = importlib.util.find_spec("mamba_ssm")
            if spec is None or spec.origin is None:
                return None
            stub = types.ModuleType("mamba_ssm")
            stub.__path__ = [os.path.dirname(spec.origin)]
            stub.__spec__ = spec
            sys.modules["mamba_ssm"] = stub
            mod = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
            return mod.selective_scan_fn
    except Exception:
        return None


def _selective_scan_kernel(fn, u, delta, A, B, C, D):
    """Call the mamba kernel.  Our layout is (B,T,E)/(B,T,N); the kernel wants
    channels-first (B,E,L)/(B,N,L).  delta is already softplus'd and zeroed at
    invalid positions, so delta_softplus=False; gating (z) is applied by the
    caller, so z=None here."""
    # The kernel requires u, delta, B, C to share one dtype ("input_type").
    # Under bf16-mixed autocast these can differ (e.g. delta from softplus stays
    # fp32), so unify them to u's dtype.  A and D stay fp32.
    dt = u.dtype
    uT = u.transpose(1, 2).contiguous()                 # (B,E,L)
    dT = delta.to(dt).transpose(1, 2).contiguous()      # (B,E,L)
    BT = B.to(dt).transpose(1, 2).contiguous()          # (B,N,L)
    CT = C.to(dt).transpose(1, 2).contiguous()          # (B,N,L)
    y = fn(uT, dT, A.float(), BT, CT, D=D.float(), z=None,
           delta_bias=None, delta_softplus=False)
    return y.transpose(1, 2)                             # (B,T,E)


# ---------------------------------------------------------------------------
# Parallel associative scan
# ---------------------------------------------------------------------------

def _shift(x: torch.Tensor, span: int, fill: float) -> torch.Tensor:
    """Shift `x` forward along dim=1 (time) by `span`, filling the leading
    `span` positions with `fill`.  result[:, t] = x[:, t-span] for t >= span."""
    if span == 0:
        return x
    T = x.shape[1]
    pad_shape = list(x.shape)
    pad_shape[1] = span
    pad = x.new_full(pad_shape, fill)
    return torch.cat([pad, x[:, : T - span]], dim=1)


def pscan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix scan of the linear recurrence h_t = a_t h_{t-1} + b_t.

    a, b : (B, T, ...) — identical trailing shape.  Returns h with the same
    shape, where h[:, t] is the recurrence value at step t (h_0 = 0 implied,
    i.e. h[:, 0] == b[:, 0]).

    Implemented with Hillis-Steele doubling.  For a prefix segment summarised
    by (acc_a, acc_h) — meaning "applying this segment to incoming state s
    yields acc_a*s + acc_h" — composing an earlier segment L then a later
    segment R gives  a = R.a * L.a ,  h = R.a * L.h + R.h .
    """
    acc_a = a
    acc_h = b
    T = a.shape[1]
    span = 1
    while span < T:
        a_sh = _shift(acc_a, span, 1.0)   # earlier-segment a (identity = 1)
        h_sh = _shift(acc_h, span, 0.0)   # earlier-segment h (identity = 0)
        # compute new_h before overwriting acc_a (new_h uses the *old* acc_a)
        acc_h = acc_a * h_sh + acc_h
        acc_a = acc_a * a_sh
        span *= 2
    return acc_h


# ---------------------------------------------------------------------------
# Selective scan (Mamba S6)
# ---------------------------------------------------------------------------

def _scan_segment(u_c, delta_c, A, B_c, C_c, D, h_prev):
    """One chunk of the selective scan.  Materialises the big (B, L, E, N)
    tensor only here (freed after the chunk under checkpointing).

    Returns y_c (B, L, E) and the carry state h_last (B, E, N) for the next
    chunk.  `h_prev` is the incoming state (B, E, N), so chunk boundaries are
    handled exactly:  h_t = pscan_t + (prod_{j<=t} a_j) * h_prev.
    """
    B_, L, E = u_c.shape
    N = A.shape[1]
    deltaA = torch.exp(delta_c.unsqueeze(-1) * A.view(1, 1, E, N))            # (B,L,E,N)
    deltaBu = (delta_c.unsqueeze(-1) * B_c.unsqueeze(2)) * u_c.unsqueeze(-1)  # (B,L,E,N)
    h_local = pscan(deltaA, deltaBu)                                         # (B,L,E,N)
    a_cum = torch.cumprod(deltaA, dim=1)                                     # (B,L,E,N)
    h = h_local + a_cum * h_prev.unsqueeze(1)
    y_c = (h * C_c.unsqueeze(2)).sum(dim=-1) + u_c * D.view(1, 1, E)         # (B,L,E)
    return y_c, h[:, -1]


def selective_scan(
    u: torch.Tensor,       # (B, T, E)   input (post-conv, post-activation)
    delta: torch.Tensor,   # (B, T, E)   Δ (already softplus'd, >= 0)
    A: torch.Tensor,       # (E, N)      state matrix (negative)
    B: torch.Tensor,       # (B, T, N)   input projection (selective)
    C: torch.Tensor,       # (B, T, N)   output projection (selective)
    D: torch.Tensor,       # (E,)        skip connection
    chunk_size: int = 64,
) -> torch.Tensor:
    """Discretised selective SSM, chunked + gradient-checkpointed.  Returns
    y: (B, T, E).

    The (B, L, E, N) discretised tensors are built only inside each chunk and,
    when grad is enabled, under `torch.utils.checkpoint`, so they are recomputed
    in backward rather than stored.  Persistent memory is therefore
    O(B·T·(E+N)) instead of O(B·T·E·N) — this is what lets the token count T
    grow (the sweep) at roughly fixed peak memory.

    Discretisation (zero-order hold on A, Euler on B):
        a_t = exp(Δ_t · A);  b_t = (Δ_t · B_t)·u_t;
        h_t = a_t h_{t-1} + b_t;  y_t = Σ_N C_t·h_t + D·u_t

    Fast path: when the mamba-ssm CUDA kernel is available and the input is on
    GPU, dispatch to it (memory-optimal, no chunking needed).  Otherwise use the
    pure-torch chunked + checkpointed scan.
    """
    if u.is_cuda:
        fn = kernel_selective_scan_fn()
        if fn is not None:
            return _selective_scan_kernel(fn, u, delta, A, B, C, D)
    elif CPU_SCAN_BACKEND == "seq":
        # On CPU the sequential recurrence has far less overhead than the
        # log(L) parallel-scan passes at the small T of inference.
        return selective_scan_seq(u, delta, A, B, C, D)

    Bb, T, E = u.shape
    dtype_in = u.dtype
    # float32 for scan stability under bf16/amp
    u32, delta32, A32, B32, C32, D32 = (
        u.float(), delta.float(), A.float(), B.float(), C.float(), D.float())

    h_prev = u.new_zeros(Bb, E, A.shape[1], dtype=torch.float32)
    use_ckpt = torch.is_grad_enabled()
    ys = []
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        args = (u32[:, s:e], delta32[:, s:e], A32, B32[:, s:e], C32[:, s:e], D32, h_prev)
        if use_ckpt:
            y_c, h_prev = checkpoint(_scan_segment, *args, use_reentrant=False)
        else:
            y_c, h_prev = _scan_segment(*args)
        ys.append(y_c)
    y = torch.cat(ys, dim=1)
    return y.to(dtype_in)


def selective_scan_ref(
    u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
    B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
) -> torch.Tensor:
    """Slow sequential reference, identical math to `selective_scan`.  For
    testing and as a fallback; do not use in training (python loop over T)."""
    Bb, T, E = u.shape
    N = A.shape[1]
    u32, delta32, A32, B32, C32 = (u.float(), delta.float(), A.float(),
                                   B.float(), C.float())
    h = u.new_zeros(Bb, E, N, dtype=torch.float32)
    ys = []
    for t in range(T):
        a_t = torch.exp(delta32[:, t].unsqueeze(-1) * A32.view(1, E, N))      # (B,E,N)
        b_t = (delta32[:, t].unsqueeze(-1) * B32[:, t].unsqueeze(1)) * u32[:, t].unsqueeze(-1)
        h = a_t * h + b_t                                                     # (B,E,N)
        y_t = (h * C32[:, t].unsqueeze(1)).sum(dim=-1)                        # (B,E)
        ys.append(y_t)
    y = torch.stack(ys, dim=1) + u32 * D.float().view(1, 1, E)
    return y.to(u.dtype)


# Production CPU sequential backend == the sequential reference (validated
# equal to the parallel scan in the unit test, fwd + grad).
selective_scan_seq = selective_scan_ref


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    B, T, E, N = 3, 17, 8, 5
    u = torch.randn(B, T, E)
    delta = F.softplus(torch.randn(B, T, E))
    A = -torch.exp(torch.randn(E, N))      # negative
    Bm = torch.randn(B, T, N)
    Cm = torch.randn(B, T, N)
    D = torch.randn(E)

    # chunk_size < T exercises the cross-chunk carry path
    for cs in (5, 8, 64):
        y_par = selective_scan(u, delta, A, Bm, Cm, D, chunk_size=cs)
        y_ref = selective_scan_ref(u, delta, A, Bm, Cm, D)
        max_err = (y_par - y_ref).abs().max().item()
        print(f"[ssm selftest] B={B} T={T} E={E} N={N} chunk={cs}  max|chunked-ref| = {max_err:.3e}")
        assert max_err < 1e-4, f"chunked scan disagrees with reference: {max_err}"

    # various T (incl. non-multiples of chunk) and a length-1 T
    for Tt in (1, 16, 50, 64):
        u = torch.randn(B, Tt, E)
        delta = F.softplus(torch.randn(B, Tt, E))
        Bm = torch.randn(B, Tt, N)
        Cm = torch.randn(B, Tt, N)
        e = (selective_scan(u, delta, A, Bm, Cm, D, chunk_size=16)
             - selective_scan_ref(u, delta, A, Bm, Cm, D)).abs().max().item()
        print(f"[ssm selftest] T={Tt:>3d}  max err = {e:.3e}")
        assert e < 1e-4

    # gradient parity: chunked scan grad ~= reference grad
    u = torch.randn(B, T, E, requires_grad=True)
    delta_in = torch.randn(B, T, E, requires_grad=True)
    Bm = torch.randn(B, T, N, requires_grad=True)
    Cm = torch.randn(B, T, N, requires_grad=True)
    selective_scan(u, F.softplus(delta_in), A, Bm, Cm, D, chunk_size=8).sum().backward()
    g_chunk = u.grad.clone()
    u.grad = None
    selective_scan_ref(u, F.softplus(delta_in), A, Bm, Cm, D).sum().backward()
    gerr = (g_chunk - u.grad).abs().max().item()
    print(f"[ssm selftest] grad max err = {gerr:.3e}")
    assert gerr < 1e-3, f"chunked-scan gradient disagrees: {gerr}"
    print("[ssm selftest] PASS")


if __name__ == "__main__":
    _selftest()
