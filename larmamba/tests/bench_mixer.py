"""
Mixer scaling benchmark — the clean cost-metric artifact.

Bypasses the (shared, coverage-based) tokenizer and feeds synthetic token
sequences of *controlled* length T directly to the sequence mixer, comparing
larmamba's bidirectional-Mamba transformer against polarmae's ViT
self-attention transformer (same embed_dim/depth) as T grows.

This isolates the question the whole project hinges on: how do compute (latency)
and peak memory scale with token count?  Mamba is O(T); self-attention compute
is O(T^2) (memory is O(T) when flash/SDPA is used, O(T^2) otherwise).

    python -m larmamba.tests.bench_mixer --T 256,512,1024,2048,4096,8192 --batch 8
"""

import argparse
import time

import torch

from polarmae.layers.transformer import make_transformer
from larmamba.mamba_encoder import MambaTransformer


def build(kind, depth, dim, flash):
    if kind == "mamba":
        return MambaTransformer(embed_dim=dim, depth=depth, mlp_ratio=4.0,
                                add_pos_at_every_layer=True)
    return make_transformer("vit_small", depth=depth,
                            add_pos_at_every_layer=True, use_flash_self_attn=flash)


@torch.no_grad()
def bench(kind, mod, B, T, dim, device, iters=10):
    x = torch.randn(B, T, dim, device=device)
    pos = torch.randn(B, T, dim, device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    if kind == "mamba":
        mod.set_centers(torch.randn(B, T, 3, device=device))
    def run():
        return mod(x, pos, mask)
    for _ in range(3):
        run()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 2**20
    return dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="256,512,1024,2048,4096,8192")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--kinds", default="mamba,attn_flash,attn_vanilla")
    args = ap.parse_args()
    device = "cuda"
    Ts = [int(t) for t in args.T.split(",")]

    print(f"# mixer scaling  batch={args.batch} dim={args.dim} depth={args.depth} "
          f"dtype=bf16-autocast device={torch.cuda.get_device_name()}")
    print(f"{'kind':13s} {'T':>6s} {'latency_ms':>11s} {'peak_MiB':>9s}")
    for kind in args.kinds.split(","):
        base = "mamba" if kind == "mamba" else "attn"
        flash = (kind != "attn_vanilla")
        for T in Ts:
            try:
                mod = build(base, args.depth, args.dim, flash).to(device).eval()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dt, peak = bench(base, mod, args.batch, T, args.dim, device)
                print(f"{kind:13s} {T:6d} {dt:11.2f} {peak:9.0f}")
                del mod; torch.cuda.empty_cache()
            except RuntimeError as e:
                msg = "OOM" if "out of memory" in str(e).lower() else str(e)[:32]
                print(f"{kind:13s} {T:6d} {msg:>11s} {'--':>9s}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
