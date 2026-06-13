"""
Encoder inference cost benchmark — peak GPU memory + latency.

Measures the *deployment* cost of the encoder alone (prepare_tokens +
transformer forward, no grad), which is what matters for inference — unlike the
training gpu.log that also includes the decoder, Chamfer loss, and probe SVMs.

Compares larmamba.MambaEncoder against polarmae's attention TransformerEncoder
across token counts (num_init_groups), to chart the core thesis: the Mamba
encoder's peak memory scales ~linearly in token count while attention scales
~quadratically.

    python -m larmamba.tests.bench_encoder --groups 256,512,1024,2048 --batch 16
"""

import argparse
import time

import torch

from polarmae.layers.encoder import TransformerEncoder
from larmamba import MambaEncoder


def make_batch(B, Nmax, device):
    """Synthetic points already in the model's input frame: center_and_scale
    maps (channel,tick) via (x-center)/600, so coords land in ~[-1,1] and the
    tokenizer's group_radius groups them into ~num_init_groups tokens (matching
    real data).  Feeding raw 0..1500 coords degenerates grouping to 1 token/pt."""
    pts = torch.zeros(B, Nmax, 4, device=device)
    lengths = torch.full((B,), Nmax, dtype=torch.long, device=device)
    ch = torch.randint(0, 1050, (B, Nmax), device=device).float()
    tk = torch.randint(0, 1125, (B, Nmax), device=device).float()
    pts[..., 0] = (ch - 525.0) / 600.0
    pts[..., 1] = (tk - 562.0) / 600.0
    pts[..., 3] = torch.rand(B, Nmax, device=device) * 2 - 1
    return pts, lengths


def build(kind, groups, ctx, device):
    tk = {"group_radius": 5 / 600, "num_init_groups": groups, "context_length": ctx}
    common = dict(num_channels=4, arch="vit_small", voxel_size=5,
                  tokenizer_kwargs=tk,
                  transformer_kwargs={"add_pos_at_every_layer": True})
    if kind == "mamba":
        return MambaEncoder(mamba_kwargs={"d_state": 16, "d_conv": 4, "expand": 2}, **common).to(device).eval()
    return TransformerEncoder(**common).to(device).eval()


@torch.no_grad()
def bench(enc, pts, lengths, iters=20):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    # warmup
    for _ in range(3):
        out = enc.prepare_tokens(pts, lengths)
        enc.transformer(out["x"], out["pos_embed"], out["emb_mask"])
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        out = enc.prepare_tokens(pts, lengths)
        enc.transformer(out["x"], out["pos_embed"], out["emb_mask"])
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1e3   # ms
    peak = torch.cuda.max_memory_allocated() / 2**20  # MiB
    T = int(out["emb_mask"].sum(-1).float().mean().item())
    return dt, peak, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="256,512,1024,2048")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--points", type=int, default=6000)
    ap.add_argument("--kinds", default="mamba,attn")
    args = ap.parse_args()
    device = "cuda"
    grouplist = [int(g) for g in args.groups.split(",")]

    print(f"# encoder inference benchmark  batch={args.batch} points={args.points} "
          f"dtype=bf16-autocast device={torch.cuda.get_device_name()}")
    print(f"{'kind':6s} {'groups':>7s} {'tokens':>7s} {'latency_ms':>11s} {'peak_MiB':>9s}")
    for kind in args.kinds.split(","):
        for g in grouplist:
            ctx = max(2 * g, 1024)
            try:
                pts, lengths = make_batch(args.batch, args.points, device)
                enc = build(kind, g, ctx, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dt, peak, T = bench(enc, pts, lengths)
                print(f"{kind:6s} {g:7d} {T:7d} {dt:11.2f} {peak:9.0f}")
                del enc; torch.cuda.empty_cache()
            except RuntimeError as e:
                msg = "OOM" if "out of memory" in str(e).lower() else str(e)[:40]
                print(f"{kind:6s} {g:7d} {'--':>7s} {'--':>11s} {msg:>9s}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
