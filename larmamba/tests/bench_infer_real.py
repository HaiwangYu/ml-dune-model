"""
Per-event inference memory + latency on REAL APA2D events (polar env).

Loads real DUNE events (so the tokenizer produces realistic ~256 tokens, not
the degenerate ~N you get from uniform-random synthetic points), runs the full
feature-extraction forward (prepare_tokens + transformer, eval/no-grad) at
batch=1 for both larmamba.MambaEncoder and polarmae's attention
TransformerEncoder, and reports peak CUDA memory + latency.  Directly
comparable to the sparse-CNN backbone benchmark (mae/dino) measured in the
ml-dune-model env.

    python -m larmamba.tests.bench_infer_real --n_events 32
"""

import argparse
import time

import torch

from polarmae.datasets.APA2D import APA2D
from polarmae.layers.encoder import TransformerEncoder
from larmamba import MambaEncoder

CENTER = torch.tensor([525.0, 562.0, 0.0])
SCALE = 1.0 / 600.0


def scale(points):
    p = points.clone()
    p[..., :3] = (p[..., :3] - CENTER.to(points.device)) * SCALE
    return p


def build(kind, device):
    common = dict(num_channels=4, arch="vit_small", voxel_size=5,
                  tokenizer_kwargs={"group_radius": 5 / 600},
                  transformer_kwargs={"add_pos_at_every_layer": True})
    if kind == "mamba":
        m = MambaEncoder(mamba_kwargs={"d_state": 16, "d_conv": 4, "expand": 2}, **common)
    else:
        m = TransformerEncoder(**common)
    return m.to(device).eval()


@torch.no_grad()
def run_one(enc, pts, lengths):
    out = enc.prepare_tokens(pts, lengths)
    enc.transformer(out["x"], out["pos_embed"], out["emb_mask"])
    return int(out["emb_mask"].sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/gpfs01/lbne/users/fm/cffm-data/prod-jay-100k-truth-2026-02-27/13874/1/009")
    ap.add_argument("--n_events", type=int, default=32)
    args = ap.parse_args()
    device = "cuda"

    ds = APA2D(data_path=args.data, apa=0, view="W", emin=1.0, emax=1.0e5,
               energy_threshold=1.0, min_points=256, max_points=8000,
               return_semantic_id=False, use_cache=True,
               cache_dir="/gpfs01/lbne/users/fm/hyu/cache/data")
    n = min(args.n_events, len(ds))
    events = []
    for i in range(n):
        s = ds[i]
        pts = scale(s["points"].unsqueeze(0).to(device))     # (1, Ni, 4)
        lengths = torch.tensor([s["points"].shape[0]], device=device)
        events.append((pts, lengths))
    npix = [int(l.item()) for _, l in events]
    print(f"# per-event inference (batch=1) on {n} real events  "
          f"active-pix mean={sum(npix)//n} min={min(npix)} max={max(npix)}  "
          f"device={torch.cuda.get_device_name()}")
    print(f"{'kind':6s} {'tokens(mean)':>12s} {'latency_ms':>11s} {'peak_MiB':>9s}")

    for kind in ["mamba", "attn"]:
        enc = build(kind, device)
        # warmup
        for pts, lengths in events[:3]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                run_one(enc, pts, lengths)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        toks, t0 = [], time.time()
        for pts, lengths in events:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks.append(run_one(enc, pts, lengths))
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n * 1e3
        peak = torch.cuda.max_memory_allocated() / 2**20
        print(f"{kind:6s} {sum(toks)/n:12.0f} {dt:11.2f} {peak:9.0f}")
        del enc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
