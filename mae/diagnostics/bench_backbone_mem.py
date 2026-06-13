"""
Inference memory + latency of the sparse-CNN backbone used by mae and dino
(MinkUNetSparseAttentionCore, registry key 'attn_default').

Runs in the ml-dune-model env (torch 2.10 + warpconvnet).  Builds the backbone
(random init — memory/latency don't depend on weights), feeds a synthetic
Voxels input of B events x N active pixels, and measures peak CUDA memory and
forward latency at inference (eval, no grad).  Lets us compare per-event
inference memory against larmamba / polarmae (measured in their own env).

    python -m mae.diagnostics.bench_backbone_mem --N 2000,4000,6000,8000 --batch 1,8
"""

import argparse
import time

import torch

from models import BACKBONE_REGISTRY
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures


def make_voxels(B, N, device):
    """Synthetic sparse event: B images, N active pixels each, in W-view-local
    (channel, tick) range, charge ~ log1p of random ADC."""
    coords_list, feats_list, offsets = [], [], [0]
    for _ in range(B):
        ch = torch.randint(0, 1050, (N, 1))
        tk = torch.randint(0, 1500, (N, 1))
        coords_list.append(torch.cat([ch, tk], dim=1))
        feats_list.append(torch.log1p(torch.rand(N, 1) * 3000.0))
        offsets.append(offsets[-1] + N)
    coords = torch.cat(coords_list, 0).int().to(device)
    feats = torch.cat(feats_list, 0).float().to(device)
    offs = torch.tensor(offsets, dtype=torch.long)   # CSR offsets stay on CPU
    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offs),
        batched_features=CatFeatures(feats, offsets=offs),
        offsets=offs,
    )


@torch.no_grad()
def bench(core, B, N, device, iters=10):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    # warmup (also lets warpconvnet build/cache kernel maps)
    for _ in range(3):
        core(make_voxels(B, N, device))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        core(make_voxels(B, N, device))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 2**20
    return dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", default="2000,4000,6000,8000")
    ap.add_argument("--batch", default="1,8")
    args = ap.parse_args()
    device = "cuda"
    cls = BACKBONE_REGISTRY["attn_default"]
    core = cls().to(device).eval().core   # Voxels -> Voxels feature extractor
    for p in core.parameters():
        p.requires_grad = False

    print(f"# sparse-CNN backbone (mae/dino, attn_default) inference  "
          f"device={torch.cuda.get_device_name()}")
    print(f"{'batch':>5s} {'N_pix':>6s} {'latency_ms':>11s} {'peak_MiB':>9s}")
    for B in [int(b) for b in args.batch.split(",")]:
        for N in [int(n) for n in args.N.split(",")]:
            try:
                dt, peak = bench(core, B, N, device)
                print(f"{B:5d} {N:6d} {dt:11.2f} {peak:9.0f}")
            except RuntimeError as e:
                msg = "OOM" if "out of memory" in str(e).lower() else str(e)[:30]
                print(f"{B:5d} {N:6d} {msg:>11s} {'--':>9s}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
