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
import os
import threading
import time

import torch

from polarmae.datasets.APA2D import APA2D
from polarmae.layers.encoder import TransformerEncoder
from larmamba import MambaEncoder

_PAGE = os.sysconf("SC_PAGE_SIZE")


def _rss_mib():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * _PAGE / 2**20


class PeakRSS:
    """Sample process RSS in a thread; report peak (MiB) over the window."""
    def __init__(self, dt=0.002):
        self.dt = dt; self.peak = 0.0; self._run = False
    def __enter__(self):
        self.peak = _rss_mib(); self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self
    def _loop(self):
        while self._run:
            self.peak = max(self.peak, _rss_mib()); time.sleep(self.dt)
    def __exit__(self, *a):
        self._run = False; self._t.join(); self.peak = max(self.peak, _rss_mib())

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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kinds", default="mamba,attn")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the encoder transformer (mixer stack)")
    args = ap.parse_args()
    if args.compile:
        import torch._dynamo as _d
        _d.config.suppress_errors = True   # fall back to eager on graph breaks
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

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
    import contextlib
    on_cuda = (device == "cuda")
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if on_cuda else contextlib.nullcontext
    devname = torch.cuda.get_device_name() if on_cuda else f"CPU x{torch.get_num_threads()} threads"

    npix = [int(l.item()) for _, l in events]
    mem_col = "peak_GPU_MiB" if on_cuda else "peakRSS_MiB"
    print(f"# per-event inference (batch=1) on {n} real events  "
          f"active-pix mean={sum(npix)//n} min={min(npix)} max={max(npix)}  device={devname}")
    print(f"{'kind':6s} {'tokens(mean)':>12s} {'latency_ms':>11s} {mem_col:>12s}")

    for kind in args.kinds.split(","):
        enc = build(kind, device)
        if args.compile:
            enc.transformer = torch.compile(enc.transformer, dynamic=True)
        for pts, lengths in events[:3]:
            with autocast():
                run_one(enc, pts, lengths)
        toks = []
        if on_cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for pts, lengths in events:
                with autocast():
                    toks.append(run_one(enc, pts, lengths))
            torch.cuda.synchronize()
            dt = (time.time() - t0) / n * 1e3
            mem = torch.cuda.max_memory_allocated() / 2**20
        else:
            with PeakRSS() as rss:
                t0 = time.time()
                for pts, lengths in events:
                    toks.append(run_one(enc, pts, lengths))
                dt = (time.time() - t0) / n * 1e3
            mem = rss.peak
        print(f"{kind:6s} {sum(toks)/n:12.0f} {dt:11.2f} {mem:12.0f}")
        del enc
        if on_cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
