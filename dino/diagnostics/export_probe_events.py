#!/usr/bin/env python3
"""Dump the unified-probe event list to a portable .npz.

Writes the first `--n_events` events of APASparseMetaDataset -- constructed
EXACTLY like `dino.diagnostics.ab_pid_probe` builds it (apply_log_transform
off, so charges are raw ADC) -- as flat CSR arrays:

    coords   [N, 2]        int32    (channel, tick) per voxel
    charge   [N]           float32  raw ADC per voxel
    offsets  [n_events+1]  int64    event i = rows offsets[i]:offsets[i+1]

Purpose: APASparseMetaDataset needs WarpConvNet (torch 2.10 env), but the
PoLAr-MAE / larmamba encoders live in the torch 2.5 env. This dump is the
bridge: `larmamba/export_pid_features.py` reads it, so its exported features
are aligned to `Subset(ds, range(n_events))` by construction -- the ordering
contract of `ab_pid_probe --external`.

    python -m dino.diagnostics.export_probe_events \
        --datadir /gpfs01/lbne/users/fm/cffm-data/prod-jay-100k-truth-2026-02-27 \
        --apa 0 --view W --n_events 500 --out events.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loader.apa_sparse_meta_dataset import APASparseMetaDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--apa", type=int, default=0)
    ap.add_argument("--view", default="W")
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--n_events", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = APASparseMetaDataset(
        datadir=args.datadir, apa=args.apa, view=args.view, use_cache=True,
        cache_dir=(args.cache_dir or None),
        return_full_metadata=True, return_pixel_truth=True,
    )
    ds.apply_log_transform = False
    n = min(args.n_events, len(ds))
    print(f"[dump] {len(ds)} events available, dumping first {n}", flush=True)

    coords, charge, offsets = [], [], [0]
    for gi in range(n):
        voxels, _meta = ds[gi]
        ci = voxels.coordinate_tensor.cpu().numpy().astype(np.int32)
        qi = voxels.feature_tensor[:, 0].cpu().numpy().astype(np.float32)
        coords.append(ci)
        charge.append(qi)
        offsets.append(offsets[-1] + len(ci))
        if (gi + 1) % 100 == 0:
            print(f"[dump] {gi + 1}/{n} events ({offsets[-1]} voxels)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        coords=np.concatenate(coords, axis=0),
        charge=np.concatenate(charge, axis=0),
        offsets=np.array(offsets, dtype=np.int64),
    )
    print(f"[dump] wrote {out}: {offsets[-1]} voxels / {n} events", flush=True)


if __name__ == "__main__":
    main()
