"""Export per-voxel larmamba2 features in the ab_pid_probe --external schema.

Reads the probe event dump (coords/charge/offsets, raw ADC), rebuilds the model
from a Lightning checkpoint's hyper_parameters, encodes ALL non-empty tiles
(uncapped — Mamba is O(T)), and assigns each voxel its containing tile's token.
Exact assignment: coordinate match rate is 1.0 by construction, no KNN blend.

    python -m larmamba2.export_pid_features --ckpt <...>.ckpt \
        --events events.npz --out lm2_feats.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from polarmae.datasets.APA2D import log_transform

from larmamba2.ssl_module import Larmamba2MAE


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--emin", type=float, default=1.0)
    ap.add_argument("--emax", type=float, default=1.0e5)
    ap.add_argument("--energy_threshold", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Larmamba2MAE.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)
    print(f"[export:larmamba2] loaded {args.ckpt} "
          f"(tile_size={model.hparams.tile_size}, dim={model.hparams.dim})", flush=True)

    npz = np.load(args.events)
    coords_all, charge_all, offs = npz["coords"], npz["charge"], npz["offsets"]
    n_events = len(offs) - 1
    print(f"[export:larmamba2] {n_events} events / {offs[-1]} voxels", flush=True)

    out_coords, out_feats, out_offs = [], [], [0]
    t0 = time.time()
    for start in range(0, n_events, args.batch):
        idxs = range(start, min(start + args.batch, n_events))
        pts_list, ev_coords, ev_keep = [], [], []
        for gi in idxs:
            a, b = int(offs[gi]), int(offs[gi + 1])
            ci, qi = coords_all[a:b], charge_all[a:b]
            m = qi > args.energy_threshold          # model-input filter (APA2D)
            p = np.zeros((int(m.sum()), 4), dtype=np.float32)
            p[:, 0] = ci[m, 0]
            p[:, 1] = ci[m, 1]
            p[:, 3] = log_transform(qi[m].astype(np.float32), xmax=args.emax, eps=args.emin)
            pts_list.append(torch.from_numpy(p))
            ev_coords.append(ci)
            ev_keep.append(m)

        lens = torch.tensor([p.shape[0] for p in pts_list], dtype=torch.long, device=device)
        pts = torch.nn.utils.rnn.pad_sequence(pts_list, batch_first=True).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            tokens, tile_mask, vox_tile = model.forward_features(pts, lens, t_max=-1)
        tokens = tokens.float()

        D = tokens.shape[-1]
        for j, (ci, m) in enumerate(zip(ev_coords, ev_keep)):
            # features for model-input voxels; below-threshold voxels get zeros
            # (they are matched by coordinate but carry no tile -> zero feature)
            feats = np.zeros((ci.shape[0], D), dtype=np.float32)
            vt = vox_tile[j, :int(lens[j])].cpu().numpy()
            fk = tokens[j][torch.from_numpy(vt).clamp(min=0).to(device)].cpu().numpy()
            fk[vt < 0] = 0.0
            feats[m] = fk
            out_coords.append(ci.astype(np.int32))
            out_feats.append(feats)
            out_offs.append(out_offs[-1] + ci.shape[0])

        done = len(out_offs) - 1
        if done % 100 < args.batch:
            print(f"[export:larmamba2] {done}/{n_events} events "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        coords=np.concatenate(out_coords, axis=0),
                        feat=np.concatenate(out_feats, axis=0),
                        offsets=np.array(out_offs, dtype=np.int64))
    print(f"[export:larmamba2] wrote {out_path}: {out_offs[-1]} voxels / "
          f"{len(out_offs) - 1} events (D={D}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
