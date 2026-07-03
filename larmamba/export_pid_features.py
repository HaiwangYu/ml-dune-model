"""Export per-voxel PoLAr-MAE / larmamba features for the unified PID probe.

Reads the event dump written by `dino.diagnostics.export_probe_events` (coords /
charge / offsets, raw ADC, event order = `Subset(ds, range(n_events))` of the
probe's APASparseMetaDataset), runs the frozen encoder, and writes the
`ab_pid_probe --external` npz schema:

    coords   [N, 2]        int32    (channel, tick) -- the dump coords, verbatim
    feat     [N, D]        float32  per-voxel feature
    offsets  [n_events+1]  int64    CSR event boundaries

Input construction replicates polarmae.datasets.APA2D exactly: drop charge <=
energy_threshold, cap at max_points (random subsample, seeded), points =
(ch, tick, 0, log_transform(charge, emax, emin)). Feature upsampling mirrors
polarmae.eval.probes._extract_per_voxel_features (inverse-distance K-NN from
token centers) but queries ALL dumped voxels -- including any the model input
dropped -- so every truth voxel gets a feature and the probe's coordinate
match rate is 1.0 by construction.

Runs in uvenv-polar-mae (torch 2.5 + pytorch3d + polarmae + larmamba).

    python -m larmamba.export_pid_features --encoder polarmae \
        --ckpt .../epoch=3-step=20000.ckpt --events events.npz --out polarmae_feats.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import knn_points

from larmamba.eval_quant import CENTER, SCALE, build_encoder, load_encoder_weights
from polarmae.datasets.APA2D import log_transform


def _xform(x):
    """polarmae val_transformations: center + scale the spatial dims."""
    x = x.clone()
    x[..., :3] = (x[..., :3] - CENTER.to(x.device)) * SCALE
    return x


@torch.no_grad()
def _forward_batch(enc, pts, lens, qry, qlens, device):
    """Encode one padded batch and K-NN-upsample token features to the query
    voxels. Returns per-voxel features (B, Q_max, D) float32."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = enc.prepare_tokens(_xform(pts), lens, ids=None)
        out_t = enc.transformer(out["x"], out["pos_embed"], out["emb_mask"])
    tokens = out_t.last_hidden_state.float()            # (B, T, D)
    centers = out["centers"][..., :3].float()           # (B, T, 3)
    emb_lens = out["emb_mask"].sum(dim=1)               # (B,)

    K = max(1, min(5, int(emb_lens.min().item())))
    dists, idx, _ = knn_points(_xform(qry), centers,
                               lengths1=qlens, lengths2=emb_lens,
                               K=K, return_sorted=False)     # (B, Q_max, K)
    weight = 1.0 / (dists + torch.finfo(dists.dtype).eps)
    weight = weight / weight.sum(dim=2, keepdim=True)
    idx = idx.clamp(min=0)

    B, Q_max, _ = qry.shape
    D = tokens.shape[-1]
    feats = torch.empty(B, Q_max, D, device=device)
    for b in range(B):
        gathered = tokens[b][idx[b]]                     # (Q_max, K, D)
        feats[b] = (gathered * weight[b].unsqueeze(-1)).sum(dim=1)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="npz from export_probe_events")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", default="mamba", choices=["mamba", "polarmae"])
    ap.add_argument("--num_groups", type=int, default=256)
    ap.add_argument("--context_length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--emin", type=float, default=1.0)
    ap.add_argument("--emax", type=float, default=1.0e5)
    ap.add_argument("--energy_threshold", type=float, default=1.0)
    ap.add_argument("--max_points", type=int, default=8000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    np.random.seed(0)

    npz = np.load(args.events)
    coords_all, charge_all, offs = npz["coords"], npz["charge"], npz["offsets"]
    n_events = len(offs) - 1
    print(f"[export:{args.encoder}] {n_events} events / {offs[-1]} voxels from "
          f"{args.events}", flush=True)

    enc = build_encoder(args.encoder, args.num_groups, args.context_length, device)
    enc = load_encoder_weights(enc, args.ckpt)

    out_coords, out_feats, out_offs = [], [], [0]
    t0 = time.time()
    for start in range(0, n_events, args.batch):
        idxs = range(start, min(start + args.batch, n_events))
        pts_list, qry_list = [], []
        ev_coords = []
        for gi in idxs:
            a, b = int(offs[gi]), int(offs[gi + 1])
            ci = coords_all[a:b]
            qi = charge_all[a:b]
            # model input: APA2D's threshold + cap + log-transform
            m = qi > args.energy_threshold
            mc, mq = ci[m], qi[m]
            if args.max_points > 0 and mc.shape[0] > args.max_points:
                sel = np.random.choice(mc.shape[0], size=args.max_points, replace=False)
                sel.sort()
                mc, mq = mc[sel], mq[sel]
            p = np.zeros((mc.shape[0], 4), dtype=np.float32)
            p[:, 0] = mc[:, 0]
            p[:, 1] = mc[:, 1]
            p[:, 3] = log_transform(mq.astype(np.float32), xmax=args.emax, eps=args.emin)
            pts_list.append(torch.from_numpy(p))
            # feature query: ALL dumped voxels
            q = np.zeros((ci.shape[0], 3), dtype=np.float32)
            q[:, 0] = ci[:, 0]
            q[:, 1] = ci[:, 1]
            qry_list.append(torch.from_numpy(q))
            ev_coords.append(ci)

        lens = torch.tensor([p.shape[0] for p in pts_list], dtype=torch.long, device=device)
        qlens = torch.tensor([q.shape[0] for q in qry_list], dtype=torch.long, device=device)
        pts = torch.nn.utils.rnn.pad_sequence(pts_list, batch_first=True).to(device)
        qry = torch.nn.utils.rnn.pad_sequence(qry_list, batch_first=True).to(device)

        feats = _forward_batch(enc, pts, lens, qry, qlens, device)
        for j, ci in enumerate(ev_coords):
            nq = ci.shape[0]
            out_coords.append(ci.astype(np.int32))
            out_feats.append(feats[j, :nq].cpu().numpy().astype(np.float32))
            out_offs.append(out_offs[-1] + nq)

        done = len(out_offs) - 1
        if done % 100 < args.batch:
            print(f"[export:{args.encoder}] {done}/{n_events} events "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        coords=np.concatenate(out_coords, axis=0),
        feat=np.concatenate(out_feats, axis=0),
        offsets=np.array(out_offs, dtype=np.int64),
    )
    print(f"[export:{args.encoder}] wrote {out_path}: {out_offs[-1]} voxels / "
          f"{len(out_offs) - 1} events (D={out_feats[0].shape[1]}, "
          f"{time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
