#!/usr/bin/env python3
"""Export per-voxel MAE backbone features in the ab_pid_probe --external schema.

Runs the frozen sparse-CNN MAE backbone over the first `--n_events` events of
APASparseMetaDataset -- constructed exactly like `dino.diagnostics.ab_pid_probe`
builds it, so event indices line up with `Subset(ds, range(n_events))` -- and
writes:

    coords   [N, 2]        int32    (channel, tick) of each OUTPUT voxel
    feat     [N, 64]       float32  backbone feature
    offsets  [n_events+1]  int64    CSR event boundaries

The probe joins rows to truth by coordinate per event, so sparse-conv voxel
reordering is harmless. Normalisation matches MAE training: log1p on raw ADC
(the probe dataset serves raw ADC; we apply `log1p_voxels` here).

Also handles sparseformer checkpoints (same SparseMAEModel pipeline with a
pluggable backbone): pass `--config` pointing at the run's config JSON and the
backbone is rebuilt from its `backbone_name`/`backbone_kwargs` fields.

    python -m mae.diagnostics.export_pid_features \
        --ckpt .../mae_epoch1.pt --datadir .../prod-jay-100k-truth-2026-02-27 \
        --apa 0 --view W --n_events 500 --out mae_ep1_feats.npz
    python -m mae.diagnostics.export_pid_features \
        --ckpt .../sf_w256/mae_epoch1.pt --config sparseformer/configs/config_sf_w256.json ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loader.apa_sparse_meta_dataset import APASparseMetaDataset
from loader.collate import voxels_meta_collate_fn
from models.mae_model import SparseMAEModel, log1p_voxels

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="",
                    help="run config JSON; its backbone_name/backbone_kwargs "
                         "rebuild a sparseformer backbone (empty -> legacy default)")
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--apa", type=int, default=0)
    ap.add_argument("--view", default="W")
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--n_events", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    backbone_obj = None
    if args.config:
        cfg = json.load(open(args.config))
        name = cfg.get("backbone_name", "")
        if name:
            from sparseformer.backbones import build_backbone
            backbone_obj = build_backbone(name, **(cfg.get("backbone_kwargs") or {}))
            print(f"[mae-export] backbone from config: {name} "
                  f"kwargs={cfg.get('backbone_kwargs')}", flush=True)
    model = SparseMAEModel(backbone=backbone_obj).to(dev)
    model.load_state_dict(ckpt["model"])
    backbone = model.backbone.eval()
    print(f"[mae-export] loaded {args.ckpt} (epoch={ckpt.get('epoch', '?')})", flush=True)

    ds = APASparseMetaDataset(
        datadir=args.datadir, apa=args.apa, view=args.view, use_cache=True,
        cache_dir=(args.cache_dir or None),
        return_full_metadata=True, return_pixel_truth=True,
    )
    ds.apply_log_transform = False
    n = min(args.n_events, len(ds))
    sub = Subset(ds, list(range(n)))
    dl = DataLoader(sub, batch_size=args.batch, collate_fn=voxels_meta_collate_fn,
                    num_workers=2)

    coords, feats, offsets = [], [], [0]
    done = 0
    for xs, _meta in dl:
        xs = xs.to(dev)
        xs = log1p_voxels(xs)   # MAE training normalisation (raw ADC -> log1p)
        out = backbone(xs)
        out_coords = out.coordinate_tensor.cpu().numpy().astype(np.int32)
        out_feats = out.feature_tensor.float().cpu().numpy().astype(np.float32)
        out_offs = out.offsets.cpu().tolist()
        for i in range(len(out_offs) - 1):
            a, b = out_offs[i], out_offs[i + 1]
            coords.append(out_coords[a:b])
            feats.append(out_feats[a:b])
            offsets.append(offsets[-1] + (b - a))
        done += len(out_offs) - 1
        if done % 100 < args.batch:
            print(f"[mae-export] {done}/{n} events", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        coords=np.concatenate(coords, axis=0),
        feat=np.concatenate(feats, axis=0),
        offsets=np.array(offsets, dtype=np.int64),
    )
    print(f"[mae-export] wrote {out_path}: {offsets[-1]} voxels / {done} events "
          f"(D={feats[0].shape[1]})", flush=True)


if __name__ == "__main__":
    main()
