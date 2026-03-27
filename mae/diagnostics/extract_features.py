"""
Extract backbone features from a trained MAE checkpoint.

Features are collected at all active voxels (non-zero charge) in the dataset.
The output .npz contains per-voxel feature vectors together with image-level
class labels, suitable for PCA / probing analysis.

Usage:
    python mae/diagnostics/extract_features.py path/to/checkpoint.pt
    python mae/diagnostics/extract_features.py path/to/checkpoint.pt --max_images=2000
    python mae/diagnostics/extract_features.py path/to/checkpoint.pt \\
        --output=./my_features.npz --batch_size=32

Output (.npz):
    backbone_features  [N_voxels, 64]   float32   backbone features at active voxels
    labels             [N_images]        int64     class label per image (-1 = unlabelled)
    positions          [N_voxels, 2]    int32     (channel, tick) voxel coordinates
    offsets            [N_images+1]     int64     CSR-style: image i → rows offsets[i]:offsets[i+1]
"""

import sys
from pathlib import Path

import fire
import numpy as np
import torch
import warp as wp
from torch.utils.data import DataLoader, Subset

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # mae/

from models.mae_model import SparseMAEModel, voxels_to_device, log1p_voxels
from loader.apa_sparse_meta_dataset import APASparseMetaDataset
from loader.collate import voxels_label_collate_fn


def _load_model(ckpt_path: Path, device: torch.device) -> SparseMAEModel:
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = SparseMAEModel().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    epoch = ckpt.get("epoch", 0)
    print(f"  Loaded epoch={epoch}")
    return model


@torch.no_grad()
def _run_loader(model: SparseMAEModel, loader: DataLoader, device: torch.device):
    """
    Run the backbone over the loader and collect per-voxel features.

    Returns
    -------
    features  : [N_voxels, 64]  float32
    labels    : [N_images]      int64
    positions : [N_voxels, 2]   int32   (channel, tick)
    offsets   : [N_images+1]    int64   CSR-style
    """
    feats_all, pos_all, labels_all, offsets = [], [], [], [0]

    for batch_idx, (vox_cpu, batch_labels) in enumerate(loader):
        vox = log1p_voxels(voxels_to_device(vox_cpu, device))

        if vox.feature_tensor.shape[0] == 0:
            # Empty batch: record zero-size entries for each image
            for lbl in batch_labels:
                labels_all.append(lbl.item())
                offsets.append(offsets[-1])
            continue

        backbone_out = model.backbone(vox)                        # Voxels [N_total, 64]
        feats_tensor = backbone_out.feature_tensor.float().cpu()  # [N_total, 64]
        coord_tensor = backbone_out.coordinate_tensor.cpu()       # [N_total, 2]
        out_offsets  = backbone_out.offsets                       # [B+1], CPU

        B = len(batch_labels)
        for b in range(B):
            start = int(out_offsets[b])
            end   = int(out_offsets[b + 1])
            feats_all.append(feats_tensor[start:end].numpy())    # [n_b, 64]
            pos_all.append(coord_tensor[start:end].int().numpy()) # [n_b, 2]
            labels_all.append(batch_labels[b].item())
            offsets.append(offsets[-1] + (end - start))

        if (batch_idx + 1) % 20 == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)} ...")

    return (
        np.concatenate(feats_all, axis=0).astype(np.float32) if feats_all else np.zeros((0, 64), dtype=np.float32),
        np.array(labels_all, dtype=np.int64),
        np.concatenate(pos_all, axis=0).astype(np.int32)     if pos_all  else np.zeros((0, 2),  dtype=np.int32),
        np.array(offsets,      dtype=np.int64),
    )


def main(
    checkpoint:  str,
    output:      str  = "",
    data_root:   str  = "/nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27",
    apa:         int  = 0,
    view:        str  = "W",
    max_images:  int  = 5000,
    batch_size:  int  = 64,
    num_workers: int  = 0,
    device:      str  = "cuda",
):
    """
    Extract MAE backbone features from a trained checkpoint for PCA / probing.

    Args:
        checkpoint:  Path to a .pt checkpoint saved by train_mae.py
        output:      Output .npz path. Defaults to <checkpoint_dir>/features_ep<N>.npz
        data_root:   Dataset root directory
        apa:         APA index (0–5)
        view:        Wire-plane view ("U", "V", or "W")
        max_images:  Max number of images to process (-1 = full dataset)
        batch_size:  Inference batch size
        num_workers: DataLoader workers (keep 0 unless warp is initialised in workers)
        device:      "cuda" or "cpu"
    """
    # Resolve device and initialise Warp on the correct GPU before any CUDA ops
    if device.startswith("cuda") and torch.cuda.is_available():
        # Ensure a specific index so torch.cuda.set_device works; default to cuda:0
        if ":" not in device:
            device = "cuda:0"
        device = torch.device(device)
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    wp.init()

    ckpt_path = Path(checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not output:
        ckpt_obj  = torch.load(ckpt_path, map_location="cpu")
        epoch     = ckpt_obj.get("epoch", 0)
        output    = str(ckpt_path.parent / f"features_ep{epoch}.npz")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    model = _load_model(ckpt_path, device)

    print(f"\nLoading dataset from {data_root} ...")
    dataset = APASparseMetaDataset(
        data_root, apa=apa, view=view, frame_name="frame_rebinned_reco",
    )
    if 0 < max_images < len(dataset):
        indices = torch.randperm(len(dataset))[:max_images].tolist()
        dataset = Subset(dataset, indices)
    print(f"  Images to process: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=voxels_label_collate_fn,
        num_workers=num_workers,
    )

    print("Extracting backbone features ...")
    features, labels, positions, offsets = _run_loader(model, loader, device)

    print(f"\n  Images:        {len(labels)}")
    print(f"  Active voxels: {features.shape[0]}")
    print(f"  Feature dim:   {features.shape[1]}")

    np.savez_compressed(
        out_path,
        backbone_features=features,
        labels=labels,
        positions=positions,
        offsets=offsets,
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
