"""
Dataset wrapper that exposes pixel-level PID labels for the SFT probe used by
mae/scripts/train_mae.py.

The wrapped base dataset is expected to yield (voxels, meta_dict) where
meta_dict contains 'pid_labels' aligned to the voxels' row order (i.e.,
APASparseMetaDataset with return_full_metadata=True and return_pixel_truth=True).

For each access we map the raw PDG codes in pid_labels to one of
{track=0, shower=1, other=2, ignore=-1} via models.mae_model.pdg_to_pixel_class
(connected-component blip detection on γ pixels).  Results are cached
per-index so repeated epochs are essentially free after the first pass.
"""

import torch
from torch.utils.data import Dataset

from models.mae_model import pdg_to_pixel_class


class SFTPixelPIDDataset(Dataset):
    def __init__(self, base, verbose: bool = False):
        self.base = base
        self._cache = {}
        self.verbose = verbose

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        voxels, meta = self.base[idx]
        if idx in self._cache:
            pix = self._cache[idx]
        else:
            pid = meta["pid_labels"]
            coords = voxels.coordinate_tensor.cpu().numpy()
            pix = pdg_to_pixel_class(pid, coords)
            self._cache[idx] = pix
            if self.verbose and (idx % 500 == 0):
                print(f"[SFTPixelPIDDataset] cached pixel-class for idx={idx}")
        return voxels, pix
