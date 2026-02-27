"""Sparse-aware pixel masking for DINO student augmentation."""

from typing import Tuple
import torch
from torch import Tensor


class SparseVoxelMasker:
    """
    Masks active pixels in a dense image before sparse conversion.

    For each image in a batch:
    1. Identify active pixel coordinates (non-zero values)
    2. Randomly select a fraction to mask
    3. Zero those pixels in the dense tensor

    The sparse backbone then excludes these zeros from its computation,
    so the student processes only the unmasked active pixels.
    """

    def __init__(self, mask_ratio: float = 0.5, seed: int = None):
        """
        Args:
            mask_ratio: Fraction of active pixels to mask (0.0 to 1.0)
            seed: Optional random seed for reproducibility
        """
        self.mask_ratio = mask_ratio
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply sparse-aware masking to a batch of dense images.

        Args:
            x: [B, 1, H, W] float32 tensor, typically from DUNEImageDataset

        Returns:
            x_student: [B, 1, H, W] modified tensor with masked pixels zeroed
            mask: [B, H, W] bool tensor, True where pixels were masked
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected single channel, got {C}"

        device = x.device
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        x_student = x.clone()

        for b in range(B):
            # Find all active (non-zero) pixels in this image
            active_coords = (x[b, 0] != 0).nonzero(as_tuple=False)  # [N, 2] (row, col)
            N = active_coords.shape[0]

            if N == 0:
                # Image has no active pixels, skip masking
                continue

            # How many pixels to mask
            n_mask = int(N * self.mask_ratio)

            if n_mask > 0:
                # Randomly select which active pixels to mask
                perm = torch.randperm(N, device=device)[:n_mask]
                masked_coords = active_coords[perm]

                # Mark these positions in the mask
                mask[b, masked_coords[:, 0], masked_coords[:, 1]] = True

                # Zero them in the student input
                x_student[b, 0, masked_coords[:, 0], masked_coords[:, 1]] = 0.0

        return x_student, mask
