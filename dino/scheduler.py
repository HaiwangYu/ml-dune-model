"""Cosine annealing schedule with optional warmup (adapted from DINOv2)."""

import numpy as np


class CosineScheduler:
    """
    Cosine annealing schedule with optional linear warmup and freeze period.

    Pre-computes the full schedule as a numpy array indexed by iteration.
    Supports:
    - Freeze period: values frozen at 0 for the first `freeze_iters` iterations
    - Warmup period: linear ramp from `start_warmup_value` to `base_value` over `warmup_iters`
    - Cosine annealing: cosine decay from `base_value` to `final_value` over remaining iterations
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
        freeze_iters: int = 0,
    ):
        """
        Args:
            base_value: Starting value after warmup (or at iteration 0 if no warmup)
            final_value: Final value at the last iteration
            total_iters: Total number of iterations in the schedule
            warmup_iters: Number of iterations for linear warmup
            start_warmup_value: Starting value before warmup ramp
            freeze_iters: Number of iterations to freeze at 0 before starting warmup
        """
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        # Freeze period (values remain 0)
        freeze_schedule = np.zeros(freeze_iters)

        # Warmup period (linear ramp from start_warmup_value to base_value)
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        # Cosine annealing period (from base_value to final_value)
        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        cosine_schedule = (
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + np.cos(np.pi * iters / len(iters)))
        )

        # Concatenate all periods
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, cosine_schedule))

        assert len(self.schedule) == total_iters, (
            f"Schedule length mismatch: {len(self.schedule)} vs {total_iters}"
        )

    def __getitem__(self, it: int) -> float:
        """Get the scheduled value at iteration `it`."""
        if it >= self.total_iters:
            return self.final_value
        else:
            return float(self.schedule[it])
