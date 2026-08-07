# larmamba2 Phase 0 — tile statistics (defaults frozen)

**Date:** 2026-08-07   **Script:** `larmamba2/tests/data_study.py`

Two samples, both with the APA2D SSL filter (W view, charge > 1, min 256
voxels, cap 8000):

- **1M SSL sample**: 286 events from `prod-jay-1M/13825/1/001` (30 files)
- **truth probe set**: 479/500 events from the `unified_probe_260702` dump

Image extent: **960 channels × ~1125–1178 ticks**. Voxels/event: median ~3.3k,
p95 at the 8000 cap. Charge: median ~714 ADC, p99 ~15.6k, max ~86k (the
log-transform to [−1, 1] stays mandatory).

## Non-empty tiles per event

| tile | sample | med | mean | p90 | p95 | p99 | max | occupancy |
|---|---|---|---|---|---|---|---|---|
| 3×3 | 1M    | 616 | 861 | 1854 | 2515 | 3653 | 4574 | 5.3/9 (59%) |
| 3×3 | truth | 572 | 798 | 1634 | 2314 | — | 4818 | 5.3/9 |
| **5×5** | **1M** | **324** | **440** | **892** | **1264** | **1921** | **2613** | **10.5/25 (42%)** |
| 5×5 | truth | 293 | 407 | 796 | 1175 | — | 2734 | 10.5/25 |
| 7×7 | 1M    | 216 | 295 | 582 | 850 | 1268 | 1772 | 15.3/49 (31%) |
| 7×7 | truth | 198 | 273 | 533 | 782 | — | 1728 | 15.5/49 |

The two samples agree closely — the truth set is not biased for this purpose.

## Frozen defaults (plan §3.1)

- **S = 5** — median ~324 tokens/event, same regime as polarmae's 256 groups;
  42% occupancy means a 5×5 patch is informative (not mostly padding, unlike
  7×7 at 31%, and not trivially dense like 3×3 at 59%).
- **T_max = 1024** at train — covers >p90 of events with zero truncation
  (vs the old tokenizer's routine overflow at 512); random tile subsample
  beyond it (augmentation, like `max_points`). **Uncapped at eval** — at p99 ≈
  1900 tiles the O(T) Mamba encoder is untroubled.
- Tile grid: 192 × 226 (5-px tiles over 960 × 1125) → tile coords fit easily
  in the 16-bit Morton range of `larmamba/serialize.py`.

Tile sizes 3 and 7 stay in the phase-3 ablation grid.
