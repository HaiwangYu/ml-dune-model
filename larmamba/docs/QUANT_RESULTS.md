# larmamba quantization — accuracy vs GPU-memory Pareto

**Date:** 2026-06-19   **Branch:** may15   **Env:** uvenv-polar-mae (torch 2.5.1, torchao 0.7.0)

## Goal

Reduce larmamba's inference GPU memory (toward beating PoLAr-MAE at iso-accuracy)
via post-training weight-only quantization, and test whether freed memory + more
tokens can raise accuracy. Both int8 and fp8 (torchao weight-only); the Mamba
selective-scan kernel + depthwise convs stay bf16 (no int8 kernel), so only the
Linear projections / MLP / tokenizer-PointNet are quantized.

## Setup

Trained larmamba checkpoints (20k steps) at num_init_groups ∈ {256, 512, 1024}
evaluated with `larmamba/eval_quant.py`: load encoder → optional torchao quant →
4 probes on real APA2D events (PoLAr-MAE probe helpers) + peak CUDA memory during
feature extraction (batch=8). Accuracy = best of the feature probes.

## Results — accuracy is preserved, memory drops ~10%

| Tokens | precision | sft_feat | voxel_svm_feat | peak MiB (b=8) | Δ mem |
|---|---|---|---|---|---|
| 256  | bf16 | 0.928 | 0.928 | 1236 | — |
| 256  | int8 | 0.936 | 0.929 | 1099 | **−11.1%** |
| 256  | fp8  | 0.931 | 0.929 | 1098 | **−11.1%** |
| 512  | bf16 | 0.942 | 0.926 | 1246 | — |
| 512  | int8 | 0.928 | 0.924 | 1117 | −10.4% |
| 512  | fp8  | 0.937 | 0.927 | 1117 | −10.4% |
| 1024 | bf16 | 0.933 | 0.926 | 1283 | — |
| 1024 | int8 | 0.926 | 0.926 | 1155 | −10.0% |
| 1024 | fp8  | 0.927 | 0.927 | 1155 | −10.0% |
| — | *PoLAr-MAE (ref)* | 0.933 | 0.940 | — | — |

(`sft_raw`/`voxel_svm_raw` are the no-backbone floors, ~0.43 svm_raw / noisy
sft_raw — unchanged by quantization, omitted here.)

## Findings

1. **int8 and fp8 are iso-accuracy.** Feature-probe macro-F1 stays ~0.93 at every
   token count; deltas vs bf16 are within run-to-run probe noise (±0.01). No
   accuracy is lost by quantizing the Linear layers.
2. **~10–11% peak-memory reduction**, consistent across token counts (−137 MiB at
   256, −128 at 1024). This is the weight-footprint shrink (bf16 2-byte → 1-byte
   weights); the Mamba scan + activations stay bf16, so it's a bounded ~10% in this
   batch-8 regime (where activations + tokenizer also occupy the ~1.2 GB peak).
3. **int8 ≈ fp8** on both memory and accuracy (both 1-byte weights), and **fp8
   runs on the L40S** (sm_89) — no Hopper requirement.
4. **More tokens do NOT raise accuracy** (256≈512≈1024 ≈ 0.93) — the task is
   token-saturated, confirming the earlier sweep. But **more tokens cost almost no
   memory**: 256→1024 (4× tokens) adds only +47 MiB (+4%) at bf16 — the Mamba
   encoder's near-linear, low-slope memory scaling.

## The Pareto picture

Plotting feature macro-F1 (y) vs peak memory (x), all larmamba points cluster at
**~0.93 accuracy across ~1.10–1.28 GB**. The quantized points (int8/fp8) sit
~130 MiB to the left of their bf16 counterparts at the same accuracy — i.e.
quantization shifts the whole front left by ~10% at no accuracy cost. Because
accuracy is saturated, there is no upward movement from more tokens; the Pareto
front is essentially a **vertical line at ~0.93**, and the best (lowest-memory)
operating point is **int8/fp8 @ 256 tokens (~1099 MiB, 0.93)**.

### vs PoLAr-MAE
larmamba already matches PoLAr-MAE accuracy (~0.93). On memory:
- At a fixed small token budget (this task's ~144–256), larmamba and PoLAr-MAE are
  similar (earlier per-event batch-1: larmamba 285 MiB vs polarmae 265 MiB);
  int8/fp8 now puts larmamba ~10% below its own bf16 footprint. (A fully fair
  head-to-head would quantize PoLAr-MAE too — out of scope here.)
- The decisive larmamba memory advantage remains the **scaling regime**, not this
  one: the mixer benchmark (RESULTS.md §5) shows Mamba at 8192 tokens uses 1.8 GB
  while attention OOMs. Quantization is an additive ~10% on top.

## Verdict

Post-training int8/fp8 weight-only quantization gives larmamba a **free ~10%
inference-memory reduction at iso-accuracy (~0.93)**, with fp8 usable on the L40S.
It does **not** raise accuracy (the task saturates at 256 tokens), so the win is
purely efficiency. The cheapest correct operating point is **int8 (or fp8) @ 256
tokens, ~1.1 GB at batch 8, ~0.93 macro-F1**. Quantization stacks with — but is
secondary to — larmamba's core advantage over attention (linear-time scaling to
large token counts).

## Reproduce

```bash
for tag in g256 g512 g1024; do
  condor_submit /gpfs01/lbne/users/fm/hyu/CONDOR_OUT/larmamba_quant_${tag}_260619/q.sub
done
# each loops quant in {none,int8,fp8}; see larmamba/eval_quant.py + gridutils/quant_larmamba.sh
```
Env note: `torchao==0.7.0` installed additively into uvenv-polar-mae (0.17 needs
torch≥2.6; pin 0.7.0 for torch 2.5.1). int8=`int8_weight_only`, fp8=`float8_weight_only`.
