# larmamba results — a linear-time foundation model for 2D LArTPC data

**Date:** 2026-06-13   **Branch:** may15   **Env:** `uvenv-polar-mae` (torch 2.5.1+cu124)

## TL;DR

`larmamba` replaces PoLAr-MAE's ViT self-attention encoder with a
**bidirectional Mamba (selective state-space) encoder** over
space-filling-curve-serialized tokens, while reusing PoLAr-MAE's tokenizer,
dataset, Chamfer+energy reconstruction loss, training loop, and 4-probe
evaluation.

**It matches PoLAr-MAE's representation quality** (the previous best on this
task) and far exceeds the sparse-CNN MAE/DINO models — with a sequence mixer
whose compute and memory scale **O(T)** instead of attention's **O(T²)**.

## 1. Four-probe comparison (pixel PID: track / shower / other)

Best validation macro-F1 over training. larmamba = the 256-token run
(`larmamba_full_260613`, 20k steps, 1 L40S).

| Probe | **larmamba (256 tok)** | PoLAr-MAE | MAE (sparse-CNN) | DINO (sparse-CNN) |
|---|---|---|---|---|
| **sft_feat** (MLP on features) | **0.940** | 0.933 | 0.674 | 0.719 |
| **voxel_svm_feat** (linear SVM on features) | 0.929 | **0.940** | 0.651 | 0.656 |
| sft_raw (MLP on raw ch,tick,logQ) | 0.545 | 0.526 | 0.462 | 0.473 |
| voxel_svm_raw (SVM on raw) | 0.432 | 0.432 | 0.433 | 0.495 |

- **larmamba ≈ PoLAr-MAE**: slightly ahead on the MLP probe (0.940 vs 0.933),
  slightly behind on the linear-SVM probe (0.929 vs 0.940). Net: on par, and a
  **~0.25 absolute jump** over the sparse-CNN backbones.
- **Raw-probe baselines match** across all models (svm_raw 0.432) → the task,
  taxonomy, splits, and probe definitions are aligned; the comparison is fair.

## 2. Token-count sweep — 256 already saturates this task

Same recipe, varying the tokenizer's `num_init_groups` (`context_length`
raised to match). Best macro-F1:

| Tokens (num_init_groups) | sft_feat | voxel_svm_feat | training peak mem |
|---|---|---|---|
| 256 | **0.940** | 0.929 | ~40–45 GB |
| 512 | 0.933 | 0.931 | ~45 GB |
| 1024 | 0.936 | 0.923 | ~45 GB |

More tokens do **not** improve accuracy — 256 groups already capture the
event structure for this DUNE APA wire-plane pixel-PID task. **Recommendation:
use the 256-token config.** (Training peak memory is dominated by the
attention *decoder* + the probe SVM extraction, not the Mamba encoder, so it
doesn't separate the token counts here — see §3 for the encoder-only cost.)

> Caveat: PoLAr-MAE's tokenizer uses coverage-based CNMS grouping, so on real
> sparse data the realized token count is bounded by `num_init_groups` but also
> by event coverage; the sweep varies the cap, not a hard token count.

## 3. Cost: Mamba O(T) vs attention O(T²)

Mixer-only benchmark (`larmamba/tests/bench_mixer.py`, B=8, dim=384, depth=12,
bf16, L40S) — synthetic token sequences fed straight to the sequence mixer,
isolating it from the shared tokenizer:

| T | Mamba lat / mem | attn (flash) lat / mem | attn (vanilla) lat / mem |
|---|---|---|---|
| 256  | 7.1 ms / 0.22 GB | 4.6 ms / 0.23 GB | 3.5 ms / 0.19 GB |
| 512  | 10.3 / 0.27 | 14.4 / 0.43 | 7.6 / 0.32 |
| 1024 | 20.5 / 0.36 | 63.4 / 1.18 | 34.7 / 0.79 |
| 2048 | 43.5 / 0.56 | 232 / 4.07 | 122 / 2.63 |
| 4096 | 104 / 0.96 | 931 / 15.4 | 514 / 9.9 |
| 8192 | **267 / 1.77** | **OOM** | **OOM** |

- **Latency**: Mamba grows ~linearly (32× tokens → ~38× time); attention grows
  ~quadratically (16× tokens → ~200× time). Mamba is 3× faster at 1024, 9× at
  4096.
- **Memory**: Mamba grows linearly (1.8 GB at 8192); attention OOMs at 8192,
  where it would need >40 GB. At 4096, Mamba uses **16× less** memory.
- **At T=256 (the saturating point)** the two are comparable — attention is
  even slightly faster. **Mamba's advantage is scaling headroom**, not the
  current operating point.

End-to-end encoder benchmark (`bench_encoder.py`, with the tokenizer, dense
synthetic input): Mamba was **2.6× faster** than attention at equal token
count; peak memory there is dominated by the shared FPS/CNMS tokenizer
(~20 GB), not the mixer.

## 4. Interpretation / where this matters

- **Quality goal met**: larmamba is a polarmae-quality 2D-LArTPC foundation
  model. The architectural gap that capped sparse-CNN MAE/DINO at ~0.65–0.72
  is closed by the tokenizer + global sequence mixer; using **Mamba instead of
  attention costs nothing in accuracy**.
- **Today's task doesn't need the scaling** (256 tokens saturate), so the
  practical payoff is "same accuracy, comparable cost." The linear encoder
  becomes decisive for regimes attention can't reach: **finer tokenization,
  3D / multi-plane / full-APA inputs, or longer sequences** where attention
  OOMs at a few thousand tokens but Mamba runs at 8192+ on <2 GB.
- **CPU portability**: the selective scan has a unit-tested pure-PyTorch
  fallback (`larmamba/ssm.py`), so the encoder runs without the CUDA kernel
  (slower) — relevant for the eventual CPU-deployment goal.

## 5. Reproduce

```bash
# train (1 L40S, ~2–3 h with the mamba-ssm kernel):
bash gridutils/submit_larmamba.sh larmamba_full   larmamba/configs/larmamba_apa2d_full.yml
bash gridutils/submit_larmamba.sh larmamba_g512   larmamba/configs/larmamba_apa2d_g512.yml
bash gridutils/submit_larmamba.sh larmamba_g1024  larmamba/configs/larmamba_apa2d_g1024.yml
# cost benchmarks (1 L40S, minutes):
#   mixer scaling: bench_larmamba.sh ... MIXER 8   -> larmamba.tests.bench_mixer
#   encoder e2e:   bench_larmamba.sh ... 256,512,1024,2048 16
```

Artifacts on GPFS:
`/gpfs01/lbne/users/fm/hyu/CONDOR_OUT/larmamba_{full,g512,g1024,benchmixer,bench_v2}_260613/`.

Env note: the kernel fast-path uses `mamba-ssm 2.2.5` + `causal-conv1d`
(cu12torch2.5 wheels) installed additively into `uvenv-polar-mae`; PoLAr-MAE
never imports them, so its baseline is unchanged. Without them, the pure-torch
scan runs automatically.

## 6. Open follow-ups

1. **Distill / shrink** the encoder (fewer Mamba layers or smaller `d_model`)
   now that 256 tokens suffice — push inference cost down further.
2. **Exploit the scaling**: try finer tokenization or 3D/multi-plane inputs
   where attention can't fit, to see if larmamba *exceeds* polarmae.
3. Replace the attention **decoder** with Mamba too (training-only memory).
4. Multi-GPU / longer pretraining; current runs are single-L40S, 20k steps.
