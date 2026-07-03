# Claude session summary — 2026-05-28 → 2026-07-02

DUNE 2D-LArTPC (channel × tick, sparse) foundation-model work. This file
indexes what was built/learned so future sessions can pick up fast. Deeper
detail lives in each sub-project's `docs/`.

## The models compared (pixel-PID: track / shower / other, 4 probes)

| Model | encoder | feature macro-F1 (sft_feat / svm_feat) | params | notes |
|---|---|---|---|---|
| mae | sparse-CNN (MinkUNet + 1 attn bottleneck) | 0.67 / 0.65 | 0.48M | charge-recon MAE |
| dino | same sparse-CNN | 0.72 / 0.66 | 0.48M | EMA contrastive |
| sparseformer (best) | sparse-CNN, stem + deep attn, 256ch | **0.77** / 0.72 | 19.5M | see below |
| **PoLAr-MAE** | FPS tokenizer + ViT-S attention | **0.93 / 0.94** | ~22M | prior best |
| **larmamba** | FPS tokenizer + bidirectional Mamba | **0.94 / 0.93** | ~26M | linear-time mixer |

**The 4 probes** (all reuse the same helpers → fair): `sft_feat` (MLP on
features), `voxel_svm_feat` (linear SVM on features), `sft_raw`/`voxel_svm_raw`
(same on raw ch,tick,logQ — the no-backbone floor ≈0.43 svm_raw across all).

## Sub-project 1: dino probe comparison (docs in mae/docs/)
Ported the 4-probe eval to dino (`dino/diagnostics/run_probes.py`), evaluated
mvicenzi's `longer_contrast_slow` checkpoints (ep10/50/100). dino teacher ≈
student, best sft_feat 0.72. Extended the comparison table + wrote
`mae/docs/comparison_methodology.html` (how the 4 probes are computed;
per-arch feature source; probe independence — no weight sharing).

## Sub-project 2: larmamba (NEW folder /larmamba, docs in larmamba/docs/)
**Goal: PoLAr-MAE accuracy at lower/better-scaling GPU memory via a linear-time
(Mamba/SSM) encoder instead of O(N²) attention.**
- Reuses PoLAr-MAE's tokenizer / dataset / Chamfer+energy loss / training loop /
  probes; swaps ONLY the encoder mixer. Runs in env `uvenv-polar-mae`
  (torch 2.5.1), `import polarmae`. Config points `encoder.class_path` at
  `larmamba.MambaEncoder`.
- **Pieces**: `ssm.py` (selective scan: mamba-ssm CUDA kernel fast-path +
  pure-torch chunked/sequential fallback, unit-tested fwd+grad); `serialize.py`
  (Morton/Z-order token ordering); `mamba_block.py` (bidirectional mixer);
  `mamba_encoder.py` (`MambaEncoder`, drop-in for `TransformerEncoder`).
- **Result**: matches PoLAr-MAE (sft_feat 0.94). **Token sweep 256/512/1024 all
  ~0.93 → task saturates at 256 tokens.**
- **Cost (RESULTS.md + larmamba_methodology.html)**: mixer scales O(T) — at 8192
  tokens Mamba uses 1.8 GB while attention OOMs; but at this task's ~144–256
  tokens larmamba ≈ polarmae (~285 vs 265 MiB, batch 1).
- **CPU**: mae/dino CANNOT run on CPU (WarpConvNet is CUDA-only). larmamba runs
  (pure-torch scan) but ~8× slower than polarmae attention at ~160 tokens;
  sequential CPU scan ~11% faster than chunked; `torch.compile` times out
  (inductor hostile to the scan loops) — real CPU speedup needs a fused kernel.
- Env additions (NOT committed, additive to uvenv-polar-mae): mamba-ssm 2.2.5
  (cu12torch2.5), causal-conv1d 1.5.0.post8, einops, huggingface_hub.

## Sub-project 3: sparseformer (NEW folder /sparseformer, docs in sparseformer/docs/)
**Goal: can the cheap sparse-CNN family be lifted to PoLAr-MAE accuracy?**
Runs in the ml-dune-model env (torch 2.10 + WarpConvNet, GPU-only). Configurable
backbone plugged into `SparseMAEModel` via `backbone_name`/`backbone_kwargs`.
- Round 1 (shape): local-geometry stem helps (+0.02), deep attention alone
  neutral, combo (stem+attn) best sft_feat **0.722**.
- Round 2 (capacity × objective): width 128→256ch (19.5M ≈ polarmae) lifts to
  **0.769**; true-MAE did NOT help (below std-MAE). All peak at epoch 1 then
  decline (MAE saturation).
- **Verdict**: even at polarmae-matched capacity + true-MAE, per-voxel sparse-CNN
  plateaus ~0.77, ~0.16 below polarmae. The representation itself is the ceiling;
  tokenizer+transformer is the necessary route to 0.9+. Cheapest inference
  though: sparse-CNN ~49 MiB/event (GPU-only), capped ~0.72–0.77.

## Sub-project 4: quantization (larmamba/docs/QUANT_RESULTS.md)
torchao int8/fp8 weight-only (torchao 0.7.0, pinned for torch 2.5; Mamba
scan+convs stay bf16). `larmamba/eval_quant.py` (--encoder mamba|polarmae).
- **int8 & fp8 iso-accuracy (~0.93), cut peak GPU mem ~9–11%** for BOTH larmamba
  and polarmae. fp8 works on L40S (sm_89). Weight-only quant leaves flash-attn
  intact.
- **Head-to-head @ 256 tokens (batch 8)**: quantized larmamba 1099 MiB vs
  polarmae 1094 MiB — **tied**; polarmae marginally ahead on bf16 mem, accuracy
  (svm 0.938 vs 0.929), speed (2–3s vs 7s). At this operating point larmamba is
  NOT cheaper than polarmae — its only edge is the high-token regime.

## Bottom-line takeaways (for future work)
1. **Tokenizer + global mixer is required** for ~0.93 on this task; sparse-CNN
   caps ~0.77 regardless of capacity/objective.
2. **larmamba = polarmae** in accuracy and (quantized) memory at this task's
   token scale. Mamba's structural win (O(T) memory, 8192-token headroom) only
   pays off if a task NEEDS many more tokens (finer tokenization / 3D /
   multi-plane) — this DUNE pixel-PID task saturates at 256, so there's no
   accuracy headroom to capture here.
3. **Quantization** is a free ~10% iso-accuracy memory cut for either transformer
   model; orthogonal to the architecture choice.
4. **Cheapest inference** = small sparse-CNN (~49 MiB/event, GPU-only, ~0.72).
   **Best accuracy** = polarmae or larmamba (~0.93). No single model wins both.

## Pixel-label pipeline (asked often)
- Raw per-pixel PDG truth: `frame_pid_1st` in the 100k-truth dataset, read by
  `loader/apa_sparse_meta_dataset.py::_read_pixel_truth` (aligned to reco voxels;
  no-truth → 0). The 1M SSL set has no truth → probes use the 100k-truth set.
- PDG → {track,shower,other}: `models/mae_model.py::pdg_to_pixel_class`
  (μ/p/π→track; e±/large-γ→shower; γ-blip/rest→other; 0→-1 ignored), called at
  SFT/probe time. dino/polarmae/larmamba all reuse it → fair comparison.

## Key artifacts (GPFS: /gpfs01/lbne/users/fm/hyu/CONDOR_OUT/)
- larmamba: `larmamba_{full,g512,g1024}_260613/`, `larmamba_quant_g{256,512,1024}_260619/`
- polarmae: `polarmae_apa2d_full_260523_v3/`, `polarmae_quant_g256_260620/`
- sparseformer: `sf_{ctrl,opt1,opt2,combo}_260614/`, `sf_{w128,w256,w128true,w256true}_260614/`
- Checkpoints live under `<run>/lightning_logs/lightning_logs/version_0/checkpoints/epoch=3-step=20000.ckpt`

## Envs
- ml-dune-model: `/gpfs01/lbne/users/fm/hyu/uvenv` (torch 2.10 + WarpConvNet) — mae/dino/sparseformer.
- polarmae/larmamba: `/gpfs01/lbne/users/fm/hyu/uvenv-polar-mae` (torch 2.5.1, pytorch3d, lightning, +mamba-ssm/torchao).
- Grid: gridutils/submit_*.sh + trainjob_*.sh; L40S pool (up to ~6 free); condor arguments= must NOT quote ckpt paths.

## Note
A later commit `aa3bd0d dino: unified leakage-free per-pixel PID probe (ported
from WC_FM_DINO)` postdates the larmamba/sparseformer/quant work above and was
not part of this session's analysis — check it before re-running dino probes.
