# larmamba2 — native-2D grid-patch tokenizer + Mamba backbone: project plan

**Date:** 2026-08-06   **Status:** plan (phase 0)   **Tracker:** (issue link added after creation)
**Folder:** `/larmamba2`   **Env:** `uvenv-polar-mae` (torch 2.5.1 + mamba-ssm + lightning)

## 1. Motivation

larmamba (round 1+2) reused PoLAr-MAE's *3D point-cloud* tokenizer on 2D
wire-plane data by hacking `z = 0`: FPS center selection + ball-query grouping +
PointNet per-group embedding. That carries real costs on 2D sparse images:

- **Wrong geometry prior.** FPS/ball-query is designed for irregular 3D point
  clouds; our data lives on a regular integer (channel, tick) grid.
- **Non-determinism + overflow.** CNMS grouping warned `more groups than the
  context length allows (… > 512)` on dense events in every training run;
  group count depends on event shape, requiring the `max_points=8000` subsample
  hack.
- **Heavy deps / GPU-only.** pytorch3d FPS/KNN keeps the tokenizer off CPU —
  the tokenizer, not the Mamba mixer, blocks a CPU inference path.
- **Cost.** FPS + ball query is a nontrivial fraction of step time vs a
  gather/scatter tiler.

**larmamba2 = grid-tile patching (native 2D) → CNN patch tokenizer → serialized
bidirectional-Mamba encoder → token→patch decoder with pixel-space L1.**
Success is judged with the leakage-free unified probe against the round-2
baselines (see §8).

## 2. Data facts (measured on the 500-event probe dump, APA2D filter applied)

W view, APA 0: image is **960 channels × ~1178 ticks**. After the SSL filter
(min 256 voxels, cap 8000): median 3018 voxels/event.

| tile | tiles/event med / mean / p90 / p95 / max | occupancy (of tile px) |
|---|---|---|
| 3×3 | 572 / 798 / 1634 / 2314 / 4818 | 5.3/9 (59%) |
| **5×5** | **293 / 407 / 796 / 1175 / 2734** | **10.5/25 (42%)** |
| 7×7 | 198 / 273 / 533 / 782 / 1728 | 15.5/49 (32%) |
| 10×10 | 134 / 183 / 343 / 499 / 1080 | 22.9/100 (23%) |

Reading: **5×5 gives a median token count (~293) comparable to polarmae's 256
groups**, with a fixed 192×236 tile grid. A **cap of 1024 tiles** covers >p90 of
events with zero truncation (vs the current tokenizer's routine overflow at
512). Phase 0 re-measures this on the 1M SSL sample (the truth set may be
biased) before freezing defaults.

## 3. Design

```
points (N,4)=(ch,tick,0,logq)                       [APA2D unchanged]
  └─ Tiler: tile_id=(ch//S, tick//S); scatter to dense (T,1,S,S) patches
       keep non-empty tiles; cap T<=T_max (random subsample if over)
  └─ PatchEmbed: CNN  (T,1,S,S) -> (T,D)   + 2D pos-embed from tile centers
  └─ Serialize: space-filling-curve order over (tile_row, tile_col)
  └─ Mask: random ratio r over tiles; encoder sees VISIBLE tokens only (MAE)
  └─ Encoder: bidirectional Mamba blocks (reuse larmamba)   -> latent tokens
  └─ Decoder: insert [MASK]+pos tokens in serial order -> mixer -> recon head
  └─ Loss: pixel-space L1(log-charge) on MASKED tiles only
```

### 3.1 Patching (grid tiler)

- `tile_id = (ch // S, tick // S)`, default **S = 5**; gather voxels per tile
  into a dense `S×S` log-charge patch (empty px = log_transform(0) = −1).
- Drop empty tiles. Cap at `T_max` (default **1024**, from §2): if exceeded,
  **random tile subsample** for SSL (an augmentation, like `max_points`);
  ablate top-charge keep. **At inference/eval: no cap** — Mamba is O(T), so all
  tiles fit; this removes the train-time truncation from eval entirely.
- Deterministic, dependency-free (pure torch scatter) → CPU-capable.
- Options to study: S ∈ {3, 5, 7}; overlapping tiles (stride < S) if boundary
  cuts hurt (§9 risk).

### 3.2 Tokenization (patch → D) — options

| | option | notes |
|---|---|---|
| **A (rec.)** | **Conv patch-embed, end-to-end**: 2–3 conv layers (or one S×S conv) 1×S×S → D=384, trained through the MAE objective | Simplest; with the pixel decoder (§3.4) the whole model *is* a masked conv autoencoder — no separate AE stage |
| B | Pretrained tiny conv-AE: train 1×S×S → d → 1×S×S offline on random tiles; frozen (or finetuned) encoder as tokenizer | Decoupled tokenizer; ablation for "does joint training matter" |
| C | VQ-VAE codebook (BEiT-style): discrete token ids; loss becomes masked-token classification | Heavier; only if A/B plateau |
| D | Linear embed baseline: flatten 25 px → Linear(25, D) | Cheap control — 5×5 is small; if D matches A, the CNN adds nothing |

Recommendation: **A primary, D as control in the same sweep**, B as one
ablation; C deferred.

### 3.3 Serialization — options

| | option | notes |
|---|---|---|
| **A (rec.)** | **Morton/Z-order** on (tile_row, tile_col) | Direct reuse of `larmamba/serialize.py` (`morton2d` already takes integer 2D coords) |
| B | Hilbert curve | Strictly better locality; small extra code; ablation |
| C | Raster tick-major (drift-time order) / channel-major | Physics-natural (tracks evolve in time); trivial |
| D | Multi-order (different curve per layer, PointMamba-style) | Only if A–C show order sensitivity |

The bidirectional mixer already scans each order both ways, so every option
gives 2 directions for free.

### 3.4 Backbone + decoder

- **Encoder:** reuse larmamba's `BiMambaMixer`/`MambaBlock` stack (vit_small
  width, D=384, ~12 blocks; `d_state 16, d_conv 4, expand 2`). Pos-embed: MLP
  on normalized tile-center (ch, tick) — same recipe as polarmae/larmamba.
- **Decoder options:**
  - **A (rec. first):** small transformer decoder, 4 blocks (polarmae recipe,
    known-good). O(T²) only in the shallow decoder, T ≤ 1024 — cheap.
  - B: Mamba decoder (BiMamba blocks) — fully linear-time end-to-end; run as
    ablation for the "pure Mamba" story.
  - Head (orthogonal): per-token `Linear(D → S²)` reshaped to S×S (rec.), or a
    small transposed-conv head (the mirror of tokenizer A); ablate.
- **Masking:** random ratio **0.6** over non-empty tiles (match polarmae);
  MAE-style — encoder consumes visible tokens only (efficiency), decoder
  inserts learnable [MASK]+pos tokens at their serialized positions.

### 3.5 Target & loss

**Agreed: no separate energy head.** polarmae needed one because its Chamfer
loss lives in coordinate space and is blind to per-point charge; here the
reconstruction target — the S×S log-charge patch — carries geometry (which
pixels are hit) and charge (their values) in one object, so a single
pixel-space loss covers both.

- **Primary (rec.):** **L1 on log-charge over all S² pixels of masked tiles
  only** (mae-repo experience: L1 on log charge is stable).
- Variants to ablate: L2 / SmoothL1; **zero-inflation-aware**: BCE on
  occupancy (hit/no-hit) + L1 on hit pixels only — guards against the trivial
  "predict −1 everywhere" solution if plain L1 under-trains charge; charge-
  weighted L1.
- Monitor at val time: recon L1 split into hit-px vs empty-px components (the
  cheap way to see if the model only learns occupancy).

### 3.6 Validation

- **Metric of record:** the leakage-free unified probe
  (`dino/diagnostics/ab_pid_probe.py`) — sft/svm × feat/raw on the same 500
  events / seed / split as every other model. Reuse
  `CONDOR_OUT/unified_probe_260702/events.npz`.
- **Per-voxel features:** each voxel belongs to exactly one tile → voxel
  feature = its tile's encoder token. Exact assignment, match rate 1.0 by
  construction, no KNN interpolation (cleaner than polarmae/larmamba's
  inverse-distance blend). New exporter `larmamba2/export_pid_features.py`
  (same npz schema); plug into `gridutils/submit_eval_epochs.sh` via a new
  encoder mode.
- **Per-epoch trailing eval** during training, reusing the round-2 infra
  (`watch_pmae20.sh` pattern). No in-training probe (DDP-unsafe + it's the old
  leaky one).

## 4. Training plan

- **Data/module:** APA2D datamodule unchanged (points format; tiling happens
  in-model). Train `prod-jay-1M/13825/1/00[1-8]`, val `009` (capped 2000).
- **Recipe:** copy round 2 — 4-GPU DDP on L40S, bf16, 20 epochs, epoch cosine
  (1-ep warmup, LR 7e-5 @ eff-batch 64), EarlyStopping `loss/val` (informational
  — stop decisions come from the probe curve), per-epoch checkpoints.
- **New Lightning entry** `larmamba2/train.py` (own LightningModule — the loss
  is not polarmae's, so we don't reuse `polarmae.tasks`); keep the
  `run_polarmae_fit.py` lessons: minimal trainer_defaults, `CKPT_DIR` env,
  **`NCCL_P2P_DISABLE=1`** on this pool.
- **Condor:** clone `submit_polarmae20.sh`/`trainjob_polarmae20.sh` →
  `*_larmamba2.sh`. 13 L40S free as of writing; ablations are ~2 h each
  (round-2 runs were ~5.4 min/epoch), so the phase-3 grid is very batchable.

## 5. Repo layout

```
larmamba2/
  __init__.py
  tiler.py                 # points -> (patches, tile_coords, mask) padded batch
  patch_embed.py           # conv / linear tokenizers (options A/D)
  serialize.py             # morton (reuse larmamba) + hilbert/raster orders
  encoder.py               # BiMamba encoder over patch tokens (reuse larmamba blocks)
  decoder.py               # transformer/mamba decoder + S^2 recon head
  ssl_module.py            # LightningModule: tile->embed->mask->encode->decode->L1
  train.py                 # LightningCLI entry (DDP-safe defaults)
  export_pid_features.py   # tile-token -> per-voxel feature npz for ab_pid_probe
  configs/{larmamba2_smoke.yml, larmamba2_full20.yml}
  tests/{test_tiler.py, test_roundtrip.py, bench_tokenizer.py}
  docs/0-plan.md (this file)
```

## 6. Phases

- **Phase 0 — data study (hours).** Re-run §2 on a 1M-sample slice; freeze
  S=5?, T_max=1024?; tile-charge stats; write `docs/1-data-study.md`.
- **Phase 1 — core build + smoke (1–2 days).** Tiler (unit-tested round-trip:
  voxels→patches→voxels), patch-embed, serialization, encoder/decoder,
  LightningModule; 1-GPU smoke on a small slice; verify recon images look sane;
  4-GPU DDP smoke (NCCL fix).
- **Phase 2 — first full run (~1 day incl. evals).** 20-epoch full-1M with
  defaults (S=5, T_max=1024, conv embed, Morton, transformer decoder, L1);
  per-epoch unified eval; compare curve to round-2 baselines.
- **Phase 3 — ablations (parallel, ~2 h each).** Tile size {3,5,7}; cap
  {512,1024,uncapped-train?}; serialization {morton, hilbert, tick-raster};
  tokenizer {conv, linear, pretrained-AE}; loss {L1, L1+occupancy-BCE, L2};
  decoder {transformer, mamba}; masking ratio {0.5,0.6,0.75}.
- **Phase 4 — head-to-head + docs.** Best larmamba2 vs polarmae-R2 vs
  larmamba-R2 (same probe table); tokenizer wall-time/memory benchmark
  (grid-tiler vs FPS+ball-query); CPU-path check; `docs/N-results.md`;
  update `docs_haiwang` session notes.

## 7. Success criteria

1. **Accuracy (primary):** unified-probe plateau `svm_feat >= 0.815` (larmamba-R2
   parity); stretch `> 0.823` (beat polarmae-R2 best).
2. **Tokenizer:** deterministic tile counts (no overflow warnings), tokenizer
   step-time ≤ FPS+ball-query, and a working no-pytorch3d CPU tokenize path.
3. **Simplicity:** no energy head, no z=0 hack, eval without train-time caps.

## 8. Baselines to beat (unified probe, 500 events, seed 0)

| model | best svm/sft | plateau (ep7–19) svm/sft |
|---|---|---|
| polarmae R2 (20 ep) | 0.823 / 0.809 (ep17) | 0.819 / 0.804 |
| larmamba R2 (20 ep) | 0.818 / 0.797 (ep19) | 0.815 / 0.795 |
| raw floors | svm_raw 0.402 · sft_raw 0.485 | (identical every run) |

## 9. Risks / mitigations

- **Tile-boundary cuts** (a 5×5 grid slices tracks arbitrarily): Mamba mixing
  across serialized neighbors should absorb it; if not → overlapping tiles or
  larger S.
- **Zero-inflation trivial minimum** (predict −1 everywhere): monitor hit-px
  vs empty-px loss split; occupancy-BCE variant on standby.
- **Linear ≈ conv tokenizer** (25-px patches are small): then take the simpler
  linear — that is a result, not a failure.
- **Cap truncation on dense events** (p95 > 1024 at S=5): random-subsample is
  an augmentation at train; eval runs uncapped.
- **Comparability discipline:** all numbers from `ab_pid_probe`, same events /
  seed; never compare across differently-configured probes (unified-probe
  lesson).
