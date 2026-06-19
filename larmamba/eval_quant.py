"""
Quantized-inference eval for a trained larmamba checkpoint.

Loads the MambaEncoder weights from a PoLAr-MAE Lightning checkpoint, optionally
quantizes the Linear layers (int8 or fp8 weight-only via torchao; the Mamba
selective-scan kernel + convs stay bf16), then runs the 4 probes
(sft_feat / voxel_svm_feat / sft_raw / voxel_svm_raw) on real APA2D events and
measures peak GPU memory during feature extraction. Produces one Pareto point
(accuracy vs peak memory) per (token count, precision).

Reuses PoLAr-MAE's probe helpers so the metrics match the training-time probes.

    python -m larmamba.eval_quant --ckpt <...>.ckpt --num_groups 256 --quant none
"""

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

from polarmae.eval.probes import (
    APA2DProbeCallback, _extract_per_voxel_features, _extract_per_voxel_raw,
    _fit_svm, _train_head, _head_predict, _confusion, _eff_purity, _macro_f1,
)
from larmamba import MambaEncoder

CENTER = torch.tensor([525.0, 562.0, 0.0])
SCALE = 1.0 / 600.0
PROBE_TRAIN = "/gpfs01/lbne/users/fm/cffm-data/prod-jay-100k-truth-2026-02-27/13874/1/00[1-8]"
PROBE_VAL   = "/gpfs01/lbne/users/fm/cffm-data/prod-jay-100k-truth-2026-02-27/13874/1/009"


class _Shim(nn.Module):
    """Minimal stand-in for the PoLArMAE LightningModule that the probe helpers
    need: .encoder, .val_transformations, .device."""
    def __init__(self, encoder, device):
        super().__init__()
        self.encoder = encoder
        self._device = device
    @property
    def device(self):
        return self._device
    def val_transformations(self, points):
        p = points.clone()
        p[..., :3] = (p[..., :3] - CENTER.to(points.device)) * SCALE
        return p


def build_encoder(num_groups, context_length, device):
    tk = {"group_radius": 5 / 600, "num_init_groups": num_groups, "context_length": context_length}
    enc = MambaEncoder(num_channels=4, arch="vit_small", voxel_size=5,
                       tokenizer_kwargs=tk,
                       transformer_kwargs={"add_pos_at_every_layer": True},
                       mamba_kwargs={"d_state": 16, "d_conv": 4, "expand": 2})
    return enc.to(device).eval()


def load_encoder_weights(enc, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    print(f"  loaded encoder: {len(enc_sd)} keys, missing={len(missing)} unexpected={len(unexpected)}")
    return enc


def apply_quant(enc, method):
    if method == "none":
        return enc, sum(p.numel() for p in enc.parameters())
    from torchao.quantization import quantize_, int8_weight_only, float8_weight_only
    q = int8_weight_only() if method == "int8" else float8_weight_only()
    quantize_(enc, q)   # swaps nn.Linear weights in-place; convs/norms untouched
    print(f"  applied {method} weight-only quantization to Linear layers")
    return enc, None


def n_classes_and_run(model, cap, device, sft_epochs=30, sft_batch=256, sft_lr=5e-3, svm_C=1.0):
    cb = APA2DProbeCallback(
        probe_data_path=PROBE_TRAIN, probe_val_data_path=PROBE_VAL,
        probe_batch_size=8, probe_num_workers=4,
        probe_dataset_kwargs=dict(apa=0, view="W", emin=1.0, emax=1.0e5,
                                  energy_threshold=1.0, min_points=256, max_points=8000,
                                  maxlen=-1, return_semantic_id=True,
                                  cache_dir="/gpfs01/lbne/users/fm/hyu/cache/data"),
        max_pixels_per_class=5000, svm_C=svm_C,
        sft_epochs=sft_epochs, sft_batch=sft_batch, sft_lr=sft_lr, train_frac=0.8,
    )
    dm = cb._get_datamodule(model)
    n_cls = dm.num_seg_classes
    class_names = [dm.seg_class_to_category[i] for i in range(n_cls)]

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    # autocast ONLY the encoder feature extraction; probe fitting stays fp32
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feat_pool, raw_pool = cb._collect_pool(model, dm)
    torch.cuda.synchronize()
    extract_s = time.time() - t0
    peak_mib = torch.cuda.max_memory_allocated() / 2**20

    rng = np.random.default_rng(0)
    ftr, fva = cb._split(feat_pool, rng)
    rtr, rva = cb._split(raw_pool, rng)

    out = {}
    # SVM probes
    out["voxel_svm_feat"] = _fit_svm(ftr.feats, ftr.labels, fva.feats, fva.labels, class_names, C=svm_C)["val_macro_f1"]
    out["voxel_svm_raw"]  = _fit_svm(rtr.feats, rtr.labels, rva.feats, rva.labels, class_names, C=svm_C)["val_macro_f1"]
    # MLP-head probes
    for tag, tr, va in [("sft_feat", ftr, fva), ("sft_raw", rtr, rva)]:
        head = _train_head(tr.feats, tr.labels, n_cls, epochs=sft_epochs, batch_size=sft_batch, lr=sft_lr, device=device)
        cm = _confusion(va.labels, _head_predict(head, va.feats, device), n_cls)
        out[tag] = _macro_f1(*_eff_purity(cm))
    return out, peak_mib, extract_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num_groups", type=int, default=256)
    ap.add_argument("--context_length", type=int, default=512)
    ap.add_argument("--quant", default="none", choices=["none", "int8", "fp8"])
    args = ap.parse_args()
    device = "cuda"

    print(f"=== larmamba quant-eval  ckpt={os.path.basename(args.ckpt)}  "
          f"num_groups={args.num_groups}  quant={args.quant} ===")
    enc = build_encoder(args.num_groups, args.context_length, device)
    enc = load_encoder_weights(enc, args.ckpt)
    enc, _ = apply_quant(enc, args.quant)
    model = _Shim(enc, torch.device(device)).to(device).eval()

    probes, peak_mib, extract_s = n_classes_and_run(model, 5000, device)

    res = dict(ckpt=os.path.basename(args.ckpt), num_groups=args.num_groups,
               quant=args.quant, peak_MiB=round(peak_mib, 1), extract_s=round(extract_s, 1),
               **{k: round(v, 4) for k, v in probes.items()})
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
