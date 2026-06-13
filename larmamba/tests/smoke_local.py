"""
Local CPU integration smoke for larmamba.

Builds a MambaEncoder + polarmae decoder inside a PoLArMAE LightningModule,
feeds a tiny synthetic batch through compute_loss, and runs a few optimizer
steps.  Verifies the encoder swap is wired correctly end-to-end without needing
a GPU or the real dataset.

    /gpfs01/lbne/users/fm/hyu/uvenv-polar-mae/bin/python -m larmamba.tests.smoke_local
"""

import torch

from polarmae.layers.encoder import TransformerEncoder  # noqa: F401 (ensures polarmae import works)
from polarmae.layers.decoder import TransformerDecoder
from polarmae.models.ssl.polarmae import PoLArMAE

from larmamba import MambaEncoder


def make_batch(B=2, Nmax=400, device="cpu"):
    """Synthetic (channel, tick, 0, log_charge) points + lengths."""
    pts = torch.zeros(B, Nmax, 4)
    lengths = torch.randint(Nmax // 2, Nmax, (B,))
    for b in range(B):
        n = int(lengths[b])
        pts[b, :n, 0] = torch.randint(0, 1050, (n,)).float()   # channel
        pts[b, :n, 1] = torch.randint(0, 1500, (n,)).float()   # tick
        pts[b, :n, 3] = torch.rand(n) * 2 - 1                  # log charge in [-1,1]
    return pts.to(device), lengths.to(device)


def main():
    torch.manual_seed(0)
    device = "cpu"

    enc = MambaEncoder(
        num_channels=4,
        arch="vit_small",
        masking_ratio=0.6,
        masking_type="rand",
        tokenizer_kwargs={"group_radius": 5 / 600},
        transformer_kwargs={"depth": 4, "add_pos_at_every_layer": True},  # shallow for speed
        mamba_kwargs={"d_state": 16, "d_conv": 4, "expand": 2},
    ).to(device)
    print(f"[smoke] MambaEncoder built  embed_dim={enc.embed_dim}  "
          f"blocks={len(enc.transformer.blocks)}  "
          f"params={sum(p.numel() for p in enc.parameters())/1e6:.2f}M")

    dec = TransformerDecoder(arch="vit_small", transformer_kwargs={"depth": 2}).to(device)

    model = PoLArMAE(
        encoder=enc,
        decoder=dec,
        loss_weights={"ae": 0.0, "chamfer": 1.0, "energy": 1.0},
    ).to(device)
    model.train()

    pts, lengths = make_batch(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(3):
        loss_dict = model.compute_loss(pts, lengths)
        loss = sum(loss_dict[k] * model.hparams.loss_weights.get(k, 1.0) for k in loss_dict)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        comp = {k: round(float(v), 4) for k, v in loss_dict.items()}
        print(f"[smoke] step {step}  loss={float(loss):.4f}  {comp}")

    # gradient sanity: at least one Mamba param has a non-trivial grad
    g = [p.grad.abs().sum().item() for n, p in model.named_parameters()
         if "mixer" in n and p.grad is not None]
    assert g and max(g) > 0, "no gradient flowed into the Mamba mixer!"
    print(f"[smoke] mixer grads OK  (sum|grad| max={max(g):.3e})")
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
