"""Local CPU smoke for larmamba2: tiler round-trip + tiny forward/backward.

    LARMAMBA_CPU_SCAN=seq python larmamba2/tests/smoke_local.py
"""
import torch

from larmamba2.ssl_module import Larmamba2MAE
from larmamba2.tiler import tile_events, untile


def rand_event(n, gen):
    """n unique (ch, tick) coords on a 960x1178 grid + logq in (-0.9, 1)."""
    flat = torch.randperm(960 * 1178, generator=gen)[:n]
    ch, tk = flat // 1178, flat % 1178
    q = torch.rand(n, generator=gen) * 1.9 - 0.9
    p = torch.zeros(n, 4)
    p[:, 0], p[:, 1], p[:, 3] = ch.float(), tk.float(), q
    return p


def test_tiler_roundtrip():
    gen = torch.Generator().manual_seed(0)
    evs = [rand_event(n, gen) for n in (700, 300, 1200)]
    lengths = torch.tensor([len(e) for e in evs])
    points = torch.nn.utils.rnn.pad_sequence(evs, batch_first=True)

    patches, coords, mask, vox_tile = tile_events(points, lengths, tile_size=5, t_max=-1)
    rec = untile(patches, coords, mask, tile_size=5)
    for b, ev in enumerate(evs):
        want = {(int(c[0]), int(c[1])): float(q) for c, q in zip(ev[:, :2], ev[:, 3])}
        got = {(int(c[0]), int(c[1])): float(q) for c, q in zip(*rec[b])}
        assert want.keys() == got.keys(), f"event {b}: coord sets differ"
        assert all(abs(want[k] - got[k]) < 1e-5 for k in want), f"event {b}: charges differ"
        assert (vox_tile[b, :len(ev)] >= 0).all()
    print("tiler round-trip OK")

    # cap path: t_max smaller than tile count -> exactly t_max tiles kept
    patches2, _, mask2, vt2 = tile_events(points, lengths, tile_size=5, t_max=50)
    assert mask2.sum(1).max() <= 50
    assert (vt2[0, :700] >= 0).sum() < 700   # some voxels dropped with tiles
    print("tiler cap OK")


def test_forward_backward():
    gen = torch.Generator().manual_seed(1)
    evs = [rand_event(n, gen) for n in (500, 900)]
    lengths = torch.tensor([len(e) for e in evs])
    points = torch.nn.utils.rnn.pad_sequence(evs, batch_first=True)

    model = Larmamba2MAE(dim=96, depth=2, decoder_depth=1, decoder_heads=4, t_max=64)
    loss = model._step({"points": points, "lengths": lengths}, "train")
    loss.backward()
    assert torch.isfinite(loss), "loss not finite"
    n_grad = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    n_par = sum(1 for _ in model.parameters())
    print(f"forward/backward OK: loss={loss.item():.4f}, {n_grad}/{n_par} params got grads")

    tokens, tmask, _ = model.forward_features(points, lengths, t_max=-1)
    assert tokens.shape[-1] == 96 and tmask.any()
    print(f"forward_features OK: tokens {tuple(tokens.shape)}")


if __name__ == "__main__":
    test_tiler_roundtrip()
    test_forward_backward()
    print("ALL SMOKE TESTS PASSED")
