"""Phase-0 data study for larmamba2 (plan §6): tile-count / occupancy / charge
statistics on a sample of the 1M SSL production, with the APA2D filter applied
(W view, charge > 1, min 256 voxels, cap 8000).

Pure h5py+numpy — runs on a login node, no GPU.

    python larmamba2/tests/data_study.py --n_files 20 --out /tmp/stats.json
"""
import argparse
import glob
import json

import h5py
import numpy as np

VIEW_W = (1600, 2650)   # channel range before rebase (matches APA2D)


def event_iter(files, rng):
    for fp in files:
        try:
            f = h5py.File(fp, "r")
        except OSError:
            continue
        for g in f.keys():
            grp = f[g]
            fr = grp.get("frame_rebinned_reco")
            if fr is None or "coords" not in fr:
                continue
            co = fr["coords"][()]
            fe = fr["features"][()]
            m = (co[:, 0] >= VIEW_W[0]) & (co[:, 0] < VIEW_W[1])
            co, fe = co[m], fe[m]
            co = co.copy(); co[:, 0] -= VIEW_W[0]
            thr = fe > 1.0
            co, fe = co[thr], fe[thr]
            if len(co) < 256:
                continue
            if len(co) > 8000:
                sel = rng.choice(len(co), 8000, replace=False)
                co, fe = co[sel], fe[sel]
            yield co, fe
        f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob",
                    default="/gpfs01/lbne/users/fm/cffm-data/prod-jay-1M-2026-02-27/13825/1/001/out_*/*pixeldata-anode0.h5")
    ap.add_argument("--n_files", type=int, default=20)
    ap.add_argument("--tile_sizes", default="3,5,7")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    files = sorted(glob.glob(args.data_glob))
    print(f"{len(files)} files match; sampling {args.n_files}")
    files = list(rng.choice(files, min(args.n_files, len(files)), replace=False))
    sizes = [int(s) for s in args.tile_sizes.split(",")]

    per_size = {s: {"tiles": [], "occ": []} for s in sizes}
    nvox_all, ch_max, tk_max = [], 0, 0
    charges = []
    n_ev = 0
    for co, fe in event_iter(files, rng):
        n_ev += 1
        nvox_all.append(len(co))
        ch_max = max(ch_max, int(co[:, 0].max()))
        tk_max = max(tk_max, int(co[:, 1].max()))
        if n_ev <= 200:
            charges.append(fe)
        for s in sizes:
            key = (co[:, 0] // s).astype(np.int64) * 100000 + co[:, 1] // s
            u, cnt = np.unique(key, return_counts=True)
            per_size[s]["tiles"].append(len(u))
            per_size[s]["occ"].append(float(cnt.mean()))

    nvox_all = np.array(nvox_all)
    print(f"\nevents={n_ev}  ch_max={ch_max}  tick_max={tk_max}")
    print(f"voxels/event: med={np.median(nvox_all):.0f} mean={nvox_all.mean():.0f} "
          f"p95={np.percentile(nvox_all,95):.0f} max={nvox_all.max()}")
    ch = np.concatenate(charges)
    print(f"charge (ADC, first 200 ev): med={np.median(ch):.1f} p99={np.percentile(ch,99):.0f} max={ch.max():.0f}")

    res = {"n_events": n_ev, "ch_max": ch_max, "tick_max": tk_max}
    for s in sizes:
        t = np.array(per_size[s]["tiles"]); o = np.array(per_size[s]["occ"])
        stats = dict(med=float(np.median(t)), mean=float(t.mean()),
                     p90=float(np.percentile(t, 90)), p95=float(np.percentile(t, 95)),
                     p99=float(np.percentile(t, 99)), max=int(t.max()),
                     occ_med=float(np.median(o)), occ_frac=float(np.median(o)) / (s * s))
        res[f"tile{s}"] = stats
        print(f"tile {s}x{s}: med={stats['med']:.0f} mean={stats['mean']:.0f} "
              f"p90={stats['p90']:.0f} p95={stats['p95']:.0f} p99={stats['p99']:.0f} "
              f"max={stats['max']}  occ={stats['occ_med']:.1f}/{s*s} ({100*stats['occ_frac']:.0f}%)")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
