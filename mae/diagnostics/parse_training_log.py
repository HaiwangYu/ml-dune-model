"""
Parse a train_mae.py stdout (e.g. condor `<id>.out`) into a structured JSON
summary of per-SSL-epoch SSL/SFT metrics and confusion matrices.

Output JSON shape:

{
  "class_names": ["track", "shower", "other"],
  "ssl_epochs": [
    {
      "epoch": 1,
      "ssl_train_l1": 3.1176,
      "ssl_val_l1":   0.7966,
      "sft_subepochs": [
        {"sft_epoch":1, "sft_ce":0.4223, "sft_acc":0.437,
                        "ref_ce":0.4479, "ref_acc":0.442},
        ...
      ],
      "sft_aggregate": {
        "sft_ce": 0.3635, "sft_acc": 0.503,
        "ref_ce": 0.4163, "ref_acc": 0.482,
        "confusion_sft": [[...], [...], [...]],
        "confusion_ref": [[...], [...], [...]],
        "per_class_sft": [{"name":"track","efficiency":0.6279,"purity":0.5426}, ...],
        "per_class_ref": [...]
      }
    },
    ...
  ]
}

Usage:
    python -m mae.diagnostics.parse_training_log <run_dir>/300.0.out
    python -m mae.diagnostics.parse_training_log <out> --output=<run_dir>/debug/sft_history.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


SSL_TRAIN_RE = re.compile(r"^\s*SSL train epoch\s+(\d+) done\s+\|\s+mean L1=([0-9.eE+-]+)")
SSL_VAL_RE   = re.compile(r"^\s*SSL val\s+epoch\s+(\d+) done\s+\|\s+mean L1=([0-9.eE+-]+)")
SFT_SUB_RE   = re.compile(
    r"^\s*SFT epoch\s+(\d+)/(\d+)\s+\|\s+SSL-feat:\s+CE=([0-9.eE+-]+)\s+acc=([0-9.]+)%"
    r"\s+\|\s+raw-charge:\s+CE=([0-9.eE+-]+)\s+acc=([0-9.]+)%"
)
EPOCH_BANNER_RE = re.compile(
    r"^\s*Epoch\s+(\d+)\s+\|\s+SSL train L1=([0-9.eE+-]+)\s+val L1=([0-9.eE+-]+)"
)
AGG_SFT_RE = re.compile(r"^\s*SSL features\s+:\s+CE=([0-9.eE+-]+)\s+acc=([0-9.]+)%")
AGG_REF_RE = re.compile(r"^\s*Raw charge ref:\s+CE=([0-9.eE+-]+)\s+acc=([0-9.]+)%")
PER_CLASS_RE = re.compile(
    r"^\s*([\w±μπγ±+\-]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
)


def _parse(text: str):
    lines = text.splitlines()
    n = len(lines)
    i = 0
    epochs = []
    class_names = None

    def _next_blank(j):
        while j < n and lines[j].strip():
            j += 1
        return j

    while i < n:
        line = lines[i]

        # ── Per-SSL-epoch SSL train / val lines  (appear at end of epoch) ──
        m = SSL_TRAIN_RE.match(line)
        if m:
            ep = int(m.group(1)); v = float(m.group(2))
            ent = _find_or_create(epochs, ep)
            ent["ssl_train_l1"] = v
            i += 1
            continue

        m = SSL_VAL_RE.match(line)
        if m:
            ep = int(m.group(1)); v = float(m.group(2))
            ent = _find_or_create(epochs, ep)
            ent["ssl_val_l1"] = v
            i += 1
            continue

        # ── Per-SFT-subepoch metrics ──
        m = SFT_SUB_RE.match(line)
        if m:
            sub_idx  = int(m.group(1))
            sft_ce   = float(m.group(3));  sft_acc = float(m.group(4)) / 100.0
            ref_ce   = float(m.group(5));  ref_acc = float(m.group(6)) / 100.0
            # We don't know the SSL epoch yet from this line alone; attach to
            # the most recently-opened entry (set by SSL train/val above).
            if epochs:
                epochs[-1].setdefault("sft_subepochs", []).append({
                    "sft_epoch": sub_idx,
                    "sft_ce": sft_ce, "sft_acc": sft_acc,
                    "ref_ce": ref_ce, "ref_acc": ref_acc,
                })
            i += 1
            continue

        # ── Epoch banner: "Epoch  N  |  SSL train L1=... val L1=..." ──
        m = EPOCH_BANNER_RE.match(line)
        if m:
            ep = int(m.group(1))
            ent = _find_or_create(epochs, ep)
            ent["ssl_train_l1"] = float(m.group(2))
            ent["ssl_val_l1"]   = float(m.group(3))
            # Read aggregate block following this line
            agg = {}
            i += 1
            while i < n:
                ln = lines[i]
                am = AGG_SFT_RE.match(ln)
                if am:
                    agg["sft_ce"]  = float(am.group(1))
                    agg["sft_acc"] = float(am.group(2)) / 100.0
                am2 = AGG_REF_RE.match(ln)
                if am2:
                    agg["ref_ce"]  = float(am2.group(1))
                    agg["ref_acc"] = float(am2.group(2)) / 100.0

                if "[SSL features]" in ln:
                    cm, per_cls, names, i = _read_block(lines, i + 1)
                    agg["confusion_sft"]  = cm
                    agg["per_class_sft"]  = per_cls
                    class_names = class_names or names
                    continue
                if "[Raw charge reference]" in ln:
                    cm, per_cls, names, i = _read_block(lines, i + 1)
                    agg["confusion_ref"]  = cm
                    agg["per_class_ref"]  = per_cls
                    class_names = class_names or names
                    continue
                if ln.startswith("=" * 5):  # banner separator → end of aggregate block
                    break
                i += 1
            ent["sft_aggregate"] = agg
            continue

        i += 1

    return {
        "class_names": class_names or [],
        "ssl_epochs":  epochs,
    }


def _find_or_create(epochs, epoch_num):
    for e in epochs:
        if e["epoch"] == epoch_num:
            return e
    e = {"epoch": epoch_num, "sft_subepochs": []}
    epochs.append(e)
    return e


def _read_block(lines, i):
    """Read a confusion-matrix block + per-class metrics block.

    Expects (header text below has been advanced past already):
      Confusion matrix  (rows = true, cols = predicted)
                   <name1>  <name2>  <name3>
        <name1>     n11      n12      n13
        ...
      Per-class metrics:
           class    efficiency   purity
        <name>      e            p
        ...
    Returns (confusion 2D list, per_class list of dicts, class_names list, new index).
    """
    n = len(lines)
    class_names = []
    confusion   = []
    per_class   = []

    # Skip until "Confusion matrix" header (already after [SSL features] marker)
    while i < n and "Confusion matrix" not in lines[i]:
        i += 1
    i += 1  # advance past the header

    # Column-names row (the line that has only class names, no leading row label)
    while i < n and not lines[i].strip():
        i += 1
    if i < n:
        class_names = lines[i].split()
        i += 1

    # Confusion rows: "  <name>   n n n"
    while i < n:
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        if "Per-class" in ln or ln.startswith("="):
            break
        parts = ln.split()
        # row label is first token(s), then integers
        # find where the integer-only tail starts
        for split_at in range(len(parts) - 1, 0, -1):
            try:
                ints = [int(p) for p in parts[split_at:]]
                if len(ints) == len(class_names):
                    confusion.append(ints)
                    break
            except ValueError:
                continue
        i += 1

    # Skip to per-class metrics body (rows after "class efficiency purity" header)
    while i < n and "Per-class" not in lines[i]:
        if lines[i].startswith("=") or "Confusion matrix" in lines[i]:
            return confusion, per_class, class_names, i
        i += 1
    # advance past header line
    i += 1
    # skip the column-header row ("class  efficiency  purity")
    while i < n and not lines[i].strip():
        i += 1
    if i < n and "class" in lines[i]:
        i += 1

    while i < n:
        ln = lines[i].strip()
        if not ln or ln.startswith("=") or "[" in ln or "Confusion matrix" in ln:
            break
        parts = ln.split()
        if len(parts) >= 3:
            try:
                eff = float(parts[-2]); pur = float(parts[-1])
                name = " ".join(parts[:-2])
                per_class.append({"name": name, "efficiency": eff, "purity": pur})
            except ValueError:
                pass
        i += 1

    return confusion, per_class, class_names, i


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("log", help="Path to train_mae stdout log (e.g. condor <id>.out)")
    p.add_argument("--output", default="",
                   help="Output JSON path; default = <log_dir>/sft_history.json")
    args = p.parse_args()

    log_path = Path(args.log).resolve()
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = log_path.parent / "sft_history.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text()
    data = _parse(text)
    out_path.write_text(json.dumps(data, indent=2))

    n_ep = len(data["ssl_epochs"])
    print(f"Parsed {n_ep} SSL epoch(s) from {log_path.name}")
    print(f"Class names: {data['class_names']}")
    for e in data["ssl_epochs"]:
        sub = e.get("sft_subepochs", [])
        agg = e.get("sft_aggregate", {})
        print(
            f"  epoch {e['epoch']}: "
            f"ssl_train={e.get('ssl_train_l1', 'NA')}  "
            f"ssl_val={e.get('ssl_val_l1', 'NA')}  "
            f"sft_subepochs={len(sub)}  "
            f"agg_sft_acc={agg.get('sft_acc', 'NA')}  "
            f"agg_ref_acc={agg.get('ref_acc', 'NA')}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
