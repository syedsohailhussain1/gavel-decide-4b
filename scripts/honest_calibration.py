"""THE DECISIVE MEASUREMENT: our calibration on data the head never saw.

The submission claims calibration axis 91.64, the highest of any listed system
(the field sits at 75-76). That number was fitted on a set the head MEMORISED -
213 of the 231 public items were in its training pairs. We proved that today.

This computes the same axis from grouped out-of-fold predictions, so every item
is scored by a head that never saw it. It imports jevbench's own
composite_v13 so the axis is computed exactly as the benchmark computes it, not
by a reimplementation of mine.

  honest OOF axis near 91  -> the advantage is real and #1 is live
  honest OOF axis near 75  -> the 91.64 was memorisation and the advantage is gone
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, r"D:\jevbench")
from jevbench import composite_v13 as v13  # noqa: E402

TS = Path(r"D:\gavel\models\training_state")
ENTRY = Path(r"D:\gavel-jevbench-entry\results")
SEED, EPOCHS, LR, BS, WIDE, K = 20260924, 60, 1e-4, 256, 512, 5

d = torch.load(TS / "combined_hiddens.pt", map_location="cpu", weights_only=False)
H, order, tg = d["hiddens"].float(), list(d["order"]), list(d["targets"])
jb = {r["item"] for r in json.loads((TS / "combined_pairs.jsonl")
                                   .read_text(encoding="utf-8"))
      if r["tier"] != "nli"}
mask = torch.tensor([o in jb for o in order])
X = H[mask]
y = torch.tensor(tg, dtype=torch.float32)[mask]
grp = [o for o, m in zip(order, mask.tolist()) if m]
items = sorted(set(grp))
idx = defaultdict(list)
for i, g in enumerate(grp):
    idx[g].append(i)
gold_pos = {g: int(np.argmax([y[i].item() for i in idx[g]])) for g in items}
print(f"4B cached: {tuple(X.shape)}  items {len(items)}")
print(f"v13 loaded from D:/jevbench: {v13.__file__}")


class Head(nn.Module):
    def __init__(self, h, w=WIDE):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


u = sorted(set(grp))
pm = np.random.RandomState(SEED).permutation(len(u))
fo = {it: pm[j] % K for j, it in enumerate(u)}
fold = np.array([fo[i] for i in grp])
oof = torch.zeros(len(y))
for f in range(K):
    tr, va = fold != f, fold == k if False else fold == f
    torch.manual_seed(SEED + f)
    m = Head(X.shape[1]).train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    lf = nn.BCEWithLogitsLoss()
    dl = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=BS, shuffle=True)
    for _ in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            lf(m(xb), yb).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        oof[va] = m(X[va])


def ece_top_label(pairs, bins=10):
    """Same statistic make_axes.py reports: binned |confidence - accuracy| on
    the top label, weighted by bin population."""
    n = len(pairs)
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [p for p in pairs if (lo <= p[0] < hi) or (b == bins - 1 and p[0] == hi)]
        if not sel:
            continue
        conf = sum(p[0] for p in sel) / len(sel)
        acc = sum(p[1] for p in sel) / len(sel)
        tot += (len(sel) / n) * abs(conf - acc)
    return tot


def brier(pairs):
    return sum((p[0] - p[1]) ** 2 for p in pairs) / len(pairs)


def report(label, pairs, n_items):
    ece = ece_top_label(pairs)
    br = brier(pairs)
    acc = sum(p[1] for p in pairs) / len(pairs)
    mean_conf = sum(p[0] for p in pairs) / len(pairs)
    axis = v13.calibration(ece)
    print(f"\n{label}")
    print(f"  n items          {n_items}")
    print(f"  accuracy         {acc:.4f}")
    print(f"  mean confidence  {mean_conf:.4f}   (overconfidence = "
          f"{mean_conf - acc:+.4f})")
    print(f"  binned ECE       {ece:.4f}")
    print(f"  Brier            {br:.4f}")
    print(f"  v13 CALIBRATION AXIS  {axis:.2f}")
    return {"n": n_items, "accuracy": acc, "mean_confidence": mean_conf,
            "overconfidence": mean_conf - acc, "binned_ece": ece,
            "brier": br, "calibration_axis": axis}


print("\n" + "=" * 70)
print("HONEST OUT-OF-FOLD calibration - every item scored by a head that")
print("never saw it. Axis computed by jevbench's own composite_v13.")
print("=" * 70)

# raw head output, softmax at temperature 1
pairs_raw = []
for g in items:
    sc = oof[idx[g]].tolist()
    z = np.array(sc)
    z = z - z.max()
    e = np.exp(z)
    p = e / e.sum()
    pick = int(np.argmax(sc))
    pairs_raw.append((float(p.max()), 1.0 if pick == gold_pos[g] else 0.0))
res_raw = report("A  OOF, softmax at T=1 (no temperature fitted at all)",
                 pairs_raw, len(items))

# temperature fitted ON THE OOF PREDICTIONS - this is the honest analogue of
# what the shipped head does, because the shipped T was fitted in-sample
best = None
for T in np.arange(0.05, 6.0, 0.05):
    pr = []
    for g in items:
        z = np.array(oof[idx[g]].tolist()) / T
        z = z - z.max()
        e = np.exp(z)
        p = e / e.sum()
        pick = int(np.argmax(z))
        pr.append((float(p.max()), 1.0 if pick == gold_pos[g] else 0.0))
    e_ = ece_top_label(pr)
    if best is None or e_ < best[0]:
        best = (e_, float(T), pr)
res_T = report(f"B  OOF, temperature refit on the OOF set (T={best[1]:.2f})",
               best[2], len(items))

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
shipped = json.loads((ENTRY / "bf16_recal_axes.json").read_text(encoding="utf-8"))
print(f"  shipped submission, IN-SAMPLE on a memorised fit set")
print(f"    hard binned ECE       {shipped['hard_binned_ece']:.4f}")
print(f"    calibration axis      {shipped['calibration_axis']:.2f}")
print(f"")
print(f"  HONEST, out-of-fold")
print(f"    binned ECE            {res_T['binned_ece']:.4f}")
print(f"    calibration axis      {res_T['calibration_axis']:.2f}")
print(f"")
print(f"  the field (#1 and #2)  75.0 and 76.3")
gap = shipped["calibration_axis"] - res_T["calibration_axis"]
print(f"")
print(f"  the claim is inflated by {gap:.1f} axis points.")

if res_T["calibration_axis"] >= 80:
    print("\n  VERDICT: the advantage substantially survives out-of-fold.")
elif res_T["calibration_axis"] >= 76:
    print("\n  VERDICT: the advantage mostly evaporates - we land in the field.")
else:
    print("\n  VERDICT: the 91.64 was memorisation. The calibration advantage is")
    print("  GONE, and with it the main reason we could have ranked #1.")

(ENTRY / "honest_calibration.json").write_text(json.dumps({
    "question": "what is our calibration on data the head never saw?",
    "axis_computed_by": v13.__file__,
    "shipped_in_sample": {"binned_ece": shipped["hard_binned_ece"],
                          "calibration_axis": shipped["calibration_axis"],
                          "note": "fitted on a set containing 213 of the 231 "
                                  "scored items"},
    "honest_oof_raw": res_raw,
    "honest_oof_temperature_refit": res_T,
    "temperature_refit": best[1],
    "field_reference": {"decider_4b_v2": 75.0, "jev_1_13_0": 76.3},
    "inflation": gap,
}, indent=2), encoding="utf-8")
print(f"\nwrote {ENTRY / 'honest_calibration.json'}")

