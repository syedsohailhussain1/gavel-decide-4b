"""Does the 4B head improve with more training items? Measured on real data.

The leader went 64.6 -> #1 on the same 4B by going from some N to 1.47M
examples. But their head is trained differently from ours. Ours is a 1.44M
probe on a FROZEN 4B trunk, so it may already be saturated at the data we have.

If this curve is flat, a million generated examples buys nothing and the pod
spend is wasted. If it is climbing, the plan is justified.

Only the 689 real JevBench pairs are used. Item-level accuracy, evaluated on
items never in the training set, by holding out a fixed 20% of ITEMS and
growing the training portion.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
SEED, EPOCHS, LR, BS, WIDE = 20260924, 60, 1e-4, 256, 512

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
print(f"4B cached rows {tuple(X.shape)}  items {len(items)}  "
      f"positives {int(y.sum())}")
HID = X.shape[1]


class Head(nn.Module):
    def __init__(self, h, w=WIDE):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_eval(tr_items, va_items, seed=SEED, epochs=EPOCHS):
    tr_rows = [i for it in tr_items for i in idx[it]]
    va_rows = [i for it in va_items for i in idx[it]]
    Xtr, ytr = X[tr_rows], y[tr_rows]
    Xva = X[va_rows]
    torch.manual_seed(seed)
    m = Head(HID).train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    lf = nn.BCEWithLogitsLoss()
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            lf(m(xb), yb).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        sc = m(Xva)
    by = defaultdict(list)
    for r, i in zip(va_rows, range(len(va_rows))):
        by[idx_of[i]].append(i)
    ok = 0
    for it, ii in by.items():
        pick = ii[int(torch.argmax(sc[ii]))]
        ok += y[va_rows[pick]] > 0.5
    return ok / len(va_items), len(tr_items), len(tr_rows)


# map local row index -> item
idx_of = {}
for it, ii in idx.items():
    for i in ii:
        idx_of[i] = it

# 5 repeated 20% item holdouts, averaged, so the number is not one lucky split
SPLITS = 5
curves = {}
for s in range(SPLITS):
    perm = np.random.RandomState(SEED + s).permutation(len(items))
    hold = {items[perm[j]] for j in range(int(0.2 * len(items)))}
    va = [k for k in items if k in hold]
    pool = [k for k in items if k not in hold]
    for n in (20, 50, 100, 170):
        tr = pool[:n]
        a, ni, nr = train_eval(tr, va, seed=SEED + s)
        curves.setdefault(n, []).append((a, nr))

print(f"\n{'='*66}")
print("4B head: item accuracy vs training-set size")
print(f"5 x 20% item holdout, averaged. {len(items)} items available.")
print(f"{'='*66}")
print(f"{'train items':>12} {'train rows':>11} {'held-out acc':>13} {'std':>7}")
slope = None
prev = None
out = []
for n in sorted(curves):
    accs = [c[0] for c in curves[n]]
    rows_ = curves[n][0][1]
    a = float(np.mean(accs))
    sd = float(np.std(accs))
    out.append({"n_train_items": n, "n_train_rows": rows_, "mean": a,
                "std": sd})
    print(f"{n:>12} {rows_:>11} {a:>13.4f} {sd:>7.4f}")
    if prev is not None and n == 50:
        slope = a - prev[1]
    prev = (n, a)

first = out[0]["mean"]
last = out[-1]["mean"]
print(f"\n  from {out[0]['n_train_items']} -> {out[-1]['n_train_items']} "
      f"training items: {first:.4f} -> {last:.4f}  "
      f"({last-first:+.4f})")
gain = (last - first) / (out[-1]["n_train_items"] / out[0]["n_train_items"])
print(f"  accuracy per 8.5x data: {gain:+.4f}  "
      f"-> extrapolating to 1.47M is {'NOT' if gain < 0.005 else ''} "
      f"promising")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "data_scaling_real.json").write_text(json.dumps({
    "question": "does the 4B head improve with more REAL data?",
    "protocol": f"{SPLITS} x 20% item holdout, identical head and recipe, "
                f"training set grown over the remaining items",
    "n_items_available": len(items),
    "curve": out,
    "first_to_last_delta": last - first,
    "verdict": "scaling data is worth pod spend" if last - first > 0.01
    else "curve is nearly flat - more data buys little, pod spend not "
         "justified on this evidence",
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'data_scaling_real.json'}")
