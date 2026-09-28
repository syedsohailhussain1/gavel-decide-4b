"""A defensible generalisation estimate, which the submission currently lacks.

The 3/18 figure is dominated by an uncovered question type, and 1/6 on the
remaining unseen items has n=6. Neither can answer "is this good".

This does a proper grouped 5-fold out-of-fold evaluation on the 213 JevBench
items the head trained on:
  - rows are per-option binary (target 1 = this option is the answer)
  - folds are grouped BY ITEM, so both options of an item stay together and no
    twin leaks across the boundary
  - the item-level prediction is the argmax over that item's held-out options
  - the head architecture and the production recipe are unchanged

The result estimates accuracy on unseen items OF THE COVERED TYPES. It says
nothing about the uncovered types, which is a separate and worse problem.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TS = Path(r"D:\gavel\models\training_state")
SEED, EPOCHS, LR, BS, WIDE, K = 20260924, 60, 1e-4, 256, 512, 5


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


d = torch.load(TS / "combined_hiddens.pt", map_location="cpu", weights_only=False)
H, order, targets = d["hiddens"].float(), list(d["order"]), list(d["targets"])
print(f"cached rows {tuple(H.shape)}  items {len(set(order))}")

# keep only JevBench-derived rows
pairs = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
jb_texts = {p["text"] for p in pairs if p["tier"] != "nli"}
# rows carry item ids in `order`; identify jevbench items by intersection
jb_items = {p["item"] for p in pairs if p["tier"] != "nli"}

mask = torch.tensor([o in jb_items for o in order])
X, y, grp = H[mask], torch.tensor(targets, dtype=torch.float32)[mask], \
    [o for o, m in zip(order, mask.tolist()) if m]
print(f"JevBench rows {tuple(X.shape)}  items {len(set(grp))}")
assert len(X) == len(y) == len(grp)

items = sorted(set(grp))
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(items))
fold_of = {items[i]: perm[j] % K for j, i in enumerate(range(len(items)))}
fold = np.array([fold_of[g] for g in grp])


def train(Xtr, ytr, seed):
    torch.manual_seed(seed)
    m = Head(Xtr.shape[1], WIDE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True,
                    generator=g)
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in dl:
            opt.zero_grad()
            lossf(m(xb), yb).backward()
            opt.step()
    return m.eval()


t0 = time.time()
oof = torch.zeros(len(y))
for k in range(K):
    tr, va = fold != k, fold == k
    m = train(X[tr], y[tr], SEED + k)
    with torch.no_grad():
        oof[va] = m(X[va])
    print(f"  fold {k}: train_rows={int(tr.sum()):4d} va_rows={int(va.sum()):4d} "
          f"va_items={len({g for g, f in zip(grp, fold) if f == k}):3d}")

# item-level accuracy: argmax over each held-out item's options
by = defaultdict(list)
for i, g in enumerate(grp):
    by[g].append(i)
n_ok = n_tot = 0
per_type = defaultdict(lambda: [0, 0])
for g, idx in by.items():
    scores = oof[idx]
    pick = idx[int(torch.argmax(scores))]
    ok = bool(y[pick] > 0.5)
    n_ok += ok
    n_tot += 1
    t = g.split("-")[1] if len(g.split("-")) > 2 else "?"
    per_type[t][1] += 1
    per_type[t][0] += ok

acc = n_ok / n_tot
print(f"\n{'=' * 58}")
print(f"GROUPED 5-FOLD OOF, item level, {n_tot} items")
print(f"accuracy = {n_ok}/{n_tot} = {acc:.4f}")
print(f"{'=' * 58}")
print("\nby question type (all covered - these are the folds' own items):")
for t in sorted(per_type):
    a, b = per_type[t]
    print(f"  {t:16} {a:3}/{b:3} = {a/b:.4f}")

chance = sum(1.0 / len(by[g]) for g in by) / len(by)
print(f"\nmean chance rate on these items: {chance:.4f}")
print(f"OOF accuracy above chance? {acc > chance}")
print(f"\n({time.time()-t0:.0f}s)")

out = Path(r"D:\gavel-jevbench-entry\results\grouped_oof_estimate.json")
out.write_text(json.dumps({
    "protocol": "grouped 5-fold OOF, folds keyed by item, per-option binary "
                "head, item prediction = argmax over held-out options",
    "n_items": n_tot, "n_correct": n_ok, "accuracy": acc,
    "mean_chance": chance,
    "by_type": {t: {"correct": a, "total": b, "acc": a / b}
                for t, (a, b) in sorted(per_type.items())},
    "recipe": {"epochs": EPOCHS, "lr": LR, "batch": BS, "width": WIDE,
               "seed": SEED, "folds": K},
    "covers": "accuracy on unseen items OF THE COVERED TYPES only; the "
              "ordinal type has zero training coverage and is excluded",
    "runtime_s": round(time.time() - t0, 1),
}, indent=2), encoding="utf-8")
print(f"wrote {out}")
