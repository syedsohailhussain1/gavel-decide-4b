"""The data curve: does more generated supervision actually help?

Trains the head at 1k / 10k / 100k generated items with identical
architecture, recipe and grouped 5-fold protocol, and reports item-level
accuracy at each point. This is the experiment that decides whether scaling to
1.47M examples - the leader's volume - is worth pursuing.

Folds are grouped by item, so both options of an item stay on the same side.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED, EPOCHS, LR, BS, WIDE, K = 20260924, 60, 1e-4, 256, 512, 5

ap = argparse.ArgumentParser()
ap.add_argument("--hiddens", default=r"D:\gavel\models\training_state\gen_hiddens.pt")
ap.add_argument("--sizes", default="1000,10000,100000")
ap.add_argument("--out", default=r"D:\gavel-jevbench-entry\results\data_curve.json")
ap.add_argument("--save-head", default="")
A = ap.parse_args()

d = torch.load(A.hiddens, map_location="cpu", weights_only=False)
X = d["hiddens"].float()
y = torch.tensor(d["targets"], dtype=torch.float32)
items = list(d["order"])
tiers = list(d["tiers"])
H = X.shape[1]
print(f"cached: {tuple(X.shape)}  items {len(set(items)):,}  "
      f"positives {int(y.sum()):,}  hidden {H}")
print(f"trunk: {d.get('trunk')}  use_cache={d.get('use_cache')}")

by_item = defaultdict(list)
for i, it in enumerate(items):
    by_item[it].append(i)
uniq = sorted(by_item)
print(f"distinct items {len(uniq):,}\n")

tier_of = {it: t for it, t in zip(items, tiers)}


def folds_for(sel_items, seed=SEED, k=K):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(sel_items))
    fo = {item: perm[j] % k for j, item in enumerate(sel_items)}
    return fo


class Head(nn.Module):
    """Same architecture as the shipped head, sized to this trunk."""

    def __init__(self, h, w=WIDE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(Xtr, ytr, epochs=EPOCHS, seed=SEED):
    torch.manual_seed(seed)
    head = Head(H)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    lf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True,
                    generator=g)
    for _ in range(epochs):
        head.train()
        for xb, yb in dl:
            opt.zero_grad()
            lf(head(xb), yb).backward()
            opt.step()
    return head.eval()


def run(n_items, epochs=EPOCHS):
    # take whole items, balanced across families by round-robin
    per = defaultdict(list)
    for it in uniq:
        per[tier_of[it]].append(it)
    chosen, fams = [], sorted(per)
    i = 0
    while len(chosen) < n_items:
        f = fams[i % len(fams)]
        if per[f]:
            chosen.append(per[f].pop())
        i += 1
    sel = sorted(chosen[:n_items])
    sel_set = set(sel)
    idx = [j for j, it in enumerate(items) if it in sel_set]
    Xs = X[idx]
    ys = y[idx]
    its = [items[j] for j in idx]

    fo = folds_for(sel)
    fold = np.array([fo[it] for it in its])
    oof = torch.zeros(len(ys))
    for k in range(K):
        tr, va = fold != k, fold == k
        h = train(Xs[tr], ys[tr], epochs, SEED + k)
        with torch.no_grad():
            oof[va] = h(Xs[va])

    g = defaultdict(list)
    for j, it in enumerate(its):
        g[it].append(j)
    ok = tot = 0
    pf = defaultdict(lambda: [0, 0])
    for it, jj in g.items():
        good = bool(ys[jj[int(torch.argmax(oof[jj]))]] > 0.5)
        ok += good
        tot += 1
        t = tier_of[it]
        pf[t][1] += 1
        pf[t][0] += good
    chance = sum(1.0 / len(jj) for jj in g.values()) / len(g)
    return ok / tot, chance, tot, len(idx), {t: a / b for t, (a, b) in pf.items()}, (X[idx], y[idx], its)


print(f"{'items':>8} {'rows':>9} {'ep':>4} {'acc':>8} {'chance':>8} {'lift':>7}")
print("-" * 50)
curve = []
# Epochs scale inversely with size so each point is a sensible compute budget
# instead of 60 epochs over 400k rows. Larger points therefore get MORE total
# row-passes, which if anything flatters them - the result is conservative.
EP_FOR = {1000: 60, 10000: 20, 100000: 5}
for n in [int(s) for s in A.sizes.split(",")]:
    ep = EP_FOR.get(n, max(3, min(60, int(2_400_000 / max(n, 1)) * 10)))
    acc, chance, tot, rows_n, pf, _ = run(n, epochs=ep)
    curve.append({"n_items": n, "n_rows": rows_n, "n_eval_items": tot,
                  "epochs": ep, "accuracy": acc, "chance": chance,
                  "lift_over_chance": acc / chance if chance else None,
                  "by_family": {t: round(a, 4) for t, a in sorted(pf.items())}})
    print(f"{n:>8,} {rows_n:>9,} {ep:>4} {acc:>8.4f} {chance:>8.4f} "
          f"{acc/chance:>6.2f}x", flush=True)

print("\nby family at the largest size:")
last = curve[-1]
for t, a in last["by_family"].items():
    print(f"  {t:16} {a:.4f}")

Path(A.out).write_text(json.dumps({
    "protocol": "grouped 5-fold OOF, folds keyed by item; identical "
                "architecture, recipe and seed at every point",
    "trunk": d.get("trunk"),
    "cached_with_prefix_cache": bool(d.get("use_cache")),
    "recipe": {"epochs": EPOCHS, "lr": LR, "batch": BS, "width": WIDE,
               "seed": SEED, "folds": K},
    "curve": curve,
    "note": "measured on self-generated data. It answers 'does more "
            "generated supervision help THIS head', not 'will it transfer to "
            "JevBench'. Transfer needs a held-out JevBench check.",
}, indent=2), encoding="utf-8")
print(f"\nwrote {A.out}")

if A.save_head:
    Xf, yf, itf = run(int(A.sizes.split(",")[-1]))[5]
    h = train(Xf, yf, EPOCHS, SEED)
    torch.save({"head": h.state_dict(), "hidden": H, "wide": WIDE,
                "temperature": 1.0, "meta_cal": None,
                "trained_on": "programmatically generated pairs only; zero "
                              "JevBench items",
                "n_rows": int(len(yf))}, A.save_head)
    print(f"wrote {A.save_head}")

