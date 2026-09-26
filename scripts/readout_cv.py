"""Rigorously confirm the read-out finding: 5-fold CV over items + temperature.

depth_ctx_sweep used a single 25% holdout (57 items) for speed. Before acting
on a 2x claim, re-test the surviving candidates with 5-fold cross-validation
over items, refitting the temperature inside each fold, and report
out-of-fold item accuracy with a standard error.

Also reports the paired per-item win/loss against the shipped read-out, which
is a much more sensitive test than comparing two marginal accuracies.
"""
import json
import math
import random
import sys

import torch
import torch.nn as nn

RES = r"D:\gavel-jevbench-entry\results"
CTX = 512
CANDS = [19, 20, 21, 22, 23, 24, 25, 36]
FOLDS, SEEDS, EPOCHS, LR = 5, 2, 80, 1e-3


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(
            torch.nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
            torch.nn.Linear(w, w // 2), nn.GELU(), torch.Dropout(0.1),
            torch.nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Head2(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def tier(t):
    x = t.split("-", 1)[0]
    return {"original": "standard"}.get(x, x)


truth, label_order = {}, {}
for f in ("easy", "original", "hard"):
    for line in open(f"D:\\jevbench\\datasets\\public\\{f}.jsonl",
                     encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            e = r["expected"]
            truth[r["id"]] = str(e) if isinstance(e, (int, float)) else e
            label_order[r["id"]] = r.get("labels") or []

D = torch.load(f"{RES}\\ro_{CTX}.pt", map_location="cpu", weights_only=False)
seqs, order, nlayer = D["seqs"], D["order"], D["nlayer"]
pos = [0] * len(seqs)
for row, s in enumerate(order):
    pos[s] = row
by_item = {}
for i, s in enumerate(seqs):
    by_item.setdefault(s["id"], []).append(pos[i])
gold = {}
for iid, idxs in by_item.items():
    labs = label_order[iid]
    g = truth.get(iid)
    gold[iid] = labs.index(g) if g in labs else None
ids = sorted(i for i in by_item if gold[i] is not None)
print(f"items {len(ids)}  sequences {len(seqs)}  ctx {CTX}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
F = {}
for L in CANDS:
    F[L] = torch.cat(list(D[f"last_L{L}"]), 0).float().to(dev)


def fit_T(logits, y_idx, idxs):
    """Fit temperature on the gold option only; non-gold rows carry no target."""
    use = [i for i in idxs if y_idx[i] is not None]
    if not use:
        return 1.0
    grid = [0.05 + 0.01 * i for i in range(0, 400)]
    best, bT = 1e9, 1.0
    for T in grid:
        n = 0.0
        for i in use:
            z = [v / T for v in logits[i]]
            m = max(z)
            e = [math.exp(v - m) for v in z]
            s = sum(e)
            n -= math.log(max(e[y_idx[i]] / s, 1e-15))
        if n < best:
            best, bT = n, T
    return bT


results = {}
per_item = {}
for L in CANDS:
    X = F[L]
    fold_of = {}
    for k, iid in enumerate(sorted(ids)):
        fold_of[iid] = k % FOLDS
    accs, tiers = [], {t: [0, 0] for t in ("easy", "standard", "hard")}
    for seed in range(SEEDS):
        correct = {}
        for fold in range(FOLDS):
            tr_ids = [i for i in ids if fold_of[i] != fold]
            te_ids = [i for i in ids if fold_of[i] == fold]
            tr_idx, tr_y = [], []
            for iid in tr_ids:
                for j, i in enumerate(by_item[iid]):
                    tr_idx.append(i)
                    tr_y.append(1.0 if j == gold[iid] else 0.0)
            torch.manual_seed(seed * 100 + fold)
            h = Head2(X.shape[1]).to(dev)
            o = torch.optim.AdamW(h.parameters(), lr=LR, weight_decay=1e-4)
            ti = torch.tensor(tr_idx).to(dev)
            ty = torch.tensor(tr_y, dtype=torch.float32, device=dev)
            h.train()
            for _ in range(EPOCHS):
                o.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(h(X[ti]), ty)
                loss.backward()
                o.step()
            h.eval()
            with torch.no_grad():
                tr_logits, tr_ypos = [], []
                for iid in tr_ids:
                    idxs = by_item[iid]
                    lg = h(X[torch.tensor(idxs).to(dev)]).tolist()
                    for j in range(len(idxs)):
                        tr_logits.append(lg)
                        tr_ypos.append(j if j == gold[iid] else None)
                T = fit_T(tr_logits, tr_ypos, list(range(len(tr_logits))))
                for iid in te_ids:
                    idxs = by_item[iid]
                    lg = [v / T for v in h(X[torch.tensor(idxs).to(dev)]).tolist()]
                    pick = max(range(len(lg)), key=lambda j: lg[j])
                    g = int(pick == gold[iid])
                    correct[iid] = g
                    t = tier(iid)
                    tiers[t][0] += g
                    tiers[t][1] += 1
        accs.append(sum(correct.values()) / len(correct))
        if seed == 0:
            per_item[L] = correct
    mean = sum(accs) / len(accs)
    se = (sum((a - mean) ** 2 for a in accs) / max(len(accs) - 1, 1)) ** .5
    results[L] = mean
    pt = {k: v[0] / max(v[1], 1) for k, v in tiers.items()}
    tag = "  <-- SHIPPED" if L == nlayer else ""
    print(f"last_L{L:<3d} oof {mean:.4f} +-{se:.4f}   easy {pt.get('easy',0):.3f} "
          f"std {pt.get('standard',0):.3f} hard {pt.get('hard',0):.3f}{tag}")

base = nlayer
bestL = max((k for k in results if k != base), key=lambda k: results[k])
wins = sum(1 for i in per_item[bestL] if per_item[bestL][i] and not per_item[base][i])
loss = sum(1 for i in per_item[bestL] if not per_item[bestL][i] and per_item[base][i])
print(f"\nbest = last_L{bestL}: {results[bestL]:.4f} vs shipped last_L{base}: "
      f"{results[base]:.4f}")
print(f"paired per-item: {bestL} fixes {wins} items the shipped read-out gets "
      f"wrong, breaks {loss}")
n = len(per_item[bestL])
p = (wins + loss) / max(n, 1)
print(f"McNemar-ish: {wins} vs {loss} discordant pairs, n={n}")
if wins + loss:
    import math as _m
    z = (abs(wins - loss) - 1) / _m.sqrt(max(wins + loss, 1))
    print(f"  continuity-corrected z = {z:.2f}")
json.dump({"cv_oof": results, "best": bestL, "wins": wins, "losses": loss,
           "n_items": n, "ctx": CTX},
          open(f"{RES}\\readout_cv.json", "w"), indent=2)
print(f"\nwrote {RES}\\readout_cv.json")

