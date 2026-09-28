"""Combine the 4B and the 300M, per item, on unseen data only.

Both scorers are honest on all 213 items:
  4B   grouped 5-fold OOF keyed by item - every item scored by a head that
       did not train on it
  300M zero-shot cosine - no training at all, so trivially unseen

Reported alongside the ensembles is the ORACLE: accuracy if we always picked
whichever model was right. That is the ceiling any router could reach, and it
is the honest way to say how much is left on the table.

Also reports agreement analysis, which bounds what is achievable: when the two
models agree the answer is usually right, and when they disagree it is close
to a coin flip, so the disagreement rate caps the headroom.
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
OUT = Path(r"D:\gavel-jevbench-entry\results")
SEED, EPOCHS, LR, BS, WIDE, K = 20260924, 60, 1e-4, 256, 512, 5

# ---------------- rebuild the item table from the real pairs
rows = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
jb = [r for r in rows if r["tier"] != "nli"]
items = defaultdict(lambda: {"state": "", "q": "", "opts": []})
for r in jb:
    t = r["text"]
    state = t.split("\nQuestion:")[0].replace("State: ", "", 1)
    q, opt = t.split("\nQuestion:", 1)[1].split("\nOption: ", 1)
    it = items[r["item"]]
    if not it["state"]:
        it["state"], it["q"] = state, q
    it["opts"].append((opt, r["target"]))
keys = sorted(items)
ptype = {k: (k.split("-")[1] if len(k.split("-")) > 2 else "?") for k in keys}
gold = {k: [t for _, t in items[k]["opts"]] for k in keys}
chance = sum(1.0 / len(items[k]["opts"]) for k in keys) / len(keys)
print(f"items {len(keys)}  chance {chance:.4f}")

# ---------------- 4B: grouped 5-fold OOF, per-item scores
d = torch.load(TS / "combined_hiddens.pt", map_location="cpu", weights_only=False)
H, order, targets = d["hiddens"].float(), list(d["order"]), list(d["targets"])
jb_items = {r["item"] for r in jb}
mask = torch.tensor([o in jb_items for o in order])
X = H[mask]
y = torch.tensor(targets, dtype=torch.float32)[mask]
grp = [o for o, m in zip(order, mask.tolist()) if m]
print(f"4B rows {tuple(X.shape)}  items {len(set(grp))}  hidden {X.shape[1]}")


def folds_for(g, k=K, seed=SEED):
    u = sorted(set(g))
    perm = np.random.RandomState(seed).permutation(len(u))
    fo = {it: perm[j] % k for j, it in enumerate(u)}
    return np.array([fo[i] for i in g])


class Head(nn.Module):
    def __init__(self, h, w=WIDE):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


fold = folds_for(grp)
oof = torch.zeros(len(y))
t0 = time.time()
for k in range(K):
    tr, va = fold != k, fold == k
    torch.manual_seed(SEED + k)
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
print(f"4B OOF done in {time.time()-t0:.0f}s")

idx_by = defaultdict(list)
for i, g in enumerate(grp):
    idx_by[g].append(i)
S4 = {}
for k, ii in idx_by.items():
    if k in gold:
        S4[k] = oof[ii].tolist()

# ---------------- 300M cosine per-item scores from the cached embeddings
ec = torch.load(TS / "gemma_embeddings.pt", map_location="cpu", weights_only=False)
E, texts = ec["emb"].float(), ec["texts"]
lookup = {t: n for n, t in enumerate(texts)}
S3 = {}
for k in keys:
    s = E[lookup[items[k]["state"]]]
    S3[k] = [float(s @ E[lookup[o]]) for o, _ in items[k]["opts"]]
print(f"300M scores for {len(S3)} items")

# ---------------- scorers and combinations
def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


S4p = {k: softmax(S4[k]) for k in keys if k in S4}
S3p = {k: softmax(np.array(S3[k]) * 8.0) for k in keys if k in S3}
# the 4B read-out cache only covers 137 of the 213 items (the 76 long_policy
# items were never cached), so the ensemble is measured on the intersection.
keys = sorted(set(S4p) & set(S3p))
missing = 213 - len(keys)
chance = sum(1.0 / len(items[k]["opts"]) for k in keys) / len(keys)
print(f"\nENSEMBLE ON {len(keys)} items (the intersection; {missing} of 213 are "
      f"absent from the 4B cache)")
print(f"chance on this subset: {chance:.4f}")
ptype = {k: ptype.get(k, "?") for k in keys}
gold = {k: gold[k] for k in keys}


def acc_from(pick, label):
    ok = sum(1 for k in keys if gold[k][pick(k)] == 1)
    per = defaultdict(lambda: [0, 0])
    for k in keys:
        per[ptype[k]][1] += 1
        per[ptype[k]][0] += gold[k][pick(k)] == 1
    a = ok / len(keys)
    print(f"  {label:40} {ok:3}/{len(keys)} = {a:.4f}   lift {a/chance:.2f}x")
    return a, per


print(f"\n{'='*66}\nsingles\n{'='*66}")
a4, per4 = acc_from(lambda k: int(np.argmax(S4p[k])), "4B probe, grouped OOF")
a3, per3 = acc_from(lambda k: int(np.argmax(S3p[k])), "300M cosine, zero-shot")

print(f"\n{'='*66}\ncombinations\n{'='*66}")
res = {}
for w in (0.3, 0.5, 0.7):
    res[f"mix {w:.1f}*4B+{1-w:.1f}*300M"] = acc_from(
        lambda k, w=w: int(np.argmax(w * S4p[k] + (1 - w) * S3p[k])),
        f"prob mix {w:.1f} 4B + {1-w:.1f} 300M")
res["log-prob sum"] = acc_from(
    lambda k: int(np.argmax(np.log(S4p[k] + 1e-9) + np.log(S3p[k] + 1e-9))),
    "sum of log-probs")
res["max-confidence"] = acc_from(
    lambda k: (int(np.argmax(S4p[k])) if S4p[k].max() >= S3p[k].max()
               else int(np.argmax(S3p[k]))),
    "whichever is more confident")


def oracle_pick(k):
    c4 = gold[k][int(np.argmax(S4p[k]))] == 1
    c3 = gold[k][int(np.argmax(S3p[k]))] == 1
    if c4 and c3:
        return int(np.argmax(S4p[k]))
    return int(np.argmax(S4p[k])) if c4 else int(np.argmax(S3p[k]))


orc, _ = acc_from(oracle_pick, "ORACLE (always pick the right model)")

print(f"\n{'='*66}\nagreement analysis - this is what bounds the headroom")
print(f"{'='*66}")
agree = [k for k in keys
         if int(np.argmax(S4p[k])) == int(np.argmax(S3p[k]))]
dis = [k for k in keys if k not in set(agree)]
a_ag = sum(1 for k in agree if gold[k][int(np.argmax(S4p[k]))] == 1) / max(len(agree), 1)
a_dis = sum(1 for k in dis
            if gold[k][int(np.argmax(S4p[k]))] == 1) / max(len(dis), 1)
print(f"  agree on {len(agree):3}/{len(keys)} items ({len(agree)/len(keys):.1%})"
      f"  accuracy there {a_ag:.4f}")
print(f"  disagree on {len(dis):3} items            accuracy there {a_dis:.4f} "
      f"(4B right {sum(1 for k in dis if gold[k][int(np.argmax(S4p[k]))]==1)})")
print(f"\n  headroom: oracle {orc:.4f} vs best single {max(a4,a3):.4f} "
      f"-> {orc - max(a4,a3):.4f} available, "
      f"and only {len(dis)} items are even in contention")

best_name = max(res, key=lambda k2: res[k2][0])
print(f"\n  best combination: {best_name} = {res[best_name][0]:.4f}")
print(f"  still short of the oracle by {orc - res[best_name][0]:.4f}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ensemble_4b_300m.json").write_text(json.dumps({
    "protocol": "4B = grouped 5-fold OOF keyed by item (unseen); "
                "300M = zero-shot, no training. No sealed data anywhere.",
    "n_items": len(keys), "chance": chance,
    "single_4b_oof": a4, "single_300m_zeroshot": a3,
    "combinations": {k2: v[0] for k2, v in res.items()},
    "best_combination": {"name": best_name, "accuracy": res[best_name][0]},
    "oracle_pick_better_model": orc,
    "headroom_over_best_single": orc - max(a4, a3),
    "agreement": {"n_agree": len(agree), "agree_rate": len(agree) / len(keys),
                  "accuracy_when_agree": a_ag,
                  "n_disagree": len(dis),
                  "accuracy_4b_when_disagree": a_dis},
    "per_type": {t: {"n": per4[t][1], "acc_4b": per4[t][0] / per4[t][1],
                     "acc_300m": per3[t][0] / per3[t][1]}
                 for t in sorted(per4)},
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'ensemble_4b_300m.json'}")

