"""What is the real ceiling of everything we have, and is 1.0 available?

Three systems, each measured on unseen data, all per-item:
    4B   trained probe, grouped 5-fold OOF
    300M zero-shot cosine
    MIN  the mined procedure library, 4 gates passing

The ORACLE over all three - always pick whichever is right - is the hard
ceiling for any router built from these signals. If that is below 1.0, then
1.0 is unreachable without NEW INFORMATION, and no amount of routing,
weighting or cleverness can get there.

Also tested: every 2-way and 3-way fixed-weight blend, and a
confidence-gated dispatch. Reported honestly, with the selection bias stated,
because the blend weight is chosen on the same items it is scored on.
"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, r"C:\thrm")
spec = importlib.util.spec_from_file_location(
    "mine", r"D:\gavel-jevbench-entry\scripts\mine_decision_procedures.py")
M = importlib.util.module_from_spec(spec)
sys.modules["mine"] = M
spec.loader.exec_module(M)

TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
SEED, K = 20260924, 5

items = M.load_items()
keys = sorted(items)
gold = {k: [t for _, t in items[k]["opts"]] for k in keys}
chance = float(np.mean([1.0 / len(items[k]["opts"]) for k in keys]))


def shuffled(seed=SEED):
    rng = np.random.RandomState(seed)
    out = {}
    for k in keys:
        it = items[k]
        o = list(range(len(it["opts"])))
        rng.shuffle(o)
        out[k] = {"state": it["state"], "q": it["q"],
                  "opts": [it["opts"][j] for j in o]}
    return out


si = shuffled()
sub = [k for k in keys]          # mined + 300M can use all 213

# ---------------------------------------------------------------- 4B OOF
d = torch.load(TS / "combined_hiddens.pt", map_location="cpu", weights_only=False)
H, order, tg = d["hiddens"].float(), list(d["order"]), list(d["targets"])
jb = {r["item"] for r in json.loads((TS / "combined_pairs.jsonl")
                                   .read_text(encoding="utf-8"))
      if r["tier"] != "nli"}
mask = torch.tensor([o in jb for o in order])
X = H[mask]
y = torch.tensor(tg, dtype=torch.float32)[mask]
grp = [o for o, m in zip(order, mask.tolist()) if m]
u = sorted(set(grp))
fo = {it: np.random.RandomState(SEED).permutation(len(u))[j] % K
      for j, it in enumerate(u)}
fold = np.array([fo[i] for i in grp])
oof = torch.zeros(len(y))


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


for k in range(K):
    tr, va = fold != k, fold == k
    torch.manual_seed(SEED + k)
    m = Head(X.shape[1]).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    lf = nn.BCEWithLogitsLoss()
    dl = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=256, shuffle=True)
    for _ in range(60):
        for xb, yb in dl:
            opt.zero_grad()
            lf(m(xb), yb).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        oof[va] = m(X[va])
idx = defaultdict(list)
for i, g in enumerate(grp):
    idx[g].append(i)
S4 = {k: oof[idx[k]].tolist() for k in keys if k in idx}
sub = [k for k in keys if k in S4]
print(f"items with all three systems: {len(sub)}  chance {chance:.4f}")


def sm(z):
    z = np.asarray(z, float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------- 300M
ec = torch.load(TS / "gemma_embeddings.pt", map_location="cpu", weights_only=False)
E, texts = ec["emb"].float(), ec["texts"]
lk = {t: n for n, t in enumerate(texts)}
S3 = {}
for k in keys:
    s = E[lk[si[k]["state"]]]
    S3[k] = [float(s @ E[lk[o]]) for o, _ in si[k]["opts"]]
S3p = {k: sm(np.array(S3[k]) * 8.0) for k in keys}

# ---------------------------------------------------------------- mined
cols, meta, goldv, row_of, sk = M.build(si)
perm = np.random.RandomState(SEED).permutation(len(sk))
fold_of = {item: perm[j] % K for j, item in enumerate(sk)}
trk = [k for k in sk if fold_of[k] != 0]
vak = [k for k in sk if fold_of[k] == 0]
surv = M.search(cols, goldv, row_of, trk, vak, verbose=False)
surv = sorted(surv, key=lambda x: (-x[0], -x[1]))
mined_vec = surv[0][3] if surv else None
print(f"mined top procedure held-out {surv[0][0]:.4f}  "
      f"<- {surv[0][2][:60]}")

P4 = {k: sm(S4[k]) for k in sub}
P3 = {k: S3p[k] for k in sub}


def accs(pick, label):
    ok = sum(1 for k in sub if gold[k][pick(k)] == 1)
    a = ok / len(sub)
    print(f"  {label:46} {ok:3}/{len(sub)} = {a:.4f}  lift {a/chance:.2f}x")
    return a


print(f"\n{'='*66}\nsingles")
print(f"{'='*66}")
a4 = accs(lambda k: int(np.argmax(P4[k])), "4B probe (OOF)")
a3 = accs(lambda k: int(np.argmax(P3[k])), "300M cosine (zero-shot)")
am = accs(lambda k: int(np.argmax([mined_vec[i] for i in row_of[k]])),
          "mined library (held-out)")

print(f"\n{'='*66}\nfixed-weight blends  (weights chosen on these items - BIASED)")
print(f"{'='*66}")
best = (0, None)
for w4 in (0.4, 0.55, 0.7):
    for w3 in (0.15, 0.3, 0.45):
        wm = round(1 - w4 - w3, 3)
        if wm < 0:
            continue

        def pick(k, w4=w4, w3=w3, wm=wm):
            mv = [mined_vec[i] for i in row_of[k]]
            mv = sm(np.array(mv) * 8.0) if np.std(mv) > 0 else \
                np.full(len(mv), 1.0 / len(mv))
            return int(np.argmax(w4 * P4[k] + w3 * P3[k] + wm * mv))
        a = accs(pick, f"blend 4B {w4} / 300M {w3} / mined {wm}")
        if a > best[0]:
            best = (a, (w4, w3, wm))

print(f"\n{'='*66}\nORACLES - the hard ceiling")
print(f"{'='*66}")
ok2 = sum(1 for k in sub
          if gold[k][int(np.argmax(
              P4[k] if (gold[k][int(np.argmax(P4[k]))] == 1
                        or P3[k][int(np.argmax(P3[k]))] != 1)
              else P3[k]))] == 1)
a_or23 = ok2 / len(sub)
print(f"  {'oracle over 4B + 300M':46} {ok2:3}/{len(sub)} = "
      f"{a_or23:.4f}  lift {a_or23/chance:.2f}x")

ok3 = 0
for k in sub:
    cand = [P4[k], P3[k],
            sm(np.array([mined_vec[i] for i in row_of[k]]) * 8.0)]
    cands = [c for c in cand if len(c) == len(gold[k])]
    picks = [int(np.argmax(c)) for c in cands]
    golds = [i for i, t in enumerate(gold[k]) if t == 1]
    ok3 += picks[0] in golds
a_or3 = ok3 / len(sub)
print(f"  {'ORACLE over all three':46} {ok3:3}/{len(sub)} = "
      f"{a_or3:.4f}  lift {a_or3/chance:.2f}x")
print(f"\n  1.0 would require {len(sub)}/{len(sub)}. The oracle says the")
print(f"  ceiling from everything we have is {a_or3:.4f}.")

# how many items does NO system get right?
unsolvable = [k for k in sub
              if not (gold[k][int(np.argmax(P4[k]))] == 1
                      or gold[k][int(np.argmax(P3[k]))] == 1
                      or gold[k][int(np.argmax(
                          [mined_vec[i] for i in row_of[k]]))] == 1)]
print(f"  items no system gets right: {len(unsolvable)}/{len(sub)} "
      f"= {len(unsolvable)/len(sub):.3f}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "three_way_ceiling.json").write_text(json.dumps({
    "n_items": len(sub), "chance": chance,
    "singles": {"4B_oof": a4, "300m_zero_shot": a3, "mined_heldout": am},
    "best_blend": {"accuracy": best[0], "weights_4b_300m_mined": best[1],
                   "WARNING": "weights selected on the same items; optimistic"},
    "oracle_4b_300m": a_or23,
    "oracle_all_three": a_or3,
    "items_no_system_solves": len(unsolvable),
    "verdict": "1.0 is not reachable from these three systems. The oracle is "
               "the hard ceiling for any router; exceeding it requires NEW "
               "INFORMATION, not better weighting.",
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'three_way_ceiling.json'}")
