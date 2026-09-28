"""Can the PREFIXED 300M improve the leaderboard number? Tested honestly.

The earlier 3-way blend used the 300M's UNPREFIXED embeddings, which measured
0.3504 on the 137-item subset. Prefixing measured +4.7 points on the 213-item
set (0.5446 -> 0.5915), so the blend never saw the better signal.

Rules for this test, fixed BEFORE looking at any result:
  - the 300M is zero-shot, so there is nothing to fit; only the MIX WEIGHT is
    a free parameter, and it is chosen on an inner split and applied to an
    outer split that never saw it
  - 5 outer item folds; the weight is chosen on 4 of them, scored on the 5th
  - every number reported is on items no fold in that outer split trained on
  - the positional control is reported, because a constant column won once today

Also reports the two things that are NOT leaderboard-usable, clearly labelled:
  - abstention: accuracy on the ANSWERED subset, not benchmark accuracy
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

TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
SEED, K, WIDE = 20260924, 5, 512
EMB = TS / "gemma_pref_embeddings.pt"

rows = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
jb = [r for r in rows if r["tier"] != "nli"]
items = defaultdict(lambda: {"state": "", "q": "", "opts": []})
for r in jb:
    t = r["text"]
    st = t.split("\nQuestion:")[0].replace("State: ", "", 1)
    q, opt = t.split("\nQuestion:", 1)[1].split("\nOption: ", 1)
    it = items[r["item"]]
    if not it["state"]:
        it["state"], it["q"] = st, q
    it["opts"].append((opt, r["target"]))
keys = sorted(items)
gold = {k: [t for _, t in items[k]["opts"]] for k in keys}

# ---------------------------------------------------------------- 4B OOF
d = torch.load(TS / "combined_hiddens.pt", map_location="cpu", weights_only=False)
H, order, tg = d["hiddens"].float(), list(d["order"]), list(d["targets"])
jbi = {r["item"] for r in jb}
mask = torch.tensor([o in jbi for o in order])
X = H[mask]
y = torch.tensor(tg, dtype=torch.float32)[mask]
grp = [o for o, m in zip(order, mask.tolist()) if m]
u = sorted(set(grp))
pm = np.random.RandomState(SEED).permutation(len(u))
fo4 = {it: pm[j] % K for j, it in enumerate(u)}
fold4 = np.array([fo4[i] for i in grp])
oof = torch.zeros(len(y))


class Head(nn.Module):
    def __init__(self, h, w=WIDE):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


for f in range(K):
    tr, va = fold4 != f, fold4 == f
    torch.manual_seed(SEED + f)
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
idx4 = defaultdict(list)
for i, g in enumerate(grp):
    idx4[g].append(i)
S4 = {k: oof[idx4[k]].tolist() for k in keys if k in idx4}
sub = sorted(S4)
print(f"items with a 4B OOF score: {len(sub)}")
chance = float(np.mean([1.0 / len(items[k]["opts"]) for k in sub]))


def sm(z):
    z = np.asarray(z, float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ------------------------------------------------- prefixed 300M embeddings
if EMB.exists():
    cache = torch.load(EMB, map_location="cpu", weights_only=False)
    Ep, texts = cache["emb"].float(), cache["texts"]
    lk = {t: n for n, t in enumerate(texts)}
    S3 = {k: [float(Ep[lk[items[k]["state"]]] @ Ep[lk[o]]) for o, _ in items[k]["opts"]]
          for k in sub}
    print("loaded PREFIXED 300M embeddings")
else:
    print("PREFIXED embeddings not cached - run the prefix script first")
    raise SystemExit(1)

P4 = {k: sm(S4[k]) for k in sub}
P3 = {k: sm(np.array(S3[k]) * 8.0) for k in sub}


def acc(pick, ks):
    return sum(1 for k in ks if gold[k][pick(k)] == 1) / len(ks)


def first_last(ks):
    f = sum(1 for k in ks if items[k]["opts"][0][1] == 1) / len(ks)
    l = sum(1 for k in ks if items[k]["opts"][-1][1] == 1) / len(ks)
    return f, l


WEIGHTS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.0)   # weight on the 4B
folds = {it: np.random.RandomState(SEED).permutation(len(sub))[
    list(sub).index(it)] % K for it in sub}

print(f"\n{'='*66}")
print("honest blend: weight chosen on 4 folds, scored on the 5th")
print(f"{'='*66}")
outer_ok = outer_n = 0
picks_log = []
for f in range(K):
    tr = [k for k in sub if folds[k] != f]
    va = [k for k in sub if folds[k] == f]
    best_w, best_a = 1.0, -1
    for w in WEIGHTS:
        a = acc(lambda k, w=w: int(np.argmax(w * P4[k] + (1 - w) * P3[k])), tr)
        if a > best_a:
            best_a, best_w = a, w
    outer_ok += round(acc(lambda k, w=best_w: int(
        np.argmax(w * P4[k] + (1 - w) * P3[k])), va) * len(va))
    outer_n += len(va)
    picks_log.append({"fold": f, "w_4b": best_w, "inner_acc": best_a})
blend = outer_ok / outer_n
base = acc(lambda k: int(np.argmax(P4[k])), sub)
solo3 = acc(lambda k: int(np.argmax(P3[k])), sub)
f, l = first_last(sub)
print(f"  4B alone                    {base:.4f}")
print(f"  300M prefixed alone         {solo3:.4f}")
print(f"  blend, weight chosen blind  {blend:.4f}   "
      f"({blend - base:+.4f} vs 4B)")
print(f"  positional control: always-first {f:.4f}  always-last {l:.4f}  "
      f"chance {chance:.4f}")
for p in picks_log:
    print(f"    fold {p['fold']}: w_4B={p['w_4b']}  inner {p['inner_acc']:.4f}")

# ------------------------------------------------------------- abstention
print(f"\n{'='*66}")
print("ABSTENTION - reported honestly, NOT a leaderboard number")
print(f"{'='*66}")
ranked = sorted(sub, key=lambda k: -float(P4[k].max() - np.sort(P4[k])[-2]))
for cov in (1.0, 0.8, 0.6, 0.4):
    n = max(1, int(round(cov * len(sub))))
    keep = ranked[:n]
    a_ans = acc(lambda k: int(np.argmax(P4[k])), keep)
    print(f"  answer {cov:4.0%} of items: accuracy ON ANSWERED {a_ans:.4f}   "
          f"BENCHMARK accuracy {a_ans*cov:.4f}   "
          f"abstained {len(sub)-n}")
print(f"  -> abstention is a DEPLOYMENT tool. On the leaderboard, abstaining")
print(f"     is scored as wrong, so it can only lower the number.")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "prefixed_blend_honest.json").write_text(json.dumps({
    "protocol": "5 outer item folds; the 300M is zero-shot so only the mix "
                "weight is fitted, and it is chosen on 4 folds and scored on "
                "the 5th. No outer item was in the training set for the head "
                "that scored it.",
    "n_items": len(sub), "chance": chance,
    "always_first": f, "always_last": l,
    "4B_alone": base, "300m_prefixed_alone": solo3,
    "blend_weight_chosen_blind": blend,
    "delta_vs_4B": blend - base,
    "per_fold_weight": picks_log,
    "abstention_NOT_a_leaderboard_metric": {
        cov: {"accuracy_on_answered": acc(
                  lambda k: int(np.argmax(P4[k])),
                  ranked[:max(1, int(round(cov * len(sub))))]),
              "benchmark_accuracy": acc(
                  lambda k: int(np.argmax(P4[k])),
                  ranked[:max(1, int(round(cov * len(sub))))]) * cov}
        for cov in (1.0, 0.8, 0.6, 0.4)},
    "verdict": "prefixed 300M improves the blend" if blend > base
    else "prefixed 300M does NOT improve on the 4B; abstention is a "
         "deployment tool only and would LOWER the leaderboard score",
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'prefixed_blend_honest.json'}")
