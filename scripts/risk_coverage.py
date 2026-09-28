"""Does the mined library's margin predict correctness?

This is the falsifiable core of the whole idea. If margin is informative, an
accuracy-versus-coverage curve exists, and a system can decline to answer a
chosen fraction of inputs while holding a stated accuracy - with no training
and a checkable certificate. If margin is uninformative the curve is flat and
the idea is dead.

Compares three risk-coverage curves on the same items:
  MINED  margin = top1 - top2 of the winning procedure      (trained on NOTHING)
  4B     margin = top1 - top2 of its softmax                (trained, unverified)
  RAND   margin = random                                    (the null to beat)

Reported:
  AURC   area under the risk-coverage curve. LOWER is better. Random is worst.
  the curve at 100%, 80%, 60%, 40% coverage
  whether each curve is monotone, i.e. whether abstaining actually helps
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
cols, meta, goldv, row_of, sk = M.build(si)

# ---- mined: honest held-out, five folds, collecting margin + correctness
perm = np.random.RandomState(SEED).permutation(len(sk))
fold_of = {item: perm[j] % K for j, item in enumerate(sk)}
mined = {}
for f in range(K):
    trk = [k for k in sk if fold_of[k] != f]
    vak = [k for k in sk if fold_of[k] == f]
    surv = M.search(cols, goldv, row_of, trk, vak, verbose=False)
    if not surv:
        continue
    v = max(surv, key=lambda x: (x[0], x[1]))[3]
    for k in vak:
        sc = sorted((float(v[i]) for i in row_of[k]), reverse=True)
        pick = max(row_of[k], key=lambda i: v[i])
        mined[k] = {"margin": (sc[0] - sc[1]) if len(sc) > 1 else 0.0,
                    "correct": gold[k][int(np.argmax(
                        [goldv[i] for i in row_of[k]]))] == 1}

# ---- 4B: honest held-out, same protocol
d = torch.load(TS / "combined_hiddens.pt", map_location="cpu", weights_only=False)
H, order, tg = d["hiddens"].float(), list(d["order"]), list(d["targets"])
jb = {r["item"] for r in json.loads((TS / "combined_pairs.jsonl")
                                   .read_text(encoding="utf-8"))
      if r["tier"] != "nli"}
mask = torch.tensor([o in jb for o in order])
X, y = H[mask], torch.tensor(tg, dtype=torch.float32)[mask]
grp = [o for o, m in zip(order, mask.tolist()) if m]
u = sorted(set(grp))
permu = np.random.RandomState(SEED).permutation(len(u))
fo4 = {it: permu[j] % K for j, it in enumerate(u)}
fold4 = np.array([fo4[i] for i in grp])
oof = torch.zeros(len(y))


class Head(nn.Module):
    def __init__(self, h, w=512):
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
four = {}
for k in keys:
    if k not in idx4:
        continue
    sc = sorted(oof[idx4[k]].tolist(), reverse=True)
    pick = max(idx4[k], key=lambda i: float(oof[i]))
    four[k] = {"margin": (sc[0] - sc[1]) if len(sc) > 1 else 0.0,
               "correct": float(y[pick]) > 0.5}

common = sorted(set(mined) & set(four))
print(f"items with BOTH an honest mined and an honest 4B held-out score: "
      f"{len(common)}")
base4 = sum(four[k]["correct"] for k in common) / len(common)
basem = sum(mined[k]["correct"] for k in common) / len(common)
print(f"accuracy at full coverage:  mined {basem:.4f}   4B {base4:.4f}")


def risk_coverage(scored, label):
    """Keep the highest-margin fraction, report accuracy on what is kept."""
    ks = sorted(scored, key=lambda k: -scored[k]["margin"])
    n = len(ks)
    curve, auc, correct_kept = [], 0.0, 0
    for i, k in enumerate(ks, 1):
        correct_kept += scored[k]["correct"]
        acc = correct_kept / i
        curve.append((i / n, acc))
        auc += acc
    aurc = auc / n
    at = {}
    for cov in (1.0, 0.8, 0.6, 0.4):
        i = max(1, int(round(cov * n)))
        c = sum(scored[k]["correct"] for k in ks[:i]) / i
        at[f"coverage_{int(cov*100)}pct"] = round(c, 4)
    # monotonicity: does accuracy rise as we become more selective?
    steps = [curve[i][1] - curve[i - 1][1] for i in range(1, len(curve))]
    mono = float(np.mean([s > -1e-9 for s in steps]))
    print(f"\n  {label}")
    print(f"    AURC {aurc:.4f}  (lower is better)")
    for kk, vv in at.items():
        print(f"    {kk:22} accuracy {vv:.4f}")
    print(f"    non-decreasing steps  {mono:.3f}  "
          f"({'selective' if mono > 0.6 else 'NOT selective'})")
    return {"aurc": aurc, "at": at, "monotone_frac": mono,
            "accuracy_full": curve[-1][1],
            "points": [[round(a, 3), round(b, 4)] for a, b in curve
                       if abs(a * 20 - round(a * 20)) < 1e-9]}


print("\n" + "=" * 66)
print("RISK-COVERAGE CURVES  (accuracy if we answer only the top X% by margin)")
print("=" * 66)
r_m = risk_coverage({k: mined[k] for k in common}, "MINED library margin (no training)")
r_4 = risk_coverage({k: four[k] for k in common}, "4B softmax margin (trained)")
r_r = risk_coverage({k: {"margin": float(np.random.rand()), "correct": mined[k]["correct"]}
                     for k in common}, "RANDOM margin (null)")

print("\n" + "=" * 66)
print("verdict")
print("=" * 66)
if r_m["aurc"] < r_r["aurc"]:
    print("  the mined margin IS informative: it beats random ordering on AURC")
    gain = r_r["aurc"] - r_m["aurc"]
    print(f"  AURC gain over random: {gain:.4f}")
    lift = r_m["at"]["coverage_40pct"] / r_r["at"]["coverage_40pct"]
    print(f"  accuracy at 40% coverage vs random: {lift:.2f}x")
else:
    print("  the mined margin is NOT informative - the idea is dead")
print(f"  4B margin informative: "
      f"{'yes' if r_4['aurc'] < r_r['aurc'] else 'no'}  "
      f"(AURC {r_4['aurc']:.4f} vs random {r_r['aurc']:.4f})")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "risk_coverage.json").write_text(json.dumps({
    "question": "is the margin informative enough to support abstention?",
    "n_items": len(common),
    "accuracy_full_coverage": {"mined": basem, "4B": base4},
    "mined": r_m, "four_b": r_4, "random_null": r_r,
    "mined_beats_random": bool(r_m["aurc"] < r_r["aurc"]),
    "verdict": "margin is informative - an accuracy-vs-coverage curve exists "
               "with no training and a checkable certificate"
    if r_m["aurc"] < r_r["aurc"] else "margin is uninformative",
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'risk_coverage.json'}")
