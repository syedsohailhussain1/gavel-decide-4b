"""Can we ensemble the four context views? Free — the runs already exist.

ctx 512/1024/2048/4096 disagree on different items because a longer window
changes what the head sees. If those errors are decorrelated, averaging the
per-option distributions is a straight accuracy win at zero inference cost
(the speed axis is unchanged across contexts, so this is nearly free at serve
time too).

Reports single-view accuracy, every subset ensemble, and error overlap, so we
can tell whether a gain is real signal or just noise on 231 items.
"""
import itertools
import json
import math
import sys
from collections import defaultdict

sys.path.insert(0, r"D:\jevbench")
from jevbench.metrics import ece_top_label
from jevbench.composite_v13 import calibration as cal_axis

RES = r"D:\gavel-jevbench-entry\results"
CTXS = (512, 1024, 2048, 4096)


def tier(t):
    x = t.split("-", 1)[0]
    return {"original": "standard"}.get(x, x)


truth = {}
for f in ("easy", "original", "hard"):
    for line in open(f"D:\\jevbench\\datasets\\public\\{f}.jsonl", encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            e = r["expected"]
            truth[r["id"]] = str(e) if isinstance(e, (int, float)) else e

runs = {}
for c in CTXS:
    d = {}
    for line in open(f"{RES}\\ctx{c}.jsonl", encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("probs"):
                d[r["task_id"]] = r
    runs[c] = d
    ok = sum(1 for r in d.values() if r.get("correct"))
    print(f"ctx {c:5d}: {ok}/{len(d)} = {ok/len(d):.4f}")

common = set.intersection(*(set(runs[c]) for c in CTXS))
print(f"\ncommon items across all four views: {len(common)}")
good = [t for t in common if truth.get(t) in runs[512][t]["probs"]]
print(f"with usable ground truth: {len(good)}")

print("\n=== single views ===")
for c in CTXS:
    ok = sum(1 for t in good if runs[c][t]["predicted"] == truth[t])
    print(f"  ctx {c:5d}: {ok}/{len(good)} = {ok/len(good):.4f}")


def probs_for(c, t):
    return runs[c][t]["probs"]


def acc_of(weights):
    """weights: {ctx: w}. Average in log space (geometric mean of probs)."""
    ok = 0
    for t in good:
        labs = list(probs_for(512, t))
        s = 0.0
        for l in labs:
            v = 0.0
            for c, w in weights.items():
                p = probs_for(c, t).get(l)
                if p is None:
                    p = 1e-9
                v += w * math.log(max(p, 1e-12))
            s += math.exp(v)
        pick = max(labs, key=lambda l: s)
        ok += int(pick == truth[t])
    return ok


print("\n=== uniform ensembles over every non-empty subset ===")
best = []
for r in range(2, len(CTXS) + 1):
    for combo in itertools.combinations(CTXS, r):
        w = {c: 1.0 / r for c in combo}
        ok = acc_of(w)
        best.append((ok, combo))
        print(f"  {'+'.join(str(c) for c in combo):24s} {ok}/{len(good)} = "
              f"{ok/len(good):.4f}")
best.sort(reverse=True)
print(f"\nbest: {best[0][1]} -> {best[0][0]}/{len(good)} = {best[0][0]/len(good):.4f}")
base = sum(1 for t in good if runs[512][t]["predicted"] == truth[t])
print(f"ctx512 alone: {base}/{len(good)} = {base/len(good):.4f}")
print(f"delta: {best[0][0]-base:+d} items")

print("\n=== error overlap (are the errors decorrelated?) ===")
wrong = {c: {t for t in good if runs[c][t]["predicted"] != truth[t]} for c in CTXS}
for c in CTXS:
    print(f"  ctx {c:5d} wrong: {len(wrong[c])}")
for a, b in itertools.combinations(CTXS, 2):
    inter = len(wrong[a] & wrong[b])
    union = len(wrong[a] | wrong[b])
    print(f"  {a:5d} vs {b:5d}: shared wrong {inter:3d}  "
          f"jaccard {inter/max(union,1):.3f}  "
          f"oracle-either-correct {len(good)-union}/{len(good)}")

# calibration of the best ensemble
combo = best[0][1]
w = {c: 1.0 / len(combo) for c in combo}
for label, sub in (("all", good), ("hard", [t for t in good if tier(t) == "hard"])):
    pairs = []
    for t in sub:
        labs = list(probs_for(512, t))
        s = 0.0
        vals = {}
        for l in labs:
            v = 0.0
            for c, ww in w.items():
                v += ww * math.log(max(probs_for(c, t).get(l, 1e-9), 1e-12))
            vals[l] = math.exp(v)
        Z = sum(vals.values())
        vals = {l: v / Z for l, v in vals.items()}
        pairs.append((max(vals.values()), 1.0 if truth[t] in vals and
                      max(vals, key=lambda x: vals[x]) == truth[t] else 0.0))
    e = ece_top_label(pairs)["ece"]
    print(f"\nensemble {'+'.join(map(str,combo))} {label}: ECE {e:.4f} "
          f"-> calibration axis {cal_axis(e):.2f}")
