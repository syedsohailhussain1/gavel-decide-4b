"""Refit temperature + meta-calibrator on the bf16 run, then re-score.

The shipped head was fitted on bitsandbytes nf4 hidden states. Running the
same weights in bf16 shifts the logits, so the fitted temperature (2.4453) and
the meta-calibrator are mis-specified for bf16, and hard-tier ECE rose from
0.0715 to 0.1042. This refits both against the recorded bf16 logits.

Ground truth comes from the JevBench task files (`expected`), joined by
task_id. It must NOT come from the run's own `predicted` field: using the
model's prediction as the fitting target is circular, and drives the
temperature to zero.

Sanity gate: the reconstruction must reproduce the run's own reported ECE
before any of its numbers are trusted.
"""
import json
import math
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, r"D:\jevbench")
from jevbench.metrics import ece_top_label
from jevbench.composite_v13 import calibration as cal_axis

RES = r"D:\gavel-jevbench-entry\results"
DATA = r"D:\jevbench\datasets\public"


def tier(t):
    x = t.split("-", 1)[0]
    return {"original": "standard"}.get(x, x)


truth = {}
for f in ("easy", "original", "hard"):
    for line in open(f"{DATA}\\{f}.jsonl", encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            e = r["expected"]
            e = e.get("label") if isinstance(e, dict) else e
            # score items carry an int level index while labels are the
            # string renderings ['0','1','2','3']; compare as text.
            truth[r["id"]] = str(e) if isinstance(e, (int, float)) else e

logits = {}
for line in open(f"{RES}\\bf16_logits.jsonl", encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        logits[r["task_id"]] = r

run = {}
for line in open(f"{RES}\\bf16_run.jsonl", encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        run[r["task_id"]] = r

items = []
for tid, d in logits.items():
    rec = run.get(tid)
    exp = truth.get(tid)
    if not rec or exp is None or exp not in d["labels"]:
        continue
    items.append({"tid": tid, "tier": tier(tid), "logits": d["logits"],
                  "labels": d["labels"], "y": d["labels"].index(exp),
                  "correct": bool(rec.get("correct"))})
print(f"paired items with ground truth: {len(items)}")
assert all(it["correct"] == (it["labels"][it["y"]] == run[it["tid"]]["predicted"])
           for it in items), "ground truth disagrees with run correctness"
print("ground truth reconciles with run correctness on every item")


def softmax_T(lg, T):
    z = [v / T for v in lg]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def nll(T):
    return -sum(math.log(max(softmax_T(it["logits"], T)[it["y"]], 1e-15))
                for it in items) / len(items)


# ---- sanity gate: reproduce the run's own reported ECE with shipped T+meta ----
hp = None
for cand in (r"D:\gavel\models\training_state\combined_head.pt",):
    try:
        import torch
        hp = torch.load(cand, map_location="cpu", weights_only=False)
        break
    except Exception:
        pass
if hp is not None and "meta_cal" in hp:
    sys.path.insert(0, r"D:\gavel-jevbench-entry\src")
    from gavel_meta import meta_rescale
    T0 = float(hp["temperature"])
    pairs = []
    for it in items:
        p = softmax_T(it["logits"], T0)
        d = meta_rescale({l: v for l, v in zip(it["labels"], p)}, hp["meta_cal"])
        pairs.append((max(d.values()), 1.0 if it["correct"] else 0.0))
    rec_all = ece_top_label(pairs)["ece"]
    hard = [it for it in items if it["tier"] == "hard"]
    hpairs = []
    for it in hard:
        p = softmax_T(it["logits"], T0)
        d = meta_rescale({l: v for l, v in zip(it["labels"], p)}, hp["meta_cal"])
        hpairs.append((max(d.values()), 1.0 if it["correct"] else 0.0))
    rec_hard = ece_top_label(hpairs)["ece"]
    print(f"\nSANITY GATE  reconstructed all={rec_all:.4f} hard={rec_hard:.4f}")
    print(f"             reported     all=0.0427 hard=0.1042")
    ok = abs(rec_hard - 0.1042) < 0.02
    print(f"             -> {'PASS' if ok else 'FAIL — do not trust the refit'}")
else:
    T0 = 2.4453
    print("\nSANITY GATE skipped (shipped head not found locally)")

# ---- temperature ----
grid = [0.05 + 0.005 * i for i in range(0, 1200)]
best = min(grid, key=nll)
print(f"\nfitted T (bf16) = {best:.3f}  nll={nll(best):.5f}")
print(f"shipped T       = {T0:.4f}  nll={nll(T0):.5f}")


def featvec(p, k):
    ps = sorted(p, reverse=True)
    top = ps[0]
    second = ps[1] if len(ps) > 1 else 0.0
    ent = -sum(v * math.log(max(v, 1e-15)) for v in p)
    return [top, top - second, ent, math.log(max(k, 2))]


X = np.array([featvec(softmax_T(it["logits"], best), len(it["logits"]))
              for it in items])
y = np.array([it["correct"] for it in items], dtype=int)
mu, sd = X.mean(0), X.std(0) + 1e-9
Xn = (X - mu) / sd

oof = np.zeros(len(y))
for tr, te in KFold(5, shuffle=True, random_state=0).split(Xn):
    oof[te] = LogisticRegression(C=0.05, max_iter=5000).fit(
        Xn[tr], y[tr]).predict_proba(Xn[te])[:, 1]
clf = LogisticRegression(C=0.05, max_iter=5000).fit(Xn, y)
w = clf.coef_[0]
b = float(clf.intercept_[0])
print(f"meta weights {np.round(w, 4).tolist()}  b={b:.4f}")


def rescale(d, p_top):
    lab = max(d, key=lambda x: d[x])
    rest, rest_old = 1.0 - p_top, 1.0 - d[lab]
    return {l: (p_top if l == lab else
                (d[l] / rest_old * rest if rest_old > 1e-12 else 0.0))
            for l in d}


def score(sub, T, p_top_fn):
    """Return (ECE, n_wrong, n_argmax_changed_by_rescale).

    The third number is the one that matters for safety: meta-rescaling must
    never change which option wins.
    """
    pairs, wrong, changed = [], 0, 0
    for n, it in enumerate(sub):
        p = softmax_T(it["logits"], T)
        d = {l: v for l, v in zip(it["labels"], p)}
        before = max(d, key=lambda x: d[x])
        pt = p_top_fn(d, n, it)
        d = rescale(d, pt)
        after = max(d, key=lambda x: d[x])
        changed += int(before != after)
        wrong += int(after != it["labels"][it["y"]])
        pairs.append((max(d.values()), 1.0 if it["correct"] else 0.0))
    return ece_top_label(pairs)["ece"], wrong, changed


def no_meta(d, n, it):
    return max(d.values())


def meta_in(d, n, it):
    f = (np.array(featvec(list(d.values()), len(d))) - mu) / sd
    return 1.0 / (1.0 + math.exp(-float((f * w).sum() + b)))


def meta_oof(d, n, it, oofv=None):
    return float(oofv[n])


hard_idx = [i for i, it in enumerate(items) if it["tier"] == "hard"]
hard = [items[i] for i in hard_idx]
oof_hard = oof[hard_idx]

print(f"\n{'config':40s} {'all ECE':>9} {'hard ECE':>9} {'wrong':>6} "
      f"{'rescale-flips':>14} {'axis':>7}")
for name, T, fn, oofv in (
        ("shipped T, no meta", T0, no_meta, None),
        ("bf16 fitted T, no meta", best, no_meta, None),
        ("bf16 fitted T + meta (in-sample)", best, meta_in, None),
        ("bf16 fitted T + meta (5-fold OOF)", best, meta_oof, oof),
):
    oa = oof if oofv is None else oofv
    ea, wa, ca = score(items, T, fn if fn is not meta_oof else
                       (lambda d, n, it: float(oa[n])))
    oh = oof if oofv is None else oof_hard
    eh, wh, ch = score(hard, T, fn if fn is not meta_oof else
                       (lambda d, n, it: float(oh[n])))
    print(f"{name:40s} {ea:9.4f} {eh:9.4f} {wh:6d} {ch:14d} {cal_axis(eh):7.2f}")

json.dump({"temperature": best,
           "meta": {"feats": ["c", "margin", "ent", "kopts"],
                    "mu": mu.tolist(), "sd": sd.tolist(),
                    "w": w.tolist(), "b": b},
           "n_items": len(items), "nll": nll(best)},
          open(f"{RES}\\bf16_calibration.json", "w"), indent=2)
print(f"\nwrote {RES}\\bf16_calibration.json")
