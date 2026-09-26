"""How far does TF-IDF + logistic regression actually get on JevBench-style
typed decisions? Measured, not guessed.

The claim to test: a sparse linear model over n-grams, running on a CPU in
milliseconds, can replace a 4B-parameter transformer on this task.

Protocol is identical to readout_cv.py so the numbers are comparable:
5-fold CV split BY ITEM (options of an item never straddle a fold), item-level
argmax accuracy, same 231 public items, same metric. The Gavel head scores
0.8216 on the 213 reconstructable items under the same metric.

Three featurisations, because the interesting question is what a linear model
can and cannot see:
  A  whole-pair TF-IDF (state+question+option) with word 1-2 grams
  B  A + character 3-5 grams (robust to morphology, spelling, IDs)
  C  B + explicit state/option overlap features (lexical grounding)

Also reports per-tier, because we expect the tiers to diverge sharply and the
divergence is the actual finding.
"""
import json
import math
import re
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from scipy.sparse import hstack, csr_matrix

sys.path.insert(0, r"D:\jevbench")
from jevbench.metrics import ece_top_label

PAIRS = r"D:\gavel\models\training_state\combined_pairs.jsonl"
PAIRS_NLI = None
FOLDS = 5


def tier(t):
    x = t.split("-", 1)[0]
    return {"original": "standard"}.get(x, x)


pairs = json.load(open(PAIRS, encoding="utf-8"))
print(f"pairs {len(pairs)}")

# rebuild item groups over public rows only, exactly as the Gavel eval does
by_item, gold = defaultdict(list), {}
for i, p in enumerate(pairs):
    if p.get("tier") == "nli":
        continue
    by_item[p["item"]].append(i)
    if str(p["target"]) == "1":
        gold.setdefault(p["item"], []).append(len(by_item[p["item"]]) - 1)
items = sorted(by_item)
print(f"public items {len(items)}  rows {sum(len(v) for v in by_item.values())}")

# split each pair text into its three parts so we can featurise parts and
# their interaction separately
ST_RE = re.compile(r"^State: (.*?)\nQuestion: (.*?)\nOption: (.*)$", re.S)
parsed = []
bad = 0
for i, p in enumerate(pairs):
    m = ST_RE.match(p["text"])
    if m:
        parsed.append((m.group(1), m.group(2), m.group(3)))
    else:
        parsed.append((p["text"], "", ""))
        bad += 1
print(f"parsed {len(parsed)-bad}/{len(parsed)} into state/question/option")

y = np.array([1 if str(p["target"]) == "1" else 0 for p in pairs])
pub_rows = [i for i, p in enumerate(pairs) if p.get("tier") != "nli"]
row_item = {}
for iid, idxs in by_item.items():
    for j, i in enumerate(idxs):
        row_item[i] = (iid, j)


def overlap_feats(rows):
    """Explicit lexical grounding: does the option's vocabulary appear in the
    state / question, and how much of it does?"""
    out = np.zeros((len(rows), 6), dtype=np.float32)
    for n, i in enumerate(rows):
        st, q, o = parsed[i]
        so, oo = set(re.findall(r"[a-z0-9]+", st.lower())), set(re.findall(r"[a-z0-9]+", o.lower()))
        qo = set(re.findall(r"[a-z0-9]+", q.lower()))
        out[n] = [
            len(oo & so) / max(len(oo), 1),          # option coverage in state
            len(oo & qo) / max(len(oo), 1),           # option coverage in question
            1.0 if oo and oo <= so else 0.0,          # option fully contained in state
            math.log1p(len(oo)),
            len(st) / 1000.0,
            len(oo & (so | qo)),
        ]
    return out


def run(name, use_char, use_overlap, C=1.0):
    t0 = time.perf_counter()
    correct = {}
    oof_conf = []
    oof_cor = []
    for fold_i, (tri, tei) in enumerate(
            KFold(FOLDS, shuffle=True, random_state=0).split(items)):
        # KFold yields POSITION arrays; map them back to item names, or
        # by_item[np.int64] silently returns [] and the fold trains on nothing.
        tr_items = [items[k] for k in tri]
        te_items = [items[k] for k in tei]
        tr_rows = [r for it in tr_items for r in by_item[it]]
        te_rows = [r for it in te_items for r in by_item[it]]
        assert tr_rows and te_rows, f"fold {fold_i} empty"
        tr_txt = [pairs[r]["text"] for r in tr_rows]
        te_txt = [pairs[r]["text"] for r in te_rows]

        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                             strip_accents="unicode", lowercase=True)
        Xtr = vw.fit_transform(tr_txt)
        Xte = vw.transform(te_txt)
        if use_char:
            vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True, lowercase=True)
            Xtr = hstack([Xtr, vc.fit_transform(tr_txt)]).tocsr()
            Xte = hstack([Xte, vc.transform(te_txt)]).tocsr()
        if use_overlap:
            Otr = csr_matrix(overlap_feats(tr_rows))
            Ote = csr_matrix(overlap_feats(te_rows))
            Xtr = hstack([Xtr, Otr]).tocsr()
            Xte = hstack([Xte, Ote]).tocsr()

        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(Xtr, y[tr_rows])
        pr = clf.predict_proba(Xte)[:, 1]
        pos = {r: n for n, r in enumerate(te_rows)}
        for iid in te_items:
            idxs = by_item[iid]
            sc = [pr[pos[r]] for r in idxs]
            pick = max(range(len(sc)), key=lambda j: sc[j])
            good = int(pick in gold[iid])
            correct[iid] = good
            oof_conf.append(max(sc))
            oof_cor.append(1.0 if good else 0.0)
    n = len(correct)
    per = defaultdict(lambda: [0, 0])
    for iid, g in correct.items():
        t = tier(iid)
        per[t][0] += g
        per[t][1] += 1
    acc = sum(correct.values()) / n
    pt = {k: v[0] / max(v[1], 1) for k, v in per.items()}
    ece = ece_top_label([(c, k) for c, k in zip(oof_conf, oof_cor)])["ece"]
    print(f"\n{name}")
    print(f"  dims {Xtr.shape[1]:>7}   OOF item acc {acc:.4f}  ({sum(correct.values())}/{n})"
          f"   [{time.perf_counter()-t0:.1f}s]")
    print("  " + "   ".join(f"{k}:{pt.get(k,0):.3f}" for k in ("easy", "standard", "hard")))
    print(f"  top-label ECE {ece:.4f} -> calibration axis {max(0,100*(1-ece/0.5)):.2f}")
    return acc, pt


print("\n" + "=" * 74)
print("TF-IDF + logistic regression, 5-fold CV by item (same protocol as the "
      "Gavel head)")
print("=" * 74)
results = {}
for name, ch, ov in (("A  word 1-2 grams", False, False),
                     ("B  A + char 3-5 grams", True, False),
                     ("C  B + overlap features", True, True)):
    results[name] = run(name, ch, ov)

print("\n" + "=" * 74)
print("reference points, same 231 items, same metric")
print("=" * 74)
print("  Gavel MLP head (frozen Qwen3-4B)   0.8216   easy 0.917  std 0.850  hard 0.762")
for name, (a, pt) in results.items():
    print(f"  {name:34s} {a:.4f}   easy {pt.get('easy',0):.3f}  "
          f"std {pt.get('standard',0):.3f}  hard {pt.get('hard',0):.3f}")
json.dump({k: {"acc": v[0], "tiers": v[1]} for k, v in results.items()},
          open(r"D:\gavel-jevbench-entry\results\tfidf_baseline.json", "w"), indent=2)
print("\nwrote results/tfidf_baseline.json")
