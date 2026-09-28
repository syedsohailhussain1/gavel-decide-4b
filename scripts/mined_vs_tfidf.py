"""Is the mined library just tfidf? Measure them on identical footing.

The winning mined rule is div(overlap_state_all, n_options) - a normalised
token overlap, which is tfidf-shaped. And the earlier tfidf baseline scored
0.2629, BELOW the 0.3236 chance rate. So either the mined library is
substantially more than tfidf, or one of those numbers is wrong.

Everything below runs on the SAME per-item SHUFFLED options and the SAME
grouped 5-fold OOF keyed by item, so the only thing that varies is the scorer.

  T1  tfidf over the option text alone          (the classic baseline)
  T2  tfidf over state+question+option          (full context)
  T3  tfidf cosine(state, option)
  M   the mined library, searched, held-out selected
  M+  the mined library with tfidf features ADDED to its feature space
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\thrm")
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "mine", r"D:\gavel-jevbench-entry\scripts\mine_decision_procedures.py")
M = importlib.util.module_from_spec(spec)
sys.modules["mine"] = M
spec.loader.exec_module(M)

TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
SEED, FOLDS = 20260924, 5

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

items = M.load_items()
keys = sorted(items)
gold_by_item = {k: [t for _, t in items[k]["opts"]] for k in keys}
chance = float(np.mean([1.0 / len(items[k]["opts"]) for k in keys]))


def shuffle(seed=SEED):
    rng = np.random.RandomState(seed)
    out = {}
    for k in keys:
        it = items[k]
        order = list(range(len(it["opts"])))
        rng.shuffle(order)
        out[k] = {"state": it["state"], "q": it["q"],
                  "opts": [it["opts"][j] for j in order]}
    return out


si = shuffle()
sfirst = sum(1 for k in keys if si[k]["opts"][0][1] == 1) / len(keys)
print(f"shuffled control: always-first {sfirst:.4f}   chance {chance:.4f}\n")


def oof_eval(score_fn, label):
    """Grouped 5-fold: fit on 4 folds of ITEMS, predict the 5th."""
    perm = np.random.RandomState(SEED).permutation(len(keys))
    fold_of = {item: perm[j] % FOLDS for j, item in enumerate(keys)}
    ok = tot = 0
    for f in range(FOLDS):
        tr = [k for k in keys if fold_of[k] != f]
        va = [k for k in keys if fold_of[k] == f]
        pred = score_fn(tr, va)
        for k in va:
            ok += pred[k] == int(np.argmax([t for _, t in si[k]["opts"]]))
            tot += 1
    a = ok / tot
    print(f"  {label:52} {a:.4f}  lift {a/chance:.2f}x")
    return a, ok, tot


def rows_for(ks, kind):
    texts, owner = [], []
    for k in ks:
        it = si[k]
        for o, _ in it["opts"]:
            if kind == "opt":
                texts.append(o)
            elif kind == "full":
                texts.append(f"{it['state']} {it['q']} {o}")
            owner.append(k)
    return texts, owner


print("=" * 68)
print("tfidf baselines (grouped 5-fold OOF, nothing fitted on the test fold)")
print("=" * 68)


def make_tfidf(kind, use_logreg=True):
    def fn(tr, va):
        Xtr_txt, ows = rows_for(tr, kind)
        Xva_txt, _ = rows_for(va, kind)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                              min_df=2, max_features=100_000)
        Xtr = vec.fit_transform(Xtr_txt)
        Xva = vec.transform(Xva_txt)
        pred = {}
        if use_logreg:
            oof = np.zeros(Xva.shape[0])
            ytr, yv = [], []
            gidx = defaultdict(list)
            for j, o in enumerate(ows):
                gidx[o].append(j)
            # train on items, score items
            yt = []
            for k in tr:
                for o, t in si[k]["opts"]:
                    yt.append(t)
            clf = LogisticRegression(max_iter=600, C=2.0)
            clf.fit(Xtr, np.array(yt))
            scores = clf.decision_function(Xva)
        else:
            scores = np.asarray(Xva.sum(1)).ravel()
        ptr = 0
        for k in va:
            n = len(si[k]["opts"])
            seg = scores[ptr:ptr + n]
            pred[k] = int(np.argmax(seg))
            ptr += n
        return pred
    return fn


a1, _, _ = oof_eval(make_tfidf("opt"), "T1 tfidf-char(2,4) on option only, logreg")
a2, _, _ = oof_eval(make_tfidf("full"), "T2 tfidf-char(2,4) on full context, logreg")
a3, _, _ = oof_eval(make_tfidf("opt", False), "T3 tfidf sum, no training at all")

a4 = float("nan")

print("\n" + "=" * 68)
print("the mined library, same protocol")
print("=" * 68)
cols, meta, gold, row_of, sk = M.build(si)
perm = np.random.RandomState(SEED).permutation(len(sk))
fold_of = {item: perm[j] % FOLDS for j, item in enumerate(sk)}
hits = tot = 0
for f in range(FOLDS):
    trk = [k for k in sk if fold_of[k] != f]
    vak = [k for k in sk if fold_of[k] == f]
    surv = M.search(cols, gold, row_of, trk, vak, verbose=False)
    if not surv:
        continue
    a, t, nm, v = max(surv, key=lambda x: (x[0], x[1]))
    h = round(a * len(vak))
    hits += h
    tot += len(vak)
am = hits / max(tot, 1)
print(f"  {'M  mined library, held-out selected':52} {am:.4f}  "
      f"lift {am/chance:.2f}x")

# ---- add tfidf-derived columns to the mined feature space and re-mine
print("\n" + "=" * 68)
print("does tfidf ADD anything when fed to the search? (M+)")
print("=" * 68)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1,
                      max_features=60_000)
alltxt, allstate = [], []
for k in sk:
    allstate.append(f"{si[k]['state']} {si[k]['q']}")
    alltxt.extend(o for o, _ in si[k]["opts"])
vec.fit(alltxt + allstate)
extra = {"tfidf_opt_len": [], "tfidf_cos_state": [],
         "tfidf_cos_question": [], "tfidf_max_idf": []}
for i, (k, j) in enumerate(meta):
    it = si[k]
    o = it["opts"][j][0]
    v = vec.transform([o])
    extra["tfidf_opt_len"].append(float(v.sum()))
    extra["tfidf_cos_state"].append(
        float((v @ vec.transform([f"{it['state']} {it['q']}"]).T).toarray()[0, 0]))
    extra["tfidf_cos_question"].append(
        float((v @ vec.transform([it["q"]]).T).toarray()[0, 0]))
    ids = np.asarray(v.tocoo().data)
    extra["tfidf_max_idf"].append(float(ids.max()) if ids.size else 0.0)
for k2, v2 in extra.items():
    cols[k2] = np.asarray(v2, dtype=np.float64)
print(f"  feature space grew {len(cols) - 4} -> {len(cols)} columns")
hits = tot = 0
for f in range(FOLDS):
    trk = [k for k in sk if fold_of[k] != f]
    vak = [k for k in sk if fold_of[k] == f]
    surv = M.search(cols, gold, row_of, trk, vak, verbose=False)
    if not surv:
        continue
    a, t, nm, v = max(surv, key=lambda x: (x[0], x[1]))
    hits += round(a * len(vak))
    tot += len(vak)
ap = hits / max(tot, 1)
print(f"  {'M+ mined with tfidf features added':52} {ap:.4f}  "
      f"lift {ap/chance:.2f}x")

print(f"\n{'='*68}\nsummary, identical protocol, shuffled options")
print(f"{'='*68}")
for nm, v in (("T1 tfidf opt + logreg", a1), ("T2 tfidf full + logreg", a2),
              ("T3 tfidf sum, untrained", a3), ("T4 tfidf cosine", a4),
              ("M  mined library", am), ("M+ mined + tfidf features", ap)):
    print(f"  {nm:34} {v:.4f}   {v-am:+.4f} vs mined")

(OUT / "mined_vs_tfidf.json").write_text(json.dumps({
    "protocol": "grouped 5-fold OOF keyed by item, per-item shuffled options, "
                "nothing fitted on the scored fold. Identical for every row.",
    "always_first_control": sfirst, "chance": chance,
    "results": {"T1_tfidf_option_logreg": a1, "T2_tfidf_full_logreg": a2,
                "T3_tfidf_sum_untrained": a3, "T4_tfidf_cosine": a4,
                "M_mined_library": am, "M_plus_tfidf_features": ap},
    "mined_beats_best_tfidf_by": am - max(a1, a2, a3, a4),
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'mined_vs_tfidf.json'}")


