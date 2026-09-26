"""Does a linear model add anything to the Gavel head? (free to test)

The TF-IDF baseline scored 0.3897 alone against the head's 0.8216 — less than
half. But a weak model that is wrong DIFFERENTLY is still useful, and a linear
model costs microseconds on top of a forward pass we already do.

So: blend the head's softmax with the linear model's probability, per item, and
sweep the weight. If the errors are decorrelated this wins for free; if they are
the same errors it does nothing. Either way it is a local, CPU-only experiment.
"""
import json
import math
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

RES = r"D:\gavel-jevbench-entry\results"
ST_RE = re.compile(r"^State: (.*?)\nQuestion: (.*?)\nOption: (.*)$", re.S)


def tier(t):
    x = t.split("-", 1)[0]
    return {"original": "standard"}.get(x, x)


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


pairs = json.load(open(r"D:\gavel\models\training_state\combined_pairs.jsonl",
                       encoding="utf-8"))
by_item, gold = defaultdict(list), {}
for i, p in enumerate(pairs):
    if p.get("tier") == "nli":
        continue
    by_item[p["item"]].append(i)
    if str(p["target"]) == "1":
        gold.setdefault(p["item"], []).append(len(by_item[p["item"]]) - 1)
items = sorted(by_item)
y = np.array([1 if str(p["target"]) == "1" else 0 for p in pairs])

parsed = []
for p in pairs:
    m = ST_RE.match(p["text"])
    parsed.append(m.groups() if m else (p["text"], "", ""))


def overlap_feats(rows):
    out = np.zeros((len(rows), 6), dtype=np.float32)
    for n, i in enumerate(rows):
        st, q, o = parsed[i]
        so = set(re.findall(r"[a-z0-9]+", st.lower()))
        qo = set(re.findall(r"[a-z0-9]+", q.lower()))
        oo = set(re.findall(r"[a-z0-9]+", o.lower()))
        out[n] = [len(oo & so) / max(len(oo), 1), len(oo & qo) / max(len(oo), 1),
                  1.0 if oo and oo <= so else 0.0, math.log1p(len(oo)),
                  len(st) / 1000.0, len(oo & (so | qo))]
    return out


CK = torch.load(r"D:\gavel\models\training_state\combined_hiddens.pt",
                map_location="cpu", weights_only=False)
HP = torch.load(r"D:\gavel\models\training_state\combined_head.pt",
                map_location="cpu", weights_only=False)
head = Head(HP["hidden"], HP.get("wide", 512))
head.load_state_dict(HP["head"])
head.eval()
T = float(HP["temperature"])
with torch.no_grad():
    HL = head(CK["hiddens"].float()).tolist()


def softmax_T(v):
    z = [x / T for x in v]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


print(f"items {len(items)}")
head_only = 0
for iid in items:
    z = [HL[i] for i in by_item[iid]]
    p = softmax_T(z)
    head_only += int(max(range(len(p)), key=lambda j: p[j]) in gold[iid])
print(f"head alone            {head_only}/{len(items)} = {head_only/len(items):.4f}")

# out-of-fold linear scores, aligned to the same protocol
lin_oof = {}
for tri, tei in KFold(5, shuffle=True, random_state=0).split(items):
    tr_items = [items[k] for k in tri]
    te_items = [items[k] for k in tei]
    tr_rows = [r for it in tr_items for r in by_item[it]]
    te_rows = [r for it in te_items for r in by_item[it]]
    tr_txt = [pairs[r]["text"] for r in tr_rows]
    te_txt = [pairs[r]["text"] for r in te_rows]
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                         strip_accents="unicode")
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                         sublinear_tf=True)
    Xtr = hstack([vw.fit_transform(tr_txt), vc.fit_transform(tr_txt),
                  csr_matrix(overlap_feats(tr_rows))]).tocsr()
    Xte = hstack([vw.transform(te_txt), vc.transform(te_txt),
                  csr_matrix(overlap_feats(te_rows))]).tocsr()
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, y[tr_rows])
    pr = clf.predict_proba(Xte)[:, 1]
    pos = {r: n for n, r in enumerate(te_rows)}
    for iid in te_items:
        lin_oof[iid] = [pr[pos[r]] for r in by_item[iid]]

print(f"linear alone (OOF)     {sum(1 for i in items if max(range(len(lin_oof[i])), key=lambda j: lin_oof[i][j]) in gold[i])}"
      f"/{len(items)}")

# are the errors decorrelated?
both_wrong = 0
head_wrong_lin_right = 0
for iid in items:
    hp_ = softmax_T([HL[i] for i in by_item[iid]])
    h_ok = max(range(len(hp_)), key=lambda j: hp_[j]) in gold[iid]
    l_ok = max(range(len(lin_oof[iid])), key=lambda j: lin_oof[iid][j]) in gold[iid]
    both_wrong += int(not h_ok and not l_ok)
    head_wrong_lin_right += int((not h_ok) and l_ok)
n = len(items)
print(f"\nerror overlap: both wrong {both_wrong}/{n}  "
      f"head wrong & linear right {head_wrong_lin_right}/{n}")
print(f"oracle(head or linear) correct: {n-both_wrong}/{n} = {(n-both_wrong)/n:.4f}")

print("\n=== blend sweep: score = (1-w)*head_softmax + w*linear ===")
best = None
for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
    ok = 0
    per = defaultdict(lambda: [0, 0])
    for iid in items:
        hp_ = softmax_T([HL[i] for i in by_item[iid]])
        lp = lin_oof[iid]
        s = [(1 - w) * a + w * b for a, b in zip(hp_, lp)]
        g = int(max(range(len(s)), key=lambda j: s[j]) in gold[iid])
        ok += g
        t = tier(iid)
        per[t][0] += g
        per[t][1] += 1
    acc = ok / n
    pt = {k: v[0] / max(v[1], 1) for k, v in per.items()}
    mark = ""
    if best is None or acc > best[0]:
        best = (acc, w)
    print(f"  w={w:<5} {ok:3d}/{n} = {acc:.4f}   " +
          "  ".join(f"{k}:{pt.get(k,0):.3f}" for k in ("easy", "standard", "hard")))
print(f"\nbest blend: w={best[1]} at {best[0]:.4f}  "
      f"(head alone {head_only/n:.4f}, delta {best[0]-head_only/n:+.4f})")
json.dump({"head": head_only / n, "best_w": best[1], "best_acc": best[0],
           "both_wrong": both_wrong, "n": n},
          open(f"{RES}\\hybrid_linear.json", "w"), indent=2)
print(f"wrote {RES}\\hybrid_linear.json")
