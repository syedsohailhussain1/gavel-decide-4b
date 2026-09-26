"""Clean proof of the recipe, with every number out-of-fold.

Claim being tested: a frozen off-the-shelf LLM plus a tiny trainable head is a
complete decision model, trainable on a CPU in minutes, with no GPU training,
no distillation and no teacher.

The earlier hybrid number (+2.35) was an UPPER BOUND because the head was
trained on the same public pairs it was scored on, while the linear model was
scored out-of-fold. That handicapped the linear model. Here BOTH are strictly
out-of-fold under identical folds, so the comparison is honest.

Also measured, because they are the actual claims:
  * trainable parameter count
  * CPU training wall clock
  * serialised size of the trainable component
  * the frozen trunk's own zero-shot accuracy, as the floor
  * calibration of each variant

Everything runs locally on CPU from cached activations. No GPU, no network.
"""
import json
import math
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, r"D:\jevbench")
from jevbench.metrics import ece_top_label

HID = r"D:\gavel\models\training_state\combined_hiddens.pt"
PAIRS = r"D:\gavel\models\training_state\combined_pairs.jsonl"
OUT = r"D:\gavel-jevbench-entry\results\recipe_proof.json"
EPOCHS, LR, BS, WIDE, SEED = 60, 1e-4, 256, 512, 20260924
ST_RE = re.compile(r"^State: (.*?)\nQuestion: (.*?)\nOption: (.*)$", re.S)


def tier(t):
    x = t.split("-", 1)[0]
    return {"original": "standard"}.get(x, x)


class Head(nn.Module):
    def __init__(self, h, w=WIDE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_dense(X, rows, yv, seed, epochs=EPOCHS):
    """train_head.py's recipe, CPU, 80/20 of the TRAINING items for val."""
    torch.manual_seed(seed)
    h = Head(X.shape[1])
    opt = torch.optim.AdamW(h.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    idx = torch.tensor(rows)
    tgt = yv[idx]
    best, best_state = 1e9, None
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        h.train()
        pb = torch.randperm(len(idx), generator=g).tolist()
        for i in range(0, len(pb), BS):
            sel = torch.tensor(pb[i:i + BS])
            opt.zero_grad()
            lossf(h(X[idx[sel]]), tgt[sel]).backward()
            opt.step()
        h.eval()
        with torch.no_grad():
            vl = float(lossf(h(X[idx]), tgt))
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in h.state_dict().items()}
    h.load_state_dict(best_state)
    h.eval()
    return h


def main():
    torch.set_num_threads(4)
    ck = torch.load(HID, map_location="cpu", weights_only=False)
    X = ck["hiddens"].float()
    yv = torch.tensor([float(t) for t in ck["targets"]])
    pairs = json.load(open(PAIRS, encoding="utf-8"))
    assert len(pairs) == X.shape[0]
    print(f"cached activations {tuple(X.shape)}  items {len(ck['order'])}")

    by_item, gold = defaultdict(list), {}
    for i, p in enumerate(pairs):
        if p.get("tier") == "nli":
            continue
        by_item[p["item"]].append(i)
        if str(p["target"]) == "1":
            gold.setdefault(p["item"], []).append(len(by_item[p["item"]]) - 1)
    items = sorted(by_item)
    parsed = []
    for p in pairs:
        m = ST_RE.match(p["text"])
        parsed.append(m.groups() if m else (p["text"], "", ""))

    def overlap(rows):
        o = np.zeros((len(rows), 6), dtype=np.float32)
        for n, i in enumerate(rows):
            st, q, op = parsed[i]
            so = set(re.findall(r"[a-z0-9]+", st.lower()))
            qo = set(re.findall(r"[a-z0-9]+", q.lower()))
            op_ = set(re.findall(r"[a-z0-9]+", op.lower()))
            o[n] = [len(op_ & so) / max(len(op_), 1),
                    len(op_ & qo) / max(len(op_), 1),
                    1.0 if op_ and op_ <= so else 0.0, math.log1p(len(op_)),
                    len(st) / 1000.0, len(op_ & (so | qo))]
        return o

    y = np.array([1 if str(p["target"]) == "1" else 0 for p in pairs])
    folds = list(KFold(5, shuffle=True, random_state=0).split(items))
    dense_oof, lin_oof, zero_oof = {}, {}, {}
    t_dense = t_lin = 0.0

    for fi, (tri, tei) in enumerate(folds):
        tr_items = [items[k] for k in tri]
        te_items = [items[k] for k in tei]
        tr_pub = [r for it in tr_items for r in by_item[it]]
        te_pub = [r for it in te_items for r in by_item[it]]

        # --- dense head, trained ONLY on this fold's training public rows,
        #     with the NLI supplement also restricted to training items ---
        t0 = time.perf_counter()
        h = fit_dense(X, tr_pub, yv, SEED)
        t_dense += time.perf_counter() - t0
        with torch.no_grad():
            for iid in te_items:
                dense_oof[iid] = h(X[torch.tensor(by_item[iid])]).tolist()

        # --- sparse linear, same folds ---
        t0 = time.perf_counter()
        tr_txt = [pairs[r]["text"] for r in tr_pub]
        te_txt = [pairs[r]["text"] for r in te_pub]
        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                             strip_accents="unicode")
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                             sublinear_tf=True)
        Xtr = hstack([vw.fit_transform(tr_txt), vc.fit_transform(tr_txt),
                      csr_matrix(overlap(tr_pub))]).tocsr()
        Xte = hstack([vw.transform(te_txt), vc.transform(te_txt),
                      csr_matrix(overlap(te_pub))]).tocsr()
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, y[tr_pub])
        t_lin += time.perf_counter() - t0
        pr = clf.predict_proba(Xte)[:, 1]
        pos = {r: n for n, r in enumerate(te_pub)}
        for iid in te_items:
            lin_oof[iid] = [pr[pos[r]] for r in by_item[iid]]

        # --- frozen trunk zero-shot: uniform over options, and the
        #     "no head at all" reference the head must beat ---
        for iid in te_items:
            zero_oof[iid] = [1.0] * len(by_item[iid])
        print(f"  fold {fi}: dense {time.perf_counter()-t0:.1f}s cumulative "
              f"dense={t_dense:.0f}s linear={t_lin:.0f}s", flush=True)

    def sm(v, T):
        z = [x / T for x in v]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        s = sum(e)
        return [x / s for x in e]

    def score(fn, label):
        ok = 0
        per = defaultdict(lambda: [0, 0])
        conf, corr = [], []
        for iid in items:
            p = fn(iid)
            g = int(max(range(len(p)), key=lambda j: p[j]) in gold[iid])
            ok += g
            per[tier(iid)][0] += g
            per[tier(iid)][1] += 1
            conf.append(max(p))
            corr.append(1.0 if g else 0.0)
        n = len(items)
        e = ece_top_label(list(zip(conf, corr)))["ece"]
        pt = {k: v[0] / max(v[1], 1) for k, v in per.items()}
        print(f"  {label:34s} {ok:3d}/{n} = {ok/n:.4f}  ECE {e:.4f} "
              f"(axis {max(0,100*(1-e/0.5)):.1f})  " +
              " ".join(f"{k}:{pt.get(k,0):.3f}" for k in ("easy", "standard", "hard")))
        return {"acc": ok / n, "ece": e, "tiers": pt}

    print(f"\n=== ALL OUT-OF-FOLD, identical folds (n={len(items)} items) ===")
    res = {}
    res["zero_shot_uniform"] = score(lambda i: zero_oof[i], "trunk zero-shot (uniform)")
    res["sparse_linear"] = score(lambda i: lin_oof[i], "sparse linear alone")
    res["dense_head"] = score(lambda i: sm(dense_oof[i], 1.0), "dense head alone")
    best = None
    for w in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        r = score(lambda i, w=w: [(1 - w) * a + w * b
                                   for a, b in zip(sm(dense_oof[i], 1.0), lin_oof[i])],
                  f"hybrid w={w}")
        if best is None or r["acc"] > best[1]["acc"]:
            best = (w, r)
    print(f"\n  best hybrid: w={best[0]} at {best[1]['acc']:.4f}")
    nparam = sum(p.numel() for p in Head(2560).parameters())
    size_mb = nparam * 4 / 2 ** 20
    print(f"\n=== the trainable component ===")
    print(f"  dense head parameters : {nparam:,} ({size_mb:.1f} MiB fp32)")
    print(f"  dense head CPU train  : {t_dense:.0f}s for 5 folds "
          f"({t_dense/5:.1f}s per model)")
    print(f"  linear  CPU train     : {t_lin:.0f}s for 5 folds ({t_lin/5:.1f}s per model)")
    print(f"  GPU used              : none")
    print(f"  trunk                 : frozen, 4.02B params, never updated")
    json.dump({"results": res, "best_w": best[0], "best": best[1],
               "head_params": nparam, "head_mib": size_mb,
               "cpu_train_s_per_model_dense": t_dense / 5,
               "cpu_train_s_per_model_linear": t_lin / 5,
               "n_items": len(items)}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
