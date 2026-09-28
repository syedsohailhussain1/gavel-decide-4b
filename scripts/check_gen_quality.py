"""Sanity + ceiling check on the generated pairs, before spending on caching.

Two questions:
  1. Is the task learnable at all from these features? If a head cannot beat
     chance on a 2k sample, generating 400k rows is pointless.
  2. Is the gold label recoverable from the TEXT alone, or only from the
     label string? We want a probe on the trunk's final hidden state, so the
     answer must live in the semantics, not in a giveaway token.

We also measure the trivial baseline: a constant "pick the first option" rule.
If that scores high, the generator has leaked position information the way the
JevBench pairs did.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TS = Path(r"D:\gavel\models\training_state")
SEED = 20260924


def load(path, n=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if n:
        rows = rows[:n]
    return rows


def group_folds(items, k=5, seed=SEED):
    uniq = sorted(set(items))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    fo = {uniq[i]: perm[j] % k for j, i in enumerate(range(len(uniq)))}
    return np.array([fo[i] for i in items])


def train_eval(X, y, groups, epochs=25, lr=1e-4, bs=256, w=512):
    fold = group_folds(groups)
    oof = torch.zeros(len(y))
    hidden = X.shape[1]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden, w), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(w // 2, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    for k in range(5):
        tr, va = fold != k, fold == k
        torch.manual_seed(SEED + k)
        m = Head()
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        lf = nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(SEED + k)
        dl = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=bs, shuffle=True,
                        generator=g)
        for _ in range(epochs):
            m.train()
            for xb, yb in dl:
                opt.zero_grad()
                lf(m(xb), yb).backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            oof[va] = m(X[va])

    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    ok = tot = 0
    per_fam = defaultdict(lambda: [0, 0])
    for g, idx in by.items():
        pick = idx[int(torch.argmax(oof[idx]))]
        good = bool(y[pick] > 0.5)
        ok += good
        tot += 1
        fam = g.split("-")[1]
        per_fam[fam][1] += 1
        per_fam[fam][0] += good
    return ok / tot, per_fam, tot


def position_baseline(rows):
    """Constant 'pick the first option' - the JevBench pairs leaked like this."""
    by = defaultdict(list)
    for r in rows:
        by[r["item"]].append(r)
    ok = sum(1 for g in by.values() if g[0]["target"] == 1)
    return ok / len(by), len(by)


def label_only_baseline(rows):
    """Can the answer be read off the option label alone, with no semantics?

    Fits a bag-of-words logistic regression on the option label text only.
    A high score here means the generator leaked via label wording.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    texts, y, groups = [], [], []
    for r in rows:
        opt = r["text"].split("\nOption: ")[-1].split(":")[0]
        texts.append(opt)
        y.append(r["target"])
        groups.append(r["item"])
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                        min_df=2).fit_transform(texts)
    y = np.asarray(y)
    fold = group_folds(groups)
    oof = np.zeros(len(y))
    for k in range(5):
        # sparse matrices need integer row indices, not a boolean mask
        idx_tr = np.where(fold != k)[0]
        idx_va = np.where(fold == k)[0]
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[idx_tr], y[idx_tr])
        oof[idx_va] = clf.predict_proba(X[idx_va])[:, 1]
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    ok = 0
    for g, idx in by.items():
        best = max(idx, key=lambda j: oof[j])
        if y[best] == 1:
            ok += 1
    return ok / len(by)


if __name__ == "__main__":
    gen = load(TS / "gen_pairs.jsonl")
    print(f"generated rows: {len(gen):,}")

    pos, nitems = position_baseline(gen)
    print(f"\n[leak check] 'always pick the first option' = {pos:.4f} "
          f"over {nitems:,} items   (a leak would show near 1.0)")

    lab = label_only_baseline(gen)
    print(f"[leak check] label-words-only TF-IDF+logreg = {lab:.4f}   "
          f"(a leak would show well above chance)")
