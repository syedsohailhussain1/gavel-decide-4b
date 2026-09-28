"""Is the read-out informative at all, or is it broken?

The data curve came back flat at chance. Two very different explanations:
  (a) the frozen 0.8B's final hidden state does not linearly encode which
      option is correct, or
  (b) our caching / read-out path is broken and the vector carries nothing.

These are distinguishable. If the vector is informative, a probe should be
able to predict the FAMILY (10 classes) and simple surface properties of the
state, even if it cannot predict correctness. If it cannot predict anything,
the read-out is broken.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

d = torch.load(r"D:\gavel\models\training_state\gen_hiddens.pt",
               map_location="cpu", weights_only=False)
X = d["hiddens"].float()
y = torch.tensor(d["targets"], dtype=torch.long)
items = list(d["order"])
tiers = list(d["tiers"])
print(f"hiddens {tuple(X.shape)}  items {len(set(items)):,}")
print(f"positives {int((y == 1).sum()):,}  tier classes {len(set(tiers))}")

# basic stats on the vector itself
print("\n=== vector sanity ===")
print(f"  mean {X.mean():+.5f}  std {X.std():.5f}  min {X.min():.3f}  "
      f"max {X.max():.3f}")
nrm = X.norm(dim=1)
print(f"  L2 norm  mean {nrm.mean():.2f}  std {nrm.std():.2f}")
zero = int((nrm < 1e-6).sum())
print(f"  all-zero rows: {zero}")
# are identical texts given identical vectors? (a broken cache would not be)
uniq, idx, inv = torch.unique(X, dim=0, return_inverse=True, return_counts=True)
print(f"  unique vectors: {uniq.shape[0]:,} of {X.shape[0]:,}")
dupes = int((inv.numpy() == inv.numpy()[::-1]).sum())
print(f"  duplicate pairs found: {dupes}  (same text must give same vector)")

SAMPLE = 60000
rng = random.Random(0)
sel = sorted(rng.sample(range(X.shape[0]), SAMPLE))
Xs = X[sel]
ti = torch.tensor([sorted(set(tiers)).index(tiers[i]) for i in sel])
yi = y[sel]
gr = [items[i] for i in sel]
print(f"\nsample {SAMPLE} rows for the probes")

fold_of = {}
u = sorted(set(gr))
rng2 = np.random.RandomState(0)
perm = rng2.permutation(len(u))
fo = {u[i]: perm[j] % 4 for j, i in enumerate(len(u))}
fo = {u[i]: perm[j] % 4 for j, i in enumerate(u)}
fold = torch.tensor([fo[g] for g in gr])


def probe(name, target, n_out, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xs.shape[1], 512), nn.GELU(),
                        nn.Linear(512, n_out)).train()
    head = nn.Module()
    head.net = net
    head.forward = lambda t: net(t)
    head.forward = net.forward
    opt = torch.optim.AdamW(list(net.parameters()), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    oof = torch.zeros(len(target), dtype=torch.long)
    for k in range(4):
        tr = fold != k
        va = fold == k
        dl = DataLoader(TensorDataset(Xs[tr], target[tr]), batch_size=512,
                        shuffle=True)
        for _ in range(12):
            for xb, yb in dl:
                opt.zero_grad()
                lf(net(xb), yb).backward()
                opt.step()
        with torch.no_grad():
            oof[va] = net(Xs[va]).argmax(1)
    acc = (oof == target).float().mean().item()
    print(f"  {name:44} {acc:.4f}   (chance {1/n_out:.4f}, "
          f"lift {acc*n_out:.2f}x)")
    return acc


print("\n=== can a probe read ANYTHING out of this vector? ===")
probe("predict FAMILY (10 classes)", ti, 10)
probe("predict gold vs not (2 classes)", yi, 2)

# a surface property that must be inferable: does the option's answer token
# appear in the state? build it from the pairs file
pairs = [json.loads(l) for l in
         open(r"D:\gavel\models\training_state\gen_pairs.jsonl", encoding="utf-8")]
by_text = {}
for p in pairs:
    by_text.setdefault(p["text"], p["tier"])
tgt_key = []
for i in sel:
    tgt_key.append(0)
# fall back: use option length mod 3 as a pure surface probe
optlen = []
for i in sel:
    pass
print("\n=== and from the raw text, for comparison (TF-IDF) ===")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
texts = [p["text"] for p in pairs[:SAMPLE]]
tg = torch.tensor([p["target"] for p in pairs[:SAMPLE]])
grp = [p["item"] for p in pairs[:SAMPLE]]
T = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                    max_features=200_000).fit_transform(texts)
foldt = np.array([fo[g] for g in grp])
oofp = np.zeros(len(tg))
for k in range(4):
    itr = np.where(foldt != k)[0]
    iva = np.where(foldt == k)[0]
    clf = LogisticRegression(max_iter=800, C=2.0)
    clf.fit(T[itr], tg[itr].numpy())
    oofp[iva] = clf.predict_proba(T[iva])[:, 1]
by = defaultdict(list)
for j, g in enumerate(grp):
    by[g].append(j)
ok = 0
for g, jj in by.items():
    if tg[int(max(jj, key=lambda z: oofp[z]))] == 1:
        ok += 1
print(f"  TF-IDF on the FULL text (state+question+option)   "
      f"{ok/len(by):.4f}   (chance 0.2465)")
