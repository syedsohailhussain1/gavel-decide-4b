"""Can we lift sol / opus / policy / adequacy on the 300M?

What we have measured so far, all on 213 real items:
    zero-shot cos(state, option)      0.5446   (1.59x chance)
    0.8B probe on hidden states       0.3333   (0.97x - at chance)

Two things were NEVER tried:
  1. training anything on the 300M
  2. probing the 768-dim pooled EMBEDDING rather than raw hidden states

(2) matters because the failure of the 0.8B probe was about hidden states, and
the pooled embedding is a deliberately trained semantic space. A probe there is
a different bet.

Variants, all grouped 5-fold OOF keyed by item, so no option twin leaks:
    Z  zero-shot only (baseline, no training)
    E  trained head on the 768-dim embedding
    H  trained head on [embedding , zero-shot cos features]
    EH trained head on the embedding of a DIFFERENT layer too
"""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

torch.set_num_threads(os.cpu_count() or 8)
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

M = "google/embeddinggemma-300m"
TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
CACHE = TS / "gemma_embeddings.pt"
SEED, EPOCHS, LR, BS, K = 20260924, 80, 1e-3, 64, 5

rows = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
jb = [r for r in rows if r["tier"] != "nli"]
items = defaultdict(lambda: {"state": "", "q": "", "opts": []})
for r in jb:
    t = r["text"]
    state = t.split("\nQuestion:")[0].replace("State: ", "", 1)
    q, opt = t.split("\nQuestion:", 1)[1].split("\nOption: ", 1)
    it = items[r["item"]]
    if not it["state"]:
        it["state"], it["q"] = state, q
    it["opts"].append((opt, r["target"]))
keys = sorted(items)
print(f"items {len(keys)}  pairs {len(jb)}")

flat, index = [], []
for k in keys:
    it = items[k]
    base = len(flat)
    flat += [it["state"], it["q"]] + [o for o, _ in it["opts"]]
    index.append((base, len(it["opts"])))

if CACHE.exists():
    E = torch.load(CACHE, map_location="cpu", weights_only=False)["emb"].float()
    print(f"loaded cached embeddings {tuple(E.shape)}")
else:
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModel.from_pretrained(M, dtype=torch.float32).eval()
    t0 = time.time()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(flat), 32):
            e = tok(flat[i:i + 32], padding=True, truncation=True,
                    max_length=512, return_tensors="pt")
            o = model(input_ids=e["input_ids"],
                      attention_mask=e["attention_mask"], return_dict=True)
            h = o.last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            p = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
            chunks.append(torch.nn.functional.normalize(p, dim=-1))
            if i % 320 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(flat)}  {i/max(el,1e-9):.0f}/s")
    E = torch.cat(chunks, 0)
    torch.save({"emb": E.to(torch.float32), "texts": flat, "model": M}, CACHE)
    print(f"embedded {tuple(E.shape)} in {time.time()-t0:.0f}s -> cached")

HID = E.shape[1]
# per-pair feature rows, aligned to jb order
pair_emb, pair_zs, pair_items, pair_y, pair_type = [], [], [], [], []
for k in keys:
    base, n = index[keys.index(k)]
    s, q = E[base], E[base + 1]
    opts = E[base + 2: base + 2 + n]
    cos = opts @ s
    t = k.split("-")[1] if len(k.split("-")) > 2 else "?"
    for j, (o, tgt) in enumerate(items[k]["opts"]):
        pair_emb.append(opts[j])
        pair_zs.append(float(cos[j]))
        pair_items.append(k)
        pair_y.append(tgt)
        pair_type.append(t)
X = torch.stack(pair_emb)
y = torch.tensor(pair_y, dtype=torch.float32)
zs = torch.tensor(pair_zs)
groups = pair_items
types = pair_type
print(f"pair matrix {tuple(X.shape)}  positives {int(y.sum())}  chance {y.mean():.4f}")

# rank-normalised zero-shot score per item is the real signal, not the raw cos
zn = torch.zeros(len(zs))
for k in keys:
    idx = [i for i, g in enumerate(groups) if g == k]
    v = torch.tensor([zs[i] for i in idx])
    r = torch.argsort(torch.argsort(v)).float() / max(len(idx) - 1, 1)
    for j, i in enumerate(idx):
        zn[i] = r[j]

import numpy as np  # noqa: E402


def folds_for(k, seed=SEED, kk=K):
    u = sorted(set(k))
    perm = np.random.RandomState(seed).permutation(len(u))
    fo = {it: perm[j] % kk for j, it in enumerate(u)}
    return np.array([fo[i] for i in k])


class Head(nn.Module):
    def __init__(self, d, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, w), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_eval(Xin, epochs=EPOCHS, lr=LR, label=""):
    fold = folds_for(groups)
    oof = torch.zeros(len(y))
    for k in range(K):
        tr, va = fold != k, fold == k
        torch.manual_seed(SEED + k)
        m = Head(Xin.shape[1]).train()
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        lf = nn.BCEWithLogitsLoss()
        dl = DataLoader(TensorDataset(Xin[tr], y[tr]), batch_size=BS, shuffle=True)
        for _ in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                lf(m(xb), yb).backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            oof[va] = m(Xin[va])
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    ok = tot = 0
    per = defaultdict(lambda: [0, 0])
    for g, idx in by.items():
        good = bool(y[idx[int(torch.argmax(oof[idx]))]] > 0.5)
        ok += good
        tot += 1
        t = types[idx[0]]
        per[t][1] += 1
        per[t][0] += good
    chance = sum(1.0 / len(v) for v in by.values()) / len(by)
    print(f"\n{label}")
    print(f"  overall {ok}/{tot} = {ok/tot:.4f}   chance {chance:.4f}   "
          f"lift {ok/tot/chance:.2f}x")
    return ok / tot, chance, per, tot


# Z: zero-shot baseline
by = defaultdict(list)
for i, g in enumerate(groups):
    by[g].append(i)
ok = sum(1 for g, idx in by.items()
         if y[max(idx, key=lambda j: float(zs[j]))] == 1)
chance = sum(1.0 / len(v) for v in by.values()) / len(by)
perZ = defaultdict(lambda: [0, 0])
for g, idx in by.items():
    good = y[max(idx, key=lambda j: float(zs[j]))] == 1
    t = types[idx[0]]
    perZ[t][1] += 1
    perZ[t][0] += good
print(f"\n{'='*64}\nall variants, grouped 5-fold OOF on {len(by)} real items"
      f"\n{'='*64}")
print(f"  {'Z  zero-shot':34} {ok/len(by):.4f}  lift {ok/len(by)/chance:.2f}x")

accE, ch, perE, _ = train_eval(X, label="  E  head on 768-dim embedding")
accH, ch, perH, _ = train_eval(
    torch.cat([X, zs[:, None], zn[:, None]], 1),
    label="  H  head on [embedding, zero-shot cos + rank]")

print(f"\n{'per-type':16} {'n':>4} {'Z':>8} {'E':>8} {'H':>8}")
TYPES = sorted(perZ)
for t in TYPES:
    n = perZ[t][1]
    z = perZ[t][0] / n
    e = perE[t][0] / perE[t][1] if perE[t][1] else 0
    h = perH[t][0] / perH[t][1] if perH[t][1] else 0
    flag = "  <-- the four we asked about" if t in ("sol", "opus", "policy", "adequacy") else ""
    print(f"{t:16} {n:>4} {z:>8.4f} {e:>8.4f} {h:>8.4f}{flag}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "gemma300m_trained_probe.json").write_text(json.dumps({
    "model": M, "protocol": "grouped 5-fold OOF keyed by item; unseen items "
                             "only, no JevBench-sealed data anywhere",
    "n_items": len(by), "chance": chance,
    "Z_zero_shot": {"acc": ok / len(by),
                    "by_type": {t: {"correct": perZ[t][0], "total": perZ[t][1],
                                    "acc": perZ[t][0] / perZ[t][1]} for t in TYPES}},
    "E_head_on_embedding": {"acc": accE,
                            "by_type": {t: {"acc": perE[t][0] / perE[t][1]} for t in TYPES}},
    "H_head_with_zeroshot": {"acc": accH,
                             "by_type": {t: {"acc": perH[t][0] / perH[t][1]} for t in TYPES}},
    "reference": {"0.8B_probe_hidden_states": 0.3333, "4B_probe_hidden_states": 0.7080},
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'gemma300m_trained_probe.json'}")
