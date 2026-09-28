"""$0 test: can the 0.8B read-out learn the REAL 689 JevBench pairs?

The generated data was fake volume (190 unique texts), so the 100k curve was
uninformative. The 689 JevBench-derived pairs are genuinely distinct text, so
they are the one dataset we have that can answer the real question:

    does a Qwen3.5-0.8B frozen final hidden state linearly encode
    "which option is correct"?

Same protocol as the 4B measurement (grouped 5-fold OOF keyed by item, same
recipe), so the two are directly comparable. Runs on the local 4GB card.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
SEED, EPOCHS, LR, BS, WIDE, K = 20260924, 60, 1e-4, 256, 512, 5
TRUNK = "Qwen/Qwen3.5-0.8B-Base"

rows = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
jb = [r for r in rows if r["tier"] != "nli"]
print(f"real JevBench pairs: {len(jb)}  items {len({r['item'] for r in jb})}")
print(f"unique texts       : {len({r['text'] for r in jb})}   "
      f"(the generated set had 190 -- this is genuinely distinct)")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(TRUNK)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(TRUNK, dtype=torch.bfloat16).eval().to(dev)
print(f"trunk loaded {time.time()-t0:.0f}s on {dev}")

# left-truncate exactly like the adapter
texts = []
for r in jb:
    t = r["text"]
    parts = t.split("\nQuestion:", 1)
    if len(parts) == 2:
        head = tok(parts[0], truncation=False)["input_ids"]
        budget = 512 - len(tok(parts[1], truncation=False)["input_ids"]) - 8
        if len(head) > budget:
            t = tok.decode(head[-max(budget, 16):]) + "\nQuestion:" + parts[1]
    texts.append(t)

enc = [tok(t, truncation=True, max_length=512)["input_ids"] for t in texts]
order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
H, B = [], 48
pad = tok.pad_token_id or 0
t0 = time.time()
with torch.no_grad():
    for b in range(0, len(order), B):
        idx = order[b:b + B]
        seqs = [enc[i] for i in idx]
        L = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):
            inp[k, L - len(s):] = torch.tensor(s)
            att[k, L - len(s):] = 1
        o = model(input_ids=inp.to(dev), attention_mask=att.to(dev),
                  output_hidden_states=True, use_cache=False, return_dict=True,
                  logits_to_keep=1)
        hs = o.hidden_states[-1].float()
        last = att.sum(1).to(dev) - 1
        H.append(hs[torch.arange(len(seqs), device=dev), last].cpu())
X = torch.cat(H, 0)
inv = torch.empty(len(order), dtype=torch.long)
for pos, i in enumerate(order):
    inv[i] = pos
X = X[inv].float()
y = torch.tensor([r["target"] for r in jb], dtype=torch.float32)
groups = [r["item"] for r in jb]
HID = X.shape[1]
print(f"cached {tuple(X.shape)} in {time.time()-t0:.0f}s  ({len(jb)/(time.time()-t0):.1f} rows/s)")
print(f"unique vectors: {torch.unique(X, dim=0).shape[0]} of {X.shape[0]}")

torch.save({"hiddens": X.to(torch.float16), "targets": y, "order": groups,
            "hidden": HID, "trunk": TRUNK}, TS / "jb08b_hiddens.pt")


class Head(nn.Module):
    def __init__(self, h, w=WIDE):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def folds_for(items, k=K, seed=SEED):
    u = sorted(set(items))
    perm = np.random.RandomState(seed).permutation(len(u))
    fo = {item: perm[j] % k for j, item in enumerate(u)}
    return np.array([fo[i] for i in items])


def train(Xtr, ytr, seed):
    torch.manual_seed(seed)
    m = Head(HID).train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    lf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True, generator=g)
    for _ in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            lf(m(xb), yb).backward()
            opt.step()
    return m.eval()


fold = folds_for(groups)
oof = torch.zeros(len(y))
for k in range(K):
    tr, va = fold != k, fold == k
    m = train(X[tr], y[tr], SEED + k)
    with torch.no_grad():
        oof[va] = m(X[va])

by = defaultdict(list)
for i, g in enumerate(groups):
    by[g].append(i)
ok = tot = 0
pt = defaultdict(lambda: [0, 0])
for g, idx in by.items():
    good = bool(y[idx[int(torch.argmax(oof[idx]))]] > 0.5)
    ok += good
    tot += 1
    t = g.split("-")[1] if len(g.split("-")) > 2 else "?"
    pt[t][1] += 1
    pt[t][0] += good
chance = sum(1.0 / len(v) for v in by.values()) / len(by)
print(f"\n{'='*58}\n0.8B on the 689 real JevBench pairs, grouped 5-fold OOF")
print(f"{'='*58}")
print(f"accuracy {ok}/{tot} = {ok/tot:.4f}   chance {chance:.4f}   "
      f"lift {ok/tot/chance:.2f}x")
print(f"\n4B reference on the same pairs: 0.7080 (137 items)")
print("\nby question type:")
for t in sorted(pt):
    a, b = pt[t]
    print(f"  {t:16} {a:3}/{b:3} = {a/b:.4f}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "qwen35_08b_jevbench_oof.json").write_text(json.dumps({
    "trunk": TRUNK, "hidden": HID, "n_pairs": len(jb),
    "n_items": tot, "n_correct": ok, "accuracy": ok / tot,
    "chance": chance, "lift_over_chance": ok / tot / chance,
    "by_type": {t: {"correct": a, "total": b, "acc": a / b}
                for t, (a, b) in sorted(pt.items())},
    "protocol": "grouped 5-fold OOF keyed by item, same recipe as the 4B run",
    "reference_4b_same_pairs": 0.7080,
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'qwen35_08b_jevbench_oof.json'}")
