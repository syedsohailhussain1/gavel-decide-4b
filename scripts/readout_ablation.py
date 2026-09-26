"""Which read-out should the Gavel head use? Pure Gavel: frozen trunk, cheap head.

The shipped head reads one vector — the last token of the final layer. This
trains the same MLP head on alternative read-outs of the SAME cached forward
pass, so any difference is the read-out and nothing else. No extra inference
cost: the forward pass already produces every vector we try.

Honest protocol: split BY ITEM (options of an item never straddle the split),
train on the train items, report item-level argmax accuracy on held-out items.
Absolute numbers are lower than the shipped 74.46% because this uses only the
689 public items with no NLI supplement; the comparison BETWEEN read-outs is
the result.
"""
import itertools
import json
import math
import random
import sys

import torch
import torch.nn as nn

RES = r"D:\gavel-jevbench-entry\results"
CTX = 512
HID = 2560


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


D = torch.load(f"{RES}\\readouts_{CTX}.pt", map_location="cpu",
               weights_only=False)
items = D["items"]
seqs = D["seqs"]
VARIANTS = ["last_L0", "last_L-4", "last_L-8", "last_L-16", "last_mid",
            "mean_L0", "max_L0", "mean_L-8"]
for v in VARIANTS:
    D[v] = torch.cat(list(D[v]), dim=0) if isinstance(D[v], list) else D[v]
    D[v] = D[v].float()
print("read-out shapes:", {v: tuple(D[v].shape) for v in ("last_L0", "mean_L0")})

# The cache walked sequences in length-sorted order but did not persist that
# permutation, so stored row j corresponds to seqs[order[j]]. Rebuild the
# texts exactly as the cache did and re-derive `order` deterministically.
texts = []
for f in ("easy", "original", "hard"):
    for line in open(f"D:\\jevbench\\datasets\\public\\{f}.jsonl",
                     encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        q = r["question"]
        descs = q.get("criteria") or {}
        for lab in (r.get("labels") or []):
            d = descs.get(lab) if isinstance(descs, dict) else None
            t = f"State: {r['state']}\nQuestion: {q.get('instructions') or ''}\nOption: {lab}"
            if d:
                t += f": {d}"
            texts.append(t)
assert len(texts) == len(seqs), f"{len(texts)} texts vs {len(seqs)} seqs"
order = sorted(range(len(seqs)), key=lambda i: len(texts[i]))
pos_of_seq = [0] * len(seqs)
for row, s in enumerate(order):
    pos_of_seq[s] = row
print(f"reconstructed order; first 5 seq->row {pos_of_seq[:5]}")

by_item = {}
for i, s in enumerate(seqs):
    by_item.setdefault(s["id"], []).append(pos_of_seq[i])

ids = sorted(by_item)
random.Random(0).shuffle(ids)
n_hold = int(len(ids) * 0.25)
hold, train = set(ids[:n_hold]), set(ids[n_hold:])
print(f"items {len(ids)}  train {len(train)}  holdout {len(hold)}")
print(f"option-sequences {len(seqs)}")

truth = {}
for f in ("easy", "original", "hard"):
    for line in open(f"D:\\jevbench\\datasets\\public\\{f}.jsonl",
                     encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            e = r["expected"]
            truth[r["id"]] = str(e) if isinstance(e, (int, float)) else e


def build(names):
    """Concatenate the named read-outs into one feature matrix."""
    return torch.cat([D[n] for n in names], dim=1)


label_order = {}
for f in ("easy", "original", "hard"):
    for line in open(f"D:\\jevbench\\datasets\\public\\{f}.jsonl",
                     encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            label_order[r["id"]] = r.get("labels") or []

# the cache stored options in the task's own label order, so the position of
# a sequence within its item equals the index of its label
gold_idx = {}
for iid, idxs in by_item.items():
    labs = label_order[iid]
    g = truth[iid]
    gold_idx[iid] = labs.index(g) if g in labs else None
print(f"items with resolvable gold: {sum(1 for v in gold_idx.values() if v is not None)}")

combos = [
    ("last_L0 (SHIPPED)", ["last_L0"]),
    ("mean_L0", ["mean_L0"]),
    ("max_L0", ["max_L0"]),
    ("mean_L-8", ["mean_L-8"]),
    ("last_L-4", ["last_L-4"]),
    ("last_L-8", ["last_L-8"]),
    ("last_L-16", ["last_L-16"]),
    ("last_mid", ["last_mid"]),
    ("last_L0+mean_L0", ["last_L0", "mean_L0"]),
    ("last_L0+mean_L0+max_L0", ["last_L0", "mean_L0", "max_L0"]),
    ("last_L0+mean_L-8", ["last_L0", "mean_L-8"]),
    ("last_L0+last_L-8", ["last_L0", "last_L-8"]),
    ("last_L0+last_L-8+mean_L0", ["last_L0", "last_L-8", "mean_L0"]),
    ("last_L0+last_L-8+mean_L0+max_L0", ["last_L0", "last_L-8", "mean_L0", "max_L0"]),
    ("ALL8", VARIANTS),
]

EPOCHS, LR, SEEDS = 60, 1e-3, 3
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'read-out':32s} {'dim':>6} {'holdout acc':>12} {'easy':>7} {'std':>7} {'hard':>7}")
results = []
for name, names in combos:
    X = build(names)
    dim = X.shape[1]
    accs, per_tier = [], {k: [0, 0] for k in ("easy", "standard", "hard")}
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        head = Head(dim).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
        tr_idx, tr_y = [], []
        for iid in train:
            g = gold_idx[iid]
            if g is None:
                continue
            for j, i in enumerate(by_item[iid]):
                tr_idx.append(i)
                tr_y.append(1.0 if j == g else 0.0)
        tr_idx_t = torch.tensor(tr_idx)
        tr_y_t = torch.tensor(tr_y, dtype=torch.float32)
        Xg = X.to(dev)
        head.train()
        for ep in range(EPOCHS):
            opt.zero_grad()
            logits = head(Xg[tr_idx_t.to(dev)])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, tr_y_t.to(dev))
            loss.backward()
            opt.step()
        head.eval()
        ok = 0
        with torch.no_grad():
            for iid in hold:
                g = gold_idx[iid]
                if g is None:
                    continue
                idxs = by_item[iid]
                lg = head(Xg[torch.tensor(idxs).to(dev)])
                pick = int(torch.argmax(lg).item())
                good = int(pick == g)
                ok += good
                t = tier(iid)
                per_tier[t][0] += good
                per_tier[t][1] += 1
        accs.append(ok)
    mean_ok = sum(accs) / len(accs)
    n_hold_items = sum(1 for iid in hold if gold_idx[iid] is not None)
    pt = {k: (v[0] / max(v[1], 1)) for k, v in per_tier.items()}
    results.append((mean_ok / max(n_hold_items, 1), name))
    print(f"{name:32s} {dim:6d} {mean_ok/max(n_hold_items,1):12.4f} "
          f"{pt.get('easy',0):7.3f} {pt.get('standard',0):7.3f} {pt.get('hard',0):7.3f}")

print("\n=== ranked ===")
for a, n in sorted(results, reverse=True):
    print(f"  {a:.4f}  {n}")
json.dump({"results": [{"acc": a, "readout": n} for a, n in
                       sorted(results, reverse=True)]},
          open(f"{RES}\\readout_ablation.json", "w"), indent=2)
print(f"\nwrote {RES}\\readout_ablation.json")
