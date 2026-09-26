"""Full depth x context sweep for the Gavel read-out. Same cached forward.

The ablation showed the shipped read-out (final layer, last token) is the worst
of fifteen. This sweeps every layer 0..36 at three context lengths, plus the
pooled variants, to find the real optimum and to test whether a better
read-out also unlocks longer context.

Protocol is unchanged from readout_ablation.py: split BY ITEM, identical
head, identical optimiser, identical seeds. Only the read-out changes.
"""
import json
import random
import sys

import torch
import torch.nn as nn

RES = r"D:\gavel-jevbench-entry\results"
CTXS = (512, 1024, 2048)
SEEDS, EPOCHS, LR = 3, 60, 1e-3


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def tier(t):
    x = t.split("-", 1)[0]
    return {"original": "standard"}.get(x, x)


truth, label_order = {}, {}
for f in ("easy", "original", "hard"):
    for line in open(f"D:\\jevbench\\datasets\\public\\{f}.jsonl",
                     encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            e = r["expected"]
            truth[r["id"]] = str(e) if isinstance(e, (int, float)) else e
            label_order[r["id"]] = r.get("labels") or []

dev = "cuda" if torch.cuda.is_available() else "cpu"
all_rows = []

for CTX in CTXS:
    D = torch.load(f"{RES}\\ro_{CTX}.pt", map_location="cpu", weights_only=False)
    seqs, items = D["seqs"], D["items"]
    order = D["order"]                       # persisted this time
    pos = [0] * len(seqs)
    for row, s in enumerate(order):
        pos[s] = row
    by_item = {}
    for i, s in enumerate(seqs):
        by_item.setdefault(s["id"], []).append(pos[i])
    gold = {}
    for iid, idxs in by_item.items():
        labs = label_order[iid]
        g = truth.get(iid)
        gold[iid] = labs.index(g) if g in labs else None
    ids = sorted(by_item)
    random.Random(0).shuffle(ids)
    n_hold = int(len(ids) * 0.25)
    hold, train = set(ids[:n_hold]), set(ids[n_hold:])

    tr_idx, tr_y = [], []
    for iid in train:
        g = gold[iid]
        if g is None:
            continue
        for j, i in enumerate(by_item[iid]):
            tr_idx.append(i)
            tr_y.append(1.0 if j == g else 0.0)
    tr_idx_t = torch.tensor(tr_idx).to(dev)
    tr_y_t = torch.tensor(tr_y, dtype=torch.float32, device=dev)
    n_hold_items = sum(1 for iid in hold if gold[iid] is not None)

    feats = {}
    for L in range(D["nlayer"] + 1):
        feats[f"last_L{L}"] = torch.cat(list(D[f"last_L{L}"]), 0).float()
    for k in ("mean_L0", "max_L0", "mean_L-16", "max_L-16", "mean_L-8"):
        feats[k] = torch.cat(list(D[k]), 0).float()

    def run(names):
        X = torch.cat([feats[n] for n in names], 1).to(dev)
        dim = X.shape[1]
        accs, tiers = [], {k: [0, 0] for k in ("easy", "standard", "hard")}
        for seed in range(SEEDS):
            torch.manual_seed(seed)
            h = Head(dim).to(dev)
            o = torch.optim.AdamW(h.parameters(), lr=LR, weight_decay=1e-4)
            h.train()
            for _ in range(EPOCHS):
                o.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    h(X[tr_idx_t]), tr_y_t)
                loss.backward()
                o.step()
            h.eval()
            ok = 0
            with torch.no_grad():
                for iid in hold:
                    g = gold[iid]
                    if g is None:
                        continue
                    lg = h(X[torch.tensor(by_item[iid]).to(dev)])
                    good = int(int(torch.argmax(lg).item()) == g)
                    ok += good
                    t = tier(iid)
                    tiers[t][0] += good
                    tiers[t][1] += 1
            accs.append(ok / max(n_hold_items, 1))
        return (sum(accs) / len(accs),
                {k: v[0] / max(v[1], 1) for k, v in tiers.items()})

    print(f"\n########## ctx={CTX}  (holdout {n_hold_items} items) ##########")
    print(f"{'read-out':16s} {'acc':>7} {'easy':>7} {'std':>7} {'hard':>7}")
    rows = []
    for L in range(D["nlayer"] + 1):
        a, pt = run([f"last_L{L}"])
        rows.append((a, f"last_L{L}", pt))
    for k in ("mean_L0", "max_L0", "mean_L-16", "max_L-16", "mean_L-8"):
        a, pt = run([k])
        rows.append((a, k, pt))
    rows.sort(reverse=True)
    for a, n, pt in rows:
        mark = "  <-- SHIPPED" if n == f"last_L{D['nlayer']}" else ""
        print(f"{n:16s} {a:7.4f} {pt.get('easy',0):7.3f} "
              f"{pt.get('standard',0):7.3f} {pt.get('hard',0):7.3f}{mark}")
    for a, n, pt in rows:
        all_rows.append({"ctx": CTX, "readout": n, "acc": a, **pt})
    del D, feats
    torch.cuda.empty_cache()

print("\n================ GLOBAL TOP 15 ================")
all_rows.sort(key=lambda r: -r["acc"])
for r in all_rows[:15]:
    print(f"  ctx {r['ctx']:5d}  {r['readout']:14s} acc {r['acc']:.4f}  "
          f"hard {r.get('hard',0):.3f}")
json.dump(all_rows, open(f"{RES}\\depth_ctx_sweep.json", "w"), indent=2)
print(f"\nwrote {RES}\\depth_ctx_sweep.json")
