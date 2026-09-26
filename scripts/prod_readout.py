"""Production read-out comparison, using the REAL train_head.py recipe.

The previous attempt scored 0.33 and was invalid: it trained full-batch, which
for 40 epochs is only 40 gradient steps, so the head was effectively
untrained. train_head.py uses mini-batches of 256 with an 80/20 item-level
split and keeps the best-validation epoch (~320 steps).

Validation gate: this harness must reproduce ~0.82 item accuracy for the
SHIPPED read-out (L36) on the SHIPPED cached hiddens. If it does not, the
harness is still broken and no layer comparison is trustworthy.
"""
import json
import random
import sys
from collections import defaultdict

import torch
import torch.nn as nn

RES = r"D:\gavel-jevbench-entry\results"
LAYERS = [19, 20, 21, 23, 36]
EPOCHS, LR, BS, SEED, WIDE = 40, 3e-4, 256, 0, 512


class Head(nn.Module):
    def __init__(self, h, w=WIDE):
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


pairs = json.load(open(r"D:\gavel\models\training_state\combined_pairs.jsonl",
                       encoding="utf-8"))
D = torch.load(f"{RES}\\pair_ro.pt", map_location="cpu", weights_only=False)
order = D["order"]
row_of = [0] * len(pairs)
for j, p in enumerate(order):
    row_of[p] = j

CK = torch.load(r"D:\gavel\models\training_state\combined_hiddens.pt",
                map_location="cpu", weights_only=False)
REAL = CK["hiddens"].float()
print(f"pairs {len(pairs)}  shipped hiddens {tuple(REAL.shape)}  "
      f"cached order permuted: {CK.get('order') is not None}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
F = {L: torch.cat(list(D[f"last_L{L}"]), 0).float().to(dev) for L in LAYERS}
REAL_D = REAL.to(dev)
y = torch.tensor([1.0 if str(p["target"]) == "1" else 0.0 for p in pairs],
                 device=dev)
rows_t = torch.tensor(row_of, device=dev)

# ALL rows participate in training, exactly as train_head.py does: its
# `order` covers all 2409 rows including the 1,720 NLI ones, so the 80/20
# split and the fit both include NLI. Evaluating on public rows only was
# dropping NLI entirely and scoring 0.385 instead of 0.82.
train_by_item = defaultdict(list)
for i, p in enumerate(pairs):
    train_by_item[str(p.get("item"))].append(i)
train_items = sorted(train_by_item)

# public items, for evaluation only
by_item, gold = defaultdict(list), {}
for i, p in enumerate(pairs):
    if p.get("tier") == "nli":
        continue
    by_item[p["item"]].append(i)
    if str(p["target"]) == "1":
        gold.setdefault(p["item"], []).append(len(by_item[p["item"]]) - 1)
items = sorted(by_item)
print(f"train items {len(train_items)} (incl NLI)   eval items {len(items)} (public)")


def train(X, seed=SEED, epochs=EPOCHS):
    """Exactly train_head.py: 80/20 item split over ALL items (NLI included),
    bs=256, best-validation epoch."""
    rng = random.Random(seed)
    perm = train_items[:]
    rng.shuffle(perm)
    cut = int(len(perm) * 0.8)
    tr_items, va_items = perm[:cut], perm[cut:]
    tr_rows = [i for it in tr_items for i in train_by_item[it]]
    va_rows = [i for it in va_items for i in train_by_item[it]]
    torch.manual_seed(seed)
    h = Head(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(h.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    ytr = y[torch.tensor(tr_rows, device=dev)]
    yva = y[torch.tensor(va_rows, device=dev)]
    tr_t = torch.tensor(tr_rows, device=dev)
    va_t = torch.tensor(va_rows, device=dev)
    best, best_state = 1e9, None
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        h.train()
        pb = torch.randperm(len(tr_rows), generator=g).tolist()
        for i in range(0, len(pb), BS):
            sel = pb[i:i + BS]
            sel_t = torch.tensor(sel, device=dev)
            opt.zero_grad()
            lossf(h(X[tr_t[sel_t]]), ytr[sel_t]).backward()
            opt.step()
        h.eval()
        with torch.no_grad():
            vl = float(lossf(h(X[va_t]), yva))
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in h.state_dict().items()}
    h.load_state_dict(best_state)
    h.eval()
    return h, best


def item_acc(h, X, rowmap):
    ok = 0
    per = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for iid in items:
            idx = torch.tensor([rowmap(i) for i in by_item[iid]], device=dev)
            lg = h(X[idx]).tolist()
            pick = max(range(len(lg)), key=lambda j: lg[j])
            g = int(pick in gold[iid])
            ok += g
            t = tier(iid)
            per[t][0] += g
            per[t][1] += 1
    n = sum(v[1] for v in per.values())
    return ok / max(n, 1), {k: v[0] / max(v[1], 1) for k, v in per.items()}, ok, n, per


print("\n########## GATE: shipped head on shipped hiddens ##########")
HP = torch.load(r"D:\gavel\models\training_state\combined_head.pt",
                map_location="cpu", weights_only=False)
sh = Head(HP["hidden"], HP.get("wide", 512))
sh.load_state_dict(HP["head"])
sh.to(dev).eval()
a, pt, ok, n, _per = item_acc(sh, REAL_D, lambda i: i)   # shipped hiddens: original order
print(f"  {ok}/{n} = {a:.4f}   " +
      "  ".join(f"{k}:{_per[k][0]}/{_per[k][1]}" for k in sorted(_per)))
GATE = 0.75
print(f"  gate (>= {GATE}): {'PASS' if a >= GATE else 'FAIL - harness still broken'}")
if a < GATE:
    print("  aborting: layer comparison would be meaningless")
    sys.exit(1)

# Isolate cache-vs-training: score the SHIPPED head on OUR cache at L36.
# If this is near 0.82 our cache is faithful and the gap above is a training
# artefact; if it is near 0.39 our cache (bf16 @ ctx384) is simply not
# comparable to the shipped 4-bit full-length cache.
a2, pt2, ok2, n2, per2 = item_acc(sh, F[36], lambda i: rows_t[i])
print(f"\n  shipped head on OUR bf16 ctx384 cache @L36: {ok2}/{n2} = {a2:.4f}")
print(f"    -> cache is {'FAITHFUL' if a2 > 0.6 else 'NOT COMPARABLE (bf16/ctx384 vs 4-bit/full)'}")
scale_r = REAL.abs().mean().item()
scale_o = F[36].abs().mean().item()
print(f"    mean |h|  shipped 4-bit cache: {scale_r:.3f}   our bf16 cache: {scale_o:.3f}"
      f"   ratio {scale_o/max(scale_r,1e-9):.2f}")

print(f"\n########## LAYER COMPARISON, real recipe, all public items ##########")
print(f"{'layer':>6} {'acc':>8} {'easy':>7} {'std':>7} {'hard':>7} {'val_loss':>9}")
res = {}
for L in LAYERS:
    h, vl = train(F[L])
    a, pt, ok, n, _per = item_acc(h, F[L], lambda i: rows_t[i])   # our cache: length-sorted
    res[L] = a
    print(f"{L:6d} {a:8.4f} {pt.get('easy',0):7.3f} {pt.get('standard',0):7.3f} "
          f"{pt.get('hard',0):7.3f} {vl:9.5f}"
          f"{'  <-- SHIPPED LAYER' if L == 36 else ''}")

best = max((k for k in res if k != 36), key=lambda k: res[k])
print(f"\nshipped L36: {res[36]:.4f}")
for L in LAYERS:
    if L != 36:
        print(f"  L{L:<3d} {res[L]:.4f}  delta {res[L]-res[36]:+.4f}")
print(f"\nbest: L{best} at {res[best]:.4f} ({res[best]-res[36]:+.4f} vs shipped)")
json.dump({"gate": a, "acc": res, "best": best},
          open(f"{RES}\\prod_readout.json", "w"), indent=2)
print(f"wrote {RES}\\prod_readout.json")




