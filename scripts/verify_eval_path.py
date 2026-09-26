"""Isolate the bug: score the SHIPPED head on the SHIPPED cached hiddens.

My reimplementation gets ~0.33 item accuracy; the shipped system gets 0.7446
on the official run and 0.62 on the pair set. Loading the real artifacts
through my own item-level evaluation tells me which half is broken: if the
real head + real hiddens score ~0.6 here, my EVALUATION is fine and my
TRAINING loop is at fault. If they score ~0.33, my EVALUATION is at fault.
"""
import json
import sys
from collections import defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, r"D:\gavel\models\training_state")


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


pairs = json.load(open(r"D:\gavel\models\training_state\combined_pairs.jsonl",
                       encoding="utf-8"))
CK = torch.load(r"D:\gavel\models\training_state\combined_hiddens.pt",
                map_location="cpu", weights_only=False)
HP = torch.load(r"D:\gavel\models\training_state\combined_head.pt",
                map_location="cpu", weights_only=False)
print("hiddens keys:", sorted(CK.keys()))
for k, v in CK.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
print("pairs:", len(pairs), " head hidden:", HP["hidden"], " wide:", HP.get("wide"))

H = CK["hiddens"] if "hiddens" in CK else CK[[k for k in CK if hasattr(CK[k], "shape")][0]]
if hasattr(H, "shape") and H.shape[0] != len(pairs):
    print(f"!! hiddens rows {H.shape[0]} != pairs {len(pairs)}  <-- MISALIGNED")

head = Head(HP["hidden"], HP.get("wide", 512))
head.load_state_dict(HP["head"])
head.eval()
T = float(HP["temperature"])

# item-level argmax over the rows present in combined_pairs
by_item, gold = defaultdict(list), {}
for i, p in enumerate(pairs):
    if p.get("tier") == "nli":
        continue
    by_item[p["item"]].append(i)
    if str(p["target"]) == "1":
        gold.setdefault(p["item"], []).append(len(by_item[p["item"]]) - 1)
print(f"items {len(by_item)}  gold-known {len(gold)}")

with torch.no_grad():
    lg = head(H.float()).tolist()
ok = 0
per = defaultdict(lambda: [0, 0])
for iid, idxs in by_item.items():
    if iid not in gold:
        continue
    z = [lg[i] / T for i in idxs]
    pick = max(range(len(z)), key=lambda j: z[j])
    g = int(pick in gold[iid])
    ok += g
    t = tier(iid)
    per[t][0] += g
    per[t][1] += 1
n = sum(v[1] for v in per.values())
print(f"\nSHIPPED head + SHIPPED hiddens, my item eval: {ok}/{n} = {ok/max(n,1):.4f}")
for k, v in sorted(per.items()):
    print(f"  {k:9s} {v[0]}/{v[1]} = {v[0]/max(v[1],1):.4f}")

# also: pairwise accuracy, which is what the cached report used
pw = sum(1 for i, p in enumerate(pairs)
         if (lg[i] > 0) == (str(p["target"]) == "1"))
print(f"\npairwise agreement: {pw}/{len(pairs)} = {pw/len(pairs):.4f}")
