import json
import os
import subprocess
import sys
from collections import defaultdict

import torch
import torch.nn as nn

RES = r"D:\gavel-jevbench-entry\results"
WORK = r"D:\gavel\models\training_state\readout"
TRAIN = r"D:\gavel\models\training_state\train_head.py"
LAYERS = [20, 21, 36]   # only these were built from the pod cache
# recovered from combined_head_report.json: val_acc 0.6119, best_ep 31
EPOCHS, LR, WIDE, SEED = "60", "1e-4", "512", "20260924"

pairs = json.load(open(r"D:\gavel\models\training_state\combined_pairs.jsonl",
                       encoding="utf-8"))


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


by_item, gold = defaultdict(list), {}
for i, p in enumerate(pairs):
    if p.get("tier") == "nli":
        continue
    by_item[p["item"]].append(i)
    if str(p["target"]) == "1":
        gold.setdefault(p["item"], []).append(len(by_item[p["item"]]) - 1)
items = sorted(by_item)


def evaluate(headfile, hiddens=None):
    hp = torch.load(headfile, map_location="cpu", weights_only=False)
    if "hiddens" in hp:                     # a combined hiddens+head file
        X = hp["hiddens"].float()
    elif hiddens is not None:               # a head file, features supplied
        X = hiddens.float()
    else:
        ck = torch.load(r"D:\gavel\models\training_state\combined_hiddens.pt",
                        map_location="cpu", weights_only=False)
        X = ck["hiddens"].float()
    h = Head(hp["hidden"], hp.get("wide", 512))
    h.load_state_dict(hp["head"])
    h.eval()
    T = float(hp["temperature"])
    with torch.no_grad():
        lg = h(X).tolist()
    ok = 0
    per = defaultdict(lambda: [0, 0])
    for iid, idxs in by_item.items():
        z = [lg[i] / T for i in idxs]
        pick = max(range(len(z)), key=lambda j: z[j])
        g = int(pick in gold[iid])
        ok += g
        t = tier(iid)
        per[t][0] += g
        per[t][1] += 1
    n = sum(v[1] for v in per.values())
    return ok, n, {k: v[0] / max(v[1], 1) for k, v in per.items()}, T


print("=== reference: shipped combined_head.pt on shipped 4-bit cache ===")
ok, n, pt, T = evaluate(r"D:\gavel\models\training_state\combined_head.pt")
print(f"  {ok}/{n} = {ok/n:.4f}  T={T:.4f}  " +
      "  ".join(f"{k}:{pt.get(k,0):.3f}" for k in ("easy", "standard", "hard")))

print(f"\n=== retrained with the RECOVERED recipe ({EPOCHS} ep, lr {LR}) ===")
print(f"{'layer':>6} {'item acc':>9} {'easy':>7} {'std':>7} {'hard':>7} {'T':>8} {'val_acc':>8} {'best_ep':>8}")
res = {}
for L in LAYERS:
    hid = os.path.join(WORK, f"pair_hiddens_L{L}.pt")
    if not os.path.exists(hid):
        print(f"{L:6d}  MISSING {hid}")
        continue
    out = os.path.join(WORK, f"head_r{L}.pt")
    r = subprocess.run([sys.executable, TRAIN, hid, out, EPOCHS, LR, WIDE, SEED],
                       capture_output=True, text=True,
                       cwd=os.path.dirname(TRAIN))
    rep = out.replace(".pt", "_report.json")
    va = be = "-"
    if os.path.exists(rep):
        j = json.load(open(rep))
        va, be = j.get("val_acc"), j.get("best_ep")
    ok, n, pt, T = evaluate(out, torch.load(hid, map_location='cpu', weights_only=False)['hiddens'])
    res[L] = ok / n
    print(f"{L:6d} {ok/n:9.4f} {pt.get('easy',0):7.3f} {pt.get('standard',0):7.3f} "
          f"{pt.get('hard',0):7.3f} {T:8.4f} {str(va):>8} {str(be):>8}"
          f"{'  <-- SHIPPED LAYER' if L == 36 else ''}")

print("\n=== deltas vs L36 on the SAME cache (apples to apples) ===")
for L in LAYERS:
    if L != 36 and L in res and 36 in res:
        print(f"  L{L:<3d} {res[L]:.4f}  delta {res[L]-res[36]:+.4f}")
best = max((k for k in res if k != 36), key=lambda k: res[k])
print(f"\nbest: L{best} {res[best]:.4f}  vs shipped-layer L36 {res[36]:.4f} "
      f"({res[best]-res[36]:+.4f})")
json.dump({"acc": res, "best": best}, open(f"{RES}\\prod_readout_final.json", "w"),
          indent=2)
print(f"wrote {RES}\\prod_readout_final.json")


