"""Build hiddens files in train_head.py's exact format, then CALL the real trainer.

Reimplementing the recipe produced 0.30-0.44 where the shipped head scores
0.8216 on the same features, so the reimplementation is the unreliable part.
train_head.py takes a hiddens file as argv[1] and expects exactly:

    {"hiddens": FloatTensor[n, 2560], "targets": [0/1, ...],
     "order": [item_name, ...], "hidden": 2560, "model": str}

We have every one of those, so we build the file and let the real trainer do
the work. No reimplementation, so no reimplementation bugs.
"""
import json
import os
import subprocess
import sys

import torch

RES = r"D:\gavel-jevbench-entry\results"
TRAIN = r"D:\gavel\models\training_state\train_head.py"
LAYERS = [20, 21, 36]
EPOCHS, LR, WIDE, SEED = "40", "3e-4", "512", "20260924"

pairs = json.load(open(r"D:\gavel\models\training_state\combined_pairs.jsonl",
                       encoding="utf-8"))
D = torch.load(f"{RES}\\pair_ro.pt", map_location="cpu", weights_only=False)
order = D["order"]
row_of = [0] * len(pairs)
for j, p in enumerate(order):
    row_of[p] = j
# put our cached read-outs back into ORIGINAL pair order
targets = [1 if str(p["target"]) == "1" else 0 for p in pairs]
item_names = [str(p.get("item")) for p in pairs]
HID = 2560

work = r"D:\gavel\models\training_state\readout"
os.makedirs(work, exist_ok=True)
made = {}
for L in LAYERS:
    X = torch.cat(list(D[f"last_L{L}"]), 0).float()          # length-sorted order
    # order[j] = pair index at sorted position j, so X[j] belongs to
    # pair order[j]. Unpack as (sorted_position, pair_index) — swapping these
    # scrambles every row and silently destroys the comparison.
    Xo = torch.empty_like(X)
    for j, p in enumerate(order):
        Xo[p] = X[j]                                          # -> original order
    path = os.path.join(work, f"pair_hiddens_L{L}.pt")
    torch.save({"hiddens": Xo.half(), "targets": targets, "order": item_names,
                "hidden": HID, "model": "Qwen3-4B-Base",
                "readout_layer": L, "cache": "bf16 ctx384 RTX PRO 6000"},
               path)
    made[L] = path
    print(f"wrote {path}  {tuple(Xo.shape)}")

for L in LAYERS:
    out = os.path.join(work, f"pair_head_L{L}.pt")
    print(f"\n=== real train_head.py on L{L} ===", flush=True)
    r = subprocess.run(
        [sys.executable, TRAIN, made[L], out, EPOCHS, LR, WIDE, SEED],
        capture_output=True, text=True, cwd=os.path.dirname(TRAIN))
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
print("\ndone")
