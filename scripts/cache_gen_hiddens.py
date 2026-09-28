"""Cache trunk read-outs for the generated pairs. Runs on the pod.

Deliberately uses use_cache=False, a plain full forward per option. The shared
prefix cache was measured to diverge 35% in relative L2 on this architecture
(hybrid linear_attention + full_attention), so it is a serving optimisation
only and must never touch training data - otherwise we would fit a head to a
different function than the one we serve.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--trunk", default="Qwen/Qwen3.5-0.8B-Base")
ap.add_argument("--pairs", default="/root/gc/gen_pairs.jsonl")
ap.add_argument("--out", default="/root/gc/gen_hiddens.pt")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--dtype", default="bf16")
A = ap.parse_args()

rows = [json.loads(l) for l in open(A.pairs, encoding="utf-8")]
if A.limit:
    rows = rows[:A.limit]
print(f"rows to cache: {len(rows):,}", flush=True)

tok = AutoTokenizer.from_pretrained(A.trunk)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
          "fp16": torch.float16, "float16": torch.float16,
          "fp32": torch.float32, "float32": torch.float32}
torch_dtype = DTYPES[A.dtype]
model = AutoModelForCausalLM.from_pretrained(
    A.trunk, dtype=torch_dtype, device_map="auto").eval()
dev = next(model.parameters()).device
print(f"trunk {A.trunk} on {dev}  "
      f"vram {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

# left-truncate, matching the adapter: never lose the option
texts = []
for r in rows:
    t = r["text"]
    parts = t.split("\nQuestion:", 1)
    if len(parts) == 2:
        head = tok(parts[0], truncation=False)["input_ids"]
        budget = A.ctx - len(tok(parts[1], truncation=False)["input_ids"]) - 8
        if len(head) > budget:
            t = tok.decode(head[-max(budget, 16):]) + "\nQuestion:" + parts[1]
    texts.append(t)

enc = [tok(t, truncation=True, max_length=A.ctx)["input_ids"] for t in texts]
order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
print(f"token lengths: min {min(len(e) for e in enc)} "
      f"median {sorted(len(e) for e in enc)[len(enc)//2]} "
      f"max {max(len(e) for e in enc)}", flush=True)

H, targets, items, tiers, t0 = [], [], [], [], time.time()
pad = tok.pad_token_id or 0
done = 0
with torch.no_grad():
    for b in range(0, len(order), A.batch):
        idx = order[b:b + A.batch]
        seqs = [enc[i] for i in idx]
        L = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):          # left pad, read last real token
            inp[k, L - len(s):] = torch.tensor(s)
            att[k, L - len(s):] = 1
        out = model(input_ids=inp.to(dev), attention_mask=att.to(dev),
                    output_hidden_states=True, use_cache=False, return_dict=True)
        hs = out.hidden_states[-1].float()               # [B, L, hidden]
        last = att.sum(1).to(dev) - 1
        vecs = hs[torch.arange(len(seqs), device=dev), last]
        H.append(vecs.to(torch.float16).cpu())
        for i in idx:
            targets.append(int(rows[i]["target"]))
            items.append(rows[i]["item"])
            tiers.append(rows[i]["tier"])
        done += len(idx)
        if done % 6400 < A.batch:
            el = time.time() - t0
            rate = done / el
            print(f"  {done:>7,}/{len(order):,}  {rate:6.0f} rows/s  "
                  f"eta {(len(order)-done)/rate/60:5.1f} min", flush=True)

X = torch.cat(H, 0)
inv = torch.empty(len(order), dtype=torch.long)
for pos, i in enumerate(order):
    inv[i] = pos
X = X[inv]
targets = [targets[i] for i in range(len(order))]
items = [items[i] for i in range(len(order))]
tiers = [tiers[i] for i in range(len(order))]

print(f"\ncached {tuple(X.shape)}  "
      f"{(time.time()-t0)/60:.1f} min", flush=True)
torch.save({"hiddens": X, "hidden": X.shape[1], "trunk": A.trunk,
            "targets": targets, "order": items, "tiers": tiers,
            "ctx": A.ctx, "dtype": A.dtype, "use_cache": False},
           A.out)
print(f"wrote {A.out}  "
      f"({Path(A.out).stat().st_size/2**30:.2f} GiB)", flush=True)
print("DONE", flush=True)
