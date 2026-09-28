"""Can EmbeddingGemma-300M make decisions WITHOUT a probe?

We measured that a trained probe on the 0.8B's final hidden state sits at
chance (0.333 vs 0.343). That is a statement about ONE mechanism: read a
hidden vector, learn to map it to correctness.

EmbeddingGemma offers a different mechanism that needs no probe at all -
semantic similarity, zero-shot. For `intent` ("which intent does this message
express?" against four intent descriptions) that is exactly the task the model
was trained for. So it may succeed where the probe failed, for reasons that
have nothing to do with parameter count.

Four scorers, all untrained:
  A  cos(state, option)
  B  cos(question, option)
  C  cos(state+question, option)
  D  cos(state, option) - cos(state, mean(all options))   <- contrastive
"""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

# torch ships single-threaded here; 1 text/sec is a threading artefact, not
# the model's real speed. Without this the embed pass takes 24 minutes.
torch.set_num_threads(os.cpu_count() or 8)
print(f"torch threads: {torch.get_num_threads()} of {os.cpu_count()} logical")

from transformers import AutoModel, AutoTokenizer  # noqa: E402

M = "google/embeddinggemma-300m"
TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")

tok = AutoTokenizer.from_pretrained(M)
model = AutoModel.from_pretrained(M, dtype=torch.float32).eval()
print(f"loaded {sum(p.numel() for p in model.parameters()):,} params")


@torch.no_grad()
def embed(texts, prefix="", bs=64):
    out = []
    for i in range(0, len(texts), bs):
        chunk = [prefix + t for t in texts[i:i + bs]]
        e = tok(chunk, padding=True, truncation=True, max_length=512,
                return_tensors="pt")
        o = model(input_ids=e["input_ids"], attention_mask=e["attention_mask"],
                  return_dict=True)
        h = o.last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
        out.append(torch.nn.functional.normalize(pooled, dim=-1))
    return torch.cat(out, 0)


# rebuild items from the real JevBench pairs
rows = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
jb = [r for r in rows if r["tier"] != "nli"]
items = defaultdict(lambda: {"state": "", "q": "", "opts": []})
for r in jb:
    t = r["text"]
    state = t.split("\nQuestion:")[0].replace("State: ", "", 1)
    rest = t.split("\nQuestion:", 1)[1]
    q, opt = rest.split("\nOption: ", 1)
    it = items[r["item"]]
    if not it["state"]:
        it["state"] = state
        it["q"] = q
    it["opts"].append((opt, r["target"]))
keys = sorted(items)
print(f"items {len(keys)}  options total {sum(len(items[k]['opts']) for k in keys)}")

flat, index = [], []
for k in keys:
    it = items[k]
    base = len(flat)
    flat.append(it["state"])
    flat.append(it["q"])
    flat.append(it["state"] + " " + it["q"])
    for o, _ in it["opts"]:
        flat.append(o)
    index.append((base, len(it["opts"])))

t0 = time.time()
V = embed(flat)
print(f"embedded {len(flat)} texts in {time.time()-t0:.1f}s "
      f"({len(flat)/(time.time()-t0):.0f}/s)  dim {V.shape[1]}")

scorers = {"A cos(state,opt)": lambda s, q, opts, mean_o: s @ opts.T,
           "B cos(question,opt)": lambda s, q, opts, mean_o: q @ opts.T,
           "C cos(state+q,opt)": lambda s, q, opts, mean_o: (s + q) / 2 @ opts.T,
           "D contrastive": lambda s, q, opts, mean_o: (s @ opts.T)
           - (s @ mean_o)}
# D needs the mean option vector, so handle it separately
res = {}
for name in ("A cos(state,opt)", "B cos(question,opt)", "C cos(state+q,opt)"):
    ok = tot = 0
    per = defaultdict(lambda: [0, 0])
    for (base, n), k in zip(index, keys):
        s, q, sq = V[base], V[base + 1], V[base + 2]
        opts = V[base + 3: base + 3 + n]
        sc = scorers[name](s, q, opts, None)
        pick = int(sc.argmax())
        good = items[k]["opts"][pick][1] == 1
        ok += good
        tot += 1
        t = k.split("-")[1] if len(k.split("-")) > 2 else "?"
        per[t][1] += 1
        per[t][0] += good
    res[name] = (ok / tot, per, tot)

ok = tot = 0
per = defaultdict(lambda: [0, 0])
for (base, n), k in zip(index, keys):
    s = V[base]
    opts = V[base + 3: base + 3 + n]
    mean_o = torch.nn.functional.normalize(opts.mean(0, keepdim=True), dim=-1)
    sc = (s @ opts.T) - (float(s @ mean_o.reshape(-1)))
    good = items[k]["opts"][int(sc.argmax())][1] == 1
    ok += good
    tot += 1
    t = k.split("-")[1] if len(k.split("-")) > 2 else "?"
    per[t][1] += 1
    per[t][0] += good
res["D contrastive"] = (ok / tot, per, tot)

chance = sum(1.0 / len(items[k]["opts"]) for k in keys) / len(keys)
print(f"\n{'='*62}\nEmbeddingGemma-300M, ZERO-SHOT on 213 real JevBench items")
print(f"chance = {chance:.4f}\n{'='*62}")
print(f"{'scorer':28} {'acc':>8} {'lift':>7}")
for name, (acc, per, tot) in res.items():
    print(f"{name:28} {acc:>8.4f} {acc/chance:>6.2f}x")

print("\nby question type, best scorer per type:")
best = max(res.items(), key=lambda kv: kv[1][0])
for t in sorted(best[1][1]):
    a, b = best[1][1][t]
    print(f"  {t:16} {a:3}/{b:3} = {a/b:.4f}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "embeddinggemma_zeroshot.json").write_text(json.dumps({
    "model": M, "params": 302863104, "protocol": "zero-shot, no probe, no training",
    "n_items": len(keys), "chance": chance,
    "scorers": {n: {"accuracy": a, "lift": a / chance, "n": t}
                for n, (a, p, t) in res.items()},
    "by_type_best_scorer": {t: {"correct": a, "total": b, "acc": a / b}
                            for t, (a, b) in best[1][1].items()},
    "contrast": "0.8B trained probe = 0.3333 vs 0.3427 chance; "
                "4B trained probe = 0.7080",
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'embeddinggemma_zeroshot.json'}")
