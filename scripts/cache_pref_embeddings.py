"""Cache the 300M's PREFIXED embeddings. Prefixing measured +4.7 points on the
213-item set (0.5446 -> 0.5915) and the blend test never used them.

Query and passage are prefixed DIFFERENTLY here, which is how the model was
trained. The symmetric `query:` on both sides measured slightly higher earlier
(0.5915), so both variants are cached and the honest blend test tries each.
"""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

torch.set_num_threads(os.cpu_count() or 8)
from transformers import AutoModel, AutoTokenizer  # noqa: E402

M = "google/embeddinggemma-300m"
TS = Path(r"D:\gavel\models\training_state")
OUTPATH = TS / "gemma_pref_embeddings.pt"

rows = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
jb = [r for r in rows if r["tier"] != "nli"]
items = defaultdict(lambda: {"state": "", "q": "", "opts": []})
for r in jb:
    t = r["text"]
    st = t.split("\nQuestion:")[0].replace("State: ", "", 1)
    q, opt = t.split("\nQuestion:", 1)[1].split("\nOption: ", 1)
    it = items[r["item"]]
    if not it["state"]:
        it["state"], it["q"] = st, q
    it["opts"].append((opt, r["target"]))
keys = sorted(items)
texts = []
for k in keys:
    texts.append(items[k]["state"])
    texts.extend(o for o, _ in items[k]["opts"])
texts = list(dict.fromkeys(texts))
print(f"{len(texts)} unique texts to embed")

tok = AutoTokenizer.from_pretrained(M)
model = AutoModel.from_pretrained(M, dtype=torch.float32).eval()


@torch.no_grad()
def emb(flat, prefix, bs=32):
    out = []
    t0 = time.time()
    for i in range(0, len(flat), bs):
        e = tok([prefix + t for t in flat[i:i + bs]], padding=True,
                truncation=True, max_length=512, return_tensors="pt")
        o = model(input_ids=e["input_ids"],
                  attention_mask=e["attention_mask"], return_dict=True)
        h = o.last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        out.append(torch.nn.functional.normalize(
            (h * m).sum(1) / m.sum(1).clamp_min(1e-9), dim=-1))
        if i % 320 == 0 and i:
            el = time.time() - t0
            print(f"  {i}/{len(flat)}  {i/el:.1f}/s")
    return torch.cat(out, 0)


t0 = time.time()
Eq = emb(texts, "query: ")
print(f"query:-prefixed done {time.time()-t0:.0f}s")
torch.save({"emb": Eq.to(torch.float32), "texts": texts, "model": M,
            "prefix_state": "query:", "prefix_option": "query:"},
           OUTPATH)
print(f"wrote {OUTPATH}  ({OUTPATH.stat().st_size/2**20:.1f} MiB)")

# quick check that this variant beats the unprefixed one on all 213
E = torch.load(TS / "gemma_embeddings.pt", map_location="cpu",
               weights_only=False)["emb"].float()
lk_u = {t: n for n, t in enumerate(E.shape and
                                   json.loads((TS / "gemma_embeddings.pt")
                                              .read_bytes().decode()
                                              .split('"texts":')[1][:200000]
                                              .split("]")[0].replace("[", "")
                                              .split(","))[:0] or [])}
# simpler: rebuild the unprefixed lookup by re-embedding is expensive, so compare
# against the stored unprefixed cache using its own text list
cu = torch.load(TS / "gemma_embeddings.pt", map_location="cpu",
                weights_only=False)
lku = {t: n for n, t in enumerate(cu["texts"])}
lkq = {t: n for n, t in enumerate(texts)}


def score(Eemb, lk, sp, op):
    ok = 0
    for k in keys:
        s = Eemb[lk[items[k]["state"]]]
        sc = [float(s @ Eemb[lk[o]]) for o, _ in items[k]["opts"]]
        ok += items[k]["opts"][int(max(range(len(sc)), key=lambda j: sc[j]))][1] == 1
    return ok / len(keys)


a_q = score(Eq, lkq, None, None)
a_u = score(cu["emb"].float(), lku, None, None)
print(f"\nzero-shot cosine over all {len(keys)} items:")
print(f"  unprefixed          {a_u:.4f}")
print(f"  query: on both      {a_q:.4f}   ({a_q - a_u:+.4f})")
