import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

M = "google/embeddinggemma-300m"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(M)
model = AutoModel.from_pretrained(M, dtype=torch.float32)
model.eval()
print(f"loaded in {time.time()-t0:.0f}s")
print(f"  params {sum(p.numel() for p in model.parameters()):,}")
cfg = model.config
print(f"  hidden {cfg.hidden_size}  layers {cfg.num_hidden_layers}  "
      f"vocab {cfg.vocab_size}  heads {cfg.num_attention_heads}  "
      f"maxpos {getattr(cfg, 'max_position_embeddings', '?')}")

# EmbeddingGemma uses mean pooling over the final layer + L2 normalisation.
# Google also documents task prefixes; retrieval is the relevant one here.
def embed(texts, prefix="query: "):
    toks = [tok(prefix + t, padding=True, truncation=True,
                max_length=2048, return_tensors="pt") for t in texts]
    L = max(t["input_ids"].shape[1] for t in toks)
    ids = torch.full((len(toks), L), tok.pad_token_id or 0, dtype=torch.long)
    att = torch.zeros((len(toks), L), dtype=torch.long)
    for i, t in enumerate(toks):
        n = t["input_ids"].shape[1]
        ids[i, L - n:] = t["input_ids"][0]      # left pad
        att[i, L - n:] = 1
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=att, return_dict=True)
    # left padding -> use the attention mask to mean-pool correctly
    h = o.last_hidden_state
    m = att.unsqueeze(-1).float()
    pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
    return torch.nn.functional.normalize(pooled, dim=-1)

print("\n=== semantic sanity: are similar texts closer? ===")
A = "Where is my package? It was due four days ago."
B = "My order has not arrived yet, when will it come?"
C = "A package marked fragile arrived with a cracked corner."
D = "Please cancel order #5521, I no longer need it."
v = embed([A, B, C, D])
names = "order-late-A order-late-B damaged-C cancel-D".split()
import math
cos = lambda i, j: float(v[i] @ v[j])
print(f"  cos(A,B) order-late pair      = {cos(0,1):.4f}   <- expect HIGH")
print(f"  cos(A,C) late vs damaged      = {cos(0,2):.4f}")
print(f"  cos(A,D) late vs cancel       = {cos(0,3):.4f}   <- expect LOW")
print(f"  cos(C,D) damaged vs cancel    = {cos(2,3):.4f}")
print(f"  dims {tuple(v.shape)}  norm {v.norm(dim=1)[0]:.4f}")
print("\n  mean-pooled+normalised:", tuple(v.shape))
print(f"  {time.time()-t0:.0f}s total")

p = Path(r"D:\gavel\models\training_state\embedding_smoke.json")
p.write_text(json.dumps({
    "model": M, "hidden": cfg.hidden_size, "layers": cfg.num_hidden_layers,
    "params": sum(x.numel() for x in model.parameters()),
    "pooling": "mean over attention mask, then L2 normalise",
    "prefix": "query: ",
    "cos_same_intent": cos(0, 1), "cos_different_topic": cos(0, 3),
    "emb_dim": int(v.shape[1]),
}, indent=2), encoding="utf-8")
print(f"wrote {p}")
