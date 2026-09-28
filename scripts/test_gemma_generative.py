"""Third mechanism on the 300M: generative scoring with the LM head.

Two mechanisms had been tried, both on the embedding as a fixed vector:
    bi-encoder cosine similarity      0.5446
    trained probe on the 768-dim vec  0.2113   (worse than chance)

Both ignore the fact that embeddinggemma-300m is a gemma3_text model and
therefore still has an LM head. That gives a THIRD mechanism, a generative
cross-encoder: score each option by the log-likelihood of its own tokens
conditioned on the state and question. No probe, no similarity, no training.

    score(option) = (1/|option|) * sum_t log P(option_t | state, question, option_<t)

Also tested, both of which had been skipped:
  - task prefixes (query: / passage:), which the model was trained with
  - per-family calibration: 9 parameters, not 768, so it cannot overfit the
    way the trained head did
"""
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

torch.set_num_threads(os.cpu_count() or 8)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

M = "google/embeddinggemma-300m"
TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")

rows = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
jb = [r for r in rows if r["tier"] != "nli"]
items = defaultdict(lambda: {"state": "", "q": "", "opts": []})
for r in jb:
    t = r["text"]
    state = t.split("\nQuestion:")[0].replace("State: ", "", 1)
    q, opt = t.split("\nQuestion:", 1)[1].split("\nOption: ", 1)
    it = items[r["item"]]
    if not it["state"]:
        it["state"], it["q"] = state, q
    it["opts"].append((opt, r["target"]))
keys = sorted(items)
print(f"items {len(keys)}  pairs {sum(len(items[k]['opts']) for k in keys)}")

tok = AutoTokenizer.from_pretrained(M)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
print(f"loaded in {time.time()-t0:.0f}s  "
      f"params {sum(p.numel() for p in model.parameters()):,}  "
      f"has lm_head: {hasattr(model, 'lm_head')}")


@torch.no_grad()
def score_pairs(pairs, bs=8, ctx_fmt="State: {s}\nQuestion: {q}\nOption:",
                ctx=512, progress=False):
    """Mean per-token log-likelihood of the option given the context.

    Two things matter here and both cost us a timeout before:

    1. LEFT-pad, so every option ends at the last position. Then
       logits_to_keep = n_opt+1 computes the 262,144-wide head over ~10
       positions instead of 512. Materialising the full head everywhere was
       ~4.3 GB of fp32 logits per batch and never finished.
    2. Left-truncate the CONTEXT to `ctx` tokens exactly like the adapter -
       question and option are never dropped, only old state. Without it one
       item ran to 3683 tokens against a 2048 limit.
    """
    out = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        ctxs = [ctx_fmt.format(s=s, q=q) for s, q, _ in chunk]
        opt_ids = [tok(" " + o, add_special_tokens=False)["input_ids"]
                   for _, _, o in chunk]
        trimmed = []
        for c, oi in zip(ctxs, opt_ids):
            k = ctx - len(oi) - 8
            ids = tok(c, add_special_tokens=False)["input_ids"]
            trimmed.append(tok.decode(ids[-max(k, 16):]) if len(ids) > max(k, 16)
                           else c)
        enc_ctx = tok(trimmed, add_special_tokens=True)["input_ids"]
        full = [c + oi for c, oi in zip(enc_ctx, opt_ids)]
        n_max = max(len(f) - len(c) for f, c in zip(full, enc_ctx))
        L = max(len(f) for f in full)
        ids = torch.full((len(full), L), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros((len(full), L), dtype=torch.long)
        for k, f in enumerate(full):
            ids[k, L - len(f):] = torch.tensor(f)      # LEFT pad
            att[k, L - len(f):] = 1
        # logits at the last n_max+1 positions; position p predicts token p+1
        logits = model(input_ids=ids, attention_mask=att, return_dict=True,
                       logits_to_keep=n_max + 1).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        for k in range(len(chunk)):
            n_o = len(opt_ids[k])
            # option token j sits at position L-n_o+j; its predictor is L-n_o+j-1,
            # which is offset (n_max+1) - (n_o - j) - 1 from the end of logp
            pos = []
            for j in range(n_o):
                abs_pos = L - n_o + j - 1
                off = (n_max + 1) - (L - abs_pos)
                pos.append(off)
            idx = torch.tensor([L - n_o + j for j in range(n_o)])
            tokvals = idx.clamp(0, L - 1)
            vals = [logp[k, off].gather(0, torch.tensor([tokvals[j]]))[0]
                    for j, off in enumerate(pos)]
            out.append(float(torch.stack(vals).mean()))
        if progress and (i // bs) % 20 == 0:
            print(f"    {i + len(chunk)}/{len(pairs)}", flush=True)
    return out


def evaluate(scores_by_item, label):
    ok = tot = 0
    per = defaultdict(lambda: [0, 0])
    for k in keys:
        sc = scores_by_item[k]
        pick = max(range(len(sc)), key=lambda j: sc[j])
        good = items[k]["opts"][pick][1] == 1
        ok += good
        tot += 1
        t = k.split("-")[1] if len(k.split("-")) > 2 else "?"
        per[t][1] += 1
        per[t][0] += good
    ch = sum(1.0 / len(items[k]["opts"]) for k in keys) / len(keys)
    print(f"  {label:44} {ok/tot:.4f}  lift {ok/tot/ch:.2f}x")
    return ok / tot, ch, per


# ---- build the pair list once
pairs = []
for k in keys:
    it = items[k]
    for o, tgt in it["opts"]:
        pairs.append((it["state"], it["q"], o))
print(f"\nscoring {len(pairs)} option-context pairs ...")
print(f"{'='*66}")

t0 = time.time()
plain = score_pairs(pairs, progress=True)
print(f"  ({(time.time()-t0)/len(pairs)*1000:.0f} ms/pair, "
      f"{len(pairs)/(time.time()-t0):.1f}/s)")

R = {k: [] for k in keys}
i = 0
for k in keys:
    for _ in items[k]["opts"]:
        R[k].append(plain[i])
        i += 1
accP, chance, perP = evaluate(R, "P1 log P(option | state, question)")

# ---- with a task prefix on the context
t0 = time.time()
pref = score_pairs(pairs, progress=True, ctx_fmt="query: State: {s}\nQuestion: {q}\nOption:")
R2 = {k: [] for k in keys}
i = 0
for k in keys:
    for _ in items[k]["opts"]:
        R2[k].append(pref[i])
        i += 1
accQ, _, perQ = evaluate(R2, "P2 same, with 'query:' prefix")

# ---- sum of the generative score and the bi-encoder cosine
emb_path = TS / "gemma_embeddings.pt"
if emb_path.exists():
    cache = torch.load(emb_path, map_location="cpu", weights_only=False)
    E = cache["emb"].float()
    flat = cache["texts"]
    lookup = {t: n for n, t in enumerate(flat)}
    zs = {}
    for k in keys:
        s = E[lookup[items[k]["state"]]]
        zs[k] = [float(s @ E[lookup[o]]) for o, _ in items[k]["opts"]]
    accZ, _, perZ = evaluate(zs, "Z  bi-encoder cosine (previous best)")

    print(f"\n  --- ensembling the two mechanisms ---")
    for w in (0.25, 0.5, 0.75):
        R3 = {}
        for k in keys:
            mx = max(zs[k])
            R3[k] = [(1 - w) * a / 10.0 + w * b for a, b in zip(R3.get(k, plain_all(k) if False else R[k]), zs[k])]
        evaluate(R3, f"P+Z  {1-w:.2f} generative + {w:.2f} cosine")

print(f"\n{'per-type':16} {'n':>4} {'P1 gen':>8} {'Z cos':>8}")
ZT = None
if emb_path.exists():
    ZT = perZ
for t in sorted(perP):
    n = perP[t][1]
    a = perP[t][0] / n
    b = (perZ[t][0] / perZ[t][1]) if (ZT and t in ZT) else 0
    print(f"{t:16} {n:>4} {a:>8.4f} {b:>8.4f}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "gemma300m_generative.json").write_text(json.dumps({
    "model": M, "mechanism": "generative cross-encoder: mean per-token "
                             "log P(option | state, question) via the LM head",
    "n_items": len(keys), "chance": chance,
    "P1_generative": {"acc": accP, "by_type": {t: perP[t][0] / perP[t][1]
                                               for t in perP}},
    "P2_with_prefix": {"acc": accQ, "by_type": {t: perQ[t][0] / perQ[t][1]
                                                 for t in perQ}},
    "reference_Z_cosine": 0.5446,
    "reference_E_trained_probe": 0.2113,
    "reference_4B_trained_probe": 0.7080,
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'gemma300m_generative.json'}")

