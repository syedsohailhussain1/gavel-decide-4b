"""Four creative, zero-training ways to use the 300M better.

What killed the trained head was 213 examples against a 768-dim space. So
every idea here avoids gradients entirely and either adds no parameters or
adds very few:

  1. ASYMMETRIC TASK PREFIXES. The model was trained with query:/passage:/
     title: prefixes. We only ever tried putting one on the context. Retrieval
     quality depends on the query and passage being prefixed DIFFERENTLY.
  2. MULTI-VIEW RANK FUSION. Several cosine views per item, fused by
     reciprocal rank. One hyper-parameter, no weights to overfit.
  3. PER-FAMILY Z-NORMALISATION. Nine parameters - one mean and scale per
     question type. The trained head had 768 and died; this has 9.
  4. RETRIEVAL-AUGMENTED FEW-SHOT. Use the 300M to retrieve k similar items
     as in-context demonstrations, then score with the generative mechanism.
     This is the real answer to "the model does not know your task": adapt in
     context instead of by gradient. Made HONEST by leave-one-fold-out, so
     demonstrations never include the item being scored or its fold.
"""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(os.cpu_count() or 8)
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

M = "google/embeddinggemma-300m"
TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
SEED, K = 20260924, 5

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
ptype = {k: (k.split("-")[1] if len(k.split("-")) > 2 else "?") for k in keys}
gold = {k: [t for _, t in items[k]["opts"]] for k in keys}
chance = sum(1.0 / len(items[k]["opts"]) for k in keys) / len(keys)
print(f"items {len(keys)}  chance {chance:.4f}")

tok = AutoTokenizer.from_pretrained(M)
enc_model = AutoModel.from_pretrained(M, dtype=torch.float32).eval()
gen_model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()


@torch.no_grad()
def embed(texts, prefix="", bs=32):
    out = []
    for i in range(0, len(texts), bs):
        e = tok([prefix + t for t in texts[i:i + bs]], padding=True,
                truncation=True, max_length=512, return_tensors="pt")
        o = enc_model(input_ids=e["input_ids"],
                      attention_mask=e["attention_mask"], return_dict=True)
        h = o.last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        out.append(torch.nn.functional.normalize(
            (h * m).sum(1) / m.sum(1).clamp_min(1e-9), dim=-1))
    return torch.cat(out, 0)


def acc_of(pick, label, quiet=False):
    ok = sum(1 for k in keys if gold[k][pick(k)] == 1)
    a = ok / len(keys)
    if not quiet:
        print(f"  {label:52} {ok:3}/{len(keys)} = {a:.4f}  lift {a/chance:.2f}x")
    return a


states = [items[k]["state"] for k in keys]
all_opts = [o for k in keys for o, _ in items[k]["opts"]]
opt_index, ptr = {}, 0
for k in keys:
    opt_index[k] = list(range(ptr, ptr + len(items[k]["opts"])))
    ptr += len(items[k]["opts"])

print(f"\nembedding {len(states)} states + {len(all_opts)} options ...")
t0 = time.time()
V_state = embed(states, prefix="query: ")
V_opt = embed(all_opts, prefix="passage: ")
V_state_n = embed(states)
V_opt_n = embed(all_opts)
print(f"  {time.time()-t0:.0f}s  dim {V_state.shape[1]}")

base = {k: [float(V_state[i] @ V_opt[j]) for j in opt_index[k]]
        for i, k in enumerate(keys)}
print(f"\n{'='*72}\n1. ASYMMETRIC PREFIXES  (query: on state, passage: on option)")
print(f"{'='*72}")
base_n = {k: [float(V_state_n[i] @ V_opt_n[j]) for j in opt_index[k]]
          for i, k in enumerate(keys)}
acc_of(lambda k: int(np.argmax(base_n[k])), "no prefix (our previous 0.5446)")
acc_of(lambda k: int(np.argmax(base[k])), "query: on state, passage: on option")

V_sq = embed(states, prefix="query: ")
V_po = embed(all_opts, prefix="query: ")
same = {k: [float(V_sq[i] @ V_po[j]) for j in opt_index[k]]
        for i, k in enumerate(keys)}
acc_of(lambda k: int(np.argmax(same[k])), "query: on both (symmetric, control)")


def rrf(score_sets, kconst=60):
    """Reciprocal rank fusion: rank-based, one hyper-parameter, no weights."""
    out = {}
    for k in keys:
        tot = np.zeros(len(score_sets[k]))
        for s in score_sets:
            r = np.argsort(np.argsort(-np.asarray(s[k])))
            tot += 1.0 / (kconst + 1 + r)
        out[k] = tot
    return out


print(f"\n{'='*72}\n2. MULTI-VIEW RECIPROCAL RANK FUSION")
print(f"{'='*72}")
qtxt = [items[k]["q"] for k in keys]
V_q = embed(qtxt, prefix="query: ")
q_view = {}
for i, k in enumerate(keys):
    q_view[k] = [float(V_q[i] @ V_opt[j]) for j in opt_index[k]]
V_sq_txt = embed([items[k]["state"] + " " + items[k]["q"] for k in keys],
                 prefix="query: ")
sq_view = {}
for i, k in enumerate(keys):
    sq_view[k] = [float(V_sq_txt[i] @ V_opt[j]) for j in opt_index[k]]
V_co = embed([o for k in keys for o, _ in items[k]["opts"]], prefix="title: ")
co_view = {}
ptr = 0
for k in keys:
    co_view[k] = [float(V_state[i] @ V_co[j]) for j in opt_index[k]]
    ptr += len(items[k]["opts"])

acc_of(lambda k: int(np.argmax(q_view[k])), "view: cos(question, option) alone")
for combo, label in (
    ([base, q_view], "RRF  state + question"),
    ([base, sq_view], "RRF  state + state&question"),
    ([base, q_view, sq_view], "RRF  state + question + state&question"),
    ([base, q_view, sq_view, co_view], "RRF  all four views"),
):
    f = rrf(combo)
    acc_of(lambda k, f=f: int(np.argmax(f[k])), label)

print(f"\n{'='*72}\n3. PER-FAMILY Z-NORMALISATION  (9 parameters, not 768)")
print(f"{'='*72}")
allv = {k: np.asarray(base[k], dtype=np.float64) for k in keys}
stats = {}
for t in sorted(set(ptype.values())):
    vs = [allv[k] for k in keys if ptype[k] == t]
    flat = np.concatenate(vs)
    stats[t] = (float(flat.mean()), float(flat.std() + 1e-9))
    print(f"  {t:16} mean {stats[t][0]:+.4f}  sd {stats[t][1]:.4f}  "
          f"(n_items {len(vs)})")
znorm = {k: (allv[k] - stats[ptype[k]][0]) / stats[ptype[k]][1] for k in keys}
acc_of(lambda k: int(np.argmax(znorm[k])), "per-family z-normalised argmax")

f2 = rrf([base, q_view, sq_view])
z2 = {k: (np.asarray(f2[k]) - stats[ptype[k]][0]) / stats[ptype[k]][1]
      for k in keys}
acc_of(lambda k: int(np.argmax(z2[k])), "RRF(3 views) + per-family z-norm")

# ---------------- 4. retrieval-augmented few-shot, leave-one-fold-out
print(f"\n{'='*72}\n4. RETRIEVAL-AUGMENTED FEW-SHOT  (honest: leave-one-fold-out)")
print(f"{'='*72}")
perm = np.random.RandomState(SEED).permutation(len(keys))
fold_of = {keys[i]: perm[j] % K for j, i in enumerate(len(keys))}
print(f"  folds: {K}, so demonstrations never come from the scored item's fold")


def demo_text(k):
    it = items[k]
    gi = [j for j, (_, t) in enumerate(it["opts"]) if t == 1][0]
    wrong = [o for j, (o, t) in enumerate(it["opts"]) if t != 1][:2]
    s = (f"State: {it['state']}\nQuestion: {it['q']}\n"
         f"Answer: {it['opts'][gi][0]}\n")
    for w in wrong:
        s += f"Not: {w}\n"
    return s.strip()


@torch.no_grad()
def gen_score(ctx_list, opt_list, bs=4, ctx=1600):
    out = []
    for i in range(0, len(ctx_list), bs):
        cs, os_ = ctx_list[i:i + bs], opt_list[i:i + bs]
        opt_ids = [tok(" " + o, add_special_tokens=False)["input_ids"] for o in os_]
        enc = tok(cs, add_special_tokens=True)["input_ids"]
        full = [c + oi for c, oi in zip(enc, opt_ids)]
        n_max = max(len(f) - len(c) for f, c in zip(full, enc))
        L = max(len(f) for f in full)
        ids = torch.full((len(full), L), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros((len(full), L), dtype=torch.long)
        for k, f in enumerate(full):
            ids[k, L - len(f):] = torch.tensor(f)
            att[k, L - len(f):] = 1
        lg = gen_model(input_ids=ids, attention_mask=att, return_dict=True,
                       logits_to_keep=n_max + 1).logits.float()
        lp = torch.log_softmax(lg, dim=-1)
        for k in range(len(cs)):
            n_o = len(opt_ids[k])
            idx = torch.tensor([L - n_o + j for j in range(n_o)])
            vals = []
            for j in range(n_o):
                off = (n_max + 1) - (n_o - j)
                vals.append(lp[k, off].gather(0, idx[j].view(1))[0])
            out.append(float(torch.stack(vals).mean()))
    return out


for K_SHOT in (0, 2, 4, 8):
    t0 = time.time()
    SC = defaultdict(list)
    for k in keys:
        if K_SHOT == 0:
            ctxs = [f"State: {items[k]['state']}\nQuestion: {items[k]['q']}\nAnswer:"]
        else:
            pool = [j for j in keys
                    if fold_of[j] != fold_of[k] and ptype[j] == ptype[k]]
            sims = V_state_n[keys.index(k)] @ V_state_n[[keys.index(j) for j in pool]].T
            top = [pool[int(i)] for i in np.argsort(-sims)[:K_SHOT]]
            demos = "\n\n".join(demo_text(t) for t in reversed(top))
            ctxs = [demos + f"\n\nState: {items[k]['state']}\n"
                            f"Question: {items[k]['q']}\nAnswer:"]
        for o, _ in items[k]["opts"]:
            SC[k].append(ctxs[0])
    # score each option against its context
    flat_ctx, flat_opt, owner = [], [], []
    for k in keys:
        for o, _ in items[k]["opts"]:
            flat_ctx.append(SC[k][0])
            flat_opt.append(o)
            owner.append(k)
    sc = gen_score(flat_ctx, flat_opt)
    S = defaultdict(list)
    for k, v in zip(owner, sc):
        S[k].append(v)
    acc_of(lambda k: int(np.argmax(S[k])),
           f"generative + {K_SHOT} retrieved demos (same type, other folds)")
    if K_SHOT == 4:
        json.dump({k: S[k] for k in keys},
                  open(TS / "gemma_fewshot_scores.json", "w"))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "gemma300m_creative.json").write_text(json.dumps({
    "model": M, "n_items": len(keys), "chance": chance,
    "zero_training": True,
    "asymmetric_prefix": {
        "none": 0.5446,
        "query_on_state_passage_on_option": float(
            np.mean([1 if gold[k][int(np.argmax(base[k]))] == 1 else 0
                     for k in keys])),
        "query_on_both": float(
            np.mean([1 if gold[k][int(np.argmax(same[k]))] == 1 else 0
                     for k in keys])),
    },
    "note": "RRF = reciprocal rank fusion over cosine views; per-family "
            "z-normalisation is 9 parameters; few-shot uses leave-one-fold-out "
            "so demonstrations are never from the scored fold",
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'gemma300m_creative.json'}")
