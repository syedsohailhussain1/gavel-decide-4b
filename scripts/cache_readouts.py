"""Cache multi-read-out hidden states for every option of every public item.

The Gavel head currently reads ONE vector: the last token of the final layer.
For long states whose decisive evidence sits mid-document, that throws the
evidence away. Mean/max pooling and earlier layers cost nothing extra — the
forward pass already computes them — so this is a pure read-out experiment
with zero inference cost if it wins.

Per option we store, at each requested context length:
  last_L{0,-4,-8,-16,mid}   last-token hidden at five depths
  mean_L0, max_L0           pooled over the sequence at the final layer
  mean_L-8                  pooled at a mid depth

All fp16. ~1000 option sequences x 8 vectors x 2560 dims x 2 bytes is small.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RES = "/root/gc/results"


def tier(t):
    x = t.split("-", 1)[0]
    return {"original": "standard"}.get(x, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(args.trunk)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.trunk, dtype=torch.bfloat16, device_map={"": 0}).eval()

    nlayer = model.config.num_hidden_layers
    depths = {"L0": nlayer, "L-4": nlayer - 4, "L-8": nlayer - 8,
              "L-16": nlayer - 16, "mid": nlayer // 2}
    print(f"layers={nlayer} depths={depths}", flush=True)

    meta = []
    seqs = []          # (item_id, option_index, text, is_gold)
    for f in ("easy", "original", "hard"):
        for line in open(f"/root/jevbench/datasets/public/{f}.jsonl",
                         encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = r["question"]
            labels = r.get("labels") or []
            if not labels:
                continue
            descs = q.get("criteria") or {}
            exp = r.get("expected")
            exp = str(exp) if isinstance(exp, (int, float)) else exp
            opts = []
            for lab in labels:
                d = descs.get(lab) if isinstance(descs, dict) else None
                t = f"State: {r['state']}\nQuestion: {q.get('instructions') or ''}\nOption: {lab}"
                if d:
                    t += f": {d}"
                opts.append(t)
            meta.append({"id": r["id"], "tier": tier(r["id"]),
                         "n_opts": len(opts), "gold": exp})
            for i, t in enumerate(opts):
                seqs.append((r["id"], i, t, opts[i].endswith(exp) if exp else False))
    print(f"items={len(meta)} option-sequences={len(seqs)}", flush=True)

    store = {}
    for L in range(nlayer + 1):
        store[f"last_L{L}"] = []
    for name in ("mean_L0", "max_L0", "mean_L-16", "max_L-16", "mean_L-8"):
        store[name] = []
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][2]))
    import time
    start = time.perf_counter()
    with torch.no_grad():
        for b in range(0, len(order), args.batch):
            chunk = [order[j] for j in range(b, min(b + args.batch, len(order)))]
            enc = []
            for i in chunk:
                ids = tok(seqs[i][2], truncation=False)["input_ids"]
                enc.append(ids[-args.ctx:] if len(ids) > args.ctx else ids)
            L = max(len(e) for e in enc)
            inp = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
            am = torch.zeros((len(enc), L), dtype=torch.long)
            for j, e in enumerate(enc):
                inp[j, :len(e)] = torch.tensor(e)
                am[j, :len(e)] = 1
            inp, am = inp.to(dev), am.to(dev)
            o = model(input_ids=inp, attention_mask=am, output_hidden_states=True,
                      use_cache=False, logits_to_keep=1, return_dict=True)
            hs = o.hidden_states
            lastidx = am.sum(1) - 1
            bidx = torch.arange(len(enc), device=dev)
            for L_ in range(nlayer + 1):
                store[f"last_L{L_}"].append(
                    hs[L_][bidx, lastidx].to(torch.float16).cpu())
            m = am.unsqueeze(-1).float()
            for tag, d in (("L0", nlayer), ("L-16", nlayer - 16), ("L-8", nlayer - 8)):
                v = hs[d].float()
                store[f"mean_{tag}"].append(
                    ((v * m).sum(1) / m.sum(1)).to(torch.float16).cpu())
                if tag != "L-8":
                    store[f"max_{tag}"].append(
                        v.masked_fill(m == 0, -1e4).max(1).values.to(torch.float16).cpu())
            if (b // args.batch) % 40 == 0:
                el = time.perf_counter() - start
                print(f"  {b+len(chunk)}/{len(order)}  {el:.0f}s", flush=True)

    out = {"ctx": args.ctx, "nlayer": nlayer, "order": order, "items": meta,
           "seqs": [{"id": s[0], "opt": s[1]} for s in seqs],
           "n_seqs": len(seqs)}
    torch.save({**store, **out}, args.out)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/2**20:.1f} MiB)  "
          f"in {time.perf_counter()-start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
