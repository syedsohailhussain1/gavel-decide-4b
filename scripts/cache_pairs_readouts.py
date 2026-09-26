"""Cache read-outs for an arbitrary pairs file, at chosen layers only.

The depth sweep proved the read-out matters more than anything else we have
tested, so the production comparison needs the FULL training recipe (689 real
pairs + 1,720 NLI supplement) at both the shipped layer and the winning one,
trained and scored identically.
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="19,20,21,23,36")
    ap.add_argument("--ctx", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda"
    layers = [int(x) for x in args.layers.split(",")]

    tok = AutoTokenizer.from_pretrained(args.trunk)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.trunk, dtype=torch.bfloat16, device_map={"": 0}).eval()
    nlayer = model.config.num_hidden_layers
    print(f"nlayer={nlayer} caching layers {layers} at ctx={args.ctx}", flush=True)

    rows = json.load(open(args.pairs, encoding="utf-8"))
    print(f"pairs {len(rows)}", flush=True)
    store = {f"last_L{L}": [] for L in layers}
    order = sorted(range(len(rows)),
                   key=lambda i: len(rows[i]["text"]))
    start = time.perf_counter()
    with torch.no_grad():
        for b in range(0, len(order), args.batch):
            chunk = [order[j] for j in range(b, min(b + args.batch, len(order)))]
            enc = []
            for i in chunk:
                ids = tok(rows[i]["text"], truncation=False)["input_ids"]
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
            bidx = torch.arange(len(enc), device=dev)
            lastidx = am.sum(1) - 1
            for LL in layers:
                store[f"last_L{LL}"].append(
                    o.hidden_states[LL][bidx, lastidx].to(torch.float16).cpu())
            if (b // args.batch) % 50 == 0:
                print(f"  {b+len(chunk)}/{len(order)} "
                      f"{time.perf_counter()-start:.0f}s", flush=True)

    torch.save({**store, "order": order, "nlayer": nlayer, "ctx": args.ctx,
                "n": len(rows), "layers": layers}, args.out)
    print(f"wrote {args.out} in {time.perf_counter()-start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
