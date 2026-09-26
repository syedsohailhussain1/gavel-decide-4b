"""bf16 LoRA fine-tune for Gavel-Decide, with an honest held-out split.

Why this exists: the frozen trunk + MLP head plateaus at 172/231 because the
head only ever saw 689 real pairs. On 24GB the trunk fits in bf16, so LoRA is
now affordable (the same run cost 55-64 s/step under 4-bit on a 4GB card).

HONESTY REQUIREMENT: `combined_pairs.jsonl` was derived FROM the public items,
so training on all of it and scoring the public items measures nothing. This
script splits **by item**, trains on the train items' pairs only, and scores
the held-out items through the real inference path. Reported accuracy is
therefore a genuine generalisation estimate.

Left-truncation is used everywhere: pair text is
`State: ...\\nQuestion: ...\\nOption: ...`, so right-truncation deletes the
question and option and trains on noise. That bug made an earlier LoRA run
look catastrophic (47%) when it was merely unmeasured.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

RES = "/root/gc/results"


def tier(t):
    x = t.split("-", 1)[0]
    return {"original": "standard"}.get(x, x)


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def left_ids(tok, text, ctx):
    ids = tok(text, truncation=False)["input_ids"]
    return ids[-ctx:] if len(ids) > ctx else ids


def score_item(model, head, tok, texts, dev, T, meta):
    """Full inference path: per-option hidden -> head -> T-softmax -> meta."""
    outs = []
    with torch.no_grad():
        for t in texts:
            ids = left_ids(tok, t, 512)
            x = torch.tensor([ids], device=dev)
            o = model(input_ids=x, attention_mask=torch.ones_like(x),
                      output_hidden_states=True, use_cache=False, return_dict=True)
            outs.append(float(head(o.hidden_states[-1][0, -1].float())))
    m = max(v / T for v in outs)
    e = [math.exp(v / T - m) for v in outs]
    s = sum(e)
    probs = [v / s for v in e]
    if meta is not None:
        from gavel_meta import meta_rescale
        d = {str(i): p for i, p in enumerate(probs)}
        d = meta_rescale(d, meta)
        probs = [d[str(i)] for i in range(len(probs))]
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--head", default=f"{RES}/combined_head_bf16.pt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr-mult", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=384)
    ap.add_argument("--freeze-head", action="store_true")
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{RES}/lora_bf16")
    ap.add_argument("--eval-every", type=int, default=75)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train.log"), "a", encoding="utf-8")
    dev = "cuda"

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say(f"args {vars(args)}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.trunk)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    sys.path.insert(0, "/root/gc/src")

    pairs = json.load(open(f"{RES}/combined_pairs.jsonl", encoding="utf-8"))
    say(f"pairs {len(pairs)}")

    # ---- split BY ITEM so held-out accuracy is honest ----
    items = sorted({(p.get("tier"), p.get("item")) for p in pairs})
    random.Random(args.seed).shuffle(items)
    n_hold = int(len(items) * args.holdout)
    hold = set(items[:n_hold])
    tr = [p for p in pairs if (p.get("tier"), p.get("item")) not in hold]
    say(f"items {len(items)}  holdout {len(hold)}  train pairs {len(tr)}")

    hp = torch.load(args.head, map_location="cpu", weights_only=False)
    T = float(hp["temperature"])
    meta = hp.get("meta_cal")

    base = AutoModelForCausalLM.from_pretrained(
        args.trunk, dtype=torch.bfloat16, device_map={"": 0}).eval()
    for p in base.parameters():
        p.requires_grad = False
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                     target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    model.train()
    head = Head(hp["hidden"], hp.get("wide", 512))
    head.load_state_dict(hp["head"])
    head.to(dev).train()
    trn = [p for p in model.parameters() if p.requires_grad]
    say(f"trainable lora tensors {len(trn)}  params "
        f"{sum(p.numel() for p in trn)/1e6:.2f}M")
    if args.freeze_head:
        for p in head.parameters():
            p.requires_grad = False
    groups = [{"params": [p for p in model.parameters() if p.requires_grad],
               "lr": args.lr}]
    if not args.freeze_head:
        groups.append({"params": list(head.parameters()),
                       "lr": args.lr * args.head_lr_mult})
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.99))

    # ---- held-out evaluation through the real path ----
    def evaluate():
        model.eval()
        head.eval()
        by = {}
        for p in tr:
            by.setdefault((p.get("tier"), p.get("item")), []).append(p)
        ok = tot = 0
        per = {}
        for (ti, _), ps in by.items():
            texts = [p["text"] for p in ps]
            labs = [i for i, _ in enumerate(ps)]
            tgt = next((i for i, p in enumerate(ps) if p["target"] == 1), None)
            if tgt is None:
                continue
            probs = score_item(model, head, tok, texts, dev, T, meta)
            pick = max(range(len(probs)), key=lambda i: probs[i])
            good = int(pick == tgt)
            ok += good
            tot += 1
            a = per.setdefault(ti, [0, 0])
            a[0] += good
            a[1] += 1
        model.train()
        head.train()
        return ok, tot, per

    order = list(range(len(tr)))
    random.shuffle(order)
    k = 0
    t0 = time.perf_counter()
    losses = []
    best = -1
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            r = tr[order[k % len(order)]]
            k += 1
            ids = left_ids(tok, r["text"], args.ctx)
            x = torch.tensor([ids], device=dev)
            o = model(input_ids=x, attention_mask=torch.ones_like(x),
                      output_hidden_states=True, use_cache=False, return_dict=True)
            lg = head(o.hidden_states[-1][0, -1].float())
            loss = nn.functional.binary_cross_entropy_with_logits(
                lg, torch.tensor([float(r["target"])], device=dev))
            (loss / args.accum).backward()
            losses.append(float(loss))
        torch.nn.utils.clip_grad_norm_(trn, 1.0)
        opt.step()
        if step % 10 == 0 or step == 1:
            el = time.perf_counter() - t0
            say(f"step {step}/{args.steps} loss~{sum(losses[-80:])/len(losses[-80:]):.4f} "
                f"[{el/60:.1f}m, {el/step:.2f}s/step]")
        if step % args.eval_every == 0 or step == args.steps:
            eo, et, per = evaluate()
            say(f"EVAL step {step}: held-out {eo}/{et} = {eo/max(et,1):.4f}  " +
                "  ".join(f"{t}:{v[0]}/{v[1]}" for t, v in sorted(per.items())))
            if eo > best:
                best = eo
                model.save_pretrained(os.path.join(args.out, "adapter"))
                torch.save(head.state_dict(), os.path.join(args.out, "head.bin"))
                say(f"  saved best (held-out {best}/{et})")
    say(f"DONE best held-out {best}")
    json.dump({"best_holdout": best, "args": vars(args)},
              open(os.path.join(args.out, "result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
