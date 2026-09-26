"""Decisive test: is our slowness the 4-bit path or the GPU?

Run inside a rented >=16GB VRAM pod. Qwen3-4B is 8.04GB in bf16, so bf16 is
impossible on the GTX 1650 (4GB) and only testable here.

Reports, for bf16 and bitsandbytes nf4 on the SAME card:
  * steady-state prefill throughput (tok/s), with cuda synchronise
  * time = a + b*tokens fit, to separate fixed overhead from compute
  * the fixed per-forward floor, which is what makes N-forwards-per-item hurt
  * whether the two precisions AGREE on decisions

If bf16 is multiples faster and does not regress accuracy, the constraint was
precision-under-memory-pressure, not the model or the algorithm.

ALL TIMING IS CUDA-SYNCHRONISED. The earlier local microbenchmarks were not,
which is why they disagreed with item-level measurements by 2-5x.
"""
import json
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Base"
REPS = 5


def sync_time(fn, reps=REPS):
    fn()  # warm up / autotune
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def fit(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in points)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    return a, b


def profile(lm, dev, tag):
    print(f"\n--- {tag} ---", flush=True)
    print(f"{'tokens':>7} {'ms':>10} {'tok/s':>10}", flush=True)
    pts = []
    for n in (128, 256, 512, 1024, 2048):
        ids = torch.randint(100, 5000, (1, n), device=dev)

        def f(ids=ids):
            with torch.no_grad():
                lm(input_ids=ids, use_cache=False, logits_to_keep=1,
                   return_dict=True)
        t, _ = sync_time(f, 3)
        pts.append((n, t * 1000))
        print(f"{n:7d} {t*1000:10.1f} {n/t:10.0f}", flush=True)
    a, b = fit(pts)
    print(f"fit: ms = {a:.1f} + {b:.3f}*tok   "
          f"(fixed {a:.0f}ms, {1000/b:.0f} tok/s marginal)", flush=True)
    ms512 = a + b * 512
    return {"fixed_ms": a, "ms_per_token": b, "points": pts,
            "ms_at_512": ms512,
            "tok_per_s_512": 512 / (ms512 / 1000.0)}


def main():
    dev = "cuda"
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[mem] {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB",
          flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    out = {"gpu": torch.cuda.get_device_name(0),
           "mem_gib": torch.cuda.get_device_properties(0).total_memory / 2 ** 30}

    # ---- bf16 ----
    lm16 = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0}).eval()
    out["bf16"] = profile(lm16, dev, "bfloat16")

    # decisions under bf16, to compare against 4-bit later
    torch.manual_seed(0)
    probe = {k: v.to(dev) for k, v in tok(
        "State: the sky is blue and water is wet. "
        "Question: what colour is the sky? Option: blue",
        return_tensors="pt", truncation=True, max_length=64).items()}
    with torch.no_grad():
        h16 = lm16(**probe, output_hidden_states=True, logits_to_keep=1,
                   return_dict=True).hidden_states[-1][:, -1].float()
    del lm16
    torch.cuda.empty_cache()

    # ---- nf4 ----
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    lm4 = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map={"": 0}).eval()
    out["nf4"] = profile(lm4, dev, "bitsandbytes nf4")
    with torch.no_grad():
        h4 = lm4(**probe, output_hidden_states=True, logits_to_keep=1,
                 return_dict=True).hidden_states[-1][:, -1].float()

    d = (h16 - h4).abs().max().item()
    rel = d / max(h16.abs().max().item(), 1e-9)
    print(f"\n[precision] last-hidden max abs diff bf16 vs nf4 = {d:.4f} "
          f"(relative {rel:.4f})", flush=True)
    out["hidden_max_abs_diff"] = d
    out["hidden_rel_diff"] = rel
    out["speedup_bf16_over_nf4"] = (out["nf4"]["tok_per_s_512"] /
                                    out["bf16"]["tok_per_s_512"])

    print(f"\n=== RESULT ===")
    print(f"512-token prefill: nf4 {out['nf4']['tok_per_s_512']:.0f} tok/s -> "
          f"bf16 {out['bf16']['tok_per_s_512']:.0f} tok/s  "
          f"= {out['speedup_bf16_over_nf4']:.1f}x", flush=True)
    print(f"fixed per-forward:  nf4 {out['nf4']['fixed_ms']:.0f}ms -> "
          f"bf16 {out['bf16']['fixed_ms']:.0f}ms", flush=True)
    print("For a 5-option item the per-forward floor x5 is "
          f"nf4 {out['nf4']['fixed_ms']*5/1000:.2f}s, "
          f"bf16 {out['bf16']['fixed_ms']*5/1000:.2f}s", flush=True)
    json.dump(out, open("results/precision_ab.json", "w"), indent=2)
    print("wrote results/precision_ab.json", flush=True)


if __name__ == "__main__":
    main()
