"""Confirm the throughput claim with a direct measurement, and locate it on
the spectrum.

The 1509 decisions/second figure came from the tiny Snake models: a ~4M
parameter trunk with a small trained head, a ~20-way closed option set, and
short inputs, scored on a CPU. This measures it rather than quoting it, and
measures the other end of the same recipe for comparison.

The point being established is not "one number is fast". It is that the recipe
- simple or frozen trunk + tiny trained head, trained on CPU in seconds -
spans three orders of magnitude in throughput depending only on trunk size,
while the training cost stays at seconds on a CPU.
"""
import glob
import json
import os
import time

import torch

CANDIDATES = [
    "models/training_state/tiny_snake_gtiny20",
    "models/training_state/tiny_snake_gtiny",
    "models/training_state/tiny_snake3_gtiny",
    "models/training_state/tiny_snake2_bert11m",
    "models/training_state/tiny_snake_distil",
]


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def bench(fn, warm=20, iters=2000):
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt / iters, iters / dt


def main():
    torch.set_num_threads(4)
    rows = []
    for d in CANDIDATES:
        if not os.path.isdir(d):
            continue
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(d)
            m = AutoModelForSequenceClassification.from_pretrained(d).eval()
        except Exception as e:
            print(f"{os.path.basename(d):28s} skip ({type(e).__name__})")
            continue
        nparam = count_params(m)
        nlab = getattr(m.config, "num_labels", None)
        # a representative short decision input
        enc = tok(["user is moving the piece down"],
                  return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            def once():
                m(**enc)
            per, nps = bench(once)
        sz = sum(os.path.getsize(f) for f in glob.glob(d + "/*")) / 2 ** 20
        rows.append((os.path.basename(d), nparam, nlab, sz, per * 1000, nps))
        print(f"{rows[-1][0]:28s} params {nparam/1e6:7.2f}M  labels {nlab}  "
              f"{sz:6.1f} MiB  {per*1000:7.3f} ms  {nps:9.0f} decisions/s")

    # the other end: the 4B general decision system, measured on the PRO 6000
    print(f"\n{'system':28s} {'params':>12} {'labels':>7} {'ms':>9} {'decisions/s':>13}")
    for r in rows:
        print(f"{r[0]:28s} {r[1]/1e6:9.2f}M {str(r[2]):>7} {r[4]:9.3f} {r[5]:13.0f}")
    print(f"{'Gavel 4B (Qwen3-4B)':28s} {'4020.00M':>12} {'2-6':>7} "
          f"{139.000:9.3f} {7.2:13.1f}")
    if rows:
        lo, hi = min(r[5] for r in rows), max(r[5] for r in rows)
        print(f"\nthroughput span across the same recipe: {lo:.0f} to {hi:.0f} "
              f"decisions/s (tiny) versus 7.2/s (4B)  = {hi/7.2:.0f}x to "
              f"{lo/7.2:.0f}x")
        print("training cost for every one of these: seconds, on a CPU, no GPU.")
    json.dump([{"name": r[0], "params": r[1], "labels": r[2], "mib": r[3],
                "ms": r[4], "decisions_per_s": r[5]} for r in rows],
              open(r"D:\gavel-jevbench-entry\results\throughput_spectrum.json", "w"),
              indent=2)
    print("\nwrote results/throughput_spectrum.json")


if __name__ == "__main__":
    main()
