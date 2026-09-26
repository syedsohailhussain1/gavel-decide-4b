"""Recompute tok_per_s_512 in precision_ab.json from the stored fit points.

The pod run of cloud_precision_ab.py wrote a summary block with a units bug:
`1000 / (a + b*512)` treats the fit (which is in milliseconds) as if it were
seconds, understating throughput ~500x. The raw per-length timings were always
correct, so the fix is a pure recomputation from those points.
"""
import json

P = r"D:\gavel-jevbench-entry\results\precision_ab.json"
p = json.load(open(P, encoding="utf-8"))
for k in ("bf16", "nf4"):
    d = p[k]
    pts = d["points"]
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    ms = a + b * 512
    d["ms_at_512"] = ms
    d["tok_per_s_512"] = 512 / (ms / 1000.0)
    print(f"{k:5s} fit ms = {a:.1f} + {b:.3f}*tok   @512 = {ms:.1f} ms  "
          f"-> {d['tok_per_s_512']:.0f} tok/s")
p["speedup_bf16_over_nf4"] = p["bf16"]["tok_per_s_512"] / p["nf4"]["tok_per_s_512"]
p["note"] = ("tok_per_s_512 recomputed from the stored fit. The pod run's "
             "summary block had a units bug (1000/ms instead of n/(ms/1000)); "
             "the raw per-length timings were always correct.")
json.dump(p, open(P, "w", encoding="utf-8"), indent=2)
print(f"\nbf16 is {p['speedup_bf16_over_nf4']:.2f}x nf4 on the same GPU")
print(f"last-hidden divergence {p['hidden_rel_diff']:.4f} relative")
