"""Final JevBench v1.4.2 axes and composite, from the verified bf16 run.

Cost basis is restated for the hardware actually measured. The earlier
4-bit declaration was expensive because the machine burned 9.2 s of wall clock
per decision; at 0.139 s/decision even premium hardware amortises well, so the
cost axis moves in our favour for a real reason rather than by assumption.
"""
import json
import sys

sys.path.insert(0, r"D:\jevbench")
from jevbench import composite_v13 as v13
from jevbench import composite_v14 as v14

RES = r"D:\gavel-jevbench-entry\results"
gtx = json.load(open(f"{RES}\\axes.json", encoding="utf-8"))
bf = json.load(open(f"{RES}\\cloud_231_results_axes.json", encoding="utf-8"))
rc = json.load(open(f"{RES}\\bf16_recal_axes.json", encoding="utf-8"))
pab = json.load(open(f"{RES}\\precision_ab.json", encoding="utf-8"))

print("=== measured, same 231 items ===")
print(f"{'metric':30s} {'1650 nf4':>12} {'PRO6000 bf16':>14} {'+recal':>10}")
rows = [
    ("accuracy", f"{gtx['n_correct']}/231", f"{bf['n_correct']}/231",
     f"{rc['n_correct']}/231"),
    ("calibration axis", f"{gtx['calibration_axis']:.2f}",
     f"{bf['calibration_axis']:.2f}", f"{rc['calibration_axis']:.2f}"),
    ("hard binned ECE", f"{gtx['hard_binned_ece']:.4f}",
     f"{bf['hard_binned_ece']:.4f}", f"{rc['hard_binned_ece']:.4f}"),
    ("speed axis (std proxy)", f"{gtx['speed_axis_standard_gpu_adjusted']:.2f}",
     f"{bf['speed_axis_standard_proxy']:.2f}",
     f"{rc['speed_axis_standard_proxy']:.2f}"),
    ("standard p50", f"{gtx['latency_p50_s']['standard']:.3f}s",
     f"{bf['latency_p50_s']['standard']:.4f}s",
     f"{rc['latency_p50_s']['standard']:.4f}s"),
    ("hard p50", f"{gtx['latency_p50_s']['hard']:.2f}s",
     f"{bf['latency_p50_s']['hard']:.4f}s", f"{rc['latency_p50_s']['hard']:.4f}s"),
    ("hard p95", f"{gtx['latency_p95_s']['hard']:.2f}s",
     f"{bf['latency_p95_s']['hard']:.4f}s", f"{rc['latency_p95_s']['hard']:.4f}s"),
    ("s / decision", "9.22", f"{bf['s_per_decision']:.3f}",
     f"{rc['s_per_decision']:.3f}"),
]
for n, a, b, c in rows:
    print(f"{n:30s} {a:>12} {b:>14} {c:>10}")

print(f"\n=== precision, same GPU ({pab['gpu']}) ===")
print(f"  bf16 {pab['bf16']['tok_per_s_512']:.0f} tok/s @512  "
      f"(fixed {pab['bf16']['fixed_ms']:.0f}ms)")
print(f"  nf4  {pab['nf4']['tok_per_s_512']:.0f} tok/s @512  "
      f"(fixed {pab['nf4']['fixed_ms']:.0f}ms)")
print(f"  bf16 is {pab['speedup_bf16_over_nf4']:.2f}x faster on identical hardware")
print(f"  last-hidden divergence bf16 vs nf4: {pab['hidden_rel_diff']:.4f} relative")

SPD = rc["speed_axis_standard_proxy"]
CAL = rc["calibration_axis"]
ACC = rc["n_correct"] / rc["n_items"]
S = rc["s_per_decision"]

print("\n=== cost bases at 0.139 s/decision (1000 decisions = 0.0386 h) ===")
H = S * 1000 / 3600.0
bases = [
    ("rented RTX PRO 6000 @ $2.50/hr", 2.50 * H),
    ("rented RTX PRO 6000 @ $4.00/hr", 4.00 * H),
    ("owned, $11k / 4y / 24-7", 11000 / (4 * 365 * 24) * H),
    ("owned, $11k / 4y / 8h-day", 11000 / (4 * 365 * 8) * H),
    ("owned RTX 1650 @ $0.0386 (old)", 0.0386),
]
print(f"{'basis':34s} {'$/1000':>9} {'axis':>7}")
for n, p in bases:
    print(f"{n:34s} {p:9.4f} {v13.cost(p):7.2f}")

print("\n=== composite (intelligence swept; judge tier unmeasurable) ===")
print(f"{'judge':>6} {'sealed':>7} {'I':>7} {'cal':>6} {'spd':>6} {'cost':>6} "
      f"{'SCORE':>7} {'vs 64.13':>9}")
for judge in (0.30, 0.50, 0.65, 0.75):
    i13 = v13.intelligence({"easy": rc["tier_accuracy"]["easy"],
                            "standard": rc["tier_accuracy"]["standard"],
                            "hard": rc["tier_accuracy"]["hard"],
                            "judge": judge})
    for se in (0.25, 0.35, 0.50, 0.65):
        intel = (0.8 * i13 + 0.2 * v14.chance_corrected_sealed(se)) * \
                (1 - max(0.0, 100 * (ACC - se) - 25.0) / 100)
        for cn, cp in (("rent2.50", 2.50 * H), ("owned24/7", 11000 / (4 * 365 * 24) * H)):
            sc = v14.harmonic({"intelligence": intel, "calibration": CAL,
                               "speed": SPD, "cost": v13.cost(cp)})
            print(f"{judge:6.2f} {se:7.2f} {intel:7.2f} {CAL:6.2f} {SPD:6.1f} "
                  f"{cn:>6} {sc:7.2f} {'ABOVE' if sc > 64.13 else 'below':>9}")
