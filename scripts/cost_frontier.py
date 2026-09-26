"""Is 'intelligence at zero cost' reachable? Cost axis as a function of
(latency x hardware capital), and what saturates it.

JevBench cost axis = 100 - 30*log10(usd_per_1000 / 0.001), capped at 100.
usd_per_1000 is amortised machine time, so it is a pure function of how long the
hardware takes to make 1000 decisions and what that hardware cost.

This computes the break-even latency for the axis to saturate, across hardware
tiers, and what our measured 0.139 s/decision buys today.
"""
import json
import sys

sys.path.insert(0, r"D:\jevbench")
from jevbench import composite_v13 as v13
from jevbench import composite_v14 as v14

OURS_S = 0.139          # measured, RTX PRO 6000, bf16, prefix cache on
LEADER_S = 0.0323       # decider-ai 1.5.0 median on one B300
SAT = 0.001             # usd per 1000 decisions = axis 100

print("=== cost axis saturation: break-even latency by hardware ===")
print("(4-year life, 24/7 dedicated duty)\n")
hw = [("RTX 3090 24GB (used)", 600), ("RTX 4090 24GB", 1700),
      ("RTX PRO 6000 24GB", 11000), ("H100 80GB", 30000)]
print(f"{'hardware':22} {'$/hr':>9} {'s/dec @axis 100':>18} {'our 0.139s':>12} "
      f"{'leader 0.032s':>14}")
for name, capex in hw:
    rate = capex / (4 * 365 * 24)
    s_sat = SAT / (rate * 1000 / 3600)
    a_ours = v13.cost(rate * OURS_S * 1000 / 3600)
    a_lead = v13.cost(rate * LEADER_S * 1000 / 3600)
    print(f"{name:22} {rate:9.4f} {s_sat:18.4f} {a_ours:12.2f} {a_lead:14.2f}")

print("\n=== the frontier: our latency vs the axis, on a $600 RTX 3090 ===")
rate = 600 / (4 * 365 * 24)
for s in (0.139, 0.10, 0.07, 0.05, 0.046, 0.04, 0.032, 0.02, 0.01):
    p = rate * s * 1000 / 3600
    print(f"  {s*1000:6.1f} ms/dec -> ${p:.6f}/1000 -> axis {v13.cost(p):6.2f}")

print("\n=== what saturating cost+ buys us in the composite ===")
print("calibration 91.64 (ours, best on board), speed 88.43 measured on a PRO 6000.")
print("Cost axis 100 vs 67.45, everything else equal, intelligence swept:\n")
print(f"{'judge':>6} {'sealed':>7} {'I':>7} {'cost=67':>9} {'cost=100':>9} {'delta':>7}")
for judge in (0.30, 0.50, 0.65, 0.75):
    i13 = v13.intelligence({"easy": 0.9167, "standard": 0.7222,
                            "hard": 0.6847, "judge": judge})
    for se in (0.25, 0.35, 0.50, 0.65):
        acc = 172 / 231
        intel = (0.8 * i13 + 0.2 * v14.chance_corrected_sealed(se)) * \
                (1 - max(0.0, 100 * (acc - se) - 25) / 100)
        a = v14.harmonic({"intelligence": intel, "calibration": 91.64,
                          "speed": 88.43, "cost": 67.45})
        b = v14.harmonic({"intelligence": intel, "calibration": 91.64,
                          "speed": 88.43, "cost": 100.0})
        print(f"{judge:6.2f} {se:7.2f} {intel:7.2f} {a:9.2f} {b:9.2f} {b-a:+7.2f}")

print("\n=== so: what is the ONE remaining variable? ===")
print("With cost saturated at 100 and speed at 88.43, the harmonic mean is")
print("gated almost entirely by intelligence. Required for target scores:")
for target in (64.13, 70, 75, 80):
    lo, hi = 1.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if v14.harmonic({"intelligence": mid, "calibration": 91.64,
                         "speed": 88.43, "cost": 100.0}) < target:
            lo = mid
        else:
            hi = mid
    print(f"  score {target:6.2f} -> intelligence >= {hi:6.2f}")
print("\nLeaders' intelligence: 48.0 - 53.1. So 64-70 is reachable on cost+speed")
print("alone IF intelligence lands at their level; 75+ is not.")
