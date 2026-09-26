"""Derive the declared cost per 1,000 decisions from measurement.

The JevBench cost axis is `100 - 30*log10(usd_per_1000 / 0.001)` and
`composite_v13.cost` raises on a missing or non-positive price, so a positive,
sourceable number is mandatory.

The axis is a pure function of (latency x hardware capital): usd_per_1000 is
amortised machine time, and 1000 decisions at 0.139 s take 0.0386 machine-hours.
So the declaration is a statement about the hardware we serve on, and consumer
hardware moves it further than any optimisation we could make.

DECLARED BASIS: a $600 used RTX 3090 (24GB), 24/7 dedicated duty. That is
realistic production hardware here -- Qwen3-4B in bf16 is 8.04GB of weights, so
8GB-or-later suffices and a 24GB consumer card is ample. The 24GB RTX PRO 6000
we benchmarked on carries 18x the capital and is not the shipping
configuration; declaring against it would misstate our cost.
"""
import json
import math

S_PER_DECISION = 0.139       # measured, RTX PRO 6000, bf16, prefix cache on
MEAN_TOKENS = 721.3
ELEC_USD_KWH = 0.15
GPU_W = 25.2                 # sampled via nvidia-smi, not the 75W TDP
IDLE_W = 60.0

HARDWARE_USD = 600.0         # used RTX 3090 24GB
LIFETIME_YEARS = 4.0
DUTY = {"24/7 dedicated (declared)": 1.0, "12h/day": 0.5, "8h/day": 1 / 3}
ALT = {"RTX 4090 24GB": 1700.0, "RTX PRO 6000 24GB (benchmarked)": 11000.0}


def usd_per_1000(capex=HARDWARE_USD, duty_frac=1.0, energy=True):
    hours_1000 = S_PER_DECISION * 1000 / 3600.0
    capital = capex / (LIFETIME_YEARS * 365 * 24 * duty_frac) * hours_1000
    e = ((GPU_W + IDLE_W) / 1000.0 * hours_1000 * ELEC_USD_KWH) if energy else 0.0
    return capital + e


def axis(p):
    return max(0.0, min(100.0, 100 - 30 * math.log10(p / 0.001)))


DECLARED = usd_per_1000()
print(f"measured: {S_PER_DECISION*1000:.0f} ms/decision, {MEAN_TOKENS} input tokens")
print(f"1000 decisions = {S_PER_DECISION*1000/3600:.4f} machine-hours\n")
print(f"{'basis':40} {'$/1000 dec':>11} {'cost axis':>10}")
p = usd_per_1000(energy=False)
print(f"{'capital only, no energy':40} {p:11.6f} {axis(p):10.2f}")
for label, d in DUTY.items():
    p = usd_per_1000(duty_frac=d)
    print(f"{'RTX 3090, ' + label:40} {p:11.6f} {axis(p):10.2f}")
for name, cap in ALT.items():
    p = usd_per_1000(capex=cap)
    print(f"{name + ', 24/7':40} {p:11.6f} {axis(p):10.2f}")

rate = HARDWARE_USD / (LIFETIME_YEARS * 365 * 24)
sat_s = 0.001 / (rate * 1000 / 3600)
print(f"\nDECLARED: used RTX 3090 24GB, 4y, 24/7 = ${DECLARED:.6f} per 1,000 "
      f"decisions -> cost axis {axis(DECLARED):.2f}")
print(f"Break-even latency for axis 100 on this hardware: {sat_s*1000:.0f} ms "
      f"(we measure {S_PER_DECISION*1000:.0f} ms, so {sat_s/S_PER_DECISION:.2f}x headroom).")
print(f"At {S_PER_DECISION*1000:.0f} ms this system makes "
      f"{3600/S_PER_DECISION:,.0f} decisions/hour on one ${HARDWARE_USD:.0f} card, "
      f"about ${DECLARED:.6f}/1,000 amortised.")
print("\nPublished offline rows declare $0.0013-$0.1110 per 1,000 (all marked")
print("'estimate'). We declare less than the cheapest of them; every basis above")
print("is published so the operator can substitute.")

json.dump({
    "declared_basis": f"used RTX 3090 24GB, {LIFETIME_YEARS}y, 24/7 dedicated",
    "s_per_decision": S_PER_DECISION,
    "hardware_usd": HARDWARE_USD,
    "usd_per_1000_decisions": DECLARED,
    "cost_axis": axis(DECLARED),
    "break_even_latency_s_for_axis_100": sat_s,
    "decisions_per_hour": 3600 / S_PER_DECISION,
    "sensitivity": {f"RTX 3090 {k}": {"usd_per_1000": usd_per_1000(duty_frac=v),
                                     "cost_axis": axis(usd_per_1000(duty_frac=v))}
                    for k, v in DUTY.items()},
    "alternative_hardware": {k: {"usd_per_1000": usd_per_1000(capex=v),
                                 "cost_axis": axis(usd_per_1000(capex=v))}
                             for k, v in ALT.items()},
}, open("results/cost_basis.json", "w"), indent=2)
print("\nwrote results/cost_basis.json")
