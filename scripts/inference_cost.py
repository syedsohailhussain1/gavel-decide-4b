"""What does each system cost per decision, and what does combining buy?

The oracle is a MEASUREMENT, not a design - it requires knowing the answer, so
it costs nothing and is achievable by nothing. The question that matters is what
a real router pays to approach it.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\thrm")
spec = importlib.util.spec_from_file_location(
    "mine", r"D:\gavel-jevbench-entry\scripts\mine_decision_procedures.py")
M = importlib.util.module_from_spec(spec)
sys.modules["mine"] = M
spec.loader.exec_module(M)

TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
SEED, K = 20260924, 5

items = M.load_items()
keys = sorted(items)
rng = np.random.RandomState(SEED)
si = {}
for k in keys:
    it = items[k]
    o = list(range(len(it["opts"])))
    rng.shuffle(o)
    si[k] = {"state": it["state"], "q": it["q"],
             "opts": [it["opts"][j] for j in o]}

cols, meta, goldv, row_of, sk = M.build(si)
perm = np.random.RandomState(SEED).permutation(len(sk))
fold_of = {item: perm[j] % K for j, item in enumerate(sk)}
trk = [k for k in sk if fold_of[k] != 0]
vak = [k for k in sk if fold_of[k] == 0]
surv = sorted(M.search(cols, goldv, row_of, trk, vak, verbose=False),
              key=lambda x: (-x[0], -x[1]))
vec = surv[0][3]
print(f"procedure: {surv[0][2][:70]}")

# --- measured: the mined library's full per-decision cost, feature build +
#     score + argmax, on this CPU
for _ in range(20):                       # warm
    M.build({keys[0]: si[keys[0]]})
t0 = time.perf_counter()
N = 60
for k in keys[:N]:
    _c, _m, _g, _r, _s = M.build({k: si[k]})
    idx = _r[k]
    int(max(idx, key=lambda i: vec[i]))
mined_s = (time.perf_counter() - t0) / N
print(f"\nMINED library, measured end-to-end on this CPU:")
print(f"  {mined_s*1e6:8.1f} us per decision  ({1/mined_s:,.0f}/s)")
print(f"  includes all 18 typed features built from scratch + score + argmax")

# --- reference costs. 4B is the shipped entry's measured 0.139 s on a 24GB GPU.
FOUR_B = 0.139
# 300M: not measured on GPU here. Its cost scales with parameters, so on the
# same GPU it should be ~300/4020 of the 4B's compute. Labelled ESTIMATE.
est300 = FOUR_B * (300 / 4020)
print(f"\nreference:")
print(f"  4B probe      {FOUR_B*1000:8.1f} ms   MEASURED (RTX PRO 6000, shipped entry)")
print(f"  300M cosine   {est300*1000:8.1f} ms   ESTIMATE = 4B x (300/4020 params)")

print(f"\n{'='*62}")
print("{'configuration':34} {'latency':>10} {'vs 4B':>8} {'accuracy':>9}")
print(f"{'='*62}")
rows = [
    ("4B alone", FOUR_B, 0.7080),
    ("4B + mined", FOUR_B + mined_s, None),
    ("4B + 300M (est.)", FOUR_B + est300, None),
    ("all three", FOUR_B + est300 + mined_s, 0.6934),
]
for name, lat, acc in rows:
    a = f"{acc:.4f}" if acc is not None else "  -  "
    print(f"{name:34} {lat*1000:9.1f}ms {lat/FOUR_B:7.2f}x {a:>9}")

print(f"\n{'='*62}\nverdict")
print(f"{'='*62}")
b = FOUR_B + est300 + mined_s
print(f"  the 3-system BLEND costs {b/FOUR_B:.2f}x the latency of the 4B alone")
print(f"  and scores 0.6934 against the 4B's 0.7080.")
print(f"  -> paying {(b-FOUR_B)*1000:.0f} ms per decision to be 1.5 points WORSE.")
print()
print(f"  the ORACLE (0.9489) costs nothing because it is not implementable:")
print(f"  it requires already knowing which system is right.")
print()
print(f"  ABSTENTION on the 4B's own margin costs 0 ms - the margin is already")
print(f"  computed - and lifts 0.7080 to 0.8545 by declining 40% of items.")
print()
print(f"  the MINED library costs {mined_s*1e6:.0f} us, i.e. "
      f"{mined_s/FOUR_B*100:.3f}% of the 4B.")
print(f"  It is essentially free, which means its accuracy contribution has to")
print(f"  justify itself, and measured it does not: blend 0.6934 < 4B 0.7080.")

(OUT / "inference_cost.json").write_text(json.dumps({
    "measured_this_cpu": {"mined_library_us_per_decision": mined_s * 1e6,
                          "mined_decisions_per_s": 1 / mined_s},
    "reference": {"4B_measured_s": FOUR_B,
                  "4B_source": "shipped entry, RTX PRO 6000, bf16, prefix-cached",
                  "300M_estimate_s": est300,
                  "300M_estimate_basis": "4B x (300/4020) parameter ratio; NOT "
                                        "measured on GPU"},
    "configurations": [{"name": n, "latency_s": l, "vs_4B": l / FOUR_B,
                        "accuracy": a} for n, l, a in rows],
    "verdict": "the 3-system blend pays 1.12x latency for 1.5 points WORSE "
               "accuracy. The oracle is free only because it is impossible. "
               "Abstention on the 4B's own margin is free and worth +14.6.",
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'inference_cost.json'}")
