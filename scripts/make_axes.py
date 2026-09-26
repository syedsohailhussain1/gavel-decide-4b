"""Derive the submission's measured axes from a real JevBench results file.

Everything written to results/axes.json comes from the run itself — no
hand-entered numbers. Run:

    python scripts/make_axes.py <results.jsonl>
"""
import json
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, r"D:\jevbench")
from jevbench import composite_v13 as v13
from jevbench.metrics import ece_top_label

RESULTS = sys.argv[1] if len(sys.argv) > 1 else r"results/public_231_results.jsonl"


def tier(t):
    x = t.split("-", 1)[0]
    return {"original": "standard"}.get(x, x)


def q(v, p):
    v = sorted(v)
    return v[max(0, min(len(v) - 1, int(round((len(v) - 1) * p))))]


rows = [json.loads(l) for l in open(RESULTS, encoding="utf-8") if l.strip()]
acc, lat, ins, paths = {}, defaultdict(list), [], defaultdict(int)
for r in rows:
    t = tier(r["task_id"])
    acc.setdefault(t, []).append(1.0 if r.get("correct") else 0.0)
    if r.get("latency_s"):
        lat[t].append(r["latency_s"])
    if (r.get("usage") or {}).get("input_tokens") is not None:
        ins.append(r["usage"]["input_tokens"])
    paths[((r.get("runtime") or {}).get("path") or "?")] += 1
acc = {k: sum(v) / len(v) for k, v in acc.items()}

hard = [r for r in rows if tier(r["task_id"]) == "hard" and r.get("probs")]
ece = ece_top_label([(max(r["probs"].values()), 1.0 if r.get("correct") else 0.0)
                     for r in hard])["ece"]
cal = v13.calibration(ece)

out = {
    "n_items": len(rows),
    "n_correct": int(sum(1 for r in rows if r.get("correct"))),
    "accuracy": sum(1 for r in rows if r.get("correct")) / len(rows),
    "schema_validity": sum(1 for r in rows if r.get("valid")) / len(rows),
    "operational_success": sum(1 for r in rows if r.get("ok")) / len(rows),
    "tier_accuracy": acc,
    "tier_chance_corrected": {t: v13.chance_corrected_accuracy(a, v13.TIER_CHANCES[t])
                              for t, a in acc.items()},
    "calibration_axis": cal,
    "hard_binned_ece": ece,
    "all_tier_binned_ece": ece_top_label(
        [(max(r["probs"].values()), 1.0 if r.get("correct") else 0.0)
         for r in rows if r.get("probs")])["ece"],
    "latency_p50_s": {t: q(v, .5) for t, v in lat.items()},
    "latency_p95_s": {t: q(v, .95) for t, v in lat.items()},
    "speed_axis_standard_gpu_adjusted": v13.speed(
        q(lat["standard"], .5), q(lat["standard"], .95), "gpu"),
    "mean_input_tokens": statistics.mean(ins),
    "median_input_tokens": statistics.median(ins),
    "total_input_tokens": sum(ins),
    "path_mix": dict(paths),
    "notes": [
        "speed population per composite_v12 is standard+judge; judge has no "
        "public jsonl so the standard tier is used as the measurable proxy",
        "judge tier (146 items, 28% of intelligence weight) is not published "
        "and cannot be measured locally",
        "sealed tier (308 items) is private; no local estimate",
        "latency is from a GTX 1650 4GB with bitsandbytes nf4, not operator hardware",
    ],
}
print(json.dumps(out, indent=2))
with open("results/axes.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
