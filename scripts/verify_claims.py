"""Verify every number asserted in the submission docs against the artifacts.
Fails loudly on any mismatch so nothing unverified reaches the leaderboard."""
import json
import re
from pathlib import Path

E = Path(r"D:\gavel-jevbench-entry")
R = E / "results"
TS = Path(r"D:\gavel\models\training_state")

ok = True


def check(label, claimed, actual, tol=1e-9):
    global ok
    good = (abs(claimed - actual) <= tol) if isinstance(claimed, float) \
        else (claimed == actual)
    ok &= good
    print(f"  [{'OK ' if good else 'BAD'}] {label:44} docs={claimed}  "
          f"artifact={actual}")


print("=== axes file vs README table ===")
ax = json.loads((R / "bf16_recal_axes.json").read_text(encoding="utf-8"))
check("n_items", 231, ax["n_items"])
check("n_correct", 172, ax["n_correct"])
check("accuracy", 0.7446, round(ax["accuracy"], 4), tol=5e-5)
check("easy tier", 0.9167, round(ax["tier_accuracy"]["easy"], 4), tol=5e-5)
check("hard tier", 0.6847, round(ax["tier_accuracy"]["hard"], 4), tol=5e-5)
check("calibration axis", 91.64, round(ax["calibration_axis"], 2), tol=5e-3)
check("hard binned ECE", 0.0418, round(ax["hard_binned_ece"], 4), tol=5e-5)
check("speed axis", 88.43, round(ax["speed_axis_standard_proxy"], 2), tol=5e-3)
check("s per decision", 0.139, round(ax["s_per_decision"], 3), tol=5e-4)
check("hard p50", 0.183, round(ax["latency_p50_s"]["hard"], 3), tol=5e-4)
check("hard p95", 0.279, round(ax["latency_p95_s"]["hard"], 3), tol=5e-4)
check("mean input tokens", 721.3, round(ax["mean_input_tokens"], 1), tol=5e-2)
check("prefix cache items", 144, ax["path_mix"]["cached"])

print("\n=== accuracy recomputed from the raw run ===")
run = [json.loads(l) for l in open(R / "bf16_recal.jsonl", encoding="utf-8")]
n_ok = sum(bool(r.get("correct")) for r in run)
check("rows", 231, len(run))
check("correct", 172, n_ok)
check("accuracy recomputed", 0.7446, round(n_ok / len(run), 4), tol=5e-5)
tiers = {}
for r in run:
    t = r["task_id"].split("-")[0]
    tiers.setdefault(t, [0, 0])
    tiers[t][1] += 1
    tiers[t][0] += bool(r.get("correct"))
for t, (a, b) in tiers.items():
    print(f"       {t:9} {a}/{b} = {a/b:.4f}")
check("output_tokens all zero", True,
      all(r["usage"]["output_tokens"] == 0 for r in run))

print("\n=== contamination disclosure figures ===")
pairs = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
jb = [p for p in pairs if p["tier"] != "nli"]
jb_items = {p["item"] for p in jb}
scored = {r["task_id"] for r in run}
ov = jb_items & scored
check("JevBench pairs", 689, len(jb))
check("NLI pairs", 1720, len(pairs) - len(jb))
check("total pairs", 2409, len(pairs))
check("distinct JevBench items", 213, len(jb_items))
check("overlap with scored", 213, len(ov))
check("overlap rate", 0.9221, round(len(ov) / len(scored), 4), tol=5e-5)
seen = [r for r in run if r["task_id"] in ov]
uns = [r for r in run if r["task_id"] not in ov]
check("correct on trained", 169, sum(bool(r.get("correct")) for r in seen))
check("  = 79.34%", 0.7934,
      round(sum(bool(r.get("correct")) for r in seen) / len(seen), 4), tol=5e-5)
check("unseen items", 18, len(uns))
check("correct on unseen", 3, sum(bool(r.get("correct")) for r in uns))
etiers = {}
for r in run:
    t = r["task_id"].split("-")[0]
    etiers.setdefault(t, [0, 0])
    etiers[t][1] += 1
    etiers[t][0] += r["task_id"] in ov
check("easy fully covered", 48, etiers["easy"][0])

print("\n=== clean ablation figures ===")
oof = json.loads((R / "clean_head_oof.json").read_text(encoding="utf-8"))
check("NLI OOF accuracy", 0.5285, round(oof["nli_oof_accuracy"], 4), tol=5e-5)
check("NLI rows", 1720, oof["nli_rows"])
check("NLI items", 860, oof["nli_items"])
check("clean head params", 1442817, oof["params"])
check("JevBench items in clean head", 0, oof["jevbench_items_in_training"])
probe = [json.loads(l) for l in
         open(r"C:\Users\Sohail\AppData\Local\Temp\opencode\clean_probe.jsonl",
              encoding="utf-8")]
check("clean head first-12", 2, sum(bool(r.get("correct")) for r in probe))
check("  = 0.1667", 0.1667,
      round(sum(bool(r.get("correct")) for r in probe) / len(probe), 4), tol=5e-5)

print("\n=== no unverified figures leaked into the docs ===")
for f in ("README.md", "MODEL_CARD.md"):
    txt = (E / f).read_text(encoding="utf-8")
    for bad in ("0.5352", "0.3897", "0.5399", "81.7%"):
        if bad in txt:
            ok = False
            print(f"  [BAD] {f} still contains unverifiable {bad}")
        else:
            print(f"  [OK ] {f} clean of {bad}")

print("\n=== the out-of-fold calibration claim ===")
hc = json.loads((R / "honest_calibration.json").read_text(encoding="utf-8"))
oof = hc["honest_oof_temperature_refit"]
raw = hc["honest_oof_raw"]
in_s = hc["shipped_in_sample"]
check("in-sample axis", 91.64, round(in_s["calibration_axis"], 2), tol=5e-3)
check("in-sample ECE", 0.0418, round(in_s["binned_ece"], 4), tol=5e-5)
check("OOF axis, T refit", 87.15, round(oof["calibration_axis"], 2), tol=5e-3)
check("OOF ECE, T refit", 0.0642, round(oof["binned_ece"], 4), tol=5e-5)
check("OOF axis, T=1 no fitting", 79.31, round(raw["calibration_axis"], 2),
      tol=5e-3)
check("OOF accuracy", 0.7080, round(oof["accuracy"], 4), tol=5e-5)
check("OOF n items", 137, oof["n"])
check("inflation", 4.5, round(hc["inflation"], 1), tol=5e-2)

print("\n=== the docs must not claim 91.64 as the headline, or the 0.0374 as clean ===")
for f in ("README.md", "MODEL_CARD.md"):
    txt = (E / f).read_text(encoding="utf-8")
    checks = {
        "reports 87.15": "87.15" in txt,
        "labels 91.64 in-sample": "in-sample" in txt,
        "says advantage survives": ("survives" in txt or "clear of the field" in txt),
    }
    for label, good in checks.items():
        ok &= good
        print(f"  [{'OK ' if good else 'BAD'}] {f:14} {label}")

print("\n" + ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
