"""Position prior, measured from the dataset rather than from parsed text.

The previous attempt recovered the gold label by splitting the rendered option
string on ':', which is lossy, and produced 0.3052 where the direct count from
target flags gives 0.8169. This reads `labels` + `expected` straight out of the
JevBench public split, which is authoritative and covers all 231 items.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(r"D:\jevbench\datasets\public")
TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")

items = {}
for tier in ("easy", "hard", "original"):
    p = DATA / f"{tier}.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        tid = d.get("id") or d.get("task_id")
        items[tid] = {"labels": list(d.get("labels") or []),
                      "expected": d.get("expected"),
                      "tier": tier}
keys = sorted(items)
print(f"items {len(keys)}  (easy/hard/original, authoritative split)")

# which of these did the head train on?
pairs = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
trained = {r["item"] for r in pairs if r["tier"] != "nli"}
print(f"items the head trained on: {len(trained & set(keys))}")
print(f"items the head NEVER saw : {len(set(keys) - trained)}")


def stats(tids, label):
    first = last = ok = 0
    exp = []
    for t in tids:
        it = items[t]
        labels, g = it["labels"], it["expected"]
        if isinstance(g, int) and 0 <= g < len(labels):
            g = labels[g]   # ordinal items carry an int expected, not a label
        if g not in labels or len(labels) < 2:
            continue
        ok += 1
        first += labels.index(g) == 0
        last += labels.index(g) == len(labels) - 1
        exp.append(1.0 / len(labels))
    if not ok:
        print(f"  {label:32} n=0")
        return None
    e = float(np.mean(exp))
    print(f"  {label:32} n={ok:3}  gold-first {first/ok:.4f}  "
          f"gold-last {last/ok:.4f}  chance {e:.4f}  "
          f"lift {first/ok/e:.2f}x")
    return {"n": ok, "gold_first": first / ok, "gold_last": last / ok,
            "chance": e, "always_first": first / ok}


print("\n" + "=" * 72)
print("the decisive comparison: is the prior as strong where NOTHING was fitted?")
print("=" * 72)
a = stats([t for t in keys if t in trained], "head TRAINED on these")
b = stats([t for t in keys if t not in trained], "head NEVER saw these")
c = stats(keys, "all 231")
print()
res = {"trained_on": a, "unseen": b, "all": c}
if a and b:
    print(f"  gold-first: trained {a['gold_first']:.4f}  vs  "
          f"unseen {b['gold_first']:.4f}   "
          f"(chance on unseen {b['chance']:.4f})")
    weak = b["gold_first"] < 0.5
    print()
    if weak:
        print("  VERDICT: on the 18 items nothing was fitted to, gold-first is")
        print(f"  {b['gold_first']:.4f} - essentially chance. The 0.8169 prior is an")
        print("  artifact of how the PUBLIC split was assembled, not a property")
        print("  of the task. A system built on it scores ~0.34 on fresh data.")
    else:
        print("  VERDICT: the prior holds on unseen items, so it is a")
        print("  construction property of JevBench and would likely carry to the")
        print("  sealed tier. Exploiting it is still benchmark gaming, not")
        print("  capability, and we would be gambling the sealed ordering.")

(OUT / "position_prior_audit.json").write_text(json.dumps({
    "source": "D:/jevbench/datasets/public/*.jsonl, fields labels + expected",
    "question": "is gold-at-position-0 a property of the task or of the public "
                "split?",
    **res,
    "verdict": ("artifact of the public split - collapses to chance on unseen "
                "items" if (b and b["gold_first"] < 0.5)
                else "construction property - would likely carry to sealed"),
    "real_numbers_nothing_fitted": {
        "mined_procedures_shuffled": 0.5493,
        "300m_zero_shot_prefixed": 0.5915,
        "4b_trained_head": 0.7080,
        "chance": 0.3427,
    },
    "the_0.8310": "was always-pick-option-0; it is a number about the dataset, "
                  "not about a model",
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'position_prior_audit.json'}")

