"""Is gold-at-position-0 a property of the benchmark, or of the public split?

0.8310 was 'always pick option 0' against 174/213 items with gold first. The
question that decides whether that is exploitable in the wild is whether the
prior survives where we have never fit anything:

  the 213 items the head trained on   vs   the 18 items it never saw

If the prior is equally strong on unseen items it is a construction property of
JevBench and would carry to the sealed tier. If it is much weaker on unseen
items then it is an artifact of how the public split was assembled, and any
system leaning on it collapses the moment it meets fresh data.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
ENTRY = Path(r"D:\gavel-jevbench-entry\results")

rows = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
jb = [r for r in rows if r["tier"] != "nli"]
items = defaultdict(lambda: {"opts": []})
for r in jb:
    items[r["item"]]["opts"].append((r["opt"] if "opt" in r else r["text"],
                                     r["target"]))
keys = sorted(items)

# rebuild in the ORIGINAL option order from the scored run, which preserves it
run = [json.loads(l) for l in open(ENTRY / "bf16_recal.jsonl", encoding="utf-8")]
prob = {r["task_id"]: r["probs"] for r in run if r.get("probs")}
ordered = []
for r in run:
    p = r.get("probs") or {}
    if not p:
        continue
    ordered.append((r["task_id"], list(p.keys())))
print(f"items with an ordered option list: {len(ordered)}")

# gold per item, from the pairs file, matched by position is unreliable;
# instead use the run's own correctness at each position via the recorded probs
# and the pair targets keyed by item+label
gold_by = {}
for r in jb:
    lab = r["text"].split("\nOption: ")[-1].split(":")[0].strip()
    if r["target"] == 1:
        gold_by[r["item"]] = lab

jb_items = {r["item"] for r in jb}
seen = [t for t, _ in ordered if t in jb_items]
unseen = [t for t, _ in ordered if t not in jb_items]


def pos_stats(tids, label):
    first = last = n = 0
    for t in tids:
        g = gold_by.get(t)
        if g is None:
            continue
        order = dict(ordered)[t]
        if g not in order:
            continue
        n += 1
        first += order.index(g) == 0
        last += order.index(g) == len(order) - 1
    if not n:
        print(f"  {label:34} n=0")
        return None
    exp = float(np.mean([1.0 / len(dict(ordered)[t]) for t in tids
                         if t in dict(ordered) and gold_by.get(t) in dict(ordered)[t]]))
    print(f"  {label:34} n={n:3}  gold-first {first/n:.4f}   "
          f"gold-last {last/n:.4f}   chance {exp:.4f}")
    return {"n": n, "gold_first": first / n, "gold_last": last / n,
            "chance": exp}


print("\n" + "=" * 66)
print("does the position prior survive where nothing was ever fitted?")
print("=" * 66)
a = pos_stats(seen, "items the head TRAINED on")
b = pos_stats(unseen, "items the head NEVER saw")
print()

res = {"trained_on": a, "unseen": b}
if a and b:
    gap = a["gold_first"] - b["gold_first"]
    print(f"  gold-first gap, trained vs unseen: {gap:+.4f}")
    if b["gold_first"] < a["gold_first"] - 0.15:
        print("\n  VERDICT: the prior is WEAK on unseen items. It is an artifact")
        print("  of how the public split was assembled, not a property of the")
        print("  task. A system built on it would collapse on fresh data, which")
        print("  is exactly the failure mode we already got burned by once.")
    else:
        print("\n  VERDICT: the prior is comparable on unseen items, so it is a")
        print("  construction property of JevBench. It might carry to the sealed")
        print("  tier - but exploiting it is benchmark gaming, not capability,")
        print("  and the sealed ordering is not something we can verify.")

print("\n" + "=" * 66)
print("what we would actually have to do to reach a real 0.83")
print("=" * 66)
print("  measured now, honestly, on items nothing was fitted to:")
print("    mined procedure library        0.5493")
print("    300M zero-shot, prefixed      0.5915")
print("    4B trained head               0.7080   <- best real number we have")
print("    chance                        0.3427")
print()
print("  to reach 0.83 legitimately the gap is +0.12 over our best, and the")
print("  only measured lever that moves that is COVERAGE + DATA VOLUME on the")
print("  4B. Not a trick.")

(OUT / "position_prior_audit.json").write_text(json.dumps({
    "question": "is gold-at-position-0 a property of the benchmark or of the "
                "public split?",
    "trained_on": a, "unseen": b,
    "conclusion": "prior is materially weaker on unseen items -> an assembly "
                  "artifact, not a task property"
    if (a and b and b["gold_first"] < a["gold_first"] - 0.15)
    else "prior holds on unseen items -> a construction property; exploiting it "
         "would still be benchmark gaming",
    "real_numbers": {"mined_procedures_shuffled": 0.5493,
                     "300m_zero_shot_prefixed": 0.5915,
                     "4b_trained_head": 0.7080,
                     "chance": 0.3427,
                     "the_0.8310_that_was_always_pick_first": 0.8169},
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'position_prior_audit.json'}")
