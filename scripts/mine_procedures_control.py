"""The control I omitted, and the honest re-run.

The mined winner was `lt(feat:overlap_state, feat:overlap_state)` - a constant
False column. argmax over a constant returns the first option, so the procedure
was literally "always pick option 0", and JevBench puts the gold option first
in most items. 0.88 was the position leak wearing a procedure costume.

THRM's rule: "A negative control has to violate the procedure." So:

  CONTROL 1  the trivial positional baselines, measured on the same items
  FIX       re-mine on per-item SHUFFLED options, so position carries zero
            information and any procedure that wins must be reading content
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\thrm")
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "mine", r"D:\gavel-jevbench-entry\scripts\mine_decision_procedures.py")
M = importlib.util.module_from_spec(spec)
sys.modules["mine"] = M
spec.loader.exec_module(M)

TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
SEED, FOLDS = 20260924, 5

items = M.load_items()
keys = sorted(items)
n = {k: len(items[k]["opts"]) for k in keys}
chance = float(np.mean([1.0 / n[k] for k in keys]))

print("=" * 70)
print("CONTROL: trivial positional baselines, no model at all")
print("=" * 70)
first = sum(1 for k in keys if items[k]["opts"][0][1] == 1)
last = sum(1 for k in keys if items[k]["opts"][-1][1] == 1)
print(f"  items where gold is option 0        : {first}/{len(keys)} "
      f"= {first/len(keys):.4f}")
print(f"  items where gold is the LAST option : {last}/{len(keys)} "
      f"= {last/len(keys):.4f}")
print(f"  chance                              : {chance:.4f}")
print("\n  => 'always pick the first option' is a {0:.4f} baseline. Any procedure"
      .format(first / len(keys)))
print("     scoring near that is reproducing position, not reasoning.")


def shuffled_items(seed):
    """Permute each item's options. Gold position becomes uniform, so no"
    "     procedure can win by reading position."""
    rng = np.random.RandomState(seed)
    out = {}
    for k in keys:
        it = items[k]
        order = list(range(n[k]))
        rng.shuffle(order)
        out[k] = {"state": it["state"], "q": it["q"],
                  "opts": [it["opts"][j] for j in order]}
    return out


print("\n" + "=" * 70)
print("FIX: re-mine on per-item SHUFFLED options (position carries no signal)")
print("=" * 70)
si = shuffled_items(SEED)
sfirst = sum(1 for k in keys if si[k]["opts"][0][1] == 1)
print(f"  after shuffling, gold-at-position-0 = {sfirst}/{len(keys)} "
      f"= {sfirst/len(keys):.4f}  (should be near chance {chance:.4f})")

cols, meta, gold, row_of, sk = M.build(si)
print(f"  base features {len(cols)}, option-rows {len(meta)}")
perm = np.random.RandomState(SEED).permutation(len(sk))
fold_of = {item: perm[j] % FOLDS for j, item in enumerate(sk)}
hits = tot = 0
bests = []
for f in range(FOLDS):
    tr = [k for k in sk if fold_of[k] != f]
    va = [k for k in sk if fold_of[k] == f]
    surv = M.search(cols, gold, row_of, tr, va, verbose=(f == 0))
    if not surv:
        print(f"  fold {f}: nothing cleared the floor")
        continue
    a_va, a_tr, name, v = max(surv, key=lambda x: (x[0], x[1]))
    h = round(a_va * len(va))
    hits += h
    tot += len(va)
    bests.append({"fold": f, "heldout": a_va, "expr": name[:160]})
    print(f"  fold {f}: held-out {a_va:.4f}  <- {name[:100]}")

held = hits / max(tot, 1)
print(f"\n  HELD-OUT on SHUFFLED options = {held:.4f} ({hits}/{tot})")
print(f"  always-first control on shuffled = {sfirst/len(keys):.4f}")
print(f"  chance = {chance:.4f}")
print(f"\n  previous (unshuffled) number was 0.8310, which was the position leak.")

# full-set library on shuffled data, for the report
surv = M.search(cols, gold, row_of, sk[:len(sk) // 2], sk[len(sk) // 2:],
                verbose=False)
surv = sorted(surv, key=lambda x: (-x[0], -x[1]))[:10]
print(f"\n  candidates over floor (shuffled): {len(surv)}")
for a, t, name, _ in surv[:6]:
    print(f"    val {a:.4f} train {t:.4f}  {name[:90]}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "mined_procedures_control.json").write_text(json.dumps({
    "why": "the first mined run scored 0.8310 with a constant-False column, "
           "i.e. 'always pick option 0'. JevBench places gold first in most "
           "items, so that number was the position leak, not a procedure.",
    "positional_control": {
        "gold_at_option_0": first / len(keys),
        "gold_at_last_option": last / len(keys),
        "chance": chance,
    },
    "shuffled": {
        "gold_at_option_0_after_shuffle": sfirst / len(keys),
        "heldout_accuracy": held,
        "hits": hits, "total": tot,
        "per_fold": bests,
    },
    "top_procedures_shuffled": [{"val": a, "train": t, "expr": n2[:200]}
                                for a, t, n2, _ in surv],
    "references": {"4B_trained_head_137items": 0.7080,
                   "300M_zero_shot_213items": 0.5915,
                   "trained_head_on_300M": 0.2113,
                   "first_mined_run_UNSHUFFLED": 0.8310},
    "gates": {"ood_abstention": "NOT IMPLEMENTED",
              "position_leak": "now measured and controlled for"},
}, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'mined_procedures_control.json'}")
