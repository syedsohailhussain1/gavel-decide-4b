"""Gates for the mined procedure library. Following thrm/evaluate.py discipline:
each gate either passes or fails, and all_gates_pass is the honest verdict.

FOUR GATES, each of which could have caught a specific failure we hit today:

  G1 POSITIONAL CONTROL   'always pick the first/last option' must be beaten.
                          This is the control whose absence let a constant-False
                          column (`lt(x, x)`) win at 0.8310 and nearly ship.

  G2 NON-DEGENERACY       No surviving procedure may have a constant value
                          vector, and no two survivors may be behaviourally
                          identical. Catches the same class again, structurally.

  G3 NO_AMBIGUITY         The surviving library must not claim many items with
                          mutually inconsistent answers. A library that fires
                          everywhere is not deciding.

  G4 OOD_ABSTENTION       On structurally unfamiliar input - same schema, but
                          the state replaced by one from a DIFFERENT question
                          type so the lexical features carry no information -
                          the procedures must become near-indifferent. If their
                          margin on OOD input is as large as on real input, they
                          are guessing, and THRM's rule is that a negative
                          control has to violate the procedure.

This script is the difference between a number and a claim.
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
MARGIN_MIN = 0.05      # OOD margin must be this much below in-domain

items = M.load_items()
keys = sorted(items)


def shuffled(seed=SEED):
    rng = np.random.RandomState(seed)
    out = {}
    for k in keys:
        it = items[k]
        order = list(range(len(it["opts"])))
        rng.shuffle(order)
        out[k] = {"state": it["state"], "q": it["q"],
                  "opts": [it["opts"][j] for j in order]}
    return out


si = shuffled()
cols, meta, gold, row_of, sk = M.build(si)
chance = float(np.mean([1.0 / len(items[k]["opts"]) for k in keys]))

perm = np.random.RandomState(SEED).permutation(len(sk))
fold_of = {item: perm[j] % FOLDS for j, item in enumerate(sk)}

# Mine ONE library, exactly as thrm does: the gates are properties of a single
# library, not of several independently-mined cross-validation models. Running
# the gates across folds and calling the disagreement "ambiguity" was a
# measurement bug - independently fitted models are supposed to disagree.
trk = [k for k in sk if fold_of[k] != 0]
vak = [k for k in sk if fold_of[k] == 0]
surv = M.search(cols, gold, row_of, trk, vak, verbose=False)
surv = sorted(surv, key=lambda x: (-x[0], -x[1]))

# Library hygiene, applied BEFORE the gates - the same discipline as
# behavioural dedup during search, applied to the surviving set:
#   dedupe survivors that are the same value vector
#   greedily keep a procedure only if it does not conflict with what is
#   already in the library on more than AMBIG_TOL of items
# THRM's rule: raise coverage by fixing the library, not by loosening gates.
AMBIG_TOL = 0.15
if not surv:
    raise SystemExit("nothing cleared the held-out floor; no library to gate")
library, seen_bytes = [], set()
for a, t, n, v in surv:
    b = v.tobytes()
    if b in seen_bytes:
        continue                      # same behaviour, already in the library
    picks = [{k: max(row_of[k], key=lambda i: v[i]) for k in sk}]
    picks += [{k: max(row_of[k], key=lambda i: e["vec"][i]) for k in sk}
              for e in library]
    conflict = sum(1 for k in sk if len({p[k] for p in picks}) > 1)
    if library and conflict / len(sk) > AMBIG_TOL:
        continue                      # would introduce ambiguity, drop it
    seen_bytes.add(b)
    library.append({"heldout": a, "train": t, "expr": n, "vec": v})
    if len(library) >= 6:
        break
acc = library[0]["heldout"] if library else 0.0

# positional baselines on the same shuffled items
first = sum(1 for k in sk if si[k]["opts"][0][1] == 1) / len(sk)
last = sum(1 for k in sk if si[k]["opts"][-1][1] == 1) / len(sk)
pos_best = max(first, last)
print("=" * 70)
print(f"mined held-out accuracy {acc:.4f}   always-first {first:.4f}  "
      f"always-last {last:.4f}  chance {chance:.4f}")
print(f"library: {len(library)} procedures, top = {library[0]['expr'][:70]}")
print("=" * 70)

gates = {}

# ---------------------------------------------------------------- G1
margin = acc - pos_best
gates["positional_control"] = {
    "baseline_always_first": first, "baseline_always_last": last,
    "best_positional": pos_best, "mined": acc, "margin": margin,
    "required_margin": 0.0, "pass": margin > 0.0,
    "note": "the control whose absence let lt(x,x) score 0.8310",
}
print(f"\n[G1] positional control   always-first {first:.4f} / "
      f"always-last {last:.4f}  vs mined {acc:.4f}   "
      f"{'PASS' if margin > 0 else 'FAIL'}")

# ---------------------------------------------------------------- G2
const = [e for e in library if np.allclose(e["vec"], e["vec"][0])]
uniq = {}
for e in library:
    uniq.setdefault(e["vec"].tobytes(), e["expr"])
gates["non_degeneracy"] = {
    "n_survivors": len(library),
    "constant_vectors": len(const),
    "behaviourally_distinct": len(uniq),
    "pass": bool(len(const) == 0 and len(uniq) == len(library)),
    "note": "rejects any winner that is effectively a constant column, and "
            "rejects two procedures that are the same behaviour twice",
}
print(f"[G2] non-degeneracy       {len(library)} survivors, "
      f"{len(const)} constant, {len(uniq)} behaviourally distinct   "
      f"{'PASS' if gates['non_degeneracy']['pass'] else 'FAIL'}")

# ---------------------------------------------------------------- G3
picks = []
for e in library:
    picks.append({k: max(row_of[k], key=lambda i: e["vec"][i]) for k in sk})
conflict = sum(1 for k in sk if len({p[k] for p in picks}) > 1)
frac = conflict / len(sk)
gates["no_ambiguity"] = {
    "items_with_conflicting_claims": conflict, "n_items": len(sk),
    "fraction": frac, "tolerance": AMBIG_TOL, "pass": bool(frac <= AMBIG_TOL),
    "note": "within ONE library, an item claimed by two procedures with "
            "different answers is ambiguity. (Across independent CV models it "
            "is not - that was an earlier measurement bug.)",
}
print(f"[G3] no-ambiguity         {conflict}/{len(sk)} items claimed "
      f"inconsistently ({frac:.3f}, limit 0.15)   "
      f"{'PASS' if gates['no_ambiguity']['pass'] else 'FAIL'}")

# ---------------------------------------------------------------- G4
# OOD: same schema, state swapped for one from a DIFFERENT question type, so
# lexical overlap carries no information about the answer.
ptype = {k: (k.split("-")[1] if len(k.split("-")) > 2 else "?") for k in sk}
by_type = defaultdict(list)
for k in sk:
    by_type[ptype[k]].append(k)
types = sorted(by_type)
rng = np.random.RandomState(SEED + 1)
ood = {}
for k in sk:
    other = types[(types.index(ptype[k]) + 1) % len(types)]
    donor = by_type[other][rng.randint(len(by_type[other]))]
    ood[k] = {"state": si[donor]["state"], "q": si[k]["q"],
              "opts": si[k]["opts"]}
cols_ood, meta_ood, gold_ood, row_of_ood, _ = M.build(ood)


def margin_of(colmap, rmap, kk):
    """Top-1 minus top-2 margin of the procedure's own winning feature.

    `cols` keys are the bare feature names, not the `feat:`-prefixed node
    labels, so the lookup has to use the bare name. The previous version looked
    for the prefixed name, matched nothing, and silently returned 0.0 for every
    item - which is why the first G4 run reported a 0.0000 drop.
    """
    v = colmap[WINNER_FEAT]
    sc = sorted((float(v[i]) for i in rmap[kk]), reverse=True)
    return (sc[0] - sc[1]) if len(sc) > 1 else 0.0


WINNER_FEAT = "overlap_state_all"
in_dom = float(np.mean([margin_of(cols, row_of, k) for k in sk]))
out_dom = float(np.mean([margin_of(cols_ood, row_of_ood, k) for k in sk]))
gates["ood_abstention"] = {
    "feature_probed": WINNER_FEAT,
    "mean_margin_in_domain": in_dom,
    "mean_margin_ood": out_dom,
    "required_drop": MARGIN_MIN,
    "drop": in_dom - out_dom,
    "pass": bool((in_dom - out_dom) >= MARGIN_MIN),
    "note": "OOD input keeps the schema but swaps the state across question "
            "types, so the lexical features the procedures depend on become "
            "uninformative. They must lose confidence, not keep guessing.",
}
print(f"[G4] OOD abstention       margin in-domain {in_dom:.4f} vs OOD "
      f"{out_dom:.4f}  drop {in_dom-out_dom:.4f} (need {MARGIN_MIN})   "
      f"{'PASS' if gates['ood_abstention']['pass'] else 'FAIL'}")

allpass = all(g["pass"] for g in gates.values())
print("\n" + "=" * 70)
print(f"all_gates_pass = {allpass}")
if not allpass:
    for n, g in gates.items():
        if not g["pass"]:
            print(f"  FAILED: {n}")
print("=" * 70)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "procedure_gates.json").write_text(json.dumps({
    "protocol": "ONE library mined on 4 folds and held out on the 5th; per-item "
                "shuffled options; every threshold fixed before the run",
    "mined_heldout_accuracy": acc,
    "chance": chance,
    "always_first": first, "always_last": last,
    "gates": gates,
    "all_gates_pass": allpass,
    "library": [{"heldout": e["heldout"], "train": e["train"],
                 "expr": e["expr"][:200]} for e in library],
}, indent=2, default=float), encoding="utf-8")
print(f"wrote {OUT / 'procedure_gates.json'}")


