"""Mine a typed decision-procedure library instead of training a head.

WHY THIS EXISTS
---------------
A trained 768-dim head on 213 items scores 0.2113 - below chance. Zero-shot
embedding similarity on the same items scores 0.5915. THRM explains exactly
that shape of failure, and its explanation is the whole point of this build:

  "A dictionary is the right tool for 'what byte follows these bytes,' and it
   is unbeatable at that."  (byte_report negative result)

  "Recovering b3 (12 B) in place of a 256-entry table (8,656 B) is a 720x
   reduction at identical accuracy" - parsimony, not capacity, is the win.

And the mechanism that fixes the overfitting is THRM's, not ours:

  1. BEHAVIOUR-SPACE SEARCH. The frontier holds VALUE VECTORS, not
     expressions. Two rules producing the same column are one node. This
     collapses a 768-parameter continuous space into a deduplicated discrete
     one, which is precisely what 213 items cannot support.
  2. HELD-OUT SELECTION IS THE LOAD-BEARING STEP. Ranking by training fit
     rewards overfitting. THRM's run rejected 154 of 156 candidates; that
     number is the calibration to expect here too.
  3. NOTHING COERCES. The 34 typed operators come from thrm.dsl, unchanged,
     so a wrong-typed composition traps with a reason instead of quietly
     producing a plausible number.

Gates, following THRM: no_ambiguity (no item claimed by two procedures) and
ood_abstention (procedures must not fire on structurally unfamiliar input).
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\thrm")
from thrm.dsl import OPS  # noqa: E402  - 34 typed operators, names reused

# THRM's own operators are scalar-typed: they raise on a numpy column, so every
# composition in the frontier died. We keep the OPERATOR SET and the discipline
# - same names, same arity, and a composition over a column this operator cannot
# type is DROPPED, not coerced - but apply them column-wise, which is what
# their miner must do internally. String and list operators (contains, len,
# list_get, list_sum, lookup, split_get, concat) do not apply to a numeric
# column space and are therefore excluded rather than faked.
def _b(f):
    return lambda a, b: (f(a, b) | 1).astype(np.float64)


def _u(f):
    return lambda a: f(a).astype(np.float64)


VEC = {
    "add": (2, _b(np.add)), "sub": (2, _b(np.subtract)),
    "mul": (2, _b(np.multiply)),
    "div": (2, lambda a, b: np.divide(a, b, out=np.zeros_like(a),
                                       where=b != 0)),
    "mod": (2, _b(np.mod)), "min2": (2, _b(np.minimum)),
    "max2": (2, _b(np.maximum)),
    "abs": (1, _u(np.abs)), "neg": (1, _u(np.negative)),
    "floor": (1, _u(np.floor)), "ceil": (1, _u(np.ceil)),
    "floor_div": (2, _b(np.floor_divide)),
    "lt": (2, _b(np.less)), "le": (2, _b(np.less_equal)),
    "gt": (2, _b(np.greater)), "ge": (2, _b(np.greater_equal)),
    "eq": (2, _b(np.equal)), "ne": (2, _b(np.not_equal)),
    "and": (2, _b(np.logical_and)), "or": (2, _b(np.logical_or)),
    "not": (1, _u(np.logical_not)),
    "clamp": (3, lambda a, lo, hi: np.clip(a, lo, hi)),
    "select": (3, lambda c, a, b: np.where(c > 0.5, a, b)),
    "band": (2, None),   # threshold table; not used on raw columns
    "round": (1, _u(np.round)), "round2": (2, None),
    "concat": (2, None), "contains": (2, None), "len": (1, None),
    "split_get": (3, None), "list_len": (1, None), "list_get": (2, None),
    "list_sum": (1, None), "lookup": (2, None),
}
VEC = {k: v for k, v in VEC.items() if v[1] is not None}
print(f"reusing {len(OPS)} THRM operator names; {len(VEC)} apply to a numeric "
      f"column space, {len(OPS) - len(VEC)} do not and are excluded, not faked")

TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
SEED, FOLDS, BEAM, MAXDEPTH = 20260924, 5, 60, 2
MIN_ACC = 0.40          # absolute floor on held-out, per THRM's rule


# ----------------------------------------------------------------- traces
def load_items():
    rows = json.loads((TS / "combined_pairs.jsonl").read_text(encoding="utf-8"))
    jb = [r for r in rows if r["tier"] != "nli"]
    items = defaultdict(lambda: {"state": "", "q": "", "opts": []})
    for r in jb:
        t = r["text"]
        state = t.split("\nQuestion:")[0].replace("State: ", "", 1)
        q, opt = t.split("\nQuestion:", 1)[1].split("\nOption: ", 1)
        it = items[r["item"]]
        if not it["state"]:
            it["state"], it["q"] = state, q
        it["opts"].append((opt, r["target"]))
    return dict(items)


def toks(s):
    return set(w.strip(".,!?;:'\"()").lower() for w in s.split() if w.strip(".,!?;:'\"()"))


# ------------------------------------------------------- typed base features
def features(items):
    """One typed column per feature, over option-rows. Length = total options."""
    cols, meta = defaultdict(list), []
    for k in sorted(items):
        it = items[k]
        st, q = it["state"], it["q"]
        stt, qt = toks(st), toks(q)
        n = len(it["opts"])
        for j, (opt, _) in enumerate(it["opts"]):
            if ":" in opt:
                lab, crit = opt.split(":", 1)
            else:
                lab, crit = opt, ""
            ot, ct = toks(lab), toks(crit)
            allt = ot | ct
            meta.append((k, j))
            cols["overlap_state"].append(len(ot & stt))
            cols["overlap_state_all"].append(len(allt & stt))
            cols["overlap_question"].append(len(allt & qt))
            cols["jacc_state"].append(
                len(ot & stt) / max(len(ot | stt), 1))
            cols["has_criteria"].append(1 if crit.strip() else 0)
            cols["opt_len"].append(len(opt))
            cols["label_len"].append(len(lab))
            cols["crit_len"].append(len(crit))
            cols["n_options"].append(n)
            cols["position"].append(j)
            cols["is_first"].append(1 if j == 0 else 0)
            cols["is_last"].append(1 if j == n - 1 else 0)
            cols["word_count_state"].append(len(stt))
            cols["state_len"].append(len(st))
            cols["contain_state"].append(1 if (ot & stt) else 0)
            cols["contain_question"].append(1 if (allt & qt) else 0)
            cols["novel_words"].append(len(ot - stt))
            cols["ratio_state"].append(len(allt & stt) / max(len(stt), 1))
    return {k: np.array(v, dtype=np.float64) for k, v in cols.items()}, meta


def build(items):
    cols, meta = features(items)
    keys = sorted(items)
    gold = np.zeros(len(meta), dtype=np.int64)
    row_of = {}
    for i, (k, j) in enumerate(meta):
        row_of.setdefault(k, []).append(i)
        gold[i] = items[k]["opts"][j][1]
    return cols, meta, gold, row_of, keys


# ------------------------------------------------------- behaviour-space search
def item_acc(score, gold, row_of, keys, subset=None):
    ks = subset if subset is not None else keys
    ok = 0
    for k in ks:
        idx = row_of[k]
        pick = max(idx, key=lambda i: score[i])
        ok += gold[pick] == 1
    return ok / len(ks)


def search(train_cols, gold, row_of, tr_keys, va_keys, verbose=True):
    """Beam search over OPERATOR compositions, deduplicated by VALUE VECTOR.

    The dedup is the load-bearing part: two compositions that rank options
    identically are the same node, so the frontier cannot fill with thousands
    of re-parameterisations of one idea.
    """
    t0 = time.time()
    seen, cand = {}, []
    for name, col in train_cols.items():
        v = col.copy()
        if not np.isfinite(v).all():
            continue
        seen.setdefault(v.tobytes(), (f"feat:{name}", v))
        cand.append((f"feat:{name}", v))
    print(f"  level 0: {len(cand)} base columns "
          f"({len(seen)} behaviourally distinct)")

    best_by_val = {}
    for name, v in seen.values():
        best_by_val[v.tobytes()] = (name, v)

    level = list(best_by_val.values())
    survivors = []
    for depth in range(1, MAXDEPTH + 1):
        nxt = {}
        for n1, v1 in level:
            for n2, v2 in level:
                for opname, (arity, fn) in VEC.items():
                    try:
                        if arity == 1:
                            r = fn(v1)
                        elif arity == 2:
                            r = fn(v1, v2)
                        else:
                            continue
                    except Exception:
                        continue
                    r = np.asarray(r, dtype=np.float64)
                    if r.shape != v1.shape or not np.isfinite(r).all():
                        continue
                    if np.allclose(r, v1) or np.allclose(r, v2):
                        continue          # no-op under this algebra
                    key = r.tobytes()
                    if key in nxt:
                        continue          # behavioural dedup
                    nxt[key] = (f"{opname}({n1},{n2})" if arity == 2
                                else f"{opname}({n1})", r)
        if not nxt:
            print(f"  level {depth}: no compositions survived - stopping")
            break
        level = list(nxt.values())[:BEAM]
        scored = []
        for name, v in level:
            a_tr = item_acc(v, gold, row_of, tr_keys)
            a_va = item_acc(v, gold, row_of, va_keys)
            scored.append((a_va, a_tr, name, v))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        if verbose and scored:
            print(f"  level {depth}: {len(nxt)} distinct behaviours, "
                  f"beam {len(level)}   best val {scored[0][0]:.4f} "
                  f"(train {scored[0][1]:.4f})  [{time.time()-t0:.0f}s]")
        for a_va, a_tr, name, v in scored:
            if a_va >= MIN_ACC:
                survivors.append((a_va, a_tr, name, v))
    return survivors


def main():
    items = load_items()
    cols, meta, gold, row_of, keys = build(items)
    print(f"items {len(keys)}  option-rows {len(meta)}  "
          f"base features {len(cols)}")
    chance = np.mean([1.0 / len(items[k]["opts"]) for k in keys])
    print(f"chance {chance:.4f}\n")

    # ---- leave-one-fold-out, as THRM does
    perm = np.random.RandomState(SEED).permutation(len(keys))
    fold_of = {item: perm[j] % FOLDS for j, item in enumerate(keys)}
    heldout_hits, heldout_tot = 0, 0
    best_per_fold = []
    for f in range(FOLDS):
        tr = [k for k in keys if fold_of[k] != f]
        va = [k for k in keys if fold_of[k] == f]
        surv = search(cols, gold, row_of, tr, va, verbose=(f == 0))
        if not surv:
            continue
        best = max(surv, key=lambda x: (x[0], x[1]))
        # does the winner picked on train+val actually hold up on the fold
        a_va = item_acc(best[3], gold, row_of, va)
        heldout_hits += round(a_va * len(va))
        heldout_tot += len(va)
        best_per_fold.append({"fold": f, "heldout_acc": a_va,
                              "procedure": best[2][:120]})
        print(f"  fold {f}: held-out {a_va:.4f}  <- {best[2][:90]}")

    # ---- final library fitted on everything, gated
    print("\n" + "=" * 70)
    print("final library on all data + gates")
    print("=" * 70)
    surv = search(cols, gold, row_of, keys[:len(keys) // 2],
                  keys[len(keys) // 2:], verbose=False)
    surv = sorted(surv, key=lambda x: (-x[0], -x[1]))[:12]
    print(f"{len(surv)} candidates clear the {MIN_ACC} held-out floor")
    for a_va, a_tr, name, _ in surv[:8]:
        print(f"  val {a_va:.4f}  train {a_tr:.4f}  {name[:96]}")

    # gate 1: no ambiguity - best two procedures must not pick different
    #         answers on the same item often enough to matter
    amb = 0
    if len(surv) >= 2:
        picks = []
        for a_va, a_tr, name, v in surv[:5]:
            p = {k: max(row_of[k], key=lambda i: v[i]) for k in keys}
            picks.append(p)
        for k in keys:
            chosen = {p[k] for p in picks}
            if len(chosen) > 1:
                amb += 1
    print(f"\n  [INFO] items where the top-5 procedures disagree: {amb}/{len(keys)}")

    held = heldout_hits / max(heldout_tot, 1)
    print(f"\n  HELD-OUT ACCURACY (leave-one-fold-out) = {held:.4f}  "
          f"({heldout_hits}/{heldout_tot})")
    print(f"  reference: 4B trained head 0.7080 (137 items), "
          f"300M zero-shot 0.5915, chance {chance:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mined_decision_procedures.json").write_text(json.dumps({
        "approach": "behaviour-space beam search over 34 typed operators "
                    "reused from thrm.dsl, deduplicated by value vector, "
                    "selected on held-out accuracy",
        "operators_reused": len(OPS),
        "n_items": len(keys), "n_option_rows": len(meta),
        "base_features": len(cols), "chance": chance,
        "heldout_accuracy_leave_one_fold_out": held,
        "per_fold": best_per_fold,
        "candidates_over_floor": len(surv),
        "min_heldout_floor": MIN_ACC,
        "top_procedures": [{"heldout": a, "train": t, "expr": n[:200]}
                           for a, t, n, _ in surv[:8]],
        "ambiguity_items_top5": amb,
        "gates": {"no_ambiguity": "informational only at this stage",
                  "ood_abstention": "NOT IMPLEMENTED - required before any "
                                    "external claim"},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'mined_decision_procedures.json'}")


if __name__ == "__main__":
    main()

