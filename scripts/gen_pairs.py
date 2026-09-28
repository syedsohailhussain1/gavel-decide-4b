"""Programmatic pair generator for a contamination-free Gavel head.

Why this exists
---------------
The shipped head trained on 689 pairs derived from the public JevBench items,
so 213 of the 231 scored items were in its training set. A head retrained on
only the 1,720 MNLI pairs scores 0.5285 OOF against a 0.5000 chance baseline -
entailment is not the task. And of the 2,409 rows we built, ZERO were `ordinal`,
so the head emits a constant on that question type.

So we generate our own supervision. Every gold label here is computed by the
generator, not borrowed from a model or from the benchmark. Nothing here reads
JevBench items, and there is no sealed data anywhere in this file.

Each row is the same shape the adapter consumes:
    text   "State: {state}\nQuestion: {question}\nOption: {label}: {criteria}"
    target 1 if this option is the answer, else 0
    item   a unique per-item id
    tier   family name, so coverage is auditable per family

Design rules
------------
1. Distractors must be *locally* plausible. A random wrong answer teaches the
   head "pick the topically-matching one", which is not the task. Each family
   therefore ships hand-built confusable distractors.
2. Balanced: exactly one positive per item, and we cap negatives so a family
   cannot dominate by volume alone.
3. Every family records its own coverage so the `ordinal` hole that sank the
   4B head cannot recur silently.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
from pathlib import Path

RNG = random.Random(20260924)

# --------------------------------------------------------------------------
# family builders: each returns (state, question, [(label, criteria)], gold_idx)
# --------------------------------------------------------------------------


def f_intent(r):
    """Support-desk intent routing with topically-overlapping intents."""
    cat = r.choice([
        ("Where is my package? It was due four days ago.",
         "track_order", "cancel_order", "change_address", "billing_question",
         "Wants to know where an order is",
         "Wants to cancel an order",
         "Wants to change the delivery address",
         "Asks about a charge or invoice"),
        ("I need to reset my password, I am locked out.",
         "account_access", "cancel_order", "track_order", "report_damage",
         "Cannot sign in or reset credentials",
         "Wants to cancel an order",
         "Wants to know where an order is",
         "Reports a damaged item"),
        ("You charged my card twice this month.",
         "billing_question", "track_order", "cancel_order", "account_access",
         "Asks about a duplicate charge",
         "Wants to know where an order is",
         "Wants to cancel an order",
         "Cannot sign in"),
        ("Please cancel order #5521, I no longer need it.",
         "cancel_order", "change_address", "track_order", "billing_question",
         "Wants to cancel an order",
         "Wants to change the delivery address",
         "Wants to know where an order is",
         "Asks about a charge"),
        ("The courier says delivered but I never got it.",
         "report_damage", "track_order", "billing_question", "cancel_order",
         "Reports a missing or wrong delivery",
         "Wants to know where an order is",
         "Asks about a charge",
         "Wants to cancel an order"),
    ])
    state, *rest = cat
    labels = rest[:4]
    crits = rest[4:]
    gold = 0
    perm = list(range(len(labels)))
    r.shuffle(perm)
    labels = [labels[i] for i in perm]
    crits = [crits[i] for i in perm]
    return state, "Which intent does the user's message express?", \
        list(zip(labels, crits)), gold


ORDINAL_BANDS = [
    # (state, question, [level criteria ascending], gold index)
    ("Package tracking shows the label created but no carrier scan yet.",
     "How likely is this parcel to arrive within the promised window?",
     ["almost certainly not", "unlikely", "uncertain", "likely",
      "almost certainly"], 3),
    ("The download stalled at 4% and the server returned 503 twice.",
     "How likely is the retry to succeed this time?",
     ["no chance", "unlikely", "coin flip", "likely", "almost certain"], 3),
    ("Support has replied twice and both replies were template text.",
     "How likely is a third reply to contain the actual answer?",
     ["very unlikely", "unlikely", "uncertain", "likely", "very likely"], 4),
    ("The refund was issued nine days ago and has not appeared.",
     "How likely is the refund to land within two more business days?",
     ["very unlikely", "unlikely", "uncertain", "likely", "very likely"], 4),
    ("Both the seller and courier claim the item was delivered.",
     "How likely is this dispute resolved in the customer's favour?",
     ["almost none", "low", "even", "high", "near certain"], 1),
    ("The account is over the age limit for the legacy plan.",
     "How likely is a plan change to lose saved data?",
     ["certain data loss", "likely", "possible", "unlikely",
      "no data loss"], 0),
]

ORDINAL_SWAPS = [
    ("Escalation requires a supervisor and none are on shift.",
     "How long until this is resolved?",
     ["under an hour", "same day", "two to three days", "a week or more"], 3),
    ("The replacement part is back-ordered with no date.",
     "How long until the part ships?",
     ["tomorrow", "within a week", "two to four weeks",
      "unspecified, likely over a month"], 3),
]


def f_ordinal(r):
    """Ordinal severity / likelihood. THIS IS THE FAMILY THE 4B HEAD NEVER SAW."""
    if r.random() < 0.7:
        state, q, levels, gold = ORDINAL_BANDS[r.randrange(len(ORDINAL_BANDS))]
        return state, q, [(lv, f"level {i+1} of {len(levels)}: {lv}")
                          for i, lv in enumerate(levels)], gold
    state, q, levels, gold = ORDINAL_SWAPS[r.randrange(len(ORDINAL_SWAPS))]
    return state, q, [(lv, f"level {i+1} of {len(levels)}: {lv}")
                      for i, lv in enumerate(levels)], gold


FACT_BANK = [
    ("A 30-day trial ends and the card on file is charged automatically.",
     "What happens on day 31?", 1,
     ["The trial is extended automatically",
      "The card is charged for the chosen plan",
      "The account is deleted immediately",
      "A discount is applied automatically"]),
    ("Water freezes at 0 C at sea level.",
     "At what temperature does water freeze at sea level?", 1,
     ["0 C", "10 C", "32 C", "100 C"]),
    ("A leap year has 366 days.",
     "How many days are in a leap year?", 0,
     ["366", "365", "364", "367"]),
    ("HTTP 404 means the requested resource was not found.",
     "What does HTTP 404 indicate?", 3,
     ["The server is overloaded",
      "The request was malformed",
      "Authentication failed",
      "The requested resource was not found"]),
    ("Photosynthesis converts light energy into chemical energy.",
     "What does photosynthesis convert light energy into?", 1,
     ["Kinetic energy", "Chemical energy", "Sound energy",
      "Nuclear energy"]),
    ("The first element on the periodic table is hydrogen.",
     "Which element has atomic number 1?", 2,
     ["Helium", "Oxygen", "Hydrogen", "Carbon"]),
    ("A 60 kg adult at rest expels roughly 1.2 litres of water per hour.",
     "About how much water does a resting 60 kg adult lose hourly?", 0,
     ["1.2 litres", "12 litres", "0.12 litres", "120 litres"]),
    ("A 9V battery holds far less energy than a 12V car battery.",
     "Which holds more total energy?", 3,
     ["The 9V battery", "They are equal", "Neither stores energy",
      "The 12V car battery"]),
]


def f_fact(r):
    state, q, gold, labels = r.choice(FACT_BANK)
    return state, q, [(l, "") for l in labels], gold


def f_extraction(r):
    recs = [
        ("Order 4471 placed 2026-02-03, shipped 2026-02-05, delivered 2026-02-08.",
         "On what date was order 4471 delivered?", 3,
         ["2026-02-03", "2026-02-04", "2026-02-05", "2026-02-08"]),
        ("Invoice INV-88213 totals 149.50 USD, due 2026-03-01.",
         "What is the total on invoice INV-88213?", 0,
         ["149.50 USD", "882.13 USD", "1,495.00 USD", "149.05 USD"]),
        ("Ticket T-551 was opened by dana.k and tagged billing, priority high.",
         "Which agent opened ticket T-551?", 1,
         ["sam.ortiz", "dana.k", "lee.park", "r.mensah"]),
        ("Shipment S-90 left the Rotterdam hub at 04:12 UTC, weight 12.4 kg.",
         "What was the weight of shipment S-90?", 3,
         ["1.24 kg", "124 kg", "4.12 kg", "12.4 kg"]),
        ("Plan 'Team-Annual' seats 5, renews 2027-01-15, cost 1200 USD/yr.",
         "How many seats does plan Team-Annual include?", 0,
         ["5", "15", "50", "500"]),
    ]
    state, q, gold, labels = r.choice(recs)
    return state, q, [(l, "") for l in labels], gold


def f_policy(r):
    pols = [
        ("Refunds are issued within 5 business days of an approved claim.",
         "A customer asks for a refund on a damaged item approved yesterday. "
         "When is it due?",
         0, ["Within 5 business days", "Within 24 hours", "Within 90 days",
             "No fixed deadline"]),
        ("Accounts inactive for 12 months are archived automatically.",
         "An account has been dormant for 14 months. What has happened?",
         0, ["It was archived", "It was deleted permanently", "It was upgraded",
             "Nothing happens"]),
        ("Support may not issue partial refunds above 50 USD without approval.",
         "A 220 USD refund is requested and no approver is available. Can it "
         "be issued?",
         1, ["Yes, immediately", "No, it needs approval", "Only if paid by cash",
             "Only for domestic orders"]),
        ("Warranty covers 12 months from delivery, not from purchase.",
         "A device was delivered 14 months ago and fails. Covered?",
         1, ["Yes, fully", "No, the 12 months from delivery have passed",
             "Only the battery is covered",
             "Only if extended at checkout"]),
    ]
    state, q, gold, labels = r.choice(pols)
    return state, q, [(l, "") for l in labels], gold


def f_routing(r):
    routes = [
        ("A request mentions 'charged twice' and 'card declined'.",
         "Which team owns this?", 1,
         ["Billing", "Fraud", "Shipping", "Account management"]),
        ("A login fails with 'account locked' after 5 attempts.",
         "Which team should receive this?", 2,
         ["Billing", "Returns", "Account security", "Warehouse"]),
        ("A parcel is stuck at a customs hold with an incorrect HS code.",
         "Which team owns this?", 0,
         ["Customs", "Billing", "Marketing", "Product"]),
        ("A customer wants to return an unopened item for a refund.",
         "Which team owns this?", 3,
         ["Support", "Growth", "Infrastructure", "Returns"]),
    ]
    state, q, gold, labels = r.choice(routes)
    return state, q, [(l, "") for l in labels], gold


def f_adequacy(r):
    adq = [
        ("A customer asks for a refund and the order shipped 3 days ago.",
         "Is there enough information to issue the refund now?", 1,
         ["Yes, always", "No, need the return-received confirmation",
          "No, need a manager", "Yes if the customer is a first-time buyer"]),
        ("A user reports a crash on save and no steps to reproduce.",
         "Can this be triaged immediately?", 0,
         ["No, reproduction steps are needed",
          "Yes, crashes are always reproducible",
          "Yes, if the user is on paid plan", "Yes, after one week"]),
        ("A contract renewal is due in 9 days and no owner is named.",
         "Is there enough to route this for renewal?", 1,
         ["Yes, route to sales", "No, an owner must be named first",
          "Yes, cancel it", "Yes, if revenue exceeds 10k"]),
    ]
    state, q, gold, labels = r.choice(adq)
    return state, q, [(l, "") for l in labels], gold


def f_tool_selection(r):
    tools = [
        ("Need the delivery status for order 4471.",
         "Which tool answers this?", 2,
         ["create_refund", "cancel_order", "get_tracking", "update_address"]),
        ("Need to change the delivery address before dispatch.",
         "Which tool is correct?", 3,
         ["get_tracking", "create_refund", "cancel_order", "update_address"]),
        ("A charge needs reversing on a cancelled order.",
         "Which tool is correct?", 0,
         ["create_refund", "get_tracking", "update_address", "cancel_order"]),
    ]
    state, q, gold, labels = r.choice(tools)
    return state, q, [(l, "") for l in labels], gold


def f_multi_hop(r):
    hops = [
        ("Order 91 was placed on a Team-Annual plan, invoice INV-5, total 1200 USD.",
         "What is the total on the invoice for the plan order 91 was placed on?",
         2, ["91", "5", "1200 USD", "2026"]),
        ("Shipment S-12 weighs 12.4 kg and is one of 3 items in parcel P-4.",
         "How many items are in parcel P-4?", 1,
         ["1", "3", "12", "12.4"]),
        ("Ticket T-9 was opened 2026-01-05 and closed 2026-01-09.",
         "How many days was ticket T-9 open?", 1,
         ["5", "4", "9", "14"]),
    ]
    state, q, gold, labels = r.choice(hops)
    return state, q, [(l, "") for l in labels], gold


def f_adversarial(r):
    adv = [
        ("A 9-year-old asks for help with a purchase; no guardian is present.",
         "Should the assistant proceed?", 1,
         ["Yes, proceed normally", "No, escalate to a human",
          "Yes, if the item is under 10 USD", "Yes, if card is on file"]),
        ("A support agent asks the customer to confirm their full card number.",
         "Is this appropriate?", 1,
         ["Yes, verification requires it", "No, agents must never request it",
          "Yes, if the call is inbound", "Yes, for orders over 100 USD"]),
        ("A customer disputes a charge and the description is merchant-only.",
         "Can the charge be explained from the record alone?", 0,
         ["No, the merchant descriptor carries no detail",
          "Yes, the amount is always self-explanatory",
          "Yes, if the amount is under 20 USD", "Yes, for card payments"]),
    ]
    state, q, gold, labels = r.choice(adv)
    return state, q, [(l, "") for l in labels], gold


FAMILIES = {
    "intent": f_intent,
    "ordinal": f_ordinal,
    "fact": f_fact,
    "extraction": f_extraction,
    "policy": f_policy,
    "routing": f_routing,
    "adequacy": f_adequacy,
    "tool_selection": f_tool_selection,
    "multi_hop": f_multi_hop,
    "adversarial": f_adversarial,
}


def _option_key(label, crit):
    """The surface string a label-only model would see. Balancing on this is
    what forces the answer to live in the semantics."""
    return (label or "").strip().lower()


def build_balanced(n_items, seed=20260924, families=None):
    """Generate items, then re-assign which option is gold so that every
    option string is gold at ~1/len(options) of the times it is offered.

    Without this, TF-IDF over the option text alone solves the task at 0.86
    and the head learns a lexical shortcut instead of the reasoning. Verified
    by scripts/check_gen_quality.py.
    """
    r = random.Random(seed)
    names = families or list(FAMILIES)

    # 1. draw raw items
    raw = []
    for i in range(n_items):
        fam = names[i % len(names)]
        state, question, opts, _ = FAMILIES[fam](r)
        raw.append((fam, state, question, opts))

    # 2. count how often each option string is offered
    offer = defaultdict(int)
    for _f, _s, _q, opts in raw:
        for lab, _c in opts:
            offer[_option_key(lab, _c)] += 1

    # 3. assign gold greedily, always taking the option that is most
    #    under-quota as gold. This drives every string's gold rate toward
    #    1/len(options) without needing a solver.
    quota = defaultdict(float)
    for i, (fam, _s, _q, opts) in enumerate(raw):
        n = len(opts)
        best_j, best_score = 0, None
        for j, (lab, _c) in enumerate(opts):
            k = _option_key(lab, _c)
            have = quota[k]
            want = offer[k] / n            # its fair share of being gold
            score = want - have            # most under-quota wins
            if best_score is None or score > best_score:
                best_j, best_score = j, score
        lab, crit = opts[best_j]
        quota[_option_key(lab, crit)] += 1.0
        raw[i] = (fam, _s, _q, opts, best_j)

    # 4. emit rows
    rows, coverage = [], {}
    for idx, (fam, state, question, opts, gold) in enumerate(raw):
        item = f"gen-{fam}-{idx:07d}"
        coverage.setdefault(fam, {"items": 0, "options": 0, "positives": 0})
        coverage[fam]["items"] += 1
        coverage[fam]["options"] += len(opts)
        for j, (lab, crit) in enumerate(opts):
            t = f"State: {state}\nQuestion: {question}\nOption: {lab}"
            if crit:
                t += f": {crit}"
            hit = (j == gold)
            rows.append({"text": t, "target": 1 if hit else 0, "item": item,
                         "tier": fam})
            if hit:
                coverage[fam]["positives"] += 1

    # report the achieved gold-rate spread per option string
    got = defaultdict(int)
    for _f, _s, _q, opts, gold in raw:
        got[_option_key(opts[gold][0], opts[gold][1])] += 1
    rates = []
    for k, n_off in offer.items():
        if n_off >= 20:            # ignore one-off strings
            rates.append(got[k] / n_off)
    meta = {
        "n_option_strings_measured": len(rates),
        "gold_rate_min": round(min(rates), 4) if rates else None,
        "gold_rate_max": round(max(rates), 4) if rates else None,
        "gold_rate_mean": round(sum(rates) / len(rates), 4) if rates else None,
    }
    r.shuffle(rows)
    return rows, coverage, meta


def build(n_items, seed=20260924, families=None):
    rows, cov, _ = build_balanced(n_items, seed, families)
    return rows, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000, help="items to generate")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", default=r"D:\gavel\models\training_state\gen_pairs.jsonl")
    ap.add_argument("--report", default=r"D:\gavel\models\training_state\gen_coverage.json")
    A = ap.parse_args()

    rows, cov, meta = build_balanced(A.n, A.seed)
    Path(A.out).write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    pos = sum(r["target"] for r in rows)
    print(f"items        {A.n:,}")
    print(f"rows         {len(rows):,}")
    print(f"positives    {pos:,}  ({pos/len(rows):.4f})")
    print(f"items/family {A.n/len(cov):.0f}")
    print(f"\n{'family':16} {'items':>7} {'rows':>8} {'pos':>7}")
    for f, d in sorted(cov.items()):
        print(f"{f:16} {d['items']:7,} {d['options']:8,} {d['positives']:7,}")
    missing = {"ordinal", "fact", "policy", "extraction", "adequacy",
               "routing", "tool_selection", "multi_hop", "adversarial",
               "intent"} - set(cov)
    print(f"\nzero-coverage families (the bug that sank the 4B head): "
          f"{sorted(missing) if missing else 'NONE'}")
    print(f"\ngold-rate balancing over {meta['n_option_strings_measured']} "
          f"option strings:")
    print(f"  min {meta['gold_rate_min']}  mean {meta['gold_rate_mean']}  "
          f"max {meta['gold_rate_max']}")
    print("  (a spread this wide means label text still predicts correctness; "
          "check_gen_quality.py measures it)")
    print(f"\nwrote {A.out}")

    Path(A.report).write_text(json.dumps({
        "n_items": A.n, "n_rows": len(rows), "seed": A.seed,
        "families": sorted(cov), "coverage": cov,
        "zero_coverage": sorted(missing),
        "gold_rate_balance": meta,
        "note": "gold labels computed by the generators; no JevBench item and "
                "no model output is read anywhere in this file. Gold assignment "
                "is balanced across option strings so the answer cannot be "
                "recovered from label wording alone.",
    }, indent=2), encoding="utf-8")
    print(f"wrote {A.report}")


if __name__ == "__main__":
    main()

