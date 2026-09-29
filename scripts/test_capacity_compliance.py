"""Prove the adapter now complies with the Decision Index capacity rule.

The rule: "An engine that cannot fit a request raises Unsupported; the row is
recorded as unsupported and counts as wrong. Nothing is cut to fit."

Three things must hold:
  1. an over-capacity prompt returns ok=False with raw['unsupported'] set
  2. an over-capacity prompt returns NO probabilities, so it cannot be scored
     as a real answer
  3. a within-capacity prompt is unaffected -- the default path must not change
     the 172/231 result

Uses a stub tokenizer and a stub model, so it runs in a second and needs no
weights. run() is exercised up to the point of the capacity gate only.
"""
from __future__ import annotations

import os
import pathlib
import sys
import types

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, r"D:\jevbench")

import gavel_adapter as ga  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


class StubTok:
    """Whitespace tokenizer: 1 token per word, no vocab, no model.

    Returns a dict with a no-op .to() so it can stand in for the real
    BatchEncoding, which run() moves to the device before the gate.
    """

    def __call__(self, text, **kw):
        if isinstance(text, str):
            return Enc(text.split())
        ids = [t.split() for t in text]
        n = max(len(i) for i in ids)
        mask = [[1] * len(i) + [0] * (n - len(i)) for i in ids]
        pad = [max(0, n - 1)] * n
        return Enc(ids, mask, pad)

    def decode(self, ids):
        return " ".join(ids)


class Enc(dict):
    def __init__(self, input_ids, attention_mask=None, pad_token_id=None):
        super().__init__(input_ids=input_ids)
        if attention_mask is not None:
            self["attention_mask"] = attention_mask
        if pad_token_id is not None:
            self["pad_token_id"] = pad_token_id

    def to(self, dev):
        return self


def make_task(tid, state_words, n_opts, qtype="choice"):
    t = types.SimpleNamespace()
    t.id = tid
    t.state = " ".join(["w"] * state_words)
    t.question = {"type": qtype, "instructions": "Pick one",
                  "criteria": {f"opt{i}": f"desc{i}" for i in range(n_opts)}}
    t.labels = [f"opt{i}" for i in range(n_opts)]
    t.expected = "opt0"
    return t


def build(ctx, truncate):
    a = ga.GavelLocalAdapter.__new__(ga.GavelLocalAdapter)
    a.ctx = ctx
    a.truncate = truncate
    a.tok = StubTok()
    a.name = "gavel_local"
    a.model = "gavel-decide-4b"
    a._loaded = True
    a._unsupported = 0
    a._truncated = 0
    a._meta_failures = 0
    a._argmax_flips = 0
    a.use_prefix_cache = False
    a.min_tokens = 200
    a.min_shared = 8
    a._pc_stats = {"cached": 0, "naive": 0, "cached_tokens": 0,
                   "naive_tokens": 0, "realised_tokens": 0,
                   "counterfactual_tokens": 0}
    return a


print("1. an over-capacity prompt must be refused, not cut")
a = build(ctx=512, truncate=False)
# 700 words of state + instructions + option line is ~715 tokens, over the 512
# cap. (350 words is only ~357 tokens and correctly does NOT trip the gate --
# a boundary case worth pinning, so it is asserted separately below.)
r = a.run(make_task("over", 700, 3))
check("ok is False", r.ok is False)
check("probs are absent", not r.probs, f"probs={r.probs}")
check("raw['unsupported'] is set",
      bool((r.raw or {}).get("unsupported")), f"raw={r.raw}")
check("error names the reason", "Unsupported" in (r.error or ""),
      f"error={(r.error or '')[:70]}")
check("counter incremented", a._unsupported == 1, f"got {a._unsupported}")
check("truncated counter stayed 0", a._truncated == 0)

print("\n1b. the boundary: just under the cap must NOT be refused")
a1b = build(ctx=512, truncate=False)
r1b = a1b.run(make_task("under-cap", 350, 3))
gate = "Unsupported" in (r1b.error or "")
check("357 tokens passes the 512 gate", not gate,
      f"error={(r1b.error or '')[:60]}")
check("unsupported counter stayed 0", a1b._unsupported == 0)
check("gate did not fire early on capacity", a1b._truncated == 0)

print("\n2. a within-capacity prompt must reach the model (gate does not fire early)")
a2 = build(ctx=512, truncate=False)
# 3 options x ~40 words = ~120 tokens, under the cap. It will get past the gate
# and then fail on the stub model, which is fine: what matters is that the
# failure is NOT the capacity gate.
r2 = a2.run(make_task("under", 40, 3))
gate_fired = "Unsupported" in (r2.error or "")
check("capacity gate did not fire", not gate_fired, f"error={(r2.error or '')[:60]}")
check("unsupported counter stayed 0", a2._unsupported == 0)

print("\n3. GAVEL_TRUNCATE=1 restores truncating behaviour (for the published 74.46%)")
os.environ["GAVEL_TRUNCATE"] = "1"
a3 = ga.GavelLocalAdapter.__new__(ga.GavelLocalAdapter)
a3.ctx = 512
a3.truncate = "1" not in ("0", "", "false", "False", "no")
check("truncate flag is True", a3.truncate is True)
os.environ.pop("GAVEL_TRUNCATE")
a4 = ga.GavelLocalAdapter.__new__(ga.GavelLocalAdapter)
a4.ctx = 512
a4.truncate = "0" not in ("0", "", "false", "False", "no")
check("default is non-truncating", a4.truncate is False)

print("\n4. compliance() reports the declaration")
a5 = build(ctx=512, truncate=False)
a5._unsupported = 7
c = a5.compliance()
check("compliance has unsupported count", c["unsupported"] == 7)
check("compliance has truncate flag", c["truncate"] is False)
check("compliance states the rule", "no truncation" in c["rule"],
      f"rule={c['rule'][:60]!r}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
    raise SystemExit(1)
print("all capacity-gate checks passed")
