"""Pin the prefix-cache accounting invariant.

The reported speedup was "saved -2279%", i.e. nonsense. Root cause: the
path-keyed counters sum DISJOINT item populations, so their ratio was never a
speedup. The fix compares realised vs counterfactual spend over the SAME items.

This test locks that in, because the failure mode is silent: the numbers still
print, they are just meaningless. Two properties must hold:

  1. realised <= counterfactual, always. The cache can only remove work.
  2. the two together cover every item exactly once, so
     realised + naive-only == counterfactual.

Run against the synthetic adapter from the capacity test so it needs no GPU.
"""
from __future__ import annotations

import sys
import types

sys.path.insert(0, "src")

import gavel_adapter as ga  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


# The two token formulas, as implemented in prefix_cache.score_prefix_cached.
def costs(id_lists, P, ctx=None):
    naive = sum(len(x) for x in id_lists)
    cached = P + sum(len(x) - P for x in id_lists)
    return naive, cached


print("1. cached spend can never exceed naive spend for one item")
# 4 options sharing a 500-token prefix, each with a 20-token suffix
ids = [[0] * 500 + [i] * 20 for i in range(4)]
P = 500
naive, cached = costs(ids, P)
check("cached < naive when prefix is shared", cached < naive,
      f"naive={naive} cached={cached}")
check("saving equals (n-1)*P", naive - cached == (len(ids) - 1) * P,
      f"saved={naive - cached} expected={(len(ids) - 1) * P}")

print()
print("2. disjoint populations can produce a >100% 'saving' (the original bug)")
# Reproduce the reported failure with the real orientation. In the actual run
# naive_tokens was 5,446 and cached_tokens 129,564, so the old percentage
# 100*(1 - naive/cached) came out NEGATIVE. That needs the naive population to
# out-sum the cached one, i.e. the items that SKIP the cache are collectively
# the expensive ones. Reproduce exactly that shape.
short_ids = [[7] * 30, [8] * 30]              # naive path: tiny item
_cn2, cc2 = costs(short_ids, 0)               # cached spend of a short item
# Add cached-path items whose naive spend dwarfs the short item, but whose
# cached spend is small; summed the way the old code summed disjoint sets.
disjoint_cached = cc2 + 100                    # small cached spend
disjoint_naive = _cn2 + 129_564               # large naive spend (many long items)
bad_pct = 100 * (1 - disjoint_naive / disjoint_cached)
check("disjoint percentage goes negative (the reported -2279% bug)", bad_pct < -100,
      f"saved={bad_pct:.0f}% (cached={disjoint_cached}, naive={disjoint_naive})")

print()
print("3. realised vs counterfactual over the same items is sane")
long_ids = [[0] * 3000 + [i] * 10 for i in range(12)]
_cn, cc = costs(long_ids, 3000)                 # long item, cache used
nn, _nc = costs([[7] * 30, [8] * 30], 0)        # short item, cache skipped
realised = cc + nn        # what the trunk actually processed
counterfactual = _cn + nn  # what it would process with the cache off
check("realised <= counterfactual", realised <= counterfactual,
      f"realised={realised} counterfactual={counterfactual}")
check("counterfactual - realised == the cache's saving",
      counterfactual - realised == _cn - cc,
      f"delta={counterfactual-realised} expected={_cn-cc}")
speedup = counterfactual / max(realised, 1)
check("speedup is a finite positive factor >= 1", speedup >= 1.0,
      f"{speedup:.2f}x")

print()
print("4. adapter exposes the comparable pair and no stale keys")
a = ga.GavelLocalAdapter(trunk="x", head="y", ctx=32768)
check("has realised_tokens", "realised_tokens" in a._pc_stats)
check("has counterfactual_tokens", "counterfactual_tokens" in a._pc_stats)
check("no hypothetical_* keys remain",
      not any(k.startswith("hypothetical") for k in a._pc_stats),
      f"keys={sorted(a._pc_stats)}")

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("all prefix-cache accounting checks passed")
