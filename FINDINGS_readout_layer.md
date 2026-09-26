# Read-out ablation: the shipped head reads the right layer (hypothesis refuted)

**Status: hypothesis REFUTED on the production recipe.** This file previously
reported the opposite. Both results are kept below because the negative result
is the useful one.

Date: 2026-09-26. Trunk `Qwen3-4B-Base`, frozen. 231 public items,
849 option sequences. All Gavel: cached forward, cheap MLP head, no trunk
training.

## Result (production recipe: `train_head.py`, 60 epochs, lr 1e-4, bs 256, seed 20260924)

| read-out | item acc | easy | std | hard | best_ep |
|---|---|---|---|---|---|
| **last_L36 (shipped)** | **0.8404** | 0.979 | 0.850 | **0.771** | 33 |
| last_L20 | 0.7465 | 1.000 | 0.750 | 0.629 | 14 |
| last_L21 | 0.7277 | 1.000 | 0.733 | 0.600 | 14 |

The shipped final-layer read-out is the best of the three, with the best hard
tier and the most stable optimisation (best_ep 33 vs 14). No change warranted.

## The earlier "L20 doubles accuracy" result was a harness artifact

A first pass reported L20 0.6645 vs L36 0.4329 out-of-fold (paired: fixes 82,
breaks 22, z = 5.79). That was measured under a broken harness in which L36
could not train properly (best_ep 5) while L20 could. It was measuring
trainability, not read-out quality. Three separate defects, each caught by a
gate:

1. **Scrambled features.** `for p, j in enumerate(order): Xo[p] = X[j]` unpacks
   as `p=position, j=order[position]`, so rows were permuted wrongly. Proof: the
   shipped head scored 0.3474 on the scrambled file where the correct answer is
   0.8216. Fixed to `for j, p in enumerate(order)`.
2. **Wrong recipe.** 40 epochs at lr 3e-4 was guessed; the real run is 60 epochs
   at lr 1e-4 (recovered from `combined_head_report.json`: val_acc 0.6119,
   best_ep 31 — a sweep reproduced val_acc 0.614 / best_ep 33).
3. **NLI rows dropped.** `train_head.py`'s split covers all 1073 items including
   the 1,720 NLI rows; excluding them trains on 689 instead of 2,409.

Also checked and ruled out: the shipped cache is `MAXT=512` but the training
pairs are short (p50 43 tokens, max 252), so **nothing is ever truncated** —
context length is not a confound here.

## What did survive: bf16 features beat 4-bit features

Same layer, same recipe, only the cache precision differs:

| cache | item acc |
|---|---|
| shipped, bitsandbytes nf4 | 0.8028 (retrained) / 0.8216 (shipped head) |
| **ours, bf16** | **0.8404** |

+1.9 points for free — we serve in bf16 regardless, so there is no inference
cost. This is 4 items out of 213, inside the noise band (SE ~0.025), so it is
suggestive rather than established. Confirming it needs a full 231-item run with
the new head, which costs GPU time we have not spent.

## Gates used (all in `scripts/prod_readout_final.py`)

- shipped head on shipped cache must score ~0.82
- shipped head on our cache must score ~0.78 (cache faithfulness)
- retrained head on shipped cache must score ~0.80 (recipe reproduction)


## Context length is NOT the lever

| ctx | best read-out | OOF acc |
|---|---|---|
| 512 | last_L19 | 0.6550 |
| 1024 | last_L17 | 0.6433 |
| 2048 | last_L21 | 0.6608 |

Flat within noise. Consistent with the field: `certo` truncates state to
**64 tokens** and still scores competitively, `openJev Verdict` uses 512 like
us, `ProgramAsWeights` uses 2048, and `decider-4b v2` uses 32k yet scores
**0.676** on the public hard tier against our **0.685**. Longer context
actively *hurt* us (hard 68.5% → 63.1% at ctx 2048) because the head was
trained on 512-truncated states.

## What is NOT yet established

**We could not produce a drop-in head that beats the shipped one end to end.**

`train_head.py` at its documented settings (40 epochs, lr 3e-4, bs 256) reaches
val_acc 0.5397 pairwise with `best_ep 5`, and fits T=2.7963 — which reproduces
`head/pair_head.pt` (T=2.796, no meta_cal), **not** the shipped
`v1/combined_head.pt` (T=2.4453 + embedded meta-calibrator, item accuracy
0.8216 on these features). The exact recipe that produced `combined_head.pt` is
not captured in a runnable script, so we cannot rebuild a better head from the
L20 read-outs yet.

Diagnostic that isolates this (kept as a gate in
`scripts/prod_readout.py`):
- shipped head on shipped 4-bit full-length cache: **0.8216** item accuracy
- shipped head on our bf16 ctx-384 cache at L36: **0.7793** (mean |h| ratio
  1.00, so our cache is faithful)
- freshly trained head on the same cache: **0.385**

So the cache is fine and the training loop is the missing piece.

## Three bugs found while measuring this

1. `combined_hiddens.pt` rows are in **original pair order**; our pod cache is
   **length-sorted**. Permuting the shipped hiddens made a correct harness
   score 0.333 instead of 0.8216. Now gated.
2. Full-batch training for 40 epochs is only **40 gradient steps**; the real
   recipe uses bs=256 (~320 steps). An effectively untrained head scored 0.33.
3. `train_head.py`'s split covers **all 1073 items including the 1,720 NLI
   rows**. Excluding NLI trains on 689 rows instead of 2,409.
