# Read-out ablation: the shipped head reads the wrong layer

Date: 2026-09-26. Trunk `Qwen/Qwen3-4B-Base`, frozen. 231 public items,
849 option sequences. All Gavel: cached forward, cheap MLP head, no trunk
training.

## The finding

The shipped head reads **one vector: the last token of the FINAL layer (L36)**.
Training the identical head on identical cached forwards, changing only which
layer is read, gives a clean inverted-U with the shipped read-out at the
bottom:

```
L0  0.175   L13 0.456   L19 0.632   L23 0.632   L30 0.614   L35 0.538
L1  0.193   L14 0.515   L20 0.632   L24 0.620   L31 0.620   L36 0.328  <-- SHIPPED
L2  0.269   L15 0.532   L21 0.626   L25 0.573   L32 0.579
L6  0.298   L16 0.556   L22 0.620   L26 0.561   L33 0.573
L8  0.386   L17 0.597                L27 0.561   L34 0.550
```

Confirmed with 5-fold CV over items, temperature refit inside each fold:

| read-out | OOF item acc | easy | std | hard |
|---|---|---|---|---|
| **last_L20** | **0.6645 ± 0.0031** | 0.938 | 0.778 | 0.473 |
| last_L36 (shipped) | 0.4329 ± 0.0429 | 0.719 | 0.451 | 0.297 |

Paired per item: L20 fixes **82**, breaks **22** (continuity-corrected
z = 5.79). The shipped read-out is also 14x less stable across seeds.

**It costs nothing at inference.** The forward pass computes all 37 hidden
states regardless; we were simply throwing 36 of them away.

Interpretation: the final layers of an LLM specialise next-token prediction, so
the last-token state is optimised for "what comes next" rather than for
representing state+question+option. A mid-stack layer carries a far more usable
summary.

Corroboration from an independent source: `decider` (the #1 JevBench entry)
reads hidden states at dedicated **answer slots** and projects them onto option
label tokens — the same principle, put the read-out where the signal is. They
do it with a full fine-tune; we get part of it for free from a frozen trunk.

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
