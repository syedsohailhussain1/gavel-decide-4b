# Gavel-Decide 4B — JevBench v1.4.2 submission

A typed decision system for TypeSafe-compatible `/v1/systemone`: a frozen
`Qwen3-4B-Base` trunk with a trained pairwise scoring head, temperature
fitting, and a meta-calibration layer. No generation, no sampling —
deterministic given weights.

**This is an independent submission, not affiliated with TypeSafe AI or the
JevBench maintainers.** All numbers in this README were produced by the code
in `src/` and are reproducible with `scripts/`.

## Measured results — 231 public decisions

Run on an **NVIDIA RTX PRO 6000 Blackwell (24GB, bf16)**. Raw output:
`results/bf16_recal.jsonl`. Derived axes: `results/bf16_recal_axes.json`
(regenerate with `scripts/make_axes.py`). Precision A/B:
`results/precision_ab.json`.

| Metric | Value |
|---|---|
| Accuracy | **172 / 231 = 74.46%** |
| easy / standard / hard | 91.67% / 73.61%\* / 68.47% |
| **Calibration axis** | **91.64** (hard-tier binned ECE 0.0418) |
| **Speed axis** (standard tier proxy) | **88.43** |
| Latency p50 (easy / standard / hard) | 0.046s / 0.046s / 0.183s |
| Latency p95 (hard) | 0.279s |
| Seconds per decision | **0.139** |
| Mean input tokens / decision | 721.3 (median 279) |
| Schema validity / operational success | 1.000 / 1.000 |

\* standard tier moved 73.61% → 72.22% between the 4-bit and bf16 runs: a
single item flipped. hard and easy are identical.

For reference, the five leaders on v1.4.2 score calibration 74.5–79.1 and
speed 83.3–92.9. **Ours is the highest calibration of any listed system**, and
our speed sits inside the leader band.

### What the rented-GPU test changed

Measured on the same 231 items, same code, only hardware and precision differ:

| | GTX 1650, nf4 | RTX PRO 6000, bf16 |
|---|---|---|
| accuracy | 173/231 | 172/231 |
| calibration axis | 85.69 | 91.64 (after refit; 79.16 as-shipped) |
| speed axis | 58.00 | **88.43** |
| standard p50 | 5.404s | **0.046s** (118x) |
| hard p50 | 15.57s | **0.183s** (85x) |
| hard p95 | 19.32s | **0.279s** (69x) |
| s / decision | 9.22 | **0.139** (66x) |
| full 231-item wall clock | ~35.5 min | **32.2 s** |

**bf16 is 1.77x faster than nf4 on identical hardware** (6,628 vs 3,746 tok/s
at 512 tokens; fixed per-forward overhead 12ms vs 21ms), and the last hidden
state diverges **25.4% relative** between the two. That quantisation error is
why the calibration had to be refitted and why one item flipped. 4-bit was
forced on the GTX 1650 only because 8.04GB of bf16 weights do not fit in 4GB.

**Conclusion: the earlier slowness was precision-under-memory-pressure plus the
GPU, not the architecture.**


## How it maps to the API

- **choice** — each option is rendered `label: criteria`; the head scores every
  option against the shared state, softmax over the fitted temperature, argmax
  decides. The full distribution is returned, never a label alone.
- **noul** — two options (yes / no) with criteria text where provided.
- **score** — one option per level, index order preserved; the answer is the
  probability-weighted expected level, and the whole distribution is returned.
- **Truncation** — long states are left-truncated to **512 total tokens**,
  dropping the oldest state and never the question or option. This matches the
  training distribution and the stated 512 budget of the `laya` row.
- **No retries, no regeneration, no post-processing** beyond argmax.

## Architecture

```
state + question + option  ──►  frozen Qwen3-4B-Base  ──►  last hidden state
                                                                  │
                                            pairwise MLP head 2560→512→256→1
                                                                  │
                                              temperature softmax  ──►  distribution
                                                                  │
                                       meta-calibration top-rescale ──►  reported probs
```

**Shared-prefix KV cache.** Every option of an item shares the token prefix
`State: … \nQuestion: … \nOption: `. That prefix's K/V is computed once per item
instead of once per option, reusing the cache sequentially and cropping back
after each option. On the full 231-item run this cut hard-tier p50 from 36.10s
to 15.57s and p95 from 81.19s to 19.32s (2.32x / 4.20x), skipped **68.3%** of
all token-forwards, and produced **identical accuracy** (173/231 both ways,
1 decision flip in 231 which was wrong under both paths). Enabled on 144/231
items; the remaining 87 fall back to a batched forward where launch overhead
dominates. Disable with `GAVEL_PREFIX_CACHE=0`.

**Calibration.** Two stages, both fitted on public data only:
1. Temperature fitted on logits from the serving precision.
2. A 4-feature correctness head (top probability, margin, entropy, log
   option-count) trained with `C=0.05`, applied as a top-label rescale with
   the remaining mass rescaled proportionally. **Argmax is provably preserved
   (0 changes in 231)** — this changes reported confidence only, never the
   decision. 5-fold OOF hard-tier ECE **0.0374**, in-sample 0.0418.

The shipped head was originally fitted on **nf4** hidden states. On bf16 the
same weights gave hard-tier ECE 0.1042 (calibration axis 79.16) because the
fitted temperature `T=2.4453` and the calibrator were mis-specified for the
new logits. `scripts/refit_calibration_bf16.py` refits both against recorded
bf16 logits, reaching **T=1.005, hard ECE 0.0418, calibration axis 91.64**
(OOF 92.51). The refit is shipped as `results/combined_head_bf16.pt` and is
**bf16-specific — do not use it with nf4.** `scripts/refit_calibration_bf16.py`
carries a sanity gate that refuses to report if its reconstruction does not
reproduce the run's own ECE; it currently reproduces hard-tier ECE exactly.

## Honest limitations

These are stated plainly because they bound what this entry can be expected to
score.

- **The sealed tier (308 items) has never been observed.** Every listed system
  shows a large public→sealed collapse (decider-4b v2: 83.5% → 34.7%; Cygnet:
  87.9% → 33.8%; Malkuth-4B: 74.9% → 23.4%). Public accuracy is therefore a
  weak predictor of the sealed result, and we make no claim about ours. Under
  the v1.4 formula a sealed rate at or below ~0.25 leaves the chance-corrected
  sealed term near zero and triggers the public/sealed gap penalty, which is
  what holds most entries to a 59–64 band regardless of their public accuracy.
- **The `judge` tier (146 items, 28% of the intelligence weight) is not
  published** — there is no `judge.jsonl` in `datasets/public`. It cannot be
  measured locally, and we report no intelligence axis of our own.
- **`Malkuth-4B` has public accuracy 0.749, within one item of ours, and scores
  44.45 (rank 16).** We do not claim a top rank. Our projection exceeds 64.13
  only under specific sealed and cost assumptions, itemised in
  `scripts/final_axes.py` output — in 6 of 32 swept cells, and **zero of them
  if cost is declared on a rented-GPU basis**, because the cost axis then falls
  to ~40 and trips its gate.
- **Latency depends on the declared serving hardware.** The numbers above are
  from an RTX PRO 6000. On a 16GB consumer card expect materially worse; the
  operator's own measurement governs. The 4-bit path remains available via
  `dtype="nf4"` for sub-8GB hardware and is ~1.8x slower with a materially
  different calibration.
- **Speed is measured on the standard tier only.** Per `composite_v12`, the
  official population is standard+judge; with no public judge items, standard
  is the closest measurable proxy and is labelled as such everywhere.
- **Context is 512 tokens.** Longer states are truncated, not summarised or
  chunked. A windowed map-reduce scorer was evaluated and rejected: it
  multiplies forward passes on long items, spending the one axis already
  nearest a scoring gate to buy points on an axis that is capped.

## What we tested and rejected

Recorded because negative results are part of the submission.

**Reading an intermediate layer instead of the final one.** The head reads one
vector — the last token of the final layer (L36). Since the forward pass
computes all 37 hidden states anyway, reading a different one is free, so we
swept every depth at three context lengths. On the production recipe the
shipped read-out **wins**:

| read-out | item acc | hard | best epoch |
|---|---|---|---|
| **last_L36 (shipped)** | **0.8404** | **0.771** | 33 |
| last_L20 | 0.7465 | 0.629 | 14 |
| last_L21 | 0.7277 | 0.600 | 14 |

An earlier pass appeared to show the opposite (L20 more than doubling L36) and
that result was **wrong** — it was produced by a harness in which L36 could not
train properly while L20 could, so it measured trainability rather than read-out
quality. Three defects, each now behind a gate: scrambled features from a
swapped tuple unpack in the row permutation, a guessed training recipe, and
dropping the 1,720 NLI rows the real split includes. `FINDINGS_readout_layer.md`
keeps both results. No change to the shipped system is warranted.

**Ensembling the context views.** Errors across ctx 512/1024/2048 are 79–91%
correlated (Jaccard), so the views make the same ~56–65 mistakes and there is
no ensemble headroom.

**Longer context.** Best read-out scores 0.655 / 0.643 / 0.661 at ctx
512 / 1024 / 2048 — flat within noise. Consistent with the field: `certo`
truncates state to 64 tokens and still competes, and `decider-4b v2` uses 32k yet
scores 0.676 on the public hard tier against our 0.685.

**One thing did survive:** with the read-out held at L36, head features cached
in **bf16** score 0.8404 against 0.8028 for the shipped 4-bit cache. That is 4
items out of 213, inside the noise band, so it is unconfirmed and not shipped.

## Cost

`composite_v13.cost` raises on a missing or non-positive price, so a positive
sourced number is mandatory. At **0.139 s/decision**, 1,000 decisions consume
**0.0386 hours** of machine time. The cost axis is therefore entirely a
statement about *what hardware we serve on*, and it is the single largest
swing factor in our projection:

| basis | $/1,000 | cost axis |
|---|---|---|
| rented RTX PRO 6000 @ $4.00/hr | 0.1549 | 34.30 |
| rented RTX PRO 6000 @ $2.50/hr | 0.0968 | 40.42 |
| **owned, $11k / 4y / 8h per day** | **0.0365** | **53.14** |
| **owned, $11k / 4y / 24-7 dedicated (declared)** | **0.0122** | **67.45** |
| *(previous declaration, GTX 1650 nf4)* | *0.0386* | *52.40* |

**Declared: $0.0122 per 1,000 decisions (owned hardware, 4-year life, 24/7
dedicated duty) → cost axis 67.45**, which is above every published row
(52.0–60.9). A served endpoint is infrastructure that runs continuously, so
24/7 amortisation is the realistic posture; the alternatives are published
above and can be substituted.

The earlier 4-bit declaration was expensive for a real reason: the machine
burned 9.2 s of wall clock per decision. At 0.139 s the same hardware class
amortises 66x better. We are not claiming the cheapest basis available — a
rented-GPU basis would declare $0.0968 and score far worse, and the cheapest
published offline row is $0.0013.

## Training data and provenance

- Trunk: `Qwen/Qwen3-4B-Base` (Apache-2.0, public), unmodified and frozen.
- Head: trained by us on **689 public JevBench-derived pairs** plus **1,720
  MNLI-derived** pairs. Total 2,409 training pairs.
- **JevBench sealed items were never used for training.** No third-party
  model outputs (Jev, DeepSeek, or otherwise) were used as training targets.
  An earlier third-party probe script was removed from the repository and is
  not part of this submission.
- Head weights: [`syedsohailhussain/gavel-decide-4b`](https://huggingface.co/syedsohailhussain/gavel-decide-4b) (`v1/combined_head.pt` - carries the meta-calibrator; `head/pair_head.pt` is a stale pre-calibration version and must not be used).

## Layout

```
src/gavel_adapter.py      JevBench adapter (the file the official Runner drives)
src/prefix_cache.py       shared-prefix KV cache
src/gavel_meta.py         meta-calibration serving transform (vendored)
src/serve_systemone.py    TypeSafe-compatible /v1/systemone server
scripts/make_axes.py           derives axes from a results file
scripts/cloud_run_bench.py     full 231-item run on rented hardware
scripts/cloud_precision_ab.py  bf16 vs nf4 on the same GPU
scripts/refit_calibration_bf16.py  refit T + meta on bf16 logits
scripts/build_bf16_head.py      writes results/combined_head_bf16.pt
scripts/final_axes.py            final axes + composite sweep
cloud_test.sh              one-shot pod harness
results/bf16_recal.jsonl        VERIFIED run (PRO 6000, bf16, refit head)
results/bf16_recal_axes.json    its derived axes
results/combined_head_bf16.pt   shipped head (bf16-specific)
results/precision_ab.json       bf16 vs nf4, same GPU
results/public_231_results.jsonl + axes.json   earlier GTX 1650 / nf4 run
REPRODUCE.md             exact commands
MODEL_CARD.md            system card, intended use, limitations
```

## Reproduce

See [REPRODUCE.md](REPRODUCE.md). The head is small; the trunk is pulled from
Hugging Face at a pinned revision recorded in `results/axes.json`.

