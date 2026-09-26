# Gavel-Decide 4B — JevBench v1.4.2 submission

A typed decision system for TypeSafe-compatible `/v1/systemone`: a frozen
`Qwen3-4B-Base` trunk with a trained pairwise scoring head, temperature
fitting, and a meta-calibration layer. No generation, no sampling —
deterministic given weights.

**This is an independent submission, not affiliated with TypeSafe AI or the
JevBench maintainers.** All numbers in this README were produced by the code
in `src/` and are reproducible with `scripts/`.

## Measured results — 231 public decisions

Full raw output: `results/public_231_results.jsonl`. Derived axes:
`results/axes.json` (regenerate with `scripts/make_axes.py`).

| Metric | Value |
|---|---|
| Accuracy | **173 / 231 = 74.89%** |
| easy / standard / hard | 91.67% / 73.61% / 68.47% |
| Chance-corrected (easy / standard / hard) | 88.36 / 61.38 / 52.48 |
| Schema validity | 1.000 |
| Operational success | 1.000 |
| **Calibration axis** | **85.69** (hard-tier binned ECE 0.0715) |
| Speed axis (standard tier proxy) | 58.00 |
| Declared cost | **$0.0386 per 1,000 decisions → cost axis 52.40** |
| Mean input tokens / decision | 721.3 (median 279) |
| Latency p50 (easy / standard / hard) | 4.36s / 5.40s / 15.57s |

On the published v1.4.2 board the five leaders score calibration
74.5–79.1. **Ours is the highest calibration of any listed system.**

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
1. Temperature fitted on pairwise logits.
2. A 4-feature correctness head (top probability, margin, entropy, log
   option-count) trained with `C=0.05`, applied as a top-label rescale with
   the remaining mass rescaled proportionally. **Argmax is provably preserved
   (0 flips in 231)** — this changes reported confidence only, never the
   decision. Out-of-fold ECE 0.032–0.057 across 4 seeds.

## Honest limitations

These are stated plainly because they bound what this entry can be expected to
score.

- **The sealed tier (308 items) has never been observed.** Every listed system
  shows a large public→sealed collapse (decider-4b v2: 83.5% → 34.7%; Cygnet:
  87.9% → 33.8%; Malkuth-4B: 74.9% → 23.4%). Public accuracy is therefore a
  weak predictor of the sealed result, and we make no claim about ours.
- **The `judge` tier (146 items, 28% of the intelligence weight) is not
  published** — there is no `judge.jsonl` in `datasets/public`. It cannot be
  measured locally.
- **`Malkuth-4B` has public accuracy 0.749, identical to ours, and scores
  44.45 (rank 16).** We do not claim a top rank.
- **Latency was measured on a GTX 1650 4GB with bitsandbytes nf4**, not on
  operator hardware. 4-bit is forced there only because 8.04GB of bf16 weights
  do not fit in 4GB. On adequate VRAM the trunk runs in bf16. Our speed axis
  will therefore differ from ours, in either direction.
- **Speed is measured on the standard tier only.** Per `composite_v12`, the
  official population is standard+judge; with no public judge items, standard
  is the closest measurable proxy and is labelled as such everywhere.
- **Context is 512 tokens.** Longer states are truncated, not summarised or
  chunked. A windowed map-reduce scorer was evaluated and rejected: it
  multiplies forward passes on long items, spending the one axis already
  nearest a scoring gate to buy points on an axis that is capped.

## Cost

`$0.0386 per 1,000 decisions`, derived from measured GPU power (25.2W sampled
via `nvidia-smi`, not the 75W TDP) and measured throughput, over 4 years at
24/7 dedicated duty. `composite_v13.cost` raises on a missing or non-positive
price, so a positive sourced number is mandatory. The full sensitivity table
is in `results/cost_basis.json`:

| basis | $/1,000 | cost axis |
|---|---|---|
| energy only (marginal) | 0.0130 | 66.56 |
| **energy + capital, 24/7 dedicated (declared)** | **0.0386** | **52.40** |
| energy + capital, 12h/day | 0.0642 | 45.78 |
| energy + capital, 8h/day | 0.0898 | 41.41 |
| energy + capital, 6h/day | 0.1153 | 38.14 |

This is not the cheapest basis available to us — see the 6h/day row, and note
the cheapest published offline row is $0.0013. We publish the whole table so
the operator can substitute.

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
src/gavel_adapter.py     JevBench adapter (the file the official Runner drives)
src/prefix_cache.py      shared-prefix KV cache
src/serve_systemone.py   TypeSafe-compatible /v1/systemone server
scripts/make_axes.py     derives results/axes.json from a results file
scripts/cost_basis.py    derives results/cost_basis.json from measurement
results/                 raw run output + derived axes
REPRODUCE.md             exact commands
MODEL_CARD.md            system card, intended use, limitations
```

## Reproduce

See [REPRODUCE.md](REPRODUCE.md). The head is small; the trunk is pulled from
Hugging Face at a pinned revision recorded in `results/axes.json`.

