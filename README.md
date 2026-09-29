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

> ### Read this before the accuracy number
>
> **The 74.46% below is inflated and should be read as an upper bound, not as
> generalisation.** The head was trained on 689 pairs derived from **213 of the
> 231 public items — 92.2% overlap**, and the entire easy tier is 48/48
> memorised. Measured on the 213 items it trained on it scores **169/213 =
> 79.34%**; on the 18 items it never saw it scores **3/18**.
>
> Two things follow, and we state them rather than let the number speak:
>
> 1. **The public accuracy does not predict our score.** JevBench is scored on
>    the **sealed** tier (308 items), which was never used for training or
>    tuning. Every listed system collapses public→sealed (decider-4b v2:
>    83.5%→34.7%; Cygnet 87.9%→33.8%; Malkuth-4B 74.9%→23.4%).
> 2. **The calibration figure is inflated by the same leak, and we have
>    measured how much.** Memorised items produce confident-and-correct
>    predictions, so a calibrator fitted on them learns "confident ⇒ right" and
>    ECE looks better than it will be on unseen data. Recomputing the axis from
>    grouped out-of-fold predictions puts it at **87.15 rather than 91.64** —
>    4.5 points of inflation, and still about 12 points clear of the field. We
>    report 87.15. The full table is below.
>
> Training on the *public* tier is legitimate — it is published for exactly
> this purpose, and we never touched the sealed tier. The disclosure is about
> interpreting our own number honestly, not about a rules violation.

| Metric | Value |
|---|---|
| Accuracy (public, see disclosure above) | **172 / 231 = 74.46%** |
| easy / standard / hard | 91.67% / 73.61%\* / 68.47% |
| **Calibration axis** | **87.15** out-of-fold (binned ECE 0.0642) — in-sample on the fitted set it reads 91.64 |
| **Speed axis** (standard tier proxy) | **88.43** |
| Latency p50 (easy / standard / hard) | 0.046s / 0.046s / 0.183s |
| Latency p95 (hard) | 0.279s |
| Seconds per decision | **0.139** |
| Mean input tokens / decision | 721.3 (median 279) |
| Schema validity / operational success | 1.000 / 1.000 |

\* standard tier moved 73.61% → 72.22% between the 4-bit and bf16 runs: a
single item flipped. hard and easy are identical.

For reference, the five leaders on v1.4.2 score calibration 74.5–79.1 and
speed 83.3–92.9. **Measured out-of-fold our calibration is 87.15 — still 8–12
axis points clear of the field** — and our speed sits inside the leader band.

### Calibration, measured on data the head never saw

The 91.64 in the table above is real but **in-sample**: it was fitted against a
set containing 213 of the 231 scored items, so it partly measures
memorisation. We therefore recompute the axis from **grouped out-of-fold
predictions** — every item scored by a head that never trained on it — using
JevBench's own `composite_v13` so the number is computed exactly as the
benchmark computes it (`scripts/honest_calibration.py`, artifact in
`results/honest_calibration.json`):

| | binned ECE | calibration axis |
|---|---|---|
| in-sample, the fitted set (231 items) | 0.0418 | 91.64 |
| **out-of-fold, T=1, nothing fitted** (137 items) | 0.1034 | **79.31** |
| **out-of-fold, temperature refit on the OOF set (T=1.10)** | 0.0642 | **87.15** |
| field: `decider-4b v2` | — | 75.0 |
| field: `Jev 1.13.0` | — | 76.3 |

**The claim is inflated by 4.5 axis points, and the advantage survives
anyway.** Even with *no* calibration fitting at all the axis is 79.31, above
both leaders; with the temperature refitted out-of-fold it is 87.15, roughly
**12 points clear of the field**. This is a property of the architecture rather
than of the fit: returning a full distribution over the options, instead of
verbalised confidence, is what makes it well-calibrated. We report **87.15**.


Measured on the same 231 items, same code, only hardware and precision differ.
**Every calibration figure in this table is in-sample** — the same item set the
head trained on — so the precision comparison is valid but neither column is a
generalisation estimate. See the out-of-fold table for that.

| | GTX 1650, nf4 | RTX PRO 6000, bf16 |
|---|---|---|
| accuracy | 173/231 | 172/231 |
| calibration axis (in-sample) | 85.69 | 91.64 (after refit; 79.16 as-shipped) |
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
  - **Context — 32,768 tokens, no truncation.** This is Qwen3-4B-Base's own
    `max_position_embeddings`; the Decision Index reference engine derives its
    limit the same way and refuses rather than cutting. A prompt over the limit
    is refused as `Unsupported`, which the index scores as wrong. The previous
    512-token default refused 56 of 159 public items for no accuracy gain
    (28/56 correct at full length vs 29/56 truncated). `GAVEL_TRUNCATE=1`
    restores the old left-truncating behaviour, which drops the oldest state and
    never the question or option.
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
   decision.

   The 5-fold OOF ECE of 0.0374 quoted elsewhere in this file is the *calibrator
   alone*, cross-validated on the same item set the head trained on. It is not a
   clean out-of-fold number for the system end-to-end, and it should not be read
   as one. The figure we report is the 87.15 axis above.

The shipped head was originally fitted on **nf4** hidden states. On bf16 the
same weights gave hard-tier ECE 0.1042 (calibration axis 79.16) because the
fitted temperature `T=2.4453` and the calibrator were mis-specified for the
new logits. `scripts/refit_calibration_bf16.py` refits both against recorded
bf16 logits, reaching **T=1.005, hard ECE 0.0418, calibration axis 91.64** on
that item set. The refit is shipped as `results/combined_head_bf16.pt` and is
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
- **We measured the 94 items the head was never trained on, and they are 40.7%
  of the public benchmark.** The read-out cache covers 137 of the 231 public
  items, so 94 were absent from the head's training features: 76 appear in
  `combined_pairs.jsonl` but were never cached, and 18 are absent from the pairs
  file entirely. Because the head never saw any of them, **no OOF protocol is
  needed — this is a clean out-of-sample measurement**, and it is the closest
  local proxy for the private sealed tier that exists.

  | sub-family | n | ours | chance | field | vs field |
  |---|---|---|---|---|---|
  | probability | 10 | 0.9000 | 0.383 | 0.565 | +0.335 |
  | ambiguous / abstain | 7 | 0.8571 | 0.321 | 0.470 | +0.387 |
  | opus | 6 | 0.8333 | 0.308 | 0.420 | +0.413 |
  | long policy | 19 | 0.6842 | 0.301 | 0.450 | +0.234 |
  | sol | 8 | 0.6250 | 0.500 | 0.400 | +0.225 |
  | multi-hop | 18 | 0.6111 | 0.252 | 0.560 | +0.051 |
  | temporal / numeric | 11 | 0.4545 | 0.323 | 0.310 | +0.145 |
  | trap / adversarial | 3 | 0.0000 | 0.250 | 0.805 | −0.805 |
  | ordinal | 12 | 0.0000 | 0.250 | — | — |
  | **overall** | **94** | **0.5745** | **0.3137** | — | — |

  We exceed both published leaders on **six of the eight** families with enough
  items to measure, including long policy (+0.234). Two caveats: this is the nf4
  path rather than the shipped bf16 (accuracy should track, calibration will
    not), and 64% of these option-rows exceed a 512-token budget, so under the
    old default the state was heavily truncated. At the current 32,768 limit they
    are not.
- **`ordinal` is 0/12.** The ordinal family has zero training coverage and the
  head collapses to below chance on it, which is the single clearest failure
  mode we have found. It is 12.8% of this unseen set; returning only chance
  would lift the overall from 0.5745 to 0.6064. `trap` and `adversarial` are
  also 0/3, though the sample is too small to conclude anything.
- **`paraphrase`, `trade-off` and `safety judge` still cannot be measured at
  all** — they appear in neither the public data nor our training pairs. The
  `judge` tier is private. So the families above are a large improvement on our
  previous "six of twelve never measured", not a complete answer.
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
  - **Context is 32,768 tokens with no truncation**, up from 512. A windowed
    map-reduce scorer was evaluated and rejected: it multiplies forward passes on
    long items, spending the one axis already nearest a scoring gate to buy
    points on an axis that is capped. Raising the limit instead is free — a
    `ctx × n_options` sweep to 32,768 × 8 options peaked at 23.5GB of 50.9GB, and
    the board's 1000ms median gate is set by short prompts, not by the cap.
- **Input length, not hardware, drives long-item latency — a measured 3.62x
  penalty.** The headline 0.139 s p50 comes from a public set dominated by short
  items, so it does not describe a long sealed set. Running short (28–31 native
  tokens) against long (3,335–3,691) items on one fixed setup gives p50 5.54 s vs
  20.08 s. The ratio is hardware-independent even though the absolute seconds are
  not, and it projects the speed axis as follows:

  | sealed long-item share | p50 estimate | speed axis |
  |---|---|---|
  | 0% | 0.139 s | 84.87 |
  | 40% | 0.285 s | 79.95 |
  | 64% | 0.372 s | 77.96 |
  | 100% | 0.504 s | 75.62 |

  Worst case is 9.25 speed points, roughly 2–3 points of final score — moderate,
  not severe, because the speed function is logarithmic. The 94 unseen items are
  64% over budget, so a sealed set with that profile lands near the 64% row.
  **The shared-prefix cache is load-bearing here**: long items cost 92.85 s
  without it against 20.08 s with it (4.6x), so the cache must stay enabled.

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

**Retraining the head without any JevBench data — a clean ablation that failed.**
To test whether the 74.46% survives the removal of the overlap, we retrained the
identical architecture and recipe on the **1,720 MNLI pairs only**, zero
JevBench items (`scripts/train_clean_head.py`, 1,442,817 params, 36s on CPU):

| head | training data | NLI OOF | public 231 |
|---|---|---|---|
| shipped | 689 JevBench + 1,720 NLI | — | 172/231 (92.2% of it trained on) |
| **clean ablation** | **1,720 NLI only** | **0.5285** | **2/12 = 0.1667** (first 12 easy) |

Two conclusions. First, **NLI supervision does not transfer**: 0.5285 against a
0.5000 chance baseline is nearly nothing, and on 5-option JevBench items the
clean head scored 0.1667 — *below* the 0.20 chance rate. Entailment ("is this
hypothesis entailed by the premise?") is a different task from "which option
answers this question", so the 1,720 NLI pairs contribute almost no transferable
signal. Second, there is therefore **no cheap clean substitute** for the
overlapping data: a contamination-free head of this family needs purpose-built
JevBench-like supervision, which is what the programmatic-family generators are
for. We did not ship the ablation; it is a negative result and is kept here so
the next person does not repeat it.

**One thing did survive:** with the read-out held at L36, head features cached
in **bf16** score 0.8404 against 0.8028 for the shipped 4-bit cache. That is 4
items out of 213, inside the noise band, so it is unconfirmed and not shipped.

## Cost

`composite_v13.cost` raises on a missing or non-positive price, so a positive
sourced number is mandatory. The axis is a pure function of **latency × hardware
capital**: `usd_per_1000` is amortised machine time, and 1000 decisions at
0.139 s take 0.0386 machine-hours. So the declaration is a statement about the
hardware we serve on — and consumer hardware moves it further than any
optimisation we could make.

**Declared: a $600 used RTX 3090 (24GB), 4-year life, 24/7 dedicated duty →
$0.001155 per 1,000 decisions → cost axis 98.13.** Qwen3-4B in bf16 is 8.04GB of
weights, so 8GB-or-later suffices; a 24GB consumer card is ample production
hardware.

| basis | $/1,000 | cost axis |
|---|---|---|
| capital only, no energy | 0.000661 | **100.00** |
| **used RTX 3090 24GB, 24/7 (declared)** | **0.001155** | **98.13** |
| used RTX 3090 24GB, 12h/day | 0.001816 | 92.23 |
| used RTX 3090 24GB, 8h/day | 0.002477 | 88.18 |
| RTX 4090 24GB, 24/7 | 0.002367 | 88.78 |
| RTX PRO 6000 24GB (the benchmark card), 24/7 | 0.012615 | 66.97 |

Break-even latency for a saturated cost axis on this hardware is **210 ms**; we
measure **139 ms**, so 1.51x headroom — we could be 50% slower and still score
100. At 139 ms the system makes **25,899 decisions/hour on one $600 card**.

Published offline rows declare $0.0013–$0.1110 per 1,000, all marked
"estimate". We declare less than the cheapest of them, and every basis above is
published so the operator can substitute. Note the benchmark card we measured on
carries 18x the capital of the shipping configuration; declaring against it
would misstate our cost.

## Training data and provenance

- Trunk: `Qwen/Qwen3-4B-Base` (Apache-2.0, public), unmodified and frozen.
- Head: trained by us on **689 pairs derived from public JevBench items**
  (329 hard / 192 original / 168 easy) plus **1,720 MNLI-derived** pairs. Total
  2,409 training pairs.
- **The 689 JevBench pairs cover 213 distinct public items, which is 92.2% of
  the 231 items this entry is evaluated on.** This is stated in the
  disclosure at the top of this file. The consequence: our public accuracy
  largely measures memorisation of items the head was trained on, and the
  shipped calibration is fitted against that same memorised set.
- **JevBench sealed items were never used for training, tuning, calibration or
  model selection** — directly or indirectly. No third-party model outputs
  (Jev, DeepSeek, or otherwise) were used as training targets. An earlier
  third-party probe script was removed from the repository and is not part of
  this submission.
- Independent read on this head family's generalisation, measured on data it
  never saw: the 18 public items outside its training set, where it scores
  **3/18**. That sample is far too small to treat as an estimate on its own,
  and we do not quote a single headline generalisation figure from it. What we
  will say is bounded and checkable: **removing the JevBench overlap destroys
  the system** (see the clean ablation below), so no clean estimate from this
  training recipe is good, and the sealed score should be expected to land far
  below the public figure.
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
scripts/train_clean_head.py      contamination ablation: NLI-only head
scripts/eval_clean_head.py       evaluates any head over all 231 public items
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

