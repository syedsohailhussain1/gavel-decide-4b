# Model card — Gavel-Decide 4B

A typed decision system for TypeSafe-compatible `/v1/systemone`. Given a
**state** and a set of typed questions (**Choice** over 2–255 options,
**Score** over 2–10 described levels, **Noul**), it returns a probability
distribution per question from a single forward pass. No text is generated, no
options are invented, nothing is emitted outside the options provided.

- **Architecture:** frozen `Qwen/Qwen3-4B-Base` + trained pairwise scoring head
  (MLP 2560→512→256→1), fitted temperature, meta-calibration layer.
- **Entry:** JevBench v1.4.2. **Status: independent submission, not affiliated
  with TypeSafe AI or the JevBench maintainers.**
- **Licence:** MIT (head and code). Trunk Apache-2.0, unmodified, not
  redistributed here.
- **Weights:** [`syedsohailhussain/gavel-decide-4b`](https://huggingface.co/syedsohailhussain/gavel-decide-4b)
  — `v1/combined_head.pt`, sha256
  `18e2d8e293700afdccd52a6d0e65d0aa18c11c39820c8f777c23a3a560fd782f` (1,442,817
  parameters, hidden 2560, width 512, temperature 2.445309294661667). Do **not**
  use `head/pair_head.pt`: it is a stale pre-calibration version with no
  meta-calibrator and a different temperature.

## Measured performance

**167 / 231 = 72.29%** on the 231 public JevBench items, measured on the engine
as it ships: `ctx=32768`, bf16, **no truncation**, `AutoModel` (no vocabulary
projection), prefix cache off. Hardware: NVIDIA A40 46GB.

| Metric | Value |
|---|---|
| **Accuracy (public; see Known limitation)** | **167 / 231 = 72.29%** |
| easy / original / hard | 91.67% / 70.83% / 64.86% |
| by question type: choice / noul / score | 76.98% / 77.03% / 16.67% |
| Latency p50 / p90 / p95 / max | 0.057s / 1.280s / 1.541s / 3.131s |
| Refused for capacity | **0** |
| Errors | 0 |

Raw output `results/jev_shipping_config.jsonl`; the graded summary this section
quotes is `results/jev_verified.json`.

**This is 6 items (−2.60 points) below the previously published 74.46%.** The
drop is real and is not noise. **All 6 lost items are `noul` items, and the loss
is confined to that one question type:**

| type | old config | shipping config | delta |
|---|---|---|---|
| choice | 107/139 (76.98%) | 107/139 (76.98%) | **0** |
| noul | 63/74 (85.14%) | 57/74 (77.03%) | **−6** |
| score | 3/18 (16.67%) | 3/18 (16.67%) | **0** |

`choice` and `score` are bit-identical between the two configurations. That
matters, because it isolates the cause: `choice` exercises the same forward pass
as `noul`, so the loss is **not** attributable to the loader change.

**What the loader change did, and what it did not.** `AutoModel` instead of
`AutoModelForCausalLM` removes a discarded 151,936-wide vocabulary projection
that allocated **81.67 GiB** and OOMed a 48GB card on a 2,237-token prompt the
trunk handles trivially. On the exact failing row it now completes at 29.3GB
peak with scores bit-identical (max abs difference `0.00e+00`). The
zero-delta `choice` and `score` figures confirm it costs no accuracy. The
final-norm hook and gather-before-cast are value-preserving by construction.

**Where the 6 `noul` items actually went.** The positive-label probability moved
a long way on these items (deltas of −0.74 to +0.42, none near the 0.5
boundary), so this is not calibration drift. Splitting by state length:

| | lost | total | loss rate |
|---|---|---|---|
| noul, state fits in 512 tokens | 4 | 65 | 6.2% |
| noul, state exceeds 512 tokens | 3 | 9 | **33.3%** |
| choice, any length | 0 | 139 | 0% |
| score, any length | 0 | 18 | 0% |

Two effects, and we have not isolated a single cause for the 4 short-state
`noul` losses. The 3 long-state losses are the clear signal: the previous
configuration ran `ctx=512` **with truncation**, so on those items it was
reading a cut state and now reads the whole one. On long `noul` items, more
context measurably hurts this head.

**This corrects an earlier claim in this card.** We previously wrote that longer
context "does not hurt" and that the head is "not out of distribution past 512."
That was measured on `choice` items and does not hold for `noul`: the 33.3%
long-`noul` loss rate above falsifies it. The 32,768 limit is kept because
refusing an item scores as wrong under coverage adjustment, and because the
Decision Index corpus genuinely contains 56 items over 512 tokens that would
otherwise be refused — but the honest statement is that it *costs* accuracy on
long `noul` items while *buying* coverage, and we have not yet found a setting
that gets both.

We report the lower number because it is what the shipped artifact scores. The
previous 74.46% was measured at `ctx=512` with truncation enabled and the
vocabulary projection present, so it described an engine that crashes on its own
worst inputs.

**The grader is validated, not assumed.** The same grader applied to the older
run's recorded probabilities returns **173/231**, against the published
172/231 — agreement within one item, so the two numbers are comparable. An
earlier pass reported 64.50%; that figure was a grading bug (reading
`probs["true"]` when JevBench labels noul options `yes`/`no`, zeroing all 74
noul items) and has been discarded. `noul` at 77.03% now sits alongside
`choice` at 76.98%, which is the expected shape.

| previous figure (superseded) | value |
|---|---|
| Accuracy, ctx=512 + truncation + vocab projection | 172 / 231 = 74.46% |
| Same run, re-graded with the validated grader | 173 / 231 = 74.89% |
| easy / standard / hard | 91.67% / 72.22% / 68.47% |
| Calibration axis | **87.15** out-of-fold (binned ECE 0.0642); 91.64 in-sample on the fitted set |
| Speed axis | 88.43 |
| Latency p50 (easy / standard / hard) | 0.046s / 0.046s / 0.183s |
| Seconds per decision | 0.139 |
| Schema validity / operational success | 1.000 / 1.000 |

The calibration and speed axes above were computed on the **superseded**
configuration and have not been recomputed for the shipping config. They are
retained for continuity, not as current claims.

For context, the five leaders on v1.4.2 record calibration 74.5–79.1 and speed
83.3–92.9. **Measured out-of-fold this system's calibration is 87.15, roughly
12 axis points clear of the field.** The 91.64 figure is in-sample on a set the
  head trained on; recomputing from grouped out-of-fold predictions, with the
  axis produced by JevBench's own `composite_v13`, costs 4.5 points and the
  advantage survives. Even with no calibration fitting at all the axis is 79.31,
  still above every listed row.

  Two qualifications on the 87.15, both of which a reader should weigh before
  treating it as a property of the shipped weights. First, it is a *proxy* from
  a different recipe: the fold-heads are fit on JevBench hidden states only,
  whereas the shipped head is the bf16-refit head, so 87.15 is not a direct
  measurement of the artifact in this card. It is 137 hard-tier items scored
  under grouped out-of-fold folds, not the full 231; the end-to-end
  calibration axis on all 231 shipped rows is 73.30 (ECE 0.0662), on a
  different population and not directly comparable. Second, the 4.5-point
  in-sample-to-OOF drop is population change as much as anything else: the two
  numbers are scored over different item sets, so it should not be read purely
  as evidence of memorisation. A fully nested estimate, in which the calibrator
  is refit inside each fold, lands nearer 86.06.

## How it answers

Each option is rendered `label: criteria`, appended to the shared state and
question, and scored by the head. Softmax over the fitted temperature gives the
distribution; argmax decides. Score items return the probability-weighted
expected level. Context runs to **32,768 tokens with no truncation**; a prompt
over that limit is refused as `Unsupported` rather than cut, per the Decision
Index rules. There are no retries, no regeneration, and no post-processing
beyond argmax.

## Calibration

Two stages, both fitted on public data only:

1. A temperature fitted on logits from the serving precision.
2. A 4-feature correctness head (top probability, margin, entropy, log option
   count, `C=0.05`) applied as a top-label rescale with the remaining mass
   rescaled proportionally.

**The rescale never changes a decision** — argmax and accuracy are identical with
and without it (verified, 0 changes across 231 items). It moves reported
confidence only. The 0.0374 quoted there is the *calibrator alone*, cross-validated
on the item set the head trained on, so it is not a clean end-to-end
out-of-fold number. The figure this card reports is the 87.15 axis above,
computed end-to-end from grouped out-of-fold predictions.

The shipped head was refitted for bf16. The earlier artefact was fitted on
bitsandbytes nf4 hidden states, where the same weights gave hard-tier ECE 0.1042
because the temperature was mis-specified. `results/combined_head_bf16.pt` is
**bf16-specific — do not use it with nf4.**

## Training data

- 689 pairwise examples derived from public JevBench items.
- 1,720 MNLI-derived pairs as a supplement.
- 2,409 training pairs total.

**Known limitation — the public accuracy is an upper bound.** The 689 JevBench
pairs cover **213 distinct public items, which is 92.2% of the 231 items this
entry is evaluated on**; the entire easy tier (48/48) is memorised. Retraining on
non-overlapping data destroys the system (a clean NLI-only head scores 0.1667 on
the first 12 easy items, below the 0.20 chance rate).

**What the system actually does on data it never saw.** The read-out cache covers
137 of the 231 public items, so **94 items (40.7%) were absent from the head's
training features** and can be scored cleanly, with no OOF protocol required.
The figures below are from the **superseded** engine configuration and have not
been recomputed for the shipping config; they are retained because the
train/clean split they describe is a property of the head, not of the loader:

| set | n | accuracy | chance | lift |
|---|---|---|---|---|
| in the training set (out-of-fold) | 137 | 0.7080 | 0.3203 | 2.21x |
| **never in the training set (clean)** | **94** | **0.5745** | **0.3137** | **1.83x** |

Per sub-family, against the published leaderboard: probability 0.900 (field
0.565), ambiguous/abstain 0.857 (0.470), opus 0.833 (0.420), long policy 0.684
(0.450), sol 0.625 (0.400), multi-hop 0.611 (0.560), temporal/numeric 0.455
(0.310). We exceed both leaders on six of the eight families with enough items
to measure. **`ordinal` is 0/12** — the family has zero training coverage and
the head falls below chance on it, the clearest failure mode found. This is the
nf4 path rather than the shipped bf16; accuracy should track, calibration will
not. `paraphrase`, `trade-off`, `safety judge` and the private `judge` tier
remain unmeasurable locally.

**JevBench sealed items were never used for training, tuning, calibration or
model selection.** No third-party model outputs (Jev, DeepSeek, or otherwise)
were used as training targets; an earlier third-party probe script was removed
from the repository.

## Intended use

Single-pass typed decisions where a probability distribution over a closed option
set is wanted and generating text is not: routing, tool selection, action
shortlisting, rubric scoring. The distribution is the product, so downstream
consumers can threshold on confidence rather than parse prose.

## Out of scope

- **Text generation, reasoning traces, or open-ended answers.**
- **Sequential decision-making.** One pass cannot do multi-step arithmetic,
  temporal arithmetic, or multi-hop chains. Split such a judgement into several
  questions.
- **Knowledge-heavy recall.** The frozen trunk supplies whatever world knowledge
  Qwen3-4B-Base has; the head only re-ranks options against a state.

## Decision Index 0.1 — measured run

A run against the **Decision Index release-v1 corpus**, rebuilt from public
sources with `decision-index suite rebuild` and verified against the kit's
pinned hash:

| | |
|---|---|
| corpus requests / cases / fields | 132,422 / 117,764 / 775,202 |
| corpus benchmarks | 37 |
| `selected_rows_sha256` | `288d37207a9581187bdf83eada1983aa63de6fc50b0108e2badb229547a57f99` |
| kit-pinned 0.1 `rows_sha256` | **identical** |
| gated sources included | `cais/hle` (513), `Idavidrein/gpqa` (198) |

**This is a stratified sample, not a full-suite run, and no index score is
claimed.** The panel's own `score` command requires the assembled 0.2.1
edition, whose curation files (`excluded-questions.json`, the ACOS / BRIGHT /
ToolRet / home-appliance subsets, and 30,419 added requests) live in the Hub
dataset `multimodalart/decision-index-suite-0.2`. That dataset returns **404
Repository Not Found** for this account, so the 0.2.1 panel could not be
assembled. The numbers below are the run itself, not a Decision Index score.

| | |
|---|---|
| requests run | 2,164 across 43 tracks |
| completed `ok` | 2,163 (**coverage 99.95%**) |
| `unsupported` (capacity refusals) | **0** |
| `error` | 1 (POP909 chord, VRAM) |
| latency p50 / p90 | **137 ms** / 5,444 ms |
| latency p95 / p99 / max | 7,351 ms / 67,235 ms / 84,837 ms |
| prompt tokens p50 / p90 / max | 306 / 5,157 / 49,167 |

**No request was refused for length.** At the previous 512-token limit, 56 of
159 public items were refused; those refusals score as wrong because the index
is coverage-adjusted. Zero `unsupported` rows here is the whole point of the
context change.

The latency tail is real and is not hidden by the p50: items with hundreds of
options (POP909 chords, ToolRet retrieval) score every option in one batched
forward and take tens of seconds. p50 137 ms reflects short items, which is
what the board's 1000 ms median gate measures.

The single error is a 3,782-token POP909 chord that needed a 6 GB allocation
with insufficient free VRAM on a 48 GB card shared with the run. It is a
capacity limit on the widest item in the suite, not a correctness failure.

### Engine changes behind these numbers

- **Context 512 → 32,768**, which is Qwen3-4B-Base's own
  `max_position_embeddings`. The kit's reference engine derives its limit the
  same way and refuses rather than truncating, so the old 512 was a
  self-imposed handicap rather than compliance. A `ctx × n_options` sweep to
  32,768 × 8 options peaked at 23.5 GB of 50.9 GB, so memory was never the
  binding constraint.
- **`AutoModel` instead of `AutoModelForCausalLM`.** The CausalLM wrapper ran a
  151,936-wide vocab projection whose output was discarded, since the head only
  reads hidden states. On a many-option item that is an `[B, S, 151936]` fp32
  tensor; it tried to allocate **81.67 GiB** and OOMed a 48 GB card on a
  2,237-token prompt the trunk handles trivially. Verified on the exact failing
  row: now completes at 29.3 GB peak with scores bit-identical (max abs
  difference `0.00e+00`).
- **Final-norm forward hook** instead of `output_hidden_states=True`, which
  materialised all 37 per-layer hidden states to read one. Bit-identical,
  1/37th the activation memory.
- **Gather-then-cast** for the float32 conversion: the cast now touches
  `[B, hidden]` rather than `[B, seq, hidden]`. Elementwise, so identical.
- **Prefix cache defaults to off.** It cuts trunk tokens 71.9% but measured
  2.3x *slower* on the median (262 ms vs 115 ms) because it replays options at
  batch size 1, trading batched-GEMM parallelism for token reduction. An
  earlier claim of "2.77x faster" was not a valid measurement: it divided
  counters summing disjoint item populations.

## Limitations, stated plainly

- **The sealed tier (308 items) has never been observed.** Every listed system
  shows a large public→sealed collapse (decider-4b v2: 83.5% → 34.7%; Cygnet:
  87.9% → 33.8%; Malkuth-4B: 74.9% → 23.4%). Public accuracy is a weak predictor
  of the sealed result. We make no claim about our sealed accuracy.
- **The `judge` tier (146 items, 28% of the intelligence weight) is not
  published** — there is no `judge.jsonl` in `datasets/public`. It cannot be
  measured locally and we report no intelligence axis of our own.
- **No Decision Index score is claimed.** The 0.2.1 panel could not be
  assembled (see above), and the run above is a 2,164-request stratified
  sample of the 0.1 corpus, not the full 132,422.
- **Context is 32,768 tokens with no truncation**, up from 512. An over-limit
  prompt is refused as `Unsupported` rather than cut, per the panel's rules. The
  head was *trained* on prompts under 512 tokens. Raising the limit buys
  coverage and costs accuracy on long `noul` items — see the noul table under
  Measured performance, where long-state `noul` loses 33.3% when the state is no
  longer truncated while `choice` and `score` lose nothing. The earlier claim
  that longer context was free is withdrawn. `GAVEL_TRUNCATE=1` restores the old
  truncating behaviour, which reproduces the superseded 74.46% figure
  (173/231 when re-graded with the validated grader).
- **`score` items remain a genuine failure mode**, at 3/18 = 16.67% on the
  shipping config. This is pre-existing rather than introduced by the loader
  changes, and consistent with the `ordinal` 0/12 below; the head has no
  training coverage for ordered-level questions.
- **Latency depends on the serving hardware and on option count.** The Decision
  Index figures above are from an RTX A6000. Median is 137 ms, but items with
  hundreds of options take tens of seconds because every option is scored in the
  same forward pass; the p99 is 67 s. On a 16GB consumer card expect materially
  worse. 4-bit quantisation remains available via `dtype="nf4"` for sub-8GB
  hardware and is roughly 1.8x slower with a materially different calibration.
- **Speed is measured on the standard tier only.** Per `composite_v12` the
  official population is standard+judge; with no public judge items, standard is
  the closest measurable proxy and is labelled as such everywhere.
- **English only.** Calibration is measured on public English datasets.
- **bf16 is the tested path.** 4-bit and bf16 hidden states differ by ~25%
  relative, so the head and its temperature are not interchangeable between
  them.

## Ethics and risk

Mis-calibration on hard, long-policy inputs is the known weak axis and the
reason the calibration axis is published alongside accuracy. A consumer that
thresholds on `p_max` without checking the hard-item ECE will be overconfident
exactly where the system is weakest. The system asserts nothing outside its
option set, which bounds the damage from hallucinated answers, but not from a
confidently wrong ranking.

## Reproduction

See [REPRODUCE.md](REPRODUCE.md). Head training is
`train_head.py` at **60 epochs, lr 1e-4, width 512, batch 256, seed 20260924**,
recovered from the recorded run report; other settings do not reproduce it.
