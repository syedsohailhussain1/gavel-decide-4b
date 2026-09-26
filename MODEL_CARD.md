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
  — `v1/combined_head.pt`. Do **not** use `head/pair_head.pt`: it is a stale
  pre-calibration version with no meta-calibrator and a different temperature.

## Measured performance

231 public JevBench decisions, run on an NVIDIA RTX PRO 6000 Blackwell in
bf16. Raw output `results/bf16_recal.jsonl`; derived axes
`results/bf16_recal_axes.json`.

| Metric | Value |
|---|---|
| Accuracy | 172 / 231 = 74.46% |
| easy / standard / hard | 91.67% / 72.22% / 68.47% |
| Calibration axis | 91.64 (hard-tier binned ECE 0.0418) |
| Speed axis | 88.43 |
| Latency p50 (easy / standard / hard) | 0.046s / 0.046s / 0.183s |
| Seconds per decision | 0.139 |
| Schema validity / operational success | 1.000 / 1.000 |

For context, the five leaders on v1.4.2 record calibration 74.5–79.1 and speed
83.3–92.9. This system's calibration is the highest of any listed row.

## How it answers

Each option is rendered `label: criteria`, appended to the shared state and
question, and scored by the head. Softmax over the fitted temperature gives the
distribution; argmax decides. Score items return the probability-weighted
expected level. Long states are **left-truncated to 512 total tokens**, which
drops the oldest state and never the question or the option. There are no
retries, no regeneration, and no post-processing beyond argmax.

## Calibration

Two stages, both fitted on public data only:

1. A temperature fitted on logits from the serving precision.
2. A 4-feature correctness head (top probability, margin, entropy, log option
   count, `C=0.05`) applied as a top-label rescale with the remaining mass
   rescaled proportionally.

**The rescale never changes a decision** — argmax and accuracy are identical with
and without it (verified, 0 changes across 231 items). It moves reported
confidence only. 5-fold out-of-fold hard-tier ECE 0.0374; in-sample 0.0418.

The shipped head was refitted for bf16. The earlier artefact was fitted on
bitsandbytes nf4 hidden states, where the same weights gave hard-tier ECE 0.1042
because the temperature was mis-specified. `results/combined_head_bf16.pt` is
**bf16-specific — do not use it with nf4.**

## Training data

- 689 pairwise examples derived from public JevBench items.
- 1,720 MNLI-derived pairs as a supplement.
- 2,409 training pairs total.

**JevBench sealed items were never used for training.** No third-party model
outputs (Jev, DeepSeek, or otherwise) were used as training targets; an earlier
third-party probe script was removed from the repository.

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

## Limitations, stated plainly

- **The sealed tier (308 items) has never been observed.** Every listed system
  shows a large public→sealed collapse (decider-4b v2: 83.5% → 34.7%; Cygnet:
  87.9% → 33.8%; Malkuth-4B: 74.9% → 23.4%). Public accuracy is a weak predictor
  of the sealed result. We make no claim about our sealed accuracy.
- **The `judge` tier (146 items, 28% of the intelligence weight) is not
  published** — there is no `judge.jsonl` in `datasets/public`. It cannot be
  measured locally and we report no intelligence axis of our own.
- **Context is 512 tokens.** Longer states are truncated, not summarised or
  chunked. We measured longer context on this benchmark and it did not help
  (hard tier 68.5% at 512 versus 63.1% at 2048), and at least one competitive
  entry truncates state to 64 tokens.
- **Latency depends on the serving hardware.** The figures above are from an
  RTX PRO 6000. On a 16GB consumer card expect materially worse. 4-bit
  quantisation remains available via `dtype="nf4"` for sub-8GB hardware and is
  roughly 1.8x slower with a materially different calibration.
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
