---
title: Gavel-Decide 4B — typed decisions
emoji: ⚖️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
short_description: Single-pass typed decisions with calibrated probabilities, no generation
pinned: false
license: apache-2.0
---

# Gavel-Decide 4B — single-pass typed decisions

Interactive demo for [`syedsohailhussain1/gavel-decide-4b`](https://github.com/syedsohailhussain1/gavel-decide-4b),
a typed-decision system built on a **frozen** `Qwen3-4B-Base` trunk with a
**1,442,817-parameter** head trained on CPU in 3.5 seconds.

Give it a *state* (the evidence) and a typed question — `choice`, `noul`, or
`score` — and it returns a **calibrated probability distribution over the
options** in one forward pass per option. **No text is generated**, so
`output_tokens` is always 0: the readout takes one vector from the final layer
(L36) and a small head maps it to a per-option score. The distribution is the
product, so a consumer can threshold on confidence instead of parsing prose.

## Why this runs on a free CPU Space

Most chat demos need a GPU because autoregressive **decode** is sequential and
memory-band-bound. This architecture never decodes — it does one prefill per
option and reads a hidden state. That removes the sequential loop entirely, so
CPU inference is viable and the Space needs no paid hardware.

## Two things worth trying

- **The shared-prefix KV cache** is on by default. Every option of an item
  shares the `State: … \nQuestion: … \nOption: ` prefix, so its K/V is computed
  once per item instead of once per option. The `path` field in the output says
  whether the cached or batched route ran.
- **Confidence is typed by how it was obtained.** The meta-calibrator is a
  top-label rescale with the remaining mass rescaled proportionally, so
  **argmax is provably preserved** — it changes reported confidence and never
  the decision. Set options close together and watch the margin shrink while the
  answer stays put.

## What this measures, honestly

The head was trained on 689 pairs derived from **213 of the 231 public JevBench
items — 92.2% overlap**, and the entire easy tier (48/48) is memorised. On the
items it trained on it scores 169/213 = 79.34%; on the 18 it never saw, 3/18.
**Published public accuracy for this architecture is therefore an upper bound,
not a generalisation estimate.**

The held-out tier was never used for training, tuning, calibration or model
selection. Retraining on non-overlapping data does not rescue it: an identical
architecture trained on the 1,720 MNLI pairs alone scores 0.5285 out-of-fold
against a 0.5000 chance baseline, and 0.1667 on the first 12 easy items — below
the 0.20 chance rate. Entailment is a different task from "which option answers
this question", so borrowed data does not transfer. A contamination-free head of
this family needs purpose-built supervision, which does not exist yet.

Because of that, treat the demo as a way to watch a frozen 4B read-out behave on
text it was never fitted to — not as a benchmark.

## Links

- Submission: [fstandhartinger/jevbench#118](https://github.com/fstandhartinger/jevbench/issues/118)
- Code, full results and `scripts/verify_claims.py`:
  [syedsohailhussain1/gavel-decide-4b](https://github.com/syedsohailhussain1/gavel-decide-4b)
- Head weights: [syedsohailhussain/gavel-decide-4b](https://huggingface.co/syedsohailhussain/gavel-decide-4b)

Trunk `Qwen/Qwen3-4B-Base` is Apache-2.0, unmodified and frozen. This Space's
own code is Apache-2.0.
