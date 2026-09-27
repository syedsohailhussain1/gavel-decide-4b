# Submission: `gavel-decide-4b` — typed decision system, frozen 4B trunk

Independent submission. Not affiliated with TypeSafe AI or the JevBench
maintainers. No Jev or DeepSeek outputs were used as training targets.

## What it is

A decision system for `POST /v1/systemone`. No generation, no sampling — the
full probability distribution over the supplied options is returned, never a
label alone. Deterministic given weights.

- **Serving code:** https://github.com/syedsohailhussain1/gavel-decide-4b at commit `68c4cd6`
- **Entry point:** `src/serve_systemone.py` (TypeSafe-compatible server)
- **Adapter:** `src/gavel_adapter.py` (the file the official `Runner` drives)
- **Shared-prefix KV cache:** `src/prefix_cache.py`
- **Serving transform:** `src/gavel_meta.py` (vendored meta-calibration)
- **Trunk:** [`Qwen/Qwen3-4B-Base`](https://huggingface.co/Qwen/Qwen3-4B-Base) (Apache-2.0), unmodified and **frozen**
- **Head:** `results/combined_head_bf16.pt`, **1,442,817 parameters**, trained on CPU in 3.5 s
- **License:** Apache-2.0 (`LICENSE`)

## Mapping

1. Every option is rendered as `State: … \nQuestion: … \nOption: <label>: <criteria>`.
   State is left-truncated to a **512-token** total budget, dropping the oldest
   state and never the question or option.
2. One prefill per item. The head reads **one vector — the last token of the
   final layer (L36)** — so nothing is decoded. **Output tokens = 0** on every
   decision, which is truthful: the system emits a distribution, not tokens.
3. Head logits → softmax at the fitted temperature → the distribution is the product.
4. A 4-feature meta-calibrator (top probability, margin, entropy, log option
   count) then applies a **top-label rescale** with the remaining mass rescaled
   proportionally. **Argmax is provably preserved — 0 changes in 231 decisions.**
   It alters reported confidence only, never the decision.
5. `choice` (2–255 options), `noul`, and `score` (2–10 levels) are all supported.

**Precision.** `dtype="bf16"` is the honest default and is what the numbers
below were measured in; it needs 8 GB of weights. `dtype="nf4"` is available
for sub-8 GB hardware and is ~1.8x slower with a materially different
calibration. The two checkpoints are **not** interchangeable.

## Cost basis — declared

`composite_v13.cost` raises on a missing or non-positive price, so a sourced
number is mandatory and cost is never an automatic 100.

**Declared configuration: a $600 used RTX 3090 (24 GB), 4-year life, 24/7
dedicated duty → $0.001155 per 1,000 decisions → cost axis 98.13.**
1000 decisions at 0.139 s take 0.0386 machine-hours.

| basis | $/1,000 | cost axis |
|---|---|---|
| capital only, no energy | 0.000661 | 100.00 |
| **used RTX 3090 24 GB, 24/7 (declared)** | **0.001155** | **98.13** |
| used RTX 3090 24 GB, 12 h/day | 0.001816 | 92.23 |
| used RTX 3090 24 GB, 8 h/day | 0.002477 | 88.18 |
| RTX 4090 24 GB, 24/7 | 0.002367 | 88.78 |
| RTX PRO 6000 24 GB (the benchmark card), 24/7 | 0.012615 | 66.97 |

Every basis is published in `scripts/cost_basis.py` so the operator can
substitute. Note the card we measured on carries ~18x the capital of the
declared shipping configuration; declaring against it would misstate our cost.

## Latency

Measured on an **RTX PRO 6000 Blackwell (24 GB, bf16)** over the full
231-item public set, weights loaded before the clock starts:

| | p50 | p95 |
|---|---|---|
| easy | 0.046 s | 0.189 s |
| standard | 0.046 s | 0.222 s |
| hard | 0.183 s | 0.279 s |

**0.139 s per decision**, 32.2 s for the full 231. We understand the standard
self-hosted adjustment (×2 + 0.15 s) applies to this row; we are not claiming
the raw figures as production latency.

## Our own public-set run

Reproduce with `scripts/cloud_run_bench.py`; raw output in
`results/bf16_recal.jsonl`, derived axes in `results/bf16_recal_axes.json`.

| tier | accuracy |
|---|---|
| easy | 44/48 = 0.9167 |
| standard | 52/72 = 0.7222 |
| hard | 76/111 = 0.6847 |
| **overall** | **172/231 = 0.7446** |

Schema validity 1.000, operational success 1.000, output tokens 0 on all 231.

## Disclosure — the public accuracy above is an upper bound

We would rather state this than have it found.

**The head was trained on 689 pairs derived from 213 of the 231 public items —
92.2% overlap — and the entire easy tier (48/48) is memorised.** Measured on
the items it trained on it scores **169/213 = 79.34%**; on the 18 items it
never saw it scores **3/18**.

Two consequences:

1. **The public figure does not predict our sealed score.** This system is
   scored on the held-out tier, which was never used for training, tuning,
   calibration or model selection. We expect it to land well below 0.7446.
2. **The calibration figure is inflated by the same leak.** Memorised items
   produce confident-and-correct predictions, so a calibrator fitted against
   them learns "confident ⇒ right" and the ECE looks better than it will on
   unseen data. Treat our calibration number as a property of the fitting set,
   not of the method.

Training on the *public* tier is legitimate — it is published for exactly this
purpose, and the held-out tier is untouched. This disclosure is about
interpreting our own number honestly.

**There is no cheap clean substitute.** We retrained the identical architecture
and recipe on the 1,720 MNLI pairs alone, zero JevBench items
(`scripts/train_clean_head.py`): **0.5285** out-of-fold against a **0.5000**
chance baseline, and **0.1667** on the first 12 easy items — *below* the 0.20
chance rate for 5-option items. Entailment is a different task from "which
option answers this question", so NLI contributes almost nothing transferable.
A contamination-free head of this family needs purpose-built supervision, which
we do not yet have. We did not ship the ablation; it is recorded as a negative
result in `README.md`.

## What we tested and rejected

Recorded in full in `README.md`, because negative results are part of the entry.

- **Intermediate-layer read-out.** Swept every depth at three context lengths.
  On the production recipe the shipped read-out wins (L36 0.8404 vs L20 0.7465
  item accuracy). An earlier pass appeared to show the reverse and was **wrong**
  — it measured trainability under a broken harness, not read-out quality.
- **Ensembling context views.** Errors across ctx 512/1024/2048 are 79–91%
  correlated (Jaccard), so there is no ensemble headroom.
- **Longer context.** Flat within noise (0.655 / 0.643 / 0.661 at 512/1024/2048).
- **Windowed map-reduce scoring.** Multiplies forward passes on long items to buy
  points on an axis that is capped.

## Verification

`scripts/verify_claims.py` re-derives all 40 published numbers from the raw
artifacts and fails on any mismatch. All 40 pass. We would rather the entry
break loudly than rot quietly.
