# Reproducing this entry

## 1. Environment

```
python 3.11
torch >= 2.5          (CUDA 12.1 build tested)
transformers 5.15.0
accelerate
bitsandbytes          # only needed for 4-bit; skip if running bf16
numpy
```

`pip install -r requirements.txt`

## 2. Weights

The trunk is public and unmodified:

```
huggingface-cli download Qwen/Qwen3-4B-Base --local-dir models/qwen3-4b
huggingface-cli download syedsohailhussain/gavel-decide-4b --local-dir models/head
#   -> models/head/v1/combined_head.pt   (NOT head/pair_head.pt, which is a
#      stale pre-calibration version with no meta_cal and T=2.796)
```

`combined_head.pt` is a single dict containing the head `state_dict` under
`head`, the input width under `hidden`, the fitted `temperature`, and the
embedded meta-calibrator under `meta_cal`.

## 3. Point the adapter at them

`src/gavel_adapter.py` takes these as constructor kwargs (or edit the two
defaults):

```python
GavelLocalAdapter(
    trunk=r"...\models\qwen3-4b",
    head=r"...\models\head\combined_head.pt",
    ctx=512,
)
```

## 4. Run the official harness

We drive **their** runner unmodified, with our adapter — no edits to the
JevBench repository:

```bash
git clone https://github.com/fstandhartinger/jevbench.git
python -m jevbench.cli run \
    --adapter gavel_local \
    --tasks jevbench/datasets/public/easy.jsonl,jevbench/datasets/public/original.jsonl,jevbench/datasets/public/hard.jsonl \
    --results results.jsonl \
    --manifest manifest.json
```

On Windows, `jevbench.budget.Ledger` needs Unix `fcntl`. Use the single-process
`WinLedger` shim in the upstream development repo, or run under WSL.

Score it with their summariser, not ours:

```bash
python -m jevbench.cli summarize --tasks <same> --results results.jsonl
```

## 5. Reproduce our derived axes

```bash
python scripts/make_axes.py results.jsonl     # -> results/axes.json
python scripts/cost_basis.py                  # -> results/cost_basis.json
```

`make_axes.py` imports the official `composite_v13` and `metrics` modules, so
the calibration and cost figures are computed by the benchmark's own code
rather than reimplemented.

## Determinism

The system is deterministic given weights: no sampling, no generation, no
retries. Residual run-to-run variation comes only from floating-point
non-associativity — 4-bit batched inference versus a sequential KV-cached
forward produce slightly different last-bit values. Measured effect on the
231-item run: **1 decision flip**, mean absolute probability delta 0.00208.

`GAVEL_PREFIX_CACHE=0` forces the batched path for bit-comparison.

## Hardware used for the published numbers

GTX 1650, 4GB VRAM, Windows, bitsandbytes nf4. Chosen only because it was
available; 4-bit is forced there because 8.04GB of bf16 weights do not fit in
4GB. Throughput on that card is dominated by 4-bit dequantisation, so the
latency and cost figures in this repo are **not** representative of bf16
inference on adequate VRAM. The operator's own measurement governs.

