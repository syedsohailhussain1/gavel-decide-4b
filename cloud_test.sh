#!/usr/bin/env bash
# Gavel cloud test on rented GPU (>=16GB VRAM so Qwen3-4B fits in bf16).
# Run inside a RunPod / Vast.ai / Lambda pod.
#
# Answers the three questions we cannot answer on the GTX 1650:
#   1. Does bf16 beat bitsandbytes nf4? (4-bit is forced locally only because
#      8.04GB of bf16 weights do not fit in 4GB.)
#   2. What is the real latency profile?
#   3. What speed / cost / accuracy do we get on real hardware?
#
#   bash cloud_test.sh
set -uo pipefail
cd /workspace 2>/dev/null || cd ~
ROOT="$PWD/gavel-cloud"
mkdir -p "$ROOT" && cd "$ROOT"

echo "=== [1/6] code ==="
[ -d gavel ] || git clone --depth 1 https://github.com/syedsohailhussain1/gavel-jevbench-entry.git gavel
[ -d jevbench ] || git clone --depth 1 https://github.com/fstandhartinger/jevbench.git jevbench
ls gavel/src/gavel_adapter.py gavel/src/gavel_meta.py jevbench/jevbench/runner.py || exit 1

echo "=== [2/6] env ==="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
pip install -q bitsandbytes accelerate 2>&1 | tail -2

echo "=== [3/6] weights ==="
python - <<'PY'
import os, torch
from huggingface_hub import snapshot_download, hf_hub_download
t = snapshot_download("Qwen/Qwen3-4B-Base",
                      allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("TRUNK", t)
# NOTE: there is no root combined_head.pt. v1/ carries the meta-calibrator
# (features c/margin/ent/kopts, T=2.4453...). head/pair_head.pt is a stale
# earlier version with no meta_cal and T=2.796 - do NOT use it.
h = hf_hub_download("syedsohailhussain/gavel-decide-4b", "v1/combined_head.pt")
d = torch.load(h, map_location="cpu", weights_only=False)
assert "meta_cal" in d, "head has no meta_cal - wrong file?"
assert abs(float(d["temperature"]) - 2.445309294661667) < 1e-9, \
    f"unexpected temperature {d['temperature']}"
print("HEAD", h)
print("  temperature", d["temperature"], "features", d["meta_cal"]["feats"])
with open("env.sh", "w") as f:
    f.write(f'export TRUNK="{t}"\nexport HEAD="{h}"\n')
PY
[ -f env.sh ] || { echo "FATAL: could not fetch weights (is the HF repo public? set HF_TOKEN)"; exit 1; }
# shellcheck disable=SC1091
source env.sh
export JEVBENCH_PATH="$ROOT/jevbench"

echo "=== [4/6] precision A/B: bf16 vs nf4 (cuda-synchronised) ==="
cd "$ROOT" && python gavel/scripts/cloud_precision_ab.py || echo "precision A/B failed"

echo "=== [5/6] full 231-item JevBench, bf16, prefix cache ON ==="
cd "$ROOT" && python gavel/scripts/cloud_run_bench.py --dtype bf16

echo "=== [5b/6] same, prefix cache OFF (for the bit-level comparison) ==="
cd "$ROOT" && python gavel/scripts/cloud_run_bench.py --dtype bf16 \
    --no-prefix-cache --out results/cloud_231_nopc.jsonl

echo "=== [6/6] done ==="
echo "artifacts:"
ls -la "$ROOT/results" 2>/dev/null || true
