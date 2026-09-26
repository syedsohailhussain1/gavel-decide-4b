#!/usr/bin/env bash
# Gavel-Decide on Qwen3.8-27B: does a 27B trunk buy hard-tier accuracy, and
# does it still run on a consumer 24GB card with our prefill-only read-out?
#
# Everything our architecture assumes still holds at 27B: one prefill per
# option, no decode, shared-prefix KV cache, tiny trainable head.
set -uo pipefail
cd /workspace 2>/dev/null || cd ~
ROOT="$PWD/gc38"
mkdir -p "$ROOT"/{src,scripts,results} && cd "$ROOT"
export TRUNK="${TRUNK:-Qwen/Qwen3.8-27B}"
echo "########## [1/6] deps ##########"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# qwen3_5 (Qwen3.8) needs transformers >= 5.15; older builds cannot load it
pip install -q --break-system-packages "transformers==5.15.0" "huggingface_hub>=0.34" accelerate bitsandbytes 2>&1 | tail -3
python -c "import transformers;print('transformers',transformers.__version__)"

echo "########## [2/6] code + data ##########"
cp -r /root/gc/src/. src/ 2>/dev/null || true
[ -d /root/gc/scripts ] && cp -r /root/gc/scripts/. scripts/ || true
[ -f /root/jevbench/datasets/public/hard.jsonl ] || git clone --depth 1 -q https://github.com/fstandhartinger/jevbench.git /root/jevbench
cp /root/gc/results/combined_pairs.jsonl results/ 2>/dev/null || true
ls -la results/combined_pairs.jsonl src/ | head -20

echo "########## [3/6] load 4-bit + smoke ##########"
python - <<'PY'
import torch, time
from transformers import AutoConfig, AutoTokenizer
import transformers
TRUNK = "Qwen/Qwen3.8-27B"
cfg = AutoConfig.from_pretrained(TRUNK)
print("config class:", type(cfg).__name__, "model_type:", cfg.model_type)
tok = AutoTokenizer.from_pretrained(TRUNK)
print("tokenizer ok, vocab", len(tok))
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
t0 = time.time()
model = None
for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
    try:
        mod = getattr(transformers, cls_name)
        model = mod.from_pretrained(TRUNK, quantization_config=bnb,
                                    device_map="auto").eval()
        print("loaded with", cls_name)
        break
    except Exception as e:
        print(f"{cls_name} failed: {type(e).__name__}: {str(e)[:160]}")
if model is None:
    raise SystemExit("could not load trunk")
print(f"load {time.time()-t0:.0f}s  mem {torch.cuda.memory_allocated()/2**30:.1f} GiB")
# the wrapper may nest the LM; find the module that yields hidden states
ids = tok("State: the sky is blue.\nQuestion: what colour?\nOption: blue",
          return_tensors="pt").to(model.device)
with torch.no_grad():
    o = model(**ids, output_hidden_states=True, use_cache=False, return_dict=True)
hs = o.hidden_states[-1]
print("hidden_states[-1]:", tuple(hs.shape), "n_layers:", len(o.hidden_states))
print("cfg layers:", getattr(cfg, "num_hidden_layers", None),
      "text_config layers:", getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None))
PY
