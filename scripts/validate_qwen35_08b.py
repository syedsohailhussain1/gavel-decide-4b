"""Validate Qwen3.5-0.8B-Base locally on the 4GB GTX 1650, for $0.

Three things must be true before it is worth generating 100k examples for it:
  1. it loads at all - it is a Qwen3_5ForConditionalGeneration multimodal
     wrapper, so AutoModelForCausalLM may not work
  2. the text submodel is reachable and yields hidden states of width 1024,
     which is what a 656,385-parameter head needs
  3. the shared-prefix cache still works - THIS IS THE RISK. The config has
     mamba_ssm_dtype / mrope_* / mamba_section, so it is a hybrid Mamba +
     attention model. Our prefix_cache.py uses DynamicCache.crop(-S), which is
     attention-specific. Mamba layers carry recurrent state, not a KV cache.
"""
import json
import time
import traceback

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

MODEL = "Qwen/Qwen3.5-0.8B-Base"
print(f"transformers {transformers.__version__}  torch {torch.__torch_version__ if hasattr(torch,'__torch_version__') else torch.__version__}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

print("\n" + "=" * 66)
print("1. CONFIG")
print("=" * 66)
cfg = AutoConfig.from_pretrained(MODEL)
print(f"  class      {type(cfg).__name__}")
print(f"  model_type {cfg.model_type}")
print(f"  arch       {getattr(cfg, 'architectures', None)}")
tc = getattr(cfg, "text_config", None)
print(f"  vision_config present: {getattr(cfg, 'vision_config', None) is not None}")
if tc is not None:
    for k in ("hidden_size", "num_hidden_layers", "vocab_size",
              "num_attention_heads", "num_key_value_heads",
              "layer_types", "mamba_ssm_dtype", "full_attention_interval"):
        if hasattr(tc, k):
            v = getattr(tc, k)
            if isinstance(v, list) and len(v) > 8:
                from collections import Counter
                v = f"{len(v)} layers: {dict(Counter(v))}"
            print(f"  {k:26} {v}")
    print(f"\n  -> head width would be {tc.hidden_size}, "
          f"head params {(tc.hidden_size*512+512)+(512*256+256)+257:,}")

print("\n" + "=" * 66)
print("2. LOAD + hidden states")
print("=" * 66)
tok = AutoTokenizer.from_pretrained(MODEL)
print(f"  tokenizer ok, vocab {len(tok)}")

model = None
for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
    cls = getattr(transformers, name, None)
    if cls is None:
        print(f"  {name:28} not exported")
        continue
    try:
        t0 = time.time()
        m = cls.from_pretrained(MODEL, dtype=torch.float16,
                                device_map="auto")
        model = m.eval()
        print(f"  {name:28} OK  ({time.time()-t0:.0f}s)  -> {type(m).__name__}")
        break
    except Exception as e:
        print(f"  {name:28} FAILED {type(e).__name__}: {str(e)[:110]}")

if model is None:
    raise SystemExit("could not load with any auto class")

print(f"  top-level children: {[n for n, _ in model.named_children()]}")
mem = torch.cuda.memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
print(f"  vram allocated: {mem:.2f} GiB")

ids = tok("State: a package is late.\nQuestion: which intent?\nOption: track_order",
          return_tensors="pt").to(model.device)
with torch.no_grad():
    o = model(input_ids=ids["input_ids"],
              attention_mask=ids.get("attention_mask"),
              output_hidden_states=True, use_cache=False, return_dict=True)
hs = o.hidden_states[-1]
print(f"  hidden_states: n={len(o.hidden_states)}  final={tuple(hs.shape)}")
print(f"  logits: {tuple(o.logits.shape)}")
assert hs.shape[-1] == tc.hidden_size, "hidden width mismatch"
print(f"  -> read-out vector is {hs.shape[-1]}-dim, matches the head")

print("\n" + "=" * 66)
print("3. SHARED-PREFIX CACHE - does it survive hybrid Mamba?")
print("=" * 66)
prompts = [f"State: a package is late.\nQuestion: which intent?\nOption: {o}"
           for o in ("track_order", "cancel_order", "change_address")]

try:
    from transformers import DynamicCache
    p0 = tok(prompts[0], return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model(input_ids=p0[:, :-4], use_cache=True, return_dict=True)
    c = out.past_key_values
    print(f"  cache type: {type(c).__name__}")
    layers = len(c) if hasattr(c, "__len__") else "?"
    print(f"  cache length: {layers}")
    print(f"  has .crop(): {hasattr(c, 'crop')}")
    # is there mamba state hiding in there?
    keys = list(c.keys()) if hasattr(c, "keys") else []
    print(f"  first few keys: {keys[:4]}")
    if hasattr(c, "crop"):
        c.crop(-4)
        print(f"  crop(-4) OK, new length = {c.get_seq_length() if hasattr(c,'get_seq_length') else '?'}")
    print("\n  VERDICT: DynamicCache.crop available -> prefix cache should port.")
except Exception as e:
    print(f"  DynamicCache path failed: {type(e).__name__}: {e}")
    traceback.print_exc(limit=2)

try:
    out2 = model(input_ids=p0, use_cache=True, return_dict=True)
    c2 = out2.past_key_values
    print(f"\n  full-sequence cache type: {type(c2).__name__}")
    if hasattr(c2, "crop"):
        c2.crop(-4)
        print("  crop works on the full-sequence cache too")
    else:
        print("  *** NO .crop() -> prefix cache will NOT port as written ***")
except Exception as e:
    print(f"  full-seq cache failed: {type(e).__name__}: {e}")

print("\n" + "=" * 66)
print("4. LATENCY - per-option prefill throughput")
print("=" * 66)
with torch.no_grad():
    for _ in range(2):
        model(input_ids=ids["input_ids"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 20
    for _ in range(N):
        model(input_ids=ids["input_ids"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N
print(f"  {dt*1000:.1f} ms per prefill of {ids['input_ids'].shape[1]} tokens")
print(f"  ~{1/dt:.0f} forwards/sec  ->  100k rows ~= "
      f"{100000/(1/dt)/60:.0f} min,  1.47M rows ~= {1.47e6/(1/dt)/60:.0f} min")
