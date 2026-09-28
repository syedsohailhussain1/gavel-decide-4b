"""Focused test: does the shared-prefix cache port to this architecture?

Correction to the earlier run: the device error was MY bug - I passed CPU
tensors to a CUDA model. The architecture is also not what I guessed. The
config says:

    layer_types: 18 linear_attention + 6 full_attention
    full_attention_interval: 4

So it is NOT Mamba/SSM. It is linear attention (constant-size recurrent state)
hybridised with full attention every 4th layer. Only 6 of 24 layers carry a
growing KV cache at all, which changes the prefix-cache arithmetic completely:
we cannot crop one uniform cache, we would have to reset linear-attention state
and crop only the 6 full-attention layers.
"""
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-0.8B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="auto").eval()
dev = next(model.parameters()).device
print(f"loaded on {dev}  ({torch.cuda.memory_allocated()/2**30:.2f} GiB)")

opts = ["track_order", "cancel_order", "change_address"]
texts = [f"State: a package is late.\nQuestion: which intent?\nOption: {o}"
         for o in opts]
ids = [tok(t, return_tensors="pt")["input_ids"].to(dev) for t in texts]
print(f"option token lengths: {[int(i.shape[1]) for i in ids]}")

print("\n" + "=" * 64)
print("A. naive: independent forward per option")
print("=" * 64)
with torch.no_grad():
    for i in ids:
        model(input_ids=i, use_cache=False)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(5):
        for i in ids:
            o = model(input_ids=i, use_cache=False, output_hidden_states=True,
                      return_dict=True)
            v = o.hidden_states[-1][0, -1].float()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    naive = (time.perf_counter() - t0) / 5
print(f"  {naive*1000:.1f} ms for 3 options")

print("\n" + "=" * 64)
print("B. cache type and crop support")
print("=" * 64)
with torch.no_grad():
    o = model(input_ids=ids[0], use_cache=True, return_dict=True)
c = o.past_key_values
print(f"  class: {type(c).__name__}")
print(f"  .crop present: {hasattr(c, 'crop')}")
print(f"  .get_seq_length present: {hasattr(c, 'get_seq_length')}")
if hasattr(c, "keys"):
    ks = list(c.keys())
    print(f"  n entries: {len(ks)}   first keys: {ks[:3]}")
    for attr in ("layers", "key_cache", "value_cache", "mamba_cache",
                 "conv_states", "ssm_states", "recurrent"):
        if hasattr(c, attr):
            v = getattr(c, attr)
            n = len(v) if hasattr(v, "__len__") else "?"
            print(f"    .{attr:14} len={n}")

if hasattr(c, "crop"):
    before = c.get_seq_length() if hasattr(c, "get_seq_length") else "?"
    try:
        c.crop(-3)
        after = c.get_seq_length() if hasattr(c, "get_seq_length") else "?"
        print(f"  crop(-3): {before} -> {after}   CROP WORKS")
    except Exception as e:
        print(f"  crop(-3) FAILED {type(e).__name__}: {e}")
else:
    print("  *** no .crop() -> our prefix_cache.py cannot be reused as-is ***")

print("\n" + "=" * 64)
print("C. does caching + reuse change the read-out vector?")
print("=" * 64)
# the real question: is a cached suffix read-out numerically equal to a
# full independent forward? if not, the cache is scoring a different thing.
with torch.no_grad():
    full = model(input_ids=ids[1], use_cache=False, output_hidden_states=True,
                 return_dict=True).hidden_states[-1][0, -1].float()
    P = 8
    pre = model(input_ids=ids[1][:, :P], use_cache=True, return_dict=True)
    cc = pre.past_key_values
    if hasattr(cc, "crop"):
        cc.crop(-1)  # drop the last prefix token, we re-feed it
    suf = ids[1][:, P:]
    out = model(input_ids=suf, past_key_values=cc, use_cache=True,
                output_hidden_states=True, return_dict=True)
    cached = out.hidden_states[-1][0, -1].float()
cos = torch.nn.functional.cosine_similarity(full[None], cached[None]).item()
rel = ((full - cached).norm() / full.norm()).item()
print(f"  full forward vs cached: cosine={cos:.6f}  relative L2={rel:.4%}")
print(f"  argmax head score equal? "
      f"{'LIKELY' if cos > 0.999 else 'NO - cache changes the read-out'}")

print("\n" + "=" * 64)
print("D. fp16 on this GPU")
print("=" * 64)
if dev.type == "cuda":
    cap = torch.cuda.get_device_capability(0)
    print(f"  compute capability {cap[0]}.{cap[1]}")
    print(f"  tensor cores for fp16: {'yes' if cap[0] >= 8 else 'NO (sm<8.0)'}")
    print("  -> a 353ms/prefill figure on this card is a HARDWARE artifact,")
    print("     not a property of the model. It says nothing about a 3090,")
    print("     an A10G or a 4090.")
