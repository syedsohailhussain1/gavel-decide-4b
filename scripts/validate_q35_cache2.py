"""Does activate_past_recording() make the shared-prefix cache work here?

If yes, our 2.3-4.2x prefix cache ports to this architecture with one added
call, and the read-out must stay numerically identical to a full forward.
If no, the 0.8B trunk cannot use our main serving optimisation as written.
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-0.8B-Base"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="auto").eval()
dev = next(model.parameters()).device

texts = [f"State: a package is late.\nQuestion: which intent?\nOption: {o}"
         for o in ("track_order", "cancel_order", "change_address")]
ids = [tok(t, return_tensors="pt")["input_ids"].to(dev) for t in texts]
P = 12  # shared prefix length

print("=" * 64)
print("1. activate_past_recording + crop")
print("=" * 64)
with torch.no_grad():
    pre = model(input_ids=ids[1][:, :P], use_cache=True, return_dict=True)
c = pre.past_key_values
print(f"  cache: {type(c).__name__}")
act = getattr(c, "activate_past_recording", None)
print(f"  activate_past_recording present: {act is not None}")
if act is None:
    raise SystemExit("  -> no recording API; prefix cache cannot port")

act()
print(f"  after activate: get_seq_length = {c.get_seq_length()}")
try:
    c.crop(-1)
    print(f"  crop(-1) OK -> get_seq_length = {c.get_seq_length()}")
    print("  VERDICT: crop works once recording is active")
except Exception as e:
    print(f"  crop failed: {type(e).__name__}: {e}")
    raise SystemExit

print("\n" + "=" * 64)
print("2. numerical equivalence: cached suffix vs full forward")
print("=" * 64)
for name, idx in (("option 1", 1), ("option 2", 2)):
    with torch.no_grad():
        full = model(input_ids=ids[idx], use_cache=False,
                     output_hidden_states=True, return_dict=True
                     ).hidden_states[-1][0, -1].float()
        pre = model(input_ids=ids[idx][:, :P], use_cache=True,
                    return_dict=True)
        cc = pre.past_key_values
        cc.activate_past_recording()
        cc.crop(-1)
        out = model(input_ids=ids[idx][:, P:], past_key_values=cc,
                    use_cache=True, output_hidden_states=True,
                    return_dict=True)
        cached = out.hidden_states[-1][0, -1].float()
    cos = torch.nn.functional.cosine_similarity(full[None], cached[None]).item()
    rel = ((full - cached).norm() / full.norm()).item()
    print(f"  {name}: cosine={cos:.6f}  relative L2={rel:.4%}  "
          f"{'IDENTICAL' if cos > 0.9999 else 'DIFFERS'}")

print("\n" + "=" * 64)
print("3. head scores: do they agree?")
print("=" * 64)
torch.manual_seed(0)
head = torch.nn.Sequential(torch.nn.Linear(1024, 512), torch.nn.GELU(),
                           torch.nn.Linear(512, 256), torch.nn.GELU(),
                           torch.nn.Linear(256, 1)).to(dev).eval()
naive_scores, cached_scores = [], []
with torch.no_grad():
    for idx in (0, 1, 2):
        o = model(input_ids=ids[idx], use_cache=False,
                  output_hidden_states=True, return_dict=True)
        naive_scores.append(float(head(o.hidden_states[-1][0, -1].float())))

    pre = model(input_ids=ids[1][:, :P], use_cache=True, return_dict=True)
    cc = pre.past_key_values
    cc.activate_past_recording()
    for idx in (0, 1, 2):
        cc.crop(-(ids[idx].shape[1] - P - 1))
        out = model(input_ids=ids[idx][:, P:], past_key_values=cc,
                    use_cache=True, output_hidden_states=True,
                    return_dict=True)
        cached_scores.append(float(head(out.hidden_states[-1][0, -1].float())))

for i, (a, b) in enumerate(zip(naive_scores, cached_scores)):
    print(f"  option {i}: naive={a:+.6f}  cached={b:+.6f}  diff={abs(a-b):.2e}")
print(f"  argmax naive  = {int(max(range(3), key=lambda i: naive_scores[i]))}")
print(f"  argmax cached = {int(max(range(3), key=lambda i: cached_scores[i]))}")
print(f"  agree: "
      f"{max(range(3), key=lambda i: naive_scores[i]) == max(range(3), key=lambda i: cached_scores[i])}")

print("\n" + "=" * 64)
print("4. memory cost of recording")
print("=" * 64)
if dev.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(input_ids=ids[0])
    base = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        pre = model(input_ids=ids[0], use_cache=True, return_dict=True)
        cc = pre.past_key_values
        cc.activate_past_recording()
    rec = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"  weights+activations {base:.3f} GiB")
    print(f"  with past recording {rec:.3f} GiB  (+{(rec-base)*1024:.0f} MiB)")
