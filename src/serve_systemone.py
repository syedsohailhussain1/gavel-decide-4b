#!/usr/bin/env python3
"""TypeSafe-compatible serving shim for the Gavel-Decide 4B entry.

Speaks /v1/systemone (the kev/decider pattern the operator already runs):
state + typed questions in, calibrated distributions out. Frozen 4B trunk
(4-bit on small VRAM, fp16/bf16 where it fits) + v1 MLP head, one batched
forward per question's options. Stdlib HTTP — no fastapi, no new deps.

Question wire format (mirrors TypeSafe; see hard_probe.py for the reference):
  {"state": {...} | "...", "questions": {qid: {
      "type": "choice", "instructions": "...",
      "criteria": {label: description, ...}}},
   "model": "gavel-decide-4b"}
  choice -> {"choice": best, "confidence": p, "probabilities": {l: p}}
  noul   -> {"choice": yes|no, "confidence": p, "probabilities": {...}}
  score  -> {"expected": x, "distribution": {level: p}} (levels in given order)

Long states left-truncated to CTX (matches training distribution; logged).

Usage:
  python serve_systemone.py [head_pt] [trunk] [--port P] [--ctx N] [--fp16]
Defaults: combined_head.pt <local qwen3-4b> 8080 512 fp16-off(4-bit)
"""
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HEAD = sys.argv[1] if len(sys.argv) > 1 else "combined_head.pt"
TRUNK = sys.argv[2] if len(sys.argv) > 2 else r"D:\gavel\models\qwen3-4b"
PORT = 8080
CTX = 512
USE_FP16 = False
for i, a in enumerate(sys.argv[3:]):
    if a == "--port":
        PORT = int(sys.argv[4 + i])
    if a == "--ctx":
        CTX = int(sys.argv[4 + i])
    if a == "--fp16":
        USE_FP16 = True

LABELS_YN = ["yes", "no"]


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, w), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(w, w // 2),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def flat(s):
    return s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)


tok = AutoTokenizer.from_pretrained(TRUNK, trust_remote_code=False)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type == "cuda" and not USE_FP16:
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True)
    lm = AutoModelForCausalLM.from_pretrained(
        TRUNK, quantization_config=bnb, device_map="auto",
        trust_remote_code=False).eval()
    print("[serve] trunk 4-bit on", dev, flush=True)
else:
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(
        TRUNK, dtype=dtype, trust_remote_code=False).eval().to(dev)
    print(f"[serve] trunk {dtype} on {dev}", flush=True)
for p in lm.parameters():
    p.requires_grad = False
hp = torch.load(HEAD, map_location="cpu", weights_only=False)
head = Head(hp["hidden"], hp.get("wide", 512))
head.load_state_dict(hp["head"])
head.to(dev).eval()
TEMP = float(hp["temperature"])
print(f"[serve] head {HEAD} T={TEMP}", flush=True)
META, META_FN = hp.get("meta_cal"), None
if META is not None:
    try:
        import sys as _sys
        from pathlib import Path as _P

        _sys.path.insert(0, str(_P(__file__).resolve().parent))
        from m2 import meta_rescale as _mr

        META_FN = _mr
        print("[serve] meta-calibration ON (top-rescale)", flush=True)
    except Exception as e:
        META, META_FN = None, None
        print(f"[serve] meta-calibration OFF ({e})", flush=True)


def score_options(state, instructions, options):
    """options: [(label, description|None)] -> ([(label, prob)], order kept)."""
    texts = []
    for lab, desc in options:
        t = f"State: {state}\nQuestion: {instructions}\nOption: {lab}"
        if desc:
            t += f": {desc}"
        texts.append(t)
    # Left-truncate per text (drop old state, never question+option).
    # Right-truncation deletes the question on longs (measured artifact).
    cut = []
    for t in texts:
        ids = tok(t, truncation=False)["input_ids"]
        cut.append(t if len(ids) <= CTX else tok.decode(ids[-CTX:]))
    enc = tok(cut, return_tensors="pt", truncation=False,
              padding=True).to(dev)
    with torch.no_grad():
        o = lm(input_ids=enc["input_ids"],
               attention_mask=enc.get("attention_mask"),
               output_hidden_states=True, use_cache=False, return_dict=True)
        hs = o.hidden_states[-1].float()
        last = (enc.get("attention_mask").sum(1) - 1).clamp_min(0)
        lg = head(hs[torch.arange(len(texts)), last].to(dev)).tolist()
    m = max(v / TEMP for v in lg)
    ex = [math.exp(v / TEMP - m) for v in lg]
    s = sum(ex)
    out = [(lab, e / s) for (lab, _), e in zip(options, ex)]
    if META_FN is not None:
        try:
            probs = META_FN({lab: p for lab, p in out}, META)
            out = [(lab, probs[lab]) for lab, _ in options]
        except Exception:
            pass
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/health"):
            return self._send({"ok": True, "model": "gavel-decide-4b"})
        return self._send({"error": "unknown route"}, 400)

    def do_POST(self):
        if self.path != "/v1/systemone":
            return self._send({"error": "unknown route"}, 400)
        try:
            body = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length", 0))))
        except Exception:
            return self._send({"error": "bad JSON"}, 400)
        state = flat(body.get("state", ""))
        questions = body.get("questions", {})
        answers, toks = {}, 0
        try:
            for qid, q in questions.items():
                qtype = q.get("type", "choice")
                instr = q.get("instructions", "")
                if qtype == "choice":
                    crit = q.get("criteria", {})
                    opts = [(k, crit[k]) for k in crit] if isinstance(
                        crit, dict) else [(c, None) for c in crit]
                elif qtype == "noul":
                    labs = list(q.get("labels", LABELS_YN))
                    crit = q.get("criteria", {}) or {}
                    opts = [(l, crit.get(l)) for l in labs]
                elif qtype == "score":
                    crit = q.get("criteria", [])
                    levels = list(crit.keys()) if isinstance(
                        crit, dict) else list(crit)
                    opts = [(l, crit[l] if isinstance(crit, dict) else None)
                            for l in levels]
                else:
                    answers[qid] = {"error": f"unknown type {qtype}"}
                    continue
                if not opts:
                    answers[qid] = {"error": "no options"}
                    continue
                toks += sum(len(tok(t).get("input_ids", t)) if isinstance(
                    t, str) else 0 for t in [state, instr])
                scored = score_options(state, instr, opts)
                order = sorted(range(len(scored)),
                               key=lambda k: -scored[k][1])
                best = scored[order[0]]
                probs = {lab: round(p, 6) for lab, p in scored}
                if qtype == "score":
                    exp = sum(k * probs[lab] for k, (lab, _) in enumerate(opts))
                    answers[qid] = {"expected": round(exp, 4),
                                    "distribution": probs,
                                    "confidence": round(best[1], 4)}
                else:
                    answers[qid] = {"choice": best[0],
                                    "confidence": round(best[1], 4),
                                    "probabilities": probs}
        except Exception as e:
            return self._send({"error": str(e)[:200]}, 400)
        return self._send({"answers": answers,
                           "model": body.get("model", "gavel-decide-4b"),
                           "usage": {"input_tokens": toks,
                                     "output_tokens": 0}})


if __name__ == "__main__":
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[serve] systemone-compat on 127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()
