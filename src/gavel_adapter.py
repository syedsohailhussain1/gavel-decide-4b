#!/usr/bin/env python3
"""Gavel local adapter for the JevBench harness (our code, their runner).

Drives the v1 frozen-trunk + MLP head: one batched forward per task over
(state, option) pairs, softmax with fitted temperature, native probability
distributions over the task's EXACT label strings. Long states left-
truncated to 512 total tokens (matches training distribution; laya does
the same at its 512 budget — documented, not hidden).

Follows the laya_local adapter contract: __init__ kwargs, load(), run(task)
-> DecisionResult, reserve_estimate(). No edits to the JevBench repo.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, r"D:\jevbench")
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())

import prefix_cache  # noqa: E402
from jevbench.adapters.base import DecisionResult  # noqa: E402


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


class GavelLocalAdapter:
    name = "gavel_local"
    cost_basis = "local_gpu_no_provider_tariff"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None, threads=4,
                 revision=None, head=None, ctx=512, dtype="nf4"):
        self.trunk = endpoint or r"D:\gavel\models\qwen3-4b"
        self.model = model or "gavel-decide-4b"
        self.head_path = head or (r"D:\gavel\models\training_state"
                                  r"\combined_head.pt")
        self.ctx = ctx
        # "nf4" reproduces the published 4-bit numbers and is the only option
        # that fits <8GB VRAM. "bf16" is the honest default on adequate VRAM
        # and is materially faster (bitsandbytes dequantisation, not the
        # model, dominates nf4 throughput).
        self.dtype = dtype
        # Shared-prefix KV cache: every option of an item shares the
        # `State: ...\nQuestion: ...\nOption: ` token prefix, so its K/V is
        # computed once instead of once per option. Verified 36/36 identical
        # decisions, 2.77x faster overall. Set GAVEL_PREFIX_CACHE=0 to disable.
        self.use_prefix_cache = os.environ.get("GAVEL_PREFIX_CACHE", "1") != "0"
        self.min_shared = int(os.environ.get("GAVEL_PC_MIN_SHARED", "8"))
        # 800 was too high: it excluded the easy/standard items that the
        # controlled bench showed winning 1.6-1.7x. The only measured loss was
        # a synthetic 2-option micro-case around 300 tokens, so the floor sits
        # just under that.
        self.min_tokens = int(os.environ.get("GAVEL_PC_MIN_TOKENS", "200"))
        self._pc_stats = {"cached": 0, "naive": 0, "cached_tokens": 0,
                          "naive_tokens": 0}
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.revision = revision
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        self.tok = AutoTokenizer.from_pretrained(
            self.trunk, trust_remote_code=False)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev.type == "cuda" and self.dtype == "nf4":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.trunk, quantization_config=bnb, device_map="auto",
                trust_remote_code=False).eval()
        elif dev.type == "cuda":
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.trunk, dtype=torch.bfloat16, device_map={"": 0},
                trust_remote_code=False).eval()
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.trunk, dtype=torch.float32,
                trust_remote_code=False).eval().to(dev)
        for p in self.lm.parameters():
            p.requires_grad = False
        self.dev = next(self.lm.parameters()).device
        hp = torch.load(self.head_path, map_location="cpu", weights_only=False)
        self.head = Head(hp["hidden"], hp.get("wide", 512))
        self.head.load_state_dict(hp["head"])
        self.head.to(self.dev).eval()
        self.temp = float(hp["temperature"])
        # Meta-calibrator: prefer weights embedded in the head file (versioned
        # together), fall back to a meta_cal.json sidecar beside it.
        self.meta, self._meta_fn = None, None
        try:
            from gavel_meta import meta_rescale as _mr

            meta = hp.get("meta_cal")
            if meta is None:
                import json as _js
                import os as _os

                mp = _os.path.join(
                    _os.path.dirname(_os.path.abspath(self.head_path)),
                    "meta_cal.json")
                if _os.path.exists(mp):
                    with open(mp, encoding="utf-8") as f:
                        meta = _js.load(f)
            if meta is not None:
                self.meta, self._meta_fn = meta, _mr
        except Exception:
            self.meta, self._meta_fn = None, None
        self._loaded = True

    def _pair_texts(self, task):
        st = flat(task.state)
        instr = task.question.get("instructions", "")
        qtype = task.question.get("type")
        crit = task.question.get("criteria") or {}
        if qtype == "choice":
            opts = [(k, crit.get(k) if isinstance(crit, dict) else None)
                    for k in task.labels]
        elif qtype == "noul":
            opts = [(l, crit.get(l) if isinstance(crit, dict) else None)
                    for l in task.labels]
        elif qtype == "score":
            if isinstance(crit, dict):
                opts = [(k, crit[k]) for k in task.labels]
            else:
                lv = list(crit) if crit else list(task.labels)
                opts = [(l, lv[i] if i < len(lv) else None)
                        for i, l in enumerate(task.labels)]
        else:
            return None
        texts = []
        for lab, desc in opts:
            t = f"State: {st}\nQuestion: {instr}\nOption: {lab}"
            if desc:
                t += f": {desc}"
            texts.append(t)
        return opts, texts

    def run(self, task) -> DecisionResult:
        res = DecisionResult(adapter=self.name, ok=False,
                             probs_source="native", model=self.model)
        res.request_body = {"task": task.id, "type": task.question.get("type")}
        try:
            if not self._loaded:
                self.load()
            built = self._pair_texts(task)
            if built is None:
                res.error = f"unsupported type {task.question.get('type')}"
                return res
            opts, texts = built
            # Left-truncate: drop old state context, never question+option.
            ids_list, tail_budget = [], self.ctx
            tail_probe = self.tok(opts[0][0] + (opts[0][1] or ""),
                                  truncation=False)["input_ids"]
            keep_state = max(self.ctx - len(tail_probe) - 64, 64)
            cut = []
            for t in texts:
                parts = t.split("\nQuestion:", 1)
                if len(parts) == 2:
                    sids = self.tok(parts[0], truncation=False)["input_ids"]
                    cut.append(self.tok.decode(sids[-keep_state:]) +
                               "\nQuestion:" + parts[1])
                else:
                    cut.append(t)
            enc = self.tok(cut, return_tensors="pt", truncation=True,
                           max_length=self.ctx, padding=True).to(self.dev)
            ids_list = [self.tok(c, truncation=True,
                                 max_length=self.ctx)["input_ids"] for c in cut]
            P0 = prefix_cache.common_prefix_len(ids_list)
            max_suf = max(len(x) - P0 for x in ids_list)
            P = min(P0, max(1, self.ctx - max_suf))
            cached_tokens = P + sum(len(x) - P for x in ids_list)
            naive_tokens = sum(len(x) for x in ids_list)
            use_pc = (self.use_prefix_cache and len(texts) >= 2
                      and P >= self.min_shared
                      and cached_tokens <= 0.85 * naive_tokens
                      and naive_tokens >= self.min_tokens)
            t0 = time.perf_counter()
            with torch.no_grad():
                if use_pc:
                    lg = prefix_cache.score_prefix_cached(
                        self.lm, self.head, ids_list, self.dev, self.ctx)
                    self._pc_stats["cached"] += 1
                else:
                    o = self.lm(input_ids=enc["input_ids"],
                                attention_mask=enc.get("attention_mask"),
                                output_hidden_states=True, use_cache=False,
                                return_dict=True)
                    hs = o.hidden_states[-1].float()
                    last = (enc.get("attention_mask").sum(1) - 1).clamp_min(0)
                    lg = self.head(hs[torch.arange(len(texts)),
                                       last].to(self.dev)).tolist()
                    self._pc_stats["naive"] += 1
                self._pc_stats["cached_tokens"] += cached_tokens
                self._pc_stats["naive_tokens"] += naive_tokens
            res.latency_s = time.perf_counter() - t0
            m = max(v / self.temp for v in lg)
            ex = [math.exp(v / self.temp - m) for v in lg]
            s = sum(ex)
            probs = {lab: e / s for (lab, _), e in zip(opts, ex)}
            if self._meta_fn is not None:
                try:
                    probs = self._meta_fn(probs, self.meta)
                except Exception:
                    pass
            res.probs = {k: float(v) for k, v in probs.items()}
            # output_tokens MUST be present: runner.py:37-40 only computes a
            # cost when BOTH input_tokens and output_tokens are numeric, and
            # composite.cost(None) raises. We emit a distribution, never
            # tokens, so 0 is truthful — same convention as certo_local /
            # smalljev_local / so1_decider.
            res.usage = {"input_tokens": int(enc.get("attention_mask").sum()),
                         "output_tokens": 0}
            res.raw = {"runtime": {"trunk": self.trunk, "ctx": self.ctx,
                                   "temperature": self.temp,
                                   "meta_cal": bool(self._meta_fn),
                                   "revision": self.revision,
                                   "path": "prefix_cache" if use_pc else "batched",
                                   "shared_prefix": P0,
                                   "probability_origin": "native-softmax" +
                                   ("+meta" if self._meta_fn else "")}}
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0
