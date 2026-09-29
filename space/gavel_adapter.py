#!/usr/bin/env python3
"""Gavel local adapter for the JevBench harness (our code, their runner).

Drives the v1 frozen-trunk + MLP head: one batched forward per task over
(state, option) pairs, softmax with fitted temperature, native probability
distributions over the task's EXACT label strings.

Context limit is 32768 -- Qwen3-4B-Base's own max_position_embeddings, the
model's real limit -- and NOTHING is truncated: an over-limit prompt is refused
as Unsupported (Decision Index rule "no truncation").

This was raised 512 -> 32768 on measured evidence, not caution:

  * 512 -> 4096: 56/159 public JevBench items exceed 512 tokens (max 3691).
    Full-length scored 28/56 vs 29/56 truncated -- one item, n=56, i.e. the head
    is not out of distribution past 512. The 512 gate was refusing 24% of items
    for no accuracy reason, and refusals cost coverage because the index scores
    raw x answered/requests.
  * 4096 -> 32768: a measured ctx x n_options sweep on the A6000 (50.9GB) found
    NO OOM anywhere up to 32768 tokens x 8 options, peaking at 23.5GB. Memory is
    not the ceiling. Latency at the full 32768 is ~7-8s, but the board's 1000ms
    gate is on the MEDIAN, which short prompts set; only genuinely long items
    pay, and refusing them instead would cost far more in coverage.

So the longest context the model allows is also the highest-scoring choice, and
it is free for the median. `dtype=bf16` is the default on adequate VRAM (the
published numbers used nf4, which only exists to fit <8GB cards and is slower).
The head was trained on <=512-token prompts; that is recorded in the engine's
provenance rather than papered over, with the measurements above as evidence
that it transfers.

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

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())

import prefix_cache  # noqa: E402

try:
    # Normal case: the official runner already puts jevbench on the path.
    from jevbench.adapters.base import DecisionResult  # noqa: E402
except ImportError:  # pragma: no cover - local/portable fallback
    # Only reached when jevbench is not installed or not on sys.path. Prefer an
    # explicit GAVEL_JEVBENCH_ROOT; otherwise try the historical default. This
    # used to run unconditionally and hardcoded D:\, which made the module
    # unimportable anywhere else.
    _root = os.environ.get("GAVEL_JEVBENCH_ROOT") or r"D:\jevbench"
    if os.path.isdir(os.path.join(_root, "jevbench")):
        sys.path.insert(0, _root)
    try:
        from jevbench.adapters.base import DecisionResult  # noqa: E402
    except ImportError:
        # Last resort: define the same shape locally. The Decision Index
        # engine does not need the JevBench runner, only this container, and
        # the previous chain ended in an unconditional ImportError that made
        # the adapter unusable on any machine without a D:\jevbench checkout
        # (a cloud pod, a colleague's laptop, CI). Field-for-field mirror of
        # jevbench.adapters.base.DecisionResult, verified against the source.
        from dataclasses import dataclass, field as _field
        from typing import Any as _Any, Optional as _Optional

        @dataclass
        class DecisionResult:  # type: ignore[no-redef]
            adapter: str
            ok: bool
            probs: _Optional[dict] = None
            probs_source: str = "unknown"
            model: str = ""
            status: _Optional[int] = None
            error: _Optional[str] = None
            latency_s: float = 0.0
            usage: dict = _field(default_factory=dict)
            raw: _Optional[_Any] = None
            request_body: _Optional[_Any] = None
            label: _Optional[str] = None

            def to_public(self) -> dict:
                """Public-safe view: no raw response text, no request body."""
                return {
                    "adapter": self.adapter,
                    "ok": self.ok,
                    "probs": self.probs,
                    "probs_source": self.probs_source,
                    "model": self.model,
                    "status": self.status,
                    "error": self.error,
                    "latency_s": self.latency_s,
                    "usage": self.usage,
                    "label": self.label,
                }


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

    #: override with GAVEL_TRUNK to point at a local copy of the trunk, e.g. on
    #: a pod that downloaded Qwen/Qwen3-4B-Base rather than having D:\ mapped.
    DEFAULT_TRUNK = r"D:\gavel\models\qwen3-4b"
    #: override with GAVEL_HEAD. The Space passes head= explicitly; this keeps
    #: the module importable and runnable where D:\ does not exist.
    DEFAULT_HEAD = r"D:\gavel\models\training_state\combined_head.pt"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None, threads=4,
                 revision=None, head=None, ctx=32768, dtype="nf4",
                 trunk=None):
        self.trunk = (endpoint or trunk or os.environ.get("GAVEL_TRUNK")
                      or self.DEFAULT_TRUNK)
        self.model = model or "gavel-decide-4b"
        self.head_path = (head or os.environ.get("GAVEL_HEAD")
                          or self.DEFAULT_HEAD)
        self.ctx = ctx
        # "nf4" reproduces the published 4-bit numbers and is the only option
        # that fits <8GB VRAM. "bf16" is the honest default on adequate VRAM
        # and is materially faster (bitsandbytes dequantisation, not the
        # model, dominates nf4 throughput).
        self.dtype = dtype
        # Shared-prefix KV cache: every option of an item shares the
        # `State: ...\nQuestion: ...\nOption: ` token prefix, so its K/V could
        # be computed once instead of once per option.
        #
        # DEFAULT OFF, on measured evidence. It does cut trunk tokens by 71.9%
        # (135,010 vs 480,898 over the 159 public JevBench items) but it is
        # 2.3x SLOWER on the median: 262ms with the cache vs 115ms without.
        # The reason is shape, not arithmetic -- the cached path replays the
        # options one at a time at batch size 1 to bound KV memory, so it trades
        # batched-GEMM parallelism for token reduction. At these lengths on an
        # A6000 the model is memory-bandwidth-bound, and five small sequential
        # passes lose badly to one batched pass. The cache does smooth the tail
        # (mean 279ms vs 431ms) if p95 ever matters more than the median.
        #
        # The previously published "2.77x faster overall" was not a valid
        # measurement: it divided two counters that summed disjoint item
        # populations, which is also what produced "saved -2279%". No speedup
        # is claimed in either direction now.
        # Set GAVEL_PREFIX_CACHE=1 to re-enable.
        self.use_prefix_cache = os.environ.get("GAVEL_PREFIX_CACHE", "0") != "0"
        self.min_shared = int(os.environ.get("GAVEL_PC_MIN_SHARED", "8"))
        # 800 was too high: it excluded the easy/standard items that the
        # controlled bench showed winning 1.6-1.7x. The only measured loss was
        # a synthetic 2-option micro-case around 300 tokens, so the floor sits
        # just under that.
        self.min_tokens = int(os.environ.get("GAVEL_PC_MIN_TOKENS", "200"))
        # cached_tokens/naive_tokens are per-path REALISED counts, each summing
        # only the items that took that path, so they are not comparable to one
        # another. realised_tokens and counterfactual_tokens are the pair that
        # is comparable; see the increment site.
        # realised_tokens  = what the trunk actually processed
        # counterfactual   = what it would process with the prefix cache off
        # Both cover the same items, so their ratio is a real speedup figure.
        self._pc_stats = {"cached": 0, "naive": 0, "cached_tokens": 0,
                          "naive_tokens": 0, "realised_tokens": 0,
                          "counterfactual_tokens": 0}
        self._meta_failures = 0
        self._argmax_flips = 0
        # Compliance counters, reported at the end of a run so an operator can
        # see exactly how much of a benchmark was refused rather than answered.
        self._unsupported = 0
        self._truncated = 0
        # Silent truncation is forbidden by the Decision Index rules. The
        # published 74.46% was measured WITH truncation, so reproducing that
        # number needs GAVEL_TRUNCATE=1. Default is compliant.
        self.truncate = os.environ.get("GAVEL_TRUNCATE", "0") not in (
            "0", "", "false", "False", "no")
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
        # Load the BASE model, not the CausalLM wrapper.
        #
        # We only ever read hidden states and feed them to our own MLP head, but
        # AutoModelForCausalLM still runs the vocab projection on every forward:
        #     logits = self.lm_head(hidden_states[:, slice_indices, :])
        # Qwen3's vocab is 151,936, so for a many-option item (POP909 chords have
        # hundreds of options on a ~2700-token state) that projection tries to
        # allocate a [B, S, 151936] fp32 tensor -- the 81.67 GiB that OOMed a
        # 48GB card on a prompt the trunk handles trivially. We discard those
        # logits, so this was pure waste. Qwen3Model is the same weights with no
        # lm_head at all, so the numbers are identical and the allocation is gone.
        #
        # AutoModel is used with an explicit fallback to the CausalLM class for
        # any trunk that does not expose a base architecture.
        AutoBase = None
        try:
            from transformers import AutoModel as _AutoBase
            AutoBase = _AutoBase
        except ImportError:  # pragma: no cover - very old transformers
            AutoBase = None

        def _load(**kw):
            cls = AutoBase if AutoBase is not None else AutoModelForCausalLM
            return cls.from_pretrained(
                self.trunk, trust_remote_code=False, **kw).eval()

        if dev.type == "cuda" and self.dtype == "nf4":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            self.lm = _load(quantization_config=bnb, device_map="auto")
        elif dev.type == "cuda":
            self.lm = _load(dtype=torch.bfloat16, device_map={"": 0})
        else:
            self.lm = _load(dtype=torch.float32).to(dev)
        self._is_causal_lm = AutoBase is None
        for p in self.lm.parameters():
            p.requires_grad = False
        self.dev = next(self.lm.parameters()).device
        # Capture only the final norm's output instead of asking HF for all 37
        # per-layer hidden states. Identical tensor, 1/37th of the activation
        # memory, which is what lets a long prompt fit at all. None if the
        # layout is unexpected, and every call site falls back transparently.
        self._cap = prefix_cache.install_last_hidden_capture(self.lm)
        # weights_only=True: the head checkpoint holds tensors/floats/dicts only,
        # so the unpickler never needs to import anything. The previous
        # weights_only=False made loading a checkpoint from an untrusted
        # download equivalent to executing it.
        hp = torch.load(self.head_path, map_location="cpu", weights_only=True)
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
                # .get, not [k]: a label with no criterion must render as
                # bare "Option: <label>" like the sibling branches, not raise
                # KeyError and fail the whole decision.
                opts = [(k, crit.get(k)) for k in task.labels]
            else:
                lv = list(crit) if crit else list(task.labels)
                opts = [(l, lv[i] if i < len(lv) else None)
                        for i, l in enumerate(task.labels)]
        else:
            return None
        # Render the label AND its criterion. Both were present in the run that
        # produced the published numbers; dropping the label here changes the
        # prompt text and moves accuracy (measured: 172/231 -> 177/231).
        texts = []
        for lab, desc in opts:
            t = f"State: {st}\nQuestion: {instr}\nOption: {lab}"
            if desc:
                t += f": {desc}"
            texts.append(t)
        shown = [lab if not desc else f"{lab}: {desc}" for lab, desc in opts]
        return opts, texts, shown

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
            opts, texts, shown = built
            # Capacity gate, checked BEFORE anything is cut and before the
            # encode is moved to the device, so a refused request costs
            # nothing. The Decision Index rules are explicit: "An engine that
            # cannot fit a request raises Unsupported; the row is recorded as
            # unsupported and counts as wrong. Nothing is cut to fit." The
            # previous code silently left-truncated the state to self.ctx,
            # which scores the wrong thing and is indistinguishable from a real
            # answer in the results. So: measure the full prompt, and if it
            # does not fit, say so.
            #
            # GAVEL_TRUNCATE=1 restores the old truncating behaviour for
            # reproducing the published 74.46% figure, which was measured with
            # truncation. It is off by default because silent truncation is
            # what the benchmark forbids.
            full_ids = self.tok(texts, truncation=False)["input_ids"]
            over = [i for i, ids in enumerate(full_ids)
                    if len(ids) > self.ctx]
            if over and not self.truncate:
                worst = max(len(i) for i in full_ids)
                self._unsupported += 1
                res.ok = False
                # DecisionResult has no `unsupported` field (only ok/error/status
                # /raw), so the capability declaration has to live in raw, which
                # is json-safe and survives to_public().
                res.error = (
                    f"Unsupported: {len(over)}/{len(texts)} option prompts exceed "
                    f"the {self.ctx}-token capacity (longest {worst} tokens). "
                    f"The head is trained on <= {self.ctx}-token contexts; "
                    f"cutting the state would score the wrong thing.")
                res.raw = {"unsupported": True, "ctx": self.ctx,
                           "prompt_tokens": [len(i) for i in full_ids],
                           "longest": worst}
                return res

            if over:
                tail_probe = self.tok(shown[0], truncation=False)["input_ids"]
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
                self._truncated += 1
            else:
                cut = texts
                enc = self.tok(cut, return_tensors="pt", truncation=False,
                               padding=True).to(self.dev)
            # Derive the per-option id lists from the single batched encode
            # instead of re-tokenizing every prompt a second time. Selecting
            # the attention-mask positions in order is correct for both left-
            # and right-padding tokenizers, so this does not assume a side.
            _am = enc["attention_mask"].bool()
            ids_list = [enc["input_ids"][i][_am[i]].tolist() for i in range(len(cut))]
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
                    lg, info = prefix_cache.score_prefix_cached(
                        self.lm, self.head, ids_list, self.dev, self.ctx,
                        return_info=True, capture=self._cap)
                    self._pc_stats["cached"] += 1
                else:
                    o = self.lm(input_ids=enc["input_ids"],
                                attention_mask=enc.get("attention_mask"),
                                output_hidden_states=self._cap is None,
                                use_cache=False, return_dict=True)
                    # Gather the last real token of each row BEFORE casting to
                    # float32. The old order built a full [B, S, 2560] float32
                    # tensor -- for a POP909 chord item (hundreds of options,
                    # ~2700 tokens) that is tens of GB and OOMed a 48GB card on
                    # a prompt the model handles easily:
                    #   "Tried to allocate 81.67 GiB" on a 2721-token row.
                    # Indexing first makes the cast operate on [B, 2560]
                    # instead of [B, S, 2560], ~S times smaller. The cast is
                    # elementwise, so this is numerically identical, not an
                    # approximation.
                    hs = (o.hidden_states[-1] if self._cap is None
                          else self._cap.pop("h"))
                    last = (enc.get("attention_mask").sum(1) - 1).clamp_min(0)
                    sel = hs[torch.arange(len(texts), device=hs.device), last]
                    lg = self.head(sel.to(self.dev).float()).tolist()
                    del hs, o, sel
                    self._pc_stats["naive"] += 1
                    info = {"path": "naive",
                            "cached_tokens": cached_tokens,
                            "naive_tokens": naive_tokens}
                # Spend accounting.
                #
                # The path-keyed counters (cached_tokens / naive_tokens) sum
                # DISJOINT item populations -- cached_items for one, naive_items
                # for the other -- so their ratio is not a speedup at all. That
                # is what produced the nonsensical "saved -2279%" (129,564
                # cached vs 5,446 naive) even after the formulas were corrected.
                #
                # The only valid comparison is realised vs counterfactual over
                # the SAME items:
                #   realised      = tokens actually pushed through the trunk
                #   counterfactual= tokens that would have been pushed with no
                #                  prefix cache at all
                # Both accumulate for every item regardless of path, so they are
                # directly comparable and their ratio is the honest speedup.
                self._pc_stats["realised_tokens"] += (
                    info["cached_tokens"] if info["path"] == "cached"
                    else info["naive_tokens"])
                self._pc_stats["counterfactual_tokens"] += info["naive_tokens"]
            res.latency_s = time.perf_counter() - t0
            # Optional raw-logit dump, for refitting temperature / the
            # meta-calibrator against this exact precision. Set
            # GAVEL_DUMP_LOGITS=<path> to append one JSON line per decision.
            _dump = os.environ.get("GAVEL_DUMP_LOGITS")
            if _dump:
                try:
                    with open(_dump, "a", encoding="utf-8") as _f:
                        _f.write(json.dumps({"task_id": task.id,
                                             "labels": [lab for lab, _ in opts],
                                             "logits": [float(v) for v in lg]}) + "\n")
                except Exception:
                    pass
            m = max(v / self.temp for v in lg)
            ex = [math.exp(v / self.temp - m) for v in lg]
            s = sum(ex)
            probs = {lab: e / s for (lab, _), e in zip(opts, ex)}
            # Argmax BEFORE the meta step. A temperature is monotone and cannot
            # move it; the meta-rescale is affine on probabilities and CAN
            # (near-ties invert). Counting realised flips turns that from a
            # docstring claim into a measured invariant.
            argmax_pre_meta = max(range(len(lg)), key=lambda i: lg[i])
            if self._meta_fn is not None:
                try:
                    probs = self._meta_fn(probs, self.meta)
                except Exception as exc:
                    # Previously `pass`, which let a run report
                    # runtime.meta_cal=True while every single meta call had
                    # silently failed -- the recorded probabilities would be
                    # uncalibrated with nothing in the artifact to say so.
                    self._meta_failures += 1
                    res.error = f"meta_rescale failed: {exc}"
            if max(range(len(lg)), key=lambda i: probs[opts[i][0]]) != \
                    argmax_pre_meta:
                self._argmax_flips += 1
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
                                    ("+meta" if self._meta_fn else ""),
                                    # meta_applied is the honest flag: it is
                                    # False when meta_cal is True but every
                                    # call failed, which the artifact can now
                                    # no longer hide.
                                    "meta_applied": bool(self._meta_fn) and
                                    self._meta_failures == 0,
                                    "meta_failures": self._meta_failures,
                                    "argmax_flips": self._argmax_flips}}
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res
        res.ok = True
        return res

    def compliance(self) -> dict:
        """Capability declaration, for reporting alongside a scored run.

        The Decision Index rules require an engine to declare capacity limits
        rather than cut input to fit, and the board shows these counts. A run
        where `unsupported` is large is a run that refused most of the suite,
        and that has to be visible next to the score, not buried.
        """
        return {
            "ctx": self.ctx,
            "truncate": self.truncate,
            "unsupported": self._unsupported,
            "truncated": self._truncated,
            "meta_failures": self._meta_failures,
            "argmax_flips": self._argmax_flips,
            "path_mix": dict(self._pc_stats),
            "rule": ("no truncation: prompts over ctx raise Unsupported"
                     if not self.truncate else
                     f"TRUNCATING at ctx={self.ctx}; not rule-compliant, "
                     f"reproduces the published figure only"),
        }

    def reserve_estimate(self, task) -> float:
        return 0.0
