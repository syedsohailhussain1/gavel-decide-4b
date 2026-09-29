"""Shared-prefix KV cache for pairwise option scoring.

Every option for one item is `State: ... \\nQuestion: ... \\nOption: <label...>`.
The token prefix up to `\\nOption: ` is byte-identical across all options of the
item, so the K/V for that prefix can be computed once and reused.

    naive   : N x (P + S)  token-forwards
    cached  : 1 x P + N x S  token-forwards

The cache is reused sequentially (one option at a time, batch 1) and cropped
back after each option, because expanding the prefix KV to batch N multiplies
KV memory by N and OOMs a 4 GB card. No padding is needed in that mode, so
there is no attention-mask bookkeeping per option.
"""
import time

import torch


def install_last_hidden_capture(lm):
    """Capture the trunk's final-norm output with a forward hook.

    `output_hidden_states=True` makes HF materialise one hidden-state tensor per
    layer -- 37 for this 36-layer trunk -- while the head reads exactly one, the
    last. At 10 options x 32k tokens that is tens of GB of tensors nobody looks
    at, and it is the binding limit on how long a prompt can be served at all.

    By construction HF's `hidden_states[-1]` IS the output of the final norm
    (the tuple gets one entry per layer, then a final entry appended after the
    norm), so hooking that module yields the *identical* tensor for 1/37th of
    the activation memory. Returns a single-slot dict that callers `.pop()`, or
    None if the module layout is not the expected one, in which case every
    caller transparently keeps the old path.
    """
    target = None
    base = getattr(lm, "model", None)
    if base is not None:
        target = getattr(base, "norm", None)
    if target is None:
        target = getattr(lm, "norm", None)
    if target is None or not hasattr(target, "register_forward_hook"):
        return None
    slot = {}

    def _hook(_mod, _inp, out):
        slot["h"] = out

    target.register_forward_hook(_hook)
    return slot


def common_prefix_len(id_lists):
    """Length of the token prefix shared by every option."""
    n = min(len(x) for x in id_lists)
    i = 0
    first = id_lists[0]
    while i < n:
        v = first[i]
        for x in id_lists:
            if x[i] != v:
                return i
        i += 1
    return i


def score_naive(lm, head, id_lists, device, pad_id=0, capture=None):
    """Reference path: one batched forward over all options (current behaviour)."""
    n = len(id_lists)
    L = max(len(x) for x in id_lists)
    inp = torch.full((n, L), pad_id, dtype=torch.long)
    am = torch.zeros((n, L), dtype=torch.long)
    last = []
    for r, ids in enumerate(id_lists):
        inp[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        am[r, :len(ids)] = 1
        last.append(len(ids) - 1)
    inp, am = inp.to(device), am.to(device)
    with torch.no_grad():
        o = lm(input_ids=inp, attention_mask=am,
               output_hidden_states=capture is None, use_cache=False,
               return_dict=True)
        # Index the last real token BEFORE the float32 cast, so the cast touches
        # [n, hidden] rather than [n, L, hidden]. Elementwise either way, so the
        # result is identical; the intermediate is L times smaller, which is what
        # keeps many-option items (POP909 chords, hundreds of options) inside
        # VRAM instead of OOMing on a full-sequence float32 copy.
        hs = (o.hidden_states[-1] if capture is None
              else capture.pop("h"))
        idx = torch.arange(n, device=device)
        lb = torch.tensor(last, device=device)
        return head(hs[idx, lb].to(device).float()).tolist()


def score_prefix_cached(lm, head, id_lists, device, ctx, min_shared=8,
                        force=False, return_info=False, capture=None):
    """Prefix-cached path. Falls back to the naive path when not worth it.

    min_shared: only use the cache when the shared prefix is at least this long,
    otherwise the extra kernel launches cost more than the tokens they save.
    """
    P0 = common_prefix_len(id_lists)
    max_suf = max(len(x) - P0 for x in id_lists)
    P = min(P0, max(1, ctx - max_suf))

    n_opts = len(id_lists)
    # What the naive batched path would push through the trunk: the true token
    # count of every option prompt.
    naive_tokens = sum(len(x) for x in id_lists)
    # What the cached path actually pushes: the shared prefix once, then each
    # option's private suffix. This is a real saving whenever the prefix is
    # non-trivial, and it is the figure the speedup claim must rest on.
    cached_tokens = P + sum(len(x) - P for x in id_lists)

    if not force and (P < min_shared or n_opts < 2
                      or cached_tokens > 0.85 * naive_tokens):
        t0 = time.perf_counter()
        scores = score_naive(lm, head, id_lists, device, capture=capture)
        dt = time.perf_counter() - t0
        if return_info:
            return scores, {"path": "naive", "shared": P0, "P": P,
                            "cached_tokens": cached_tokens,
                            "naive_tokens": naive_tokens, "secs": dt}
        return scores

    pre = torch.tensor([id_lists[0][:P]], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        # The zero-suffix branch below reads the prefix's own final hidden state,
        # so it is read out of the hook slot (or hidden_states on the fallback
        # path) before the loop overwrites it. Fires when two options are
        # token-identical after ctx truncation.
        o = lm(input_ids=pre, attention_mask=torch.ones_like(pre),
               use_cache=True, output_hidden_states=capture is None,
               return_dict=True)
        pre_h = o.hidden_states[-1] if capture is None else capture.get("h")
    cache = o.past_key_values
    last_h = None
    scores = []
    try:
        for ids in id_lists:
            suf = ids[P:]
            if not suf:
                h = pre_h[0, -1]
            else:
                t = torch.tensor([suf], dtype=torch.long, device=device)
                S = t.shape[1]
                pos = torch.arange(P, P + S, device=device).unsqueeze(0)
                am = torch.ones(1, P + S, dtype=torch.long, device=device)
                with torch.no_grad():
                    o2 = lm(input_ids=t, attention_mask=am, position_ids=pos,
                            past_key_values=cache, use_cache=True,
                            output_hidden_states=capture is None,
                            return_dict=True)
                h = ((o2.hidden_states[-1] if capture is None
                      else capture.pop("h"))[0, -1])
                last_h = h
                # Negative = remove S tokens from the end (restores length P).
                # Positive is LEGACY absolute-length semantics and would
                # truncate the cache to S, corrupting every later option.
                cache.crop(-S)
            scores.append(head(h.float()).item())
    finally:
        del cache
    dt = time.perf_counter() - t0
    if return_info:
        return scores, {"path": "cached", "shared": P0, "P": P,
                        "cached_tokens": cached_tokens,
                        "naive_tokens": naive_tokens, "secs": dt,
                        "n_opts": n_opts}
    return scores
