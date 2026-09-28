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


def score_naive(lm, head, id_lists, device, pad_id=0):
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
               output_hidden_states=True, use_cache=False, return_dict=True)
        hs = o.hidden_states[-1].float()
        idx = torch.arange(n, device=device)
        lb = torch.tensor(last, device=device)
        return head(hs[idx, lb]).tolist()


def score_prefix_cached(lm, head, id_lists, device, ctx, min_shared=8,
                        force=False, return_info=False):
    """Prefix-cached path. Falls back to the naive path when not worth it.

    min_shared: only use the cache when the shared prefix is at least this long,
    otherwise the extra kernel launches cost more than the tokens they save.
    """
    P0 = common_prefix_len(id_lists)
    max_suf = max(len(x) - P0 for x in id_lists)
    P = min(P0, max(1, ctx - max_suf))

    n_opts = len(id_lists)
    naive_tokens = sum(min(len(x), ctx) for x in id_lists)
    cached_tokens = P + sum(len(x) - P for x in id_lists)

    if not force and (P < min_shared or n_opts < 2
                      or cached_tokens > 0.85 * naive_tokens):
        t0 = time.perf_counter()
        scores = score_naive(lm, head, id_lists, device)
        dt = time.perf_counter() - t0
        if return_info:
            return scores, {"path": "naive", "shared": P0, "P": P,
                            "cached_tokens": cached_tokens,
                            "naive_tokens": naive_tokens, "secs": dt}
        return scores

    pre = torch.tensor([id_lists[0][:P]], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        o = lm(input_ids=pre, attention_mask=torch.ones_like(pre),
               use_cache=True, return_dict=True)
    cache = o.past_key_values
    last_h = None
    scores = []
    try:
        for ids in id_lists:
            suf = ids[P:]
            if not suf:
                h = o.hidden_states[-1][0, -1]
            else:
                t = torch.tensor([suf], dtype=torch.long, device=device)
                S = t.shape[1]
                pos = torch.arange(P, P + S, device=device).unsqueeze(0)
                am = torch.ones(1, P + S, dtype=torch.long, device=device)
                with torch.no_grad():
                    o2 = lm(input_ids=t, attention_mask=am, position_ids=pos,
                            past_key_values=cache, use_cache=True,
                            output_hidden_states=True, return_dict=True)
                h = o2.hidden_states[-1][0, -1]
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
