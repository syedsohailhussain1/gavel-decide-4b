"""Meta-calibration for Gavel-Decide (serving-time transform only).

Sets top-label confidence to P(correct) from an L2-logistic correctness head
over decision signals, then rescales the remaining mass proportionally.
Argmax and accuracy are unchanged by construction; only reported confidence
moves.

Vendored into this submission so `gavel_adapter.py` has no external module
dependency. Upstream this function lives in `m2.py` next to the fitting code;
only the serving-time transform is needed to reproduce our published numbers.

probs: {label: prob}, already temperature-scaled.
meta:  artifact dict with feats / mu / sd / w / b.
       Features: c (top prob), margin (top - second), ent (entropy in nats),
                 kopts (log of the number of options).
Unknown features raise rather than degrade silently.
"""
import math

SUPPORTED = ("c", "margin", "ent", "kopts")


def meta_rescale(probs, meta):
    ps = sorted(probs.values(), reverse=True)
    top = ps[0]
    second = ps[1] if len(ps) > 1 else 0.0
    ent = -sum(v * math.log(max(v, 1e-15)) for v in probs.values())
    table = {"c": top, "margin": top - second, "ent": ent,
             "kopts": math.log(max(len(probs), 2))}
    xs = []
    for f in meta["feats"]:
        if f not in table:
            raise ValueError(f"meta feature not servable: {f}")
        xs.append(table[f])
    mu, sd, w, b = meta["mu"], meta["sd"], meta["w"], meta["b"]
    z = sum(wi * (x - m) / s for wi, x, m, s in zip(w, xs, mu, sd)) + b
    p_top = 1.0 / (1.0 + math.exp(-z))
    labels = list(probs.keys())
    top_lab = max(labels, key=lambda k: probs[k])
    rest = 1.0 - p_top
    rest_old = 1.0 - probs[top_lab]
    out = {}
    for lab in labels:
        if lab == top_lab:
            out[lab] = p_top
        else:
            out[lab] = probs[lab] / rest_old * rest if rest_old > 1e-12 else 0.0
    return out
