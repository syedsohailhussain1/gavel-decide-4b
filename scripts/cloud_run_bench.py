"""Run the full 231-item public JevBench on rented hardware, in bf16.

Self-contained: does not import the upstream dev repo's hardcoded Windows
paths. Builds the adapter with explicit paths, drives the official
`jevbench.runner.Runner` unmodified, then derives the axes with the official
`composite_v13` / `metrics` code.

    python scripts/cloud_run_bench.py [--dtype bf16|nf4] [--no-prefix-cache]
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--dtype", default="bf16", choices=["bf16", "nf4"])
ap.add_argument("--trunk", default=os.environ.get("TRUNK", "Qwen/Qwen3-4B-Base"))
ap.add_argument("--head", default=os.environ.get("HEAD", "combined_head.pt"))
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--out", default="results/cloud_231_results.jsonl")
ap.add_argument("--no-prefix-cache", action="store_true")
ap.add_argument("--limit", type=int, default=0)
A = ap.parse_args()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if A.no_prefix_cache:
    os.environ["GAVEL_PREFIX_CACHE"] = "0"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.environ.get("JEVBENCH_PATH", "/workspace/jevbench"))
sys.path.insert(0, os.path.abspath("."))

from jevbench.runner import Runner            # noqa: E402
from jevbench.tasks import load_jsonl         # noqa: E402


class Ledger:
    """Single-process ledger; upstream's needs Unix fcntl, harmless to inline."""

    def __init__(self, path, cap_usd=25):
        self.path, self.cap, self.spent, self.n = path, cap_usd, 0.0, 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        open(path, "a").close()

    def _w(self, o):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(o) + "\n")

    def reserve(self, amount, info):
        self.n += 1
        rid = f"r{self.n}"
        self._w({"op": "reserve", "rid": rid, "amount": amount, "info": info})
        return rid

    def settle(self, rid, cost, info):
        self.spent += cost or 0.0
        self._w({"op": "settle", "rid": rid, "cost": cost, "spent": self.spent,
                 "info": info})


def main():
    import gavel_adapter as GA

    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[dtype] {A.dtype}   prefix_cache="
          f"{os.environ.get('GAVEL_PREFIX_CACHE', '1')}", flush=True)

    ad = GA.GavelLocalAdapter(endpoint=A.trunk, head=A.head, ctx=A.ctx,
                              dtype=A.dtype)
    t0 = time.perf_counter()
    ad.load()
    print(f"[load] {time.perf_counter()-t0:.0f}s  dev={ad.dev} "
          f"T={ad.temp:.4f}  meta={ad._meta_fn is not None}", flush=True)

    tasks = []
    for p in ("easy", "original", "hard"):
        tasks += load_jsonl(f"{os.environ.get('JEVBENCH_PATH', '/workspace/jevbench')}"
                            f"/datasets/public/{p}.jsonl")
    if A.limit:
        tasks = tasks[:A.limit]
    print(f"[tasks] {len(tasks)}", flush=True)

    os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
    led = Ledger(A.out.replace(".jsonl", "_ledger.json"))
    runner = Runner(ad, led, raw_dir=A.out.replace(".jsonl", "_raw"),
                    default_reserve_usd=0.0)
    t0 = time.perf_counter()
    recs = runner.run_all(tasks, results_path=A.out, progress_every=20)
    el = time.perf_counter() - t0

    ok = sum(1 for r in recs if r.get("correct"))
    tot = sum(1 for r in recs if r.get("status") == "ok")
    s = getattr(ad, "_pc_stats", {})
    print(f"\n[done] {ok}/{tot} correct in {el/60:.1f} min "
          f"({el/max(tot,1):.2f}s/decision)", flush=True)
    if s:
        print(f"[prefix-cache] cached={s.get('cached')} naive={s.get('naive')} "
              f"skipped={100*(1-s.get('cached_tokens',0)/max(s.get('naive_tokens',1),1)):.1f}%"
              f" of token-forwards", flush=True)

    # ---- axes, via the official code ----
    from jevbench import composite_v13 as v13
    from jevbench.metrics import ece_top_label

    def tier(t):
        x = t.split("-", 1)[0]
        return {"original": "standard"}.get(x, x)

    def q(v, p):
        v = sorted(v)
        return v[max(0, min(len(v) - 1, int(round((len(v) - 1) * p))))]

    acc, lat = {}, defaultdict(list)
    for r in recs:
        t = tier(r["task_id"])
        acc.setdefault(t, []).append(1.0 if r.get("correct") else 0.0)
        if r.get("latency_s"):
            lat[t].append(r["latency_s"])
    acc = {k: sum(v) / len(v) for k, v in acc.items()}
    hard = [r for r in recs if tier(r["task_id"]) == "hard" and r.get("probs")]
    ece = ece_top_label([(max(r["probs"].values()),
                          1.0 if r.get("correct") else 0.0) for r in hard])["ece"]

    axes = {
        "dtype": A.dtype,
        "gpu": torch.cuda.get_device_name(0),
        "n_items": len(recs), "n_correct": ok,
        "accuracy": ok / max(tot, 1),
        "tier_accuracy": acc,
        "tier_chance_corrected": {t: v13.chance_corrected_accuracy(
            a, v13.TIER_CHANCES[t]) for t, a in acc.items()},
        "calibration_axis": v13.calibration(ece),
        "hard_binned_ece": ece,
        "latency_p50_s": {t: q(v, .5) for t, v in lat.items()},
        "latency_p95_s": {t: q(v, .95) for t, v in lat.items()},
        "speed_axis_standard_proxy": v13.speed(
            q(lat["standard"], .5), q(lat["standard"], .95), "gpu"),
        "wall_clock_s": el,
        "s_per_decision": el / max(tot, 1),
        "mean_input_tokens": statistics.mean(
            [r["usage"]["input_tokens"] for r in recs if r.get("usage")]),
        "path_mix": dict(getattr(ad, "_pc_stats", {})),
    }
    print("\n=== axes ===", flush=True)
    print(json.dumps(axes, indent=2), flush=True)
    out = A.out.replace(".jsonl", "_axes.json")
    json.dump(axes, open(out, "w"), indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
