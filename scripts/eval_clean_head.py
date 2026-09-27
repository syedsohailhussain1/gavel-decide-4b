"""Evaluate the CLEAN head (NLI-only, zero JevBench items) on all 231 public
items, through the production adapter path. This is the honest number.

Reads the same datasets the official runner reads, builds the same task shape
the adapter expects, and scores with combined_head_nli_only.pt.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"D:\gavel-jevbench-entry\src")

DATA = Path(r"D:\jevbench\datasets\public")
ENTRY = Path(r"D:\gavel-jevbench-entry\results")

ap = argparse.ArgumentParser()
ap.add_argument("--head", default=str(ENTRY / "clean_head_nli_only.pt"))
ap.add_argument("--dtype", default="nf4")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=str(ENTRY / "clean_run.jsonl"))
A = ap.parse_args()


class Task:
    """Minimal duck-type of the JevBench task the adapter consumes."""

    def __init__(self, d):
        self.id = d.get("task_id") or d.get("id")
        self.state = d.get("state")
        self.labels = list(d.get("labels") or [])
        q = d.get("question") or {}
        self.question = {
            "type": q.get("type", "choice"),
            "instructions": q.get("instructions", ""),
            "criteria": q.get("criteria") or {},
        }
        # v1.4.2 items carry the answer in `expected`; older shapes used
        # gold/answer/label. Accept all so the script is not silently wrong.
        self.gold = (d.get("expected") or d.get("gold")
                     or d.get("answer") or d.get("label"))
        if isinstance(self.gold, int) and self.labels:
            self.gold_label = self.labels[self.gold]
        else:
            self.gold_label = self.gold


def load_all():
    out = []
    for tier in ("easy", "hard", "original"):
        p = DATA / f"{tier}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                d["_tier"] = tier
                out.append(d)
    return out


def main():
    from gavel_adapter import GavelLocalAdapter

    tasks = load_all()
    if A.limit:
        tasks = tasks[:A.limit]
    print(f"{len(tasks)} public items; head={Path(A.head).name} dtype={A.dtype}")

    ad = GavelLocalAdapter(model="clean-nli-only", dtype=A.dtype, head=A.head)
    t0 = time.time()
    ad.load()
    print(f"loaded in {time.time()-t0:.1f}s\n")

    out_f = open(A.out, "w", encoding="utf-8")
    tiers = defaultdict(lambda: [0, 0])
    n_ok = 0
    lat = []
    for i, d in enumerate(tasks):
        t = Task(d)
        if t.gold_label is None:
            raise SystemExit(f"{t.id}: could not locate the gold field; "
                             f"item keys = {list(d.keys())}")
        r = ad.run(t)
        probs = r.probs or {}
        pred = max(probs.items(), key=lambda kv: kv[1])[0] if probs else None
        good = (pred == t.gold_label)
        tiers[d["_tier"]][1] += 1
        tiers[d["_tier"]][0] += good
        n_ok += good
        lat.append(r.latency_s or 0)
        out_f.write(json.dumps({
            "task_id": t.id, "tier": d["_tier"], "correct": bool(good),
            "predicted": pred, "gold": t.gold_label, "probs": probs,
            "latency_s": r.latency_s, "strict_valid": bool(probs),
            "model": "gavel-decide-4b-clean-nli",
            "usage": r.usage, "error": r.error,
        }) + "\n")
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(tasks)}  acc so far {n_ok/(i+1):.4f}")
    out_f.close()

    print(f"\n{'='*58}\nCLEAN HEAD on 231 public items")
    print(f"{'='*58}")
    print(f"overall  {n_ok}/{len(tasks)} = {n_ok/len(tasks):.4f}")
    for k in sorted(tiers):
        a, b = tiers[k]
        print(f"  {k:9} {a:3}/{b:3} = {a/b:.4f}")
    lat.sort()
    print(f"\nlatency p50 {lat[len(lat)//2]:.4f}s  "
          f"p95 {lat[int(len(lat)*0.95)]:.4f}s  "
          f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
