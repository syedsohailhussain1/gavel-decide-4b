"""Measure the ACTUAL gavel_snake.py --brain paths, not a stand-in.

My earlier spectrum benchmark used raw transformers on a 64-token input and
reported 667 decisions/s. That is not any of the demo's real decision paths, so
the number was not the one that matters. `--brain` has four options and two of
them are not neural networks at all:

  server    Gavel /ask over keep-alive HTTP (logistic)
  onnx      local tiny transformer, batch-1 ONNX on CPU   (the default)
  immortal  provably-safe Hamiltonian search               (algorithmic)
  router    BFS hunt + flood survival                     (algorithmic)

The algorithmic ones are the interesting case: they return confidence 1.0 by
construction because the answer is proved, not fitted. That is a categorically
different claim from "a small model is quick", and it deserves a real number.
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, r"D:\gavel\models\training_state")

try:
    import torch
    torch.set_num_threads(4)
except Exception:
    torch = None

results = {}


def timeit(fn, warm, iters):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return statistics.median(ts), 1.0 / statistics.median(ts), ts[0]


# ---------- algorithmic brains: the real classes from gavel_snake ----------
try:
    from tetris_game import *  # noqa: F401,F403  (not required, kept out)
except Exception:
    pass

try:
    from gavel_snake import ImmortalBrain, RouterBrain
    HAVE_SNAKE = True
except Exception as e:
    print(f"gavel_snake import failed: {type(e).__name__}: {e}")
    HAVE_SNAKE = False

if HAVE_SNAKE:
    class FakeSnake:
        """Minimal stand-in with the attributes the policies read."""
        def __init__(self, n=20):
            self.n = n
            self.w = self.h = n
            self.snake = [(n - 1, 9), (n - 1, 10), (n - 1, 11), (n - 1, 12)]
            self.head = (n - 1, 12)
            self.food = (5, 5)
            self.dir = (0, 1)
            self.alive = True
            self.score = 0
            self.obstacles = None
            self.steps = 0
            self.done = False
            self.deaths = 0
            self.cause = None

    for pol in ("jumps-bfs", "jumps-manhattan", "lookahead", "cycle"):
        try:
            br = ImmortalBrain(pol)
            g = FakeSnake()
            for _ in range(50):
                br.decide_game(g)
            ms, dps, bst = timeit(lambda: br.decide_game(g), 200, 3000)
            results[f"immortal:{pol}"] = {"ms": ms, "dps": dps, "best_ms": bst,
                                          "kind": "algorithmic"}
            print(f"  immortal/{pol:18s} {ms*1000:8.1f} us  {dps:9.0f} decisions/s")
        except Exception as e:
            print(f"  immortal/{pol:18s} unavailable: {type(e).__name__}: {e}")

    try:
        g = FakeSnake()
        g.obstacles = {(7, 7), (8, 7), (9, 7), (10, 7), (11, 7)}
        rb = RouterBrain()
        for _ in range(50):
            rb.decide_game(g)
        ms, dps, bst = timeit(lambda: rb.decide_game(g), 200, 3000)
        results["router"] = {"ms": ms, "dps": dps, "best_ms": bst,
                             "kind": "algorithmic"}
        print(f"  router{'':26s} {ms*1000:8.1f} us  {dps:9.0f} decisions/s")
    except Exception as e:
        print(f"  router unavailable: {type(e).__name__}: {e}")

# ---------- the real OnnxBrain ----------
try:
    from gavel_snake import OnnxBrain
    import glob
    cand = sorted(glob.glob(r"D:\gavel\models\training_state\tiny_snake_gtiny*\**\*.onnx",
                            recursive=True))
    if not cand:
        cand = sorted(glob.glob(r"D:\gavel\models\training_state\tiny_snake_gtiny*\*.onnx"))
    if cand:
        ob = OnnxBrain(cand[0])
        txt = "user is moving the piece down"
        for _ in range(50):
            ob.decide(txt)
        ms, dps, bst = timeit(lambda: ob.decide(txt), 100, 1500)
        results["onnx"] = {"ms": ms, "dps": dps, "best_ms": bst,
                           "kind": "learned", "file": os.path.basename(cand[0])}
        print(f"  onnx{'':28s} {ms*1000:8.1f} us  {dps:9.0f} decisions/s")
    else:
        print("  onnx: no exported .onnx found")
except Exception as e:
    print(f"  onnx unavailable: {type(e).__name__}: {e}")

# ---------- server brain: HTTP round trip, reported separately ----------
print("\n=== summary: real --brain paths ===")
print(f"{'brain':30} {'kind':12} {'us':>10} {'decisions/s':>13}")
for k, v in sorted(results.items(), key=lambda kv: -kv[1]["dps"]):
    print(f"{k:30} {v['kind']:12} {v['ms']*1000:10.1f} {v['dps']:13.0f}")
alg = [v["dps"] for v in results.values() if v["kind"] == "algorithmic"]
lrn = [v["dps"] for v in results.values() if v["kind"] == "learned"]
if alg:
    print(f"\n  algorithmic best : {max(alg):.0f} decisions/s")
if lrn:
    print(f"  learned best     : {max(lrn):.0f} decisions/s")
if alg and lrn:
    print(f"  ratio            : {max(alg)/max(lrn):.0f}x")
json.dump(results, open(r"D:\gavel-jevbench-entry\results\brain_spectrum.json", "w"),
          indent=2)
print("\nwrote results/brain_spectrum.json")
