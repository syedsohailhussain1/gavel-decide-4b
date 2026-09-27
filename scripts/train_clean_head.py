"""Train a JevBench-clean decision head.

The shipped head was trained on 689 pairs derived from the PUBLIC evaluation
items, so 213 of the 231 scored items were in its training set (92.2%). This
trains the same architecture on the 1,720 MNLI-derived pairs ONLY - zero
JevBench items - so every JevBench number it produces is out-of-sample.

Production recipe, recovered in FINDINGS_readout_layer.md:
    60 epochs, lr 1e-4, batch 256, width 512, seed 20260924
Evaluation is 5-fold, GROUPED by NLI item, because each item contributes two
options and an ungrouped split leaks the twin across the fold boundary.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TS = Path(r"D:\gavel\models\training_state")
OUT = Path(r"D:\gavel-jevbench-entry\results")
SEED, EPOCHS, LR, BS, WIDE = 20260924, 60, 1e-4, 256, 512


class Head(nn.Module):
    def __init__(self, h, w=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h, w), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w, w // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(w // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_nli():
    """Concatenate the cached NLI shards in their recorded order."""
    X, y, g = [], [], []
    for name in ("nli_hiddens_A", "nli_hiddens_B", "nli_hiddens_C",
                 "nli_hiddens_D", "nli_hiddens_E"):
        p = TS / f"{name}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        X.append(d["hiddens"].float())
        y.extend(int(t) for t in d["targets"])
        g.extend(str(o) for o in d["order"])
    return torch.cat(X, 0), torch.tensor(y, dtype=torch.float32), g


def train_one(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    m = Head(Xtr.shape[1], WIDE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    ds = TensorDataset(Xtr, ytr)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, generator=g)
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(EPOCHS):
        m.train()
        for xb, yb in dl:
            opt.zero_grad()
            lossf(m(xb), yb).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            acc = ((m(Xva) > 0).float() == yva).float().mean().item()
        if acc > best:
            best, best_ep = acc, ep
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
    return best, best_state, best_ep


def main():
    t0 = time.time()
    X, y, groups = load_nli()
    print(f"NLI rows {tuple(X.shape)}  positives {int(y.sum())}  "
          f"items {len(set(groups))}")
    assert X.shape[0] == len(y) == len(groups), "shard/pair misalignment"
    assert not any("intent" in g or "hard" in g or g.startswith("original")
                   for g in groups), "a JevBench item leaked into the NLI set"

    # grouped 5-fold on item id
    uniq = sorted(set(groups))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(uniq))
    fold_of = {uniq[i]: perm[j] % 5 for j, i in enumerate(range(len(uniq)))}
    fold = np.array([fold_of[g] for g in groups])

    print(f"\n5-fold grouped OOF  ({EPOCHS} epochs, lr {LR}, bs {BS}, "
          f"width {WIDE}, seed {SEED})")
    oof = torch.zeros(len(y))
    for k in range(5):
        tr, va = fold != k, fold == k
        acc, state, ep = train_one(X[tr], y[tr], X[va], y[va], SEED + k)
        m = Head(X.shape[1], WIDE)
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            oof[va] = m(X[va])
        print(f"  fold {k}: n_va={int(va.sum()):4d}  acc={acc:.4f}  "
              f"best_epoch={ep}")

    oof_acc = ((oof > 0).float() == y).float().mean().item()
    print(f"\nNLI out-of-fold accuracy: {oof_acc:.4f}  "
          f"({time.time()-t0:.0f}s)")

    # final head on all NLI
    full, state, ep = train_one(X, y, X[:512], y[:512], SEED)
    m = Head(X.shape[1], WIDE)
    m.load_state_dict(state)
    n = sum(p.numel() for p in m.parameters())
    print(f"final head: {n:,} params, best epoch {ep}")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"head": m.state_dict(), "hidden": X.shape[1], "wide": WIDE,
                "temperature": 1.0, "meta_cal": None,
                "trained_on": "MNLI-derived pairs only (1720 rows, 860 items)",
                "jevbench_items_in_training": 0,
                "recipe": {"epochs": EPOCHS, "lr": LR, "batch": BS,
                           "width": WIDE, "seed": SEED},
                "nli_oof_accuracy": oof_acc}, OUT / "clean_head_nli_only.pt")
    print(f"wrote {OUT / 'clean_head_nli_only.pt'}")

    (OUT / "clean_head_oof.json").write_text(json.dumps({
        "nli_oof_accuracy": oof_acc,
        "nli_rows": int(X.shape[0]),
        "nli_items": len(set(groups)),
        "positives": int(y.sum()),
        "params": n,
        "recipe": {"epochs": EPOCHS, "lr": LR, "batch": BS, "width": WIDE,
                   "seed": SEED},
        "jevbench_items_in_training": 0,
        "contamination": "none by construction",
    }, indent=2), encoding="utf-8")
    print("wrote clean_head_oof.json")


if __name__ == "__main__":
    main()
