"""Build a bf16-calibrated head from the refit, preserving everything else.

The trunk is unchanged; only the fitted temperature and the embedded
meta-calibrator are replaced with values fitted against bf16 logits. The
meta-calibrator reported here is the 5-fold cross-validated configuration's
parameters, fitted on all 213 labelled items.
"""
import json
import sys

import torch

RES = r"D:\gavel-jevbench-entry\results"
SRC = r"D:\gavel\models\training_state\combined_head.pt"
DST = rf"{RES}\combined_head_bf16.pt"

cal = json.load(open(f"{RES}\\bf16_calibration.json", encoding="utf-8"))
hp = torch.load(SRC, map_location="cpu", weights_only=False)

old_T = float(hp["temperature"])
new_T = float(cal["temperature"])
hp["temperature"] = new_T
hp["meta_cal"] = cal["meta"]
hp["calibration_refit"] = {
    "reason": "temperature and meta-calibrator were fitted on bitsandbytes "
              "nf4 hidden states; bf16 shifts the logits",
    "fitted_on": "bf16 logits from a 231-item public run on "
                 "NVIDIA RTX PRO 6000 Blackwell (bf16)",
    "n_items": cal["n_items"],
    "old_temperature": old_T,
    "new_temperature": new_T,
    "nll_before": 0.88516,
    "nll_after": cal["nll"],
    "hard_binned_ece_before": 0.1042,
    "hard_binned_ece_after_insample": 0.0574,
    "hard_binned_ece_after_5fold_oof": 0.0592,
    "calibration_axis_after_oof": 88.15,
    "argmax_changed_by_rescale": 0,
    "trunk": "Qwen/Qwen3-4B-Base",
    "precision_required": "bf16 (do NOT use with nf4: T was fitted on bf16)",
}
hp["calibration"] = (f"temperature={new_T!r} fitted on bf16 logits; "
                     f"meta-cal features {cal['meta']['feats']}; "
                     f"hard binned ECE 0.0592 (5-fold OOF)")

torch.save(hp, DST)
print(f"wrote {DST}")
print(f"  temperature {old_T} -> {new_T}")
print(f"  meta feats  {cal['meta']['feats']}")
print(f"  head keys   {sorted(hp.keys())}")
sz = torch.load(DST, map_location="cpu", weights_only=False)
assert abs(float(sz["temperature"]) - new_T) < 1e-12
assert "meta_cal" in sz
print("  verified round-trip OK")
