"""Gavel-Decide 4B - interactive typed-decision demo.

State the evidence, give a typed question, get a calibrated distribution over
the options in a single forward pass per option. Nothing is generated: the
system reads one hidden vector per option and returns probabilities, so the
distribution is the product and you can threshold on confidence instead of
parsing prose.

Runs on a free CPU Space. That is not a trick - the architecture never decodes,
so there is no autoregressive loop to be slow. See README.md in this folder.
"""
from __future__ import annotations

import os
import time

import gradio as gr

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

ADAPTER = None
LOAD_ERR = None


def get_adapter():
    """Load once, lazily, and never block the UI thread on repeat calls."""
    global ADAPTER, LOAD_ERR
    if ADAPTER is not None or LOAD_ERR is not None:
        return ADAPTER
    try:
        from gavel_adapter import GavelLocalAdapter
        ADAPTER = GavelLocalAdapter(
            trunk=os.environ.get("GAVEL_TRUNK", "Qwen/Qwen3-4B-Base"),
            model="gavel-decide-4b",
            head=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "combined_head_bf16.pt"),
            dtype=os.environ.get("GAVEL_DTYPE", "bf16"),
            ctx=512,
        )
        ADAPTER.load()
    except Exception as e:  # noqa: BLE001
        LOAD_ERR = f"{type(e).__name__}: {e}"
        ADAPTER = None
    return ADAPTER


class Task:
    """Duck-type of the JevBench task the adapter consumes."""

    def __init__(self, tid, state, labels, qtype, instructions, criteria):
        self.id = tid
        self.state = state
        self.labels = labels
        self.question = {"type": qtype, "instructions": instructions,
                         "criteria": criteria or {}}


def parse_options(raw):
    """`label` or `label | criteria`, one per line. Blank lines ignored."""
    labels, criteria = [], {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            lab, crit = line.split("|", 1)
            lab, crit = lab.strip(), crit.strip()
        else:
            lab, crit = line, ""
        if not lab:
            continue
        labels.append(lab)
        if crit:
            criteria[lab] = crit
    return labels, criteria


EXAMPLES = [
    ("Where is my package? I ordered it last week and it still hasn't arrived.",
     "Which intent does the user's message express?",
     "track_order | Wants to know where an order is or when it arrives\n"
     "cancel_order | Wants to cancel an order\n"
     "change_address | Wants to change the delivery address\n"
     "report_damage | Reports a problem with an item\n"
     "billing_question | Asks about a charge, invoice or payment"),
    ("Order #88 shipped yesterday. The customer asks where it is.",
     "What is the true status of order #88?",
     "it is in transit\nit was returned to sender\nit was cancelled\n"
     "it was lost in the carrier network"),
    ("A package marked fragile arrives with a cracked corner and the item inside is chipped.",
     "What is the correct next action?",
     "photograph the damage and open a claim\nignore it and close the ticket\n"
     "ask the customer to pay for shipping\nwait and see if it gets worse"),
]


def decide(state, instructions, qtype, options, want_latency):
    ad = get_adapter()
    if ad is None:
        return ({"status": "model failed to load", "error": LOAD_ERR},
                "", "", "")
    labels, criteria = parse_options(options)
    if len(labels) < 2:
        return ({"status": "need at least 2 options", "n_options": len(labels)},
                "", "", "")
    if qtype == "score" and not criteria:
        criteria = {lab: f"level {i + 1} of {len(labels)}" for i, lab in enumerate(labels)}

    task = Task("demo-001", state, labels, qtype,
                instructions or "choose the best option", criteria)
    t0 = time.perf_counter()
    r = ad.run(task)
    wall = time.perf_counter() - t0

    if getattr(r, "error", None):
        return ({"status": "decision failed", "error": r.error}, "", "", "")

    probs = r.probs or {}
    if not probs:
        return ({"status": "no distribution returned"}, "", "", "")

    pred = max(probs.items(), key=lambda kv: kv[1])[0]
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    top = ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    ent = -sum(p * __import__("math").log(p + 1e-12) for p in probs.values())

    meta = {
        "status": "ok",
        "n_options": len(labels),
        "predicted": pred,
        "confidence": round(top, 4),
        "margin": round(top - second, 4),
        "entropy": round(ent, 4),
        "output_tokens": (r.usage or {}).get("output_tokens", 0),
        "path": ((r.raw or {}).get("runtime") or {}).get("path", "?"),
        "shared_prefix": ((r.raw or {}).get("runtime") or {}).get("shared_prefix", "?"),
        "trunk": str(getattr(ad, "trunk", "")).split("/")[-1] or "Qwen3-4B-Base",
        "dtype": getattr(ad, "dtype", "?"),
    }
    if want_latency:
        meta["latency_s"] = round(r.latency_s or 0.0, 4)
        meta["wall_s"] = round(wall, 3)

    dist = [[lab, round(p, 5)] for lab, p in ranked]
    conf = (f"**{pred}**  ·  p={top:.3f}  ·  margin={top - second:.3f}\n\n"
            + ("Confident enough to act on.\n\n" if top >= 0.60
               else "**Low confidence — consider abstaining or escalating.**\n\n"))
    return meta, dist, conf, pred


def warm():
    ad = get_adapter()
    if ad is None:
        return f"load failed: {LOAD_ERR}"
    return ("ready — frozen `Qwen3-4B-Base` trunk + 1,442,817-parameter head. "
            "Output tokens are always 0: the system never generates text.")


with gr.Blocks(title="Gavel-Decide 4B — typed decisions", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Gavel-Decide 4B — single-pass typed decisions\n"
        "Give it the **state** (the evidence) and a typed question. It returns a "
        "**calibrated probability distribution over the options** in one forward "
        "pass per option — no text is generated, so `output_tokens` is always 0 "
        "and the distribution itself is the product.\n\n"
        "Frozen `Qwen3-4B-Base` trunk (never updated) + a **1,442,817-parameter** "
        "head trained on CPU in 3.5 s. Two-stage argmax-preserving calibration, so "
        "the rescale changes confidence and never the decision."
    )

    with gr.Row():
        with gr.Column(scale=3):
            state = gr.Textbox(label="State (the evidence)", lines=4,
                               placeholder="Where is my package? I ordered it last week...")
            instructions = gr.Textbox(label="Question / instructions", lines=2,
                                     value="Which intent does the user's message express?")
            qtype = gr.Radio(["choice", "noul", "score"], value="choice",
                             label="Question type")
            options = gr.Textbox(
                label="Options — one per line",
                lines=6,
                value=EXAMPLES[0][2],
                placeholder="label | optional criteria text")
            run = gr.Button("Decide", variant="primary")
        with gr.Column(scale=2):
            gr.Markdown("### Try an example")
            for i, (s, q, o) in enumerate(EXAMPLES):
                gr.Button(f"Example {i + 1}").click(
                    lambda s=s, q=q, o=o: (s, q, o),
                    inputs=None,
                    outputs=[state, instructions, options])

    out_meta = gr.JSON(label="Decision detail")
    out_dist = gr.Dataframe(headers=["option", "probability"], label="Distribution",
                            interactive=False, wrap=True)
    out_conf = gr.Markdown()
    out_pred = gr.Textbox(label="Answer (argmax)", interactive=False)

    status = gr.Markdown()
    run.click(decide, [state, instructions, qtype, options, gr.State(True)],
              [out_meta, out_dist, out_conf, out_pred])
    demo.load(warm, outputs=status)

    gr.Markdown(
        "---\n"
        "### What this measures, honestly\n"
        "The head was trained on 689 pairs derived from **213 of the 231 public "
        "JevBench items (92.2% overlap)**, so published *public* accuracy for this "
        "architecture is an upper bound rather than a generalisation estimate. The "
        "held-out tier was never used for training, tuning or calibration. Retraining "
        "on non-overlapping data does not rescue it (an NLI-only head scores 0.1667, "
        "below the 0.20 chance rate) — purpose-built supervision is the missing piece, "
        "and it does not exist yet.\n\n"
        "Play with your own inputs: the interesting question is how a frozen 4B "
        "read-out behaves on text it was never fitted to.\n\n"
        "Submission: [fstandhartinger/jevbench#118](https://github.com/fstandhartinger/jevbench/issues/118) · "
        "code: [syedsohailhussain1/gavel-decide-4b](https://github.com/syedsohailhussain1/gavel-decide-4b)"
    )

if __name__ == "__main__":
    demo.queue().launch()
