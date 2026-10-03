#!/usr/bin/env python3
"""What a prompt edit costs at MiniMaxH3ReferenceConditioning, and how much of
it is the reference VAE encode.

    <comfy venv python> bench/measure_prompt_edit_conditioning.py OUT.json [--arms vae,no_vae] [--host 127.0.0.1:8188]

Submits the conditioning side of the shipped ref2va finish graph alone (the
encoder and VAE loaders, the still and its append node, the resolution node,
the conditioning node) with a text preview as the only output, so no DiT loads
and no sampler runs. Each submission changes the prompt, which is the one
thing that makes the conditioning node execute again. The arms alternate after
one cold run:

  vae      the graph as shipped: text encode plus the reference VAE encode
  no_vae   `vae` and `audio_vae` unwired: the same Qwen presentation, no encode
  q<N>     (`--arms vae,q1024,q512`) the encoder's view of the still at an N
           short edge through the append node's `qwen_view`, VAE copy unchanged:
           how the node's time moves with the number of vision tokens

Per run it records the conditioning node's seconds from the websocket feed.
`vae` minus `no_vae` is what a prompt edit pays for an encode whose result
cannot differ. Needs a running ComfyUI. Not what a render pays: the DiT is not
resident here, so core's memory management has an easier job than after a
render.
"""
import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
from bench_e2e_h3 import run_once  # noqa: E402

GRAPH = "workflows/distill_experiments/h3_probe_r2v_step_switch_pdd8_flashgen_h080_savelat_api.json"
KEEP = ("2", "3", "4", "15", "50", "27", "5")
COND = "5"


def graph(arm, n):
    full = json.loads((REPO / GRAPH).read_text())
    g = {k: json.loads(json.dumps(full[k])) for k in KEEP}
    assert g[COND]["class_type"] == "MiniMaxH3ReferenceConditioning"
    if arm == "no_vae":
        del g[COND]["inputs"]["vae"], g[COND]["inputs"]["audio_vae"]
        del g["3"], g["4"]
    elif arm.startswith("q"):
        g["50"]["inputs"]["qwen_view"] = "separate"
        g["50"]["inputs"]["qwen_view.qwen_short_edge"] = int(arm[1:])
    # A prompt edit: one word appended, different every run.
    g[COND]["inputs"]["prompt"] = g[COND]["inputs"]["prompt"].rstrip() + f" Take {n}."
    g["900"] = {"class_type": "PreviewAny", "inputs": {"source": [COND, 0]}}
    return g


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--arms", default="vae,no_vae")
    args = ap.parse_args()
    client = uuid.uuid4().hex
    arms = args.arms.split(",")
    order = ["vae"] + [a for _ in range(args.pairs) for a in arms]
    rows = []
    for n, arm in enumerate(order):
        total, per_node, err = await run_once(args.host, graph(arm, n), client, 900.0)
        row = {"n": n, "arm": arm, "cold": n == 0, "total_s": total, "error": err,
               "conditioning_node_s": per_node.get(COND),
               "per_node_s": {k: round(v, 3) for k, v in per_node.items()}}
        rows.append(row)
        print(json.dumps(row), flush=True)
    warm = {a: [r["conditioning_node_s"] for r in rows if r["arm"] == a and not r["cold"]]
            for a in arms}
    record = {
        "what": "seconds in MiniMaxH3ReferenceConditioning on a prompt edit, per arm",
        "arms": arms,
        "graph": GRAPH, "nodes_kept": list(KEEP), "state": "no DiT resident; encoder and VAE loaded by the cold run",
        "rows": rows, "warm_conditioning_node_s": warm,
    }
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
