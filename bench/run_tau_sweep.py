#!/usr/bin/env python3
"""Queue the scenes of a tau sweep on a server armed with `H3_SOL_SWEEP`.

    bench/run_tau_sweep.py --base workflows/distill_experiments/<graph>_api.json \\
        --scene market \\
        --scene dialogue=workflows/h3_image_ref_plus_text_to_video_dialogue_api.json \\
        --out data/sparse/captures/<sweep dir>/scenes.json

The sweep (`sol_tau_sweep.py`) measures whatever the armed server renders, so
a calibration over several scenes is several renders of ONE graph family with
the scene changed. A scene named alone is the base graph as shipped. A scene
with `=donor` is the base graph with the donor's prompt and its reference
chain (`LoadImage` and `MiniMaxH3AppendRefImage` nodes) in place of its own:
samplers, sigmas, LoRAs and the sparse nodes stay the base's. A scene with
`=image:FILE` is the base graph with its one `LoadImage` pointed at another
still and nothing else changed, prompt included: the test of whether a
calibration carries to a different reference. Built in memory and queued; no
file under `workflows/` is written.

Renders one at a time, same seed (the base graph's, or `--seed`). The server
must already be armed; this tool cannot see its environment, so read the
sweep's ARMED line in the server log before trusting a record. `--out` keeps
what was queued (graphs included, so keep it out of git) and each prompt id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

REF_CLASSES = ("LoadImage", "MiniMaxH3AppendRefImage")
CONDITIONING = "MiniMaxH3ReferenceConditioning"
#: Output nodes whose prefix is renamed so a sweep render is not taken for a
#: render of the base graph: class -> input.
PREFIX_INPUTS = {"VHS_VideoCombine": "filename_prefix", "SaveLatent": "filename_prefix"}


def _one(graph: dict, class_type: str) -> tuple[str, dict]:
    hits = [(nid, n) for nid, n in graph.items() if n.get("class_type") == class_type]
    if len(hits) != 1:
        raise SystemExit(f"expected one {class_type} node, found {len(hits)}")
    return hits[0]


def scene_graph(base: dict, donor: dict | None, name: str, seed: int | None, image: str | None = None) -> dict:
    """The base graph with the donor's prompt and reference chain, or with
    its one `LoadImage` pointed at `image`."""
    g = json.loads(json.dumps(base))
    if image is not None:
        _one(g, "LoadImage")[1]["inputs"]["image"] = image
    if donor is not None:
        cid, cond = _one(g, CONDITIONING)
        _did, dcond = _one(donor, CONDITIONING)
        for nid in [nid for nid, n in g.items() if n.get("class_type") in REF_CLASSES]:
            del g[nid]
        rename = {nid: f"scene_{nid}" for nid, n in donor.items() if n.get("class_type") in REF_CLASSES}
        for nid, new in rename.items():
            node = json.loads(json.dumps(donor[nid]))
            for key, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in rename:
                    node["inputs"][key] = [rename[str(value[0])], value[1]]
            g[new] = node
        link = dcond["inputs"]["references"]
        cond["inputs"]["references"] = [rename[str(link[0])], link[1]]
        cond["inputs"]["prompt"] = dcond["inputs"]["prompt"]
        g[cid] = cond
    for node in g.values():
        key = PREFIX_INPUTS.get(node.get("class_type"))
        if key and isinstance(node["inputs"].get(key), str):
            head, _, tail = node["inputs"][key].rpartition("/")
            node["inputs"][key] = f"{head}/tau_sweep_{name}_{tail}" if head else f"tau_sweep_{name}_{tail}"
        if seed is not None and node.get("class_type") == "RandomNoise":
            node["inputs"]["noise_seed"] = seed
    return g


def _http(host: str, path: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://{host}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def render(host: str, graph: dict, timeout_s: float) -> dict:
    """Queue one graph and wait for `/history` to hold it. Returns its entry."""
    prompt_id = _http(host, "/prompt", {"prompt": graph})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        entry = _http(host, f"/history/{prompt_id}").get(prompt_id)
        if entry is not None:
            status = entry.get("status", {})
            return {"prompt_id": prompt_id, "seconds": round(time.time() - t0, 1),
                    "status": status.get("status_str"), "completed": bool(status.get("completed"))}
        time.sleep(15)
    return {"prompt_id": prompt_id, "seconds": round(time.time() - t0, 1), "status": "timed out", "completed": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the graph every scene is rendered on")
    ap.add_argument("--scene", action="append", required=True, metavar="NAME[=DONOR.json | =image:FILE]")
    ap.add_argument("--seed", type=int, default=None, help="default: the base graph's own")
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--timeout", type=float, default=7200.0, help="seconds per render")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = json.loads(Path(args.base).read_text())
    out = {"base": args.base, "seed": args.seed, "scenes": {}}
    for spec in args.scene:
        name, _, donor_path = spec.partition("=")
        image = donor_path.removeprefix("image:") if donor_path.startswith("image:") else None
        donor = json.loads(Path(donor_path).read_text()) if donor_path and image is None else None
        graph = scene_graph(base, donor, name, args.seed, image)
        sha = hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        print(f"{name}: queueing ({donor_path or 'the base graph'})", flush=True)
        result = render(args.host, graph, args.timeout)
        out["scenes"][name] = {"donor": donor_path or None, "graph_sha256": sha, "graph": graph, **result}
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"{name}: {result['status']} in {result['seconds']} s, prompt {result['prompt_id']}", flush=True)
        if not result["completed"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
