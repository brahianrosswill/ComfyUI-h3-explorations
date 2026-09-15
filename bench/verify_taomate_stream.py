#!/usr/bin/env python3
"""Run the TaoMate stream sampler's equality check on the server and record it.

`docs/h3_taomate.md` section 7.3 step 2. The graph is
`workflows/h3_probe_taomate_3step_api.json` rewired for the check:

- sage, Sol and the chain assert are removed, because the sampler refuses them
  and the stock reference must run the same plain attention;
- `KSamplerSelect` is replaced by `MiniMaxH3TaoMateStreamSampler` in
  `verify_whole_clip` mode;
- the canvas and length are set explicitly, small by default;
- decoding and the video writer are replaced by `PreviewAny` on the sampler's
  latent, so nothing is decoded and nothing is written to the output share.

In that mode the node runs core's own euler sampler first, then its hooked loop
over the same inputs, and logs the deviation between the two latents. This
script submits the graph, waits for it, reads that line from the server's log
buffer, and writes the record. A cheaper canvas proves the harness and the hook,
not a render (CLAUDE.md, "1344x768 is a trained canvas").

    <comfy venv python> bench/verify_taomate_stream.py [--width 864 --height 480 --length 124]
        [--record bench/results/<date>_taomate_verify_whole_clip.json]

Exit codes: 0 the hooked loop matched within `MATCH_REL_RMS`, 1 it did not,
2 the run failed or no report was found.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "workflows" / "h3_probe_taomate_3step_api.json"
REMOVE = ("MiniMaxH3SageAttention", "MiniMaxH3SolAttn", "SageChainAssert",
          "VAEDecode", "VAEDecodeAudio", "VHS_VideoCombine", "MiniMaxH3Resolution")
#: Reasoned: the hooked loop computes the same attention with torch's SDPA on
#: the same inputs, so any difference is kernel dispatch and float rounding.
#: A wiring defect (a wrong position, timestep or row order) moves the latent
#: by orders of magnitude more than this.
MATCH_REL_RMS = 1e-3
REPORT_TAG = "[taomate] verify_whole_clip "


def _post(host, path, payload):
    req = urllib.request.Request(f"http://{host}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _get(host, path):
    with urllib.request.urlopen(f"http://{host}{path}", timeout=60) as resp:
        return json.loads(resp.read())


def build_graph(width: int, height: int, length: int) -> dict:
    doc = json.loads(PROBE.read_text(encoding="utf-8"))
    by_class = {}
    for nid, node in doc.items():
        by_class.setdefault(node["class_type"], []).append(nid)
    # Rewire the guider past the removed attention chain: to whatever the
    # chain's first node took as its model.
    sage = doc[by_class["MiniMaxH3SageAttention"][0]]
    guider = doc[by_class["BasicGuider"][0]]
    guider["inputs"]["model"] = sage["inputs"]["model"]
    cond = doc[by_class["MiniMaxH3Conditioning"][0]]
    cond["inputs"].update(width=width, height=height, length=length)
    for cls in REMOVE:
        for nid in by_class.get(cls, []):
            del doc[nid]
    doc[by_class["KSamplerSelect"][0]] = {
        "class_type": "MiniMaxH3TaoMateStreamSampler",
        "inputs": {"mode": "verify_whole_clip", "cache_device": "gpu"}}
    sampler_out = by_class["SamplerCustomAdvanced"][0]
    doc["900"] = {"class_type": "PreviewAny", "inputs": {"source": [sampler_out, 0]}}
    return doc


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the TaoMate stream sampler's whole-clip equality check.")
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--length", type=int, default=124)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--record", type=Path)
    args = ap.parse_args(argv)

    graph = build_graph(args.width, args.height, args.length)
    graph_sha = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()
    started = time.time()
    prompt_id = _post(args.host, "/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    print(f"queued {prompt_id} ({args.width}x{args.height}, {args.length} frames)", flush=True)
    status = None
    while time.time() - started < args.timeout:
        hist = _get(args.host, f"/history/{prompt_id}")
        if prompt_id in hist:
            status = hist[prompt_id].get("status", {})
            break
        time.sleep(5)
    if status is None:
        print("FAIL  timed out waiting for the run")
        return 2
    if status.get("status_str") != "success":
        errors = [m for m in status.get("messages", []) if m[0] == "execution_error"]
        print(f"FAIL  run did not succeed: {json.dumps(errors)[:2000]}")
        return 2

    logs = _get(args.host, "/internal/logs")
    text = logs if isinstance(logs, str) else json.dumps(logs)
    lines = [line for line in text.replace("\\n", "\n").split("\n") if REPORT_TAG in line]
    if not lines:
        print("FAIL  the run succeeded but no verify report is in the server log buffer")
        return 2
    raw = lines[-1].split(REPORT_TAG, 1)[1].strip()
    report = json.loads(raw.replace('\\"', '"'))
    passed = all(report[s]["rel_rms"] <= MATCH_REL_RMS for s in ("video", "audio"))
    for stream in ("video", "audio"):
        r = report[stream]
        print(f"  {stream}: exact {r['exact']}, max_abs {r['max_abs']:.3e}, rel_rms {r['rel_rms']:.3e}")
    print(("ok    " if passed else "FAIL  ") + f"hooked whole-clip loop vs core euler "
          f"(bound rel_rms {MATCH_REL_RMS}, reasoned)")

    if args.record is not None:
        record = {
            "date": dt.date.today().isoformat(),
            "what": ("TaoMate stream sampler, verify_whole_clip: core's euler sampler and the sampler's "
                     "own loop through its block attention hook, same graph inputs, latents compared"),
            "produced_by": "bench/verify_taomate_stream.py",
            "graph_from": str(PROBE.relative_to(REPO)),
            "graph_sha256": graph_sha,
            "prompt_id": prompt_id,
            "canvas": f"{args.width}x{args.height}",
            "length": args.length,
            "repo_commit": git_head(REPO),
            "comfy_commit": git_head(REPO.parent.parent),
            "match_bound_rel_rms": MATCH_REL_RMS,
            "report": report,
            "passed": passed,
            "is_not": "a render or a quality statement; a small canvas proves the hook reproduces core",
        }
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"record {args.record}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
