#!/usr/bin/env python3
"""Run the TaoMate stream sampler on the server, for its checks and its throwaway render.

`docs/h3_taomate.md` section 7.3. Two modes, each a rewired TaoMate probe graph.

**`verify_whole_clip`** (step 2). From `workflows/h3_probe_taomate_3step_api.json`:
- sage, Sol and the chain assert are removed, because the sampler refuses them
  and the stock reference must run the same plain attention;
- `KSamplerSelect` becomes `MiniMaxH3TaoMateStreamSampler` in that mode;
- the canvas and length are set explicitly, small by default;
- decoding and the video writer become `PreviewAny` on the sampler's latent,
  so nothing is decoded and nothing is written to the output share.

The node runs core's own euler sampler first, then its hooked loop over the
same inputs, and logs the deviation between the two latents. This script
reads that line and grades it.

**`stream`** (steps 4 and 5). From
`workflows/h3_probe_taomate_3step_audio_freeze_api.json`, with the same
attention chain removed and the sampler in stream mode. The track, prompt,
seed and output prefix come from the flags. It decodes and writes a clip, and
records the per-chunk log lines: wall time, peak allocated memory, cache
tokens.

A cheaper canvas proves the harness and the hook, not a render (CLAUDE.md,
"1344x768 is a trained canvas").

    <comfy venv python> bench/verify_taomate_stream.py --mode verify_whole_clip \\
        [--width 864 --height 480 --length 124] [--record bench/results/<date>_<name>.json]
    <comfy venv python> bench/verify_taomate_stream.py --mode stream --prompt-id t2va_studio_dancer \\
        --audio "Drum Machine Pulse.mp3" --seed 1101 --prefix Video/h3_taomate_stream_throwaway ...

Exit codes: 0 passed (verify: within `MATCH_REL_RMS`; stream: the run
succeeded and logged every chunk), 1 verify out of bound, 2 the run failed or
its log lines were not found.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))

import taomate_streaming as tm  # noqa: E402

PROBES = {
    "verify_whole_clip": REPO / "workflows" / "h3_probe_taomate_3step_api.json",
    "stream": REPO / "workflows" / "h3_probe_taomate_3step_audio_freeze_api.json",
}
ATTENTION_CHAIN = ("MiniMaxH3SageAttention", "MiniMaxH3SolAttn", "SageChainAssert")
#: Reasoned: the hooked loop computes the same attention with torch's SDPA on
#: the same inputs, so any difference is kernel dispatch and float rounding.
#: A wiring defect (a wrong position, timestep or row order) moves the latent
#: by orders of magnitude more than this.
MATCH_REL_RMS = 1e-3
VERIFY_TAG = "[taomate] verify_whole_clip "
CHUNK_TAG = "[taomate] request "
_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)")


def _post(host, path, payload):
    req = urllib.request.Request(f"http://{host}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _get(host, path):
    with urllib.request.urlopen(f"http://{host}{path}", timeout=60) as resp:
        return json.loads(resp.read())


def build_graph(args) -> dict:
    doc = json.loads(PROBES[args.mode].read_text(encoding="utf-8"))
    by_class = {}
    for nid, node in doc.items():
        by_class.setdefault(node["class_type"], []).append(nid)
    # Rewire the guider past the removed attention chain, to whatever its
    # first node took as the model.
    sage = doc[by_class["MiniMaxH3SageAttention"][0]]
    doc[by_class["BasicGuider"][0]]["inputs"]["model"] = sage["inputs"]["model"]
    cond = doc[by_class["MiniMaxH3Conditioning"][0]]
    cond["inputs"].update(width=args.width, height=args.height, length=args.length)
    remove = ATTENTION_CHAIN + ("MiniMaxH3Resolution",)
    if args.mode == "verify_whole_clip":
        remove += ("VAEDecode", "VAEDecodeAudio", "VHS_VideoCombine")
    for cls in remove:
        for nid in by_class.get(cls, []):
            del doc[nid]
    doc[by_class["KSamplerSelect"][0]] = {
        "class_type": "MiniMaxH3TaoMateStreamSampler",
        "inputs": {"mode": args.mode, "cache_device": args.cache_device}}
    if args.mode == "verify_whole_clip":
        doc["900"] = {"class_type": "PreviewAny",
                      "inputs": {"source": [by_class["SamplerCustomAdvanced"][0], 0]}}
        return doc
    if args.prompt_id:
        import prompts
        cond["inputs"]["prompt"] = prompts.text(args.prompt_id)
    if args.seed is not None:
        doc[by_class["RandomNoise"][0]]["inputs"]["noise_seed"] = args.seed
    if args.audio:
        doc[by_class["LoadAudio"][0]]["inputs"]["audio"] = args.audio
    doc[by_class["VHS_VideoCombine"][0]]["inputs"]["filename_prefix"] = args.prefix
    return doc


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log_lines_since(host, since_iso: str, tag: str) -> list[str]:
    logs = _get(host, "/internal/logs")
    text = logs if isinstance(logs, str) else json.dumps(logs)
    out = []
    for line in text.replace("\\n", "\n").split("\n"):
        m = _STAMP.match(line.strip())
        if tag in line and m and m.group(1) >= since_iso:
            out.append(line.strip())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the TaoMate stream sampler's check or throwaway render.")
    ap.add_argument("--mode", choices=tuple(PROBES), default="verify_whole_clip")
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--length", type=int, default=tm.frame_count(1))
    ap.add_argument("--cache-device", default=None, choices=("cpu_pinned", "cpu", "gpu"),
                    help="default: gpu for verify (no cache is used), cpu_pinned for stream")
    ap.add_argument("--prompt-id", help="stream: a prompt bank id")
    ap.add_argument("--audio", help="stream: a file in ComfyUI's input directory")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--prefix", default="Video/h3_taomate_stream_throwaway")
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--record", type=Path)
    args = ap.parse_args(argv)
    if args.cache_device is None:
        args.cache_device = "gpu" if args.mode == "verify_whole_clip" else "cpu_pinned"

    graph = build_graph(args)
    graph_sha = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()
    since = dt.datetime.now().isoformat()
    started = time.time()
    prompt_id = _post(args.host, "/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    print(f"queued {prompt_id}: {args.mode}, {args.width}x{args.height}, {args.length} frames", flush=True)
    history = None
    while time.time() - started < args.timeout:
        hist = _get(args.host, f"/history/{prompt_id}")
        if prompt_id in hist:
            history = hist[prompt_id]
            break
        time.sleep(5)
    if history is None:
        print("FAIL  timed out waiting for the run")
        return 2
    status = history.get("status", {})
    if status.get("status_str") != "success":
        errors = [m for m in status.get("messages", []) if m[0] == "execution_error"]
        print(f"FAIL  run did not succeed: {json.dumps(errors)[:3000]}")
        return 2
    wall = time.time() - started

    record = {
        "date": dt.date.today().isoformat(),
        "mode": args.mode,
        "produced_by": "bench/verify_taomate_stream.py",
        "graph_from": str(PROBES[args.mode].relative_to(REPO)),
        "graph_sha256": graph_sha,
        "prompt_id": prompt_id,
        "canvas": f"{args.width}x{args.height}",
        "length": args.length,
        "cache_device": args.cache_device,
        "repo_commit": git_head(REPO),
        "comfy_commit": git_head(REPO.parent.parent),
        "submit_to_finish_s": round(wall, 1),
        "is_not": "a quality statement; a small canvas proves the harness and the hook",
    }
    if args.mode == "verify_whole_clip":
        lines = log_lines_since(args.host, since, VERIFY_TAG)
        if not lines:
            print("FAIL  the run succeeded but no verify report is in the server log buffer")
            return 2
        report = json.loads(lines[-1].split(VERIFY_TAG, 1)[1].strip().replace('\\"', '"'))
        passed = all(report[s]["rel_rms"] <= MATCH_REL_RMS for s in ("video", "audio"))
        for stream in ("video", "audio"):
            r = report[stream]
            print(f"  {stream}: exact {r['exact']}, max_abs {r['max_abs']:.3e}, rel_rms {r['rel_rms']:.3e}")
        print(("ok    " if passed else "FAIL  ") + "hooked whole-clip loop vs core euler "
              f"(bound rel_rms {MATCH_REL_RMS}, reasoned)")
        record.update(what=("core's euler sampler and the sampler's own loop through its block attention "
                            "hook, same graph inputs, latents compared"),
                      match_bound_rel_rms=MATCH_REL_RMS, report=report, passed=passed)
        code = 0 if passed else 1
    else:
        lines = log_lines_since(args.host, since, CHUNK_TAG)
        expected = len(tm.run_plan(tm.requests_for(
            ((args.length - 5) // 17) * 5 + 2) or 1))
        outputs = [f"{i.get('subfolder', '')}/{i['filename']}" for o in history.get("outputs", {}).values()
                   for v in o.values() if isinstance(v, list) for i in v
                   if isinstance(i, dict) and "filename" in i]
        for line in lines:
            print("  " + line.split(" - ", 1)[-1])
        passed = len(lines) >= expected
        print(("ok    " if passed else "FAIL  ") + f"{len(lines)} chunk log line(s), expected {expected}; "
              f"outputs {outputs}")
        record.update(what="the TaoMate stream sampler's chunked, cached run with a frozen track",
                      prompt_bank_id=args.prompt_id, audio=args.audio, seed=args.seed,
                      chunk_log=lines, outputs=outputs, passed=passed)
        code = 0 if passed else 2

    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"record {args.record}")
    return code


if __name__ == "__main__":
    sys.exit(main())
