"""The tau sweep: Sol at every tau of a grid against the fallback, on the same
q/k/v, per head and per segment, inside a live render. The calibration
instrument behind `sparse_tables/` (`bench/calibrate_sparse_table.py`).

    H3_SOL_SWEEP="dir=<path>[,taus=0.25:0.5:1][,capture=<label>][,floor=1]" <comfy>/start.sh

Armed, every DiT attention call the sparse node is on is computed once on the
chained fallback and once per tau on Sol, and **the fallback's output is what
the model gets**. So the render is the graph with the sparse node removed,
every tau sees the same inputs, and a fixed seed gives the same inputs again.
The node's `dense_blocks` and its sigma window are ignored while armed: all
blocks and all steps are measured, on an unedited shipped graph.

**Why one render and not one per tau.** `sol_block_probe.py` measures the tau
a render ran at, on that render's own trajectory. A table needs every head's
error at every tau on identical inputs, and the kernel gives that from one
call per tau, because a head's routing reads no other head
(`bench/measure_per_head_tau_on_capture.py`). The 2026-10-02 campaign varied
tau across renders with the seed free and could not be read
(`bench/results/2026-10-02_sol_campaign_reanalysis.md`).

**The swept call routes every query row** (`sink_q` off), with the node's own
exact keys, quantizer and token routing. A query block's output does not
depend on which other rows run exact, so the video rows are the shipped
call's, bit for bit, and the conditioning rows give the routed error of each
segment in the same pass (measured: `bench/results/2026-10-03_tau_sweep.md`,
"Premises"). The one difference is the query block that straddles the end of
the conditioning prefix: its video rows are exact in a shipped call and
routed here.

`floor=1` also computes stock bf16 attention on each call and records the
fallback's own error against it, the unit a budget is stated in. It costs
more than the rest of the sweep's kernels together, so arm it for one render.

The record is `sol_sweep_<stamp>.jsonl` in `dir`: a header, a `render` row per
prompt, a `config` row per node configuration, and a `cell` row per call with,
per segment and head, the reference energy, the squared error at each tau and
the routed key blocks at each tau. No tensors. Timings of an armed render are
void; the per-call milliseconds in a cell are synchronised and are the
kernels' own.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import torch

_LOG = "[h3-sol-sweep]"
SCHEMA = 1
BLOCK = 64
#: The grid when the spec names none. **Inherited** from
#: `bench/measure_per_head_tau_on_capture.py::TAUS`, so a live cell and an
#: offline cell are priced on the same values; both sides of the shipped 1.0.
DEFAULT_TAUS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
#: Heads converted to float32 at a time by the metrics. **Reasoned**: four
#: heads of a 120k-row call are about a quarter GiB per temporary, beside an
#: attention call that already sets the render's peak.
HEAD_CHUNK = 4

_armed: bool | None = None
_spec: dict = {}
_raw_spec: str = ""


def _parse(spec: str) -> dict:
    out = {"dir": None, "taus": DEFAULT_TAUS, "capture": None, "floor": False}
    for part in spec.split(","):
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if key == "dir":
            out["dir"] = os.path.expanduser(val)
        elif key == "taus" and val:
            out["taus"] = tuple(float(x) for x in val.split(":"))
        elif key == "capture" and val:
            out["capture"] = val
        elif key == "floor":
            out["floor"] = val == "1"
    return out


def arm(spec: str | None) -> bool:
    """Set the arming state from a spec string; the server reads the
    environment through this once, tests call it directly."""
    global _armed, _spec, _raw_spec
    _raw_spec = spec or ""
    try:
        parsed = _parse(_raw_spec)
        if not parsed["taus"] or any(t < 0 for t in parsed["taus"]) or len(set(parsed["taus"])) != len(parsed["taus"]):
            raise ValueError("taus must be distinct and non-negative")
    except ValueError as exc:
        print(f"{_LOG} H3_SOL_SWEEP not understood ({exc}); sweep disabled")
        parsed = _parse("")
    _armed = bool(_raw_spec) and bool(parsed["dir"])
    if _raw_spec and not parsed["dir"]:
        print(f"{_LOG} H3_SOL_SWEEP set but no dir=; sweep disabled")
    _spec = parsed
    _writer_reset()
    if _armed:
        print(f"{_LOG} ARMED: dir={parsed['dir']} taus={list(parsed['taus'])} floor={parsed['floor']}; "
              f"every attention call under the sparse node returns the FALLBACK's output and "
              f"also runs Sol once per tau. This is not a Sol render and its timings are void")
    return _armed


def enabled() -> bool:
    if _armed is None:
        arm(os.environ.get("H3_SOL_SWEEP", ""))
    return bool(_armed)


def spec() -> dict:
    enabled()
    return dict(_spec, spec=_raw_spec)


# ---------------------------------------------------------------------------
# Writer: one lock, one append-only jsonl
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_jsonl = None
_path: str | None = None
_seq = 0
_renders_seen: dict = {}
_configs_written: set = set()


def _writer_reset() -> None:
    global _jsonl, _path, _seq
    try:
        if _jsonl is not None:
            _jsonl.close()
    except OSError:
        pass
    _jsonl, _path, _seq = None, None, 0
    _renders_seen.clear()
    _configs_written.clear()


def path() -> str | None:
    return _path


def _obs():
    try:
        from . import sol_observe
    except ImportError:
        import sol_observe  # type: ignore
    return sol_observe


def _header() -> dict:
    builds: dict = {"comfy_kitchen": None, "pack_commit": None}
    try:
        import importlib.metadata as md
        builds["comfy_kitchen"] = md.version("comfy-kitchen")
    except Exception:                                  # noqa: BLE001 -- identity, not the render
        pass
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run(["git", "-C", here, "rev-parse", "--short=12", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        builds["pack_commit"] = r.stdout.strip() if r.returncode == 0 else None
    except Exception:                                  # noqa: BLE001
        pass
    return {"kind": "header", "schema": SCHEMA, "spec": _raw_spec, "taus": list(_spec["taus"]),
            "floor": _spec["floor"], "capture": _spec.get("capture"), "builds": builds,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "trajectory": "fallback: the model gets the chained fallback's output on every swept call",
            "swept_call": "the node's Sol call with sink_q off (every query row routed) and tau varied; "
                          "exact keys, quantizer and token routing are the node's",
            "fields": {
                "den": "per segment, per head: sum |fallback|^2",
                "num": "per tau, per segment, per head: sum |sol - fallback|^2 (float32 inputs, float64 sums)",
                "routed": "per tau, per segment, per head: key blocks the kernel attended exactly, summed "
                          "over the query blocks whose first token is in the segment (blk_cnt)",
                "query_blocks": "per segment: how many query blocks that is; a row of them attends NTB at most",
                "floor": "floor=1 only. den: sum |stock|^2; num: sum |fallback - stock|^2, stock being "
                         "torch's scaled_dot_product_attention on the call's own bf16 q, k, v",
                "ms": "synchronised wall milliseconds of each kernel call, metrics excluded"},
            "note": "timings of an armed render are void"}


def _write(row: dict) -> None:
    global _jsonl, _path, _seq
    with _lock:
        if _jsonl is None:
            d = Path(_spec["dir"])
            d.mkdir(parents=True, exist_ok=True)
            _path = str(d / f"sol_sweep_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
            _jsonl = open(_path, "a", encoding="utf-8")
            _jsonl.write(json.dumps(_header()) + "\n")
        _seq += 1
        _jsonl.write(json.dumps(dict(row, seq=_seq)) + "\n")
        _jsonl.flush()


def _ensure_render(prompt_id) -> None:
    """A render row the first time a prompt id is seen, through the route
    recorder's helpers so the records describe a render the same way."""
    if prompt_id is None or prompt_id in _renders_seen:
        return
    _renders_seen[prompt_id] = len(_renders_seen)
    obs = _obs()
    row = {"kind": "render", "prompt_id": prompt_id, "process_render_index": _renders_seen[prompt_id],
           "graph_sha256": None, "workflow_file": None, "rendered": None}
    try:
        graph, why = obs._running_prompt(prompt_id)
        row["match"] = why
        if graph is not None:
            sha = obs.graph_sha256(graph)
            row.update({"graph_sha256": sha, "workflow_file": obs._shipped_graph_hashes().get(sha),
                        "rendered": obs._describe_prompt(graph), "summary": obs.graph_summary(graph)})
    except Exception as exc:                          # noqa: BLE001 -- identity, not the render
        row["match"] = f"could not read the running prompt: {exc}"
    _write(row)


def _ensure_config(settings: dict) -> str:
    digest = _obs().config_digest(settings)
    if digest not in _configs_written:
        _configs_written.add(digest)
        _write({"kind": "config", "digest": digest, "settings": settings})
    return digest


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _bhnd(x: torch.Tensor, heads: int, skip_output_reshape: bool) -> torch.Tensor:
    """(B, H, N, D) view of an attention output in either layout."""
    if skip_output_reshape:
        return x
    b, n, hd = x.shape
    return x.view(b, n, heads, hd // heads).transpose(1, 2)


def segment_head_sums(a: torch.Tensor, r: torch.Tensor | None, spans) -> torch.Tensor:
    """float64 (segments, heads): sum over a segment's rows of |a - r|^2, or of
    |a|^2 with `r` None. `a` and `r` are (B, H, N, D); `spans` is [(start, stop)]."""
    H = a.shape[1]
    out = torch.zeros((len(spans), H), dtype=torch.float64, device=a.device)
    for h in range(0, H, HEAD_CHUNK):
        x = a[:, h:h + HEAD_CHUNK].float()
        if r is not None:
            x = x - r[:, h:h + HEAD_CHUNK].float()
        row = (x * x).sum(dim=-1, dtype=torch.float64)            # (B, chunk, N)
        del x
        for j, (s0, s1) in enumerate(spans):
            out[j, h:h + HEAD_CHUNK] = row[:, :, s0:s1].sum(dim=(0, 2))
    return out


def segment_query_blocks(spans) -> list[tuple[int, int]]:
    """Per segment, the [first, stop) query blocks whose first token is in it."""
    return [((s0 + BLOCK - 1) // BLOCK, (s1 + BLOCK - 1) // BLOCK) for s0, s1 in spans]


def segment_head_routed(counts: torch.Tensor, qblocks) -> torch.Tensor:
    """int64 (segments, heads): the kernel's routed counts (B, H, NTB) summed
    over each segment's query blocks."""
    return torch.stack([counts[:, :, b0:b1].sum(dim=(0, 2), dtype=torch.int64) for b0, b1 in qblocks])


def _floats(t: torch.Tensor) -> list:
    return [[float(f"{x:.9g}") for x in row] for row in t.tolist()]


def _timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, round((time.perf_counter() - t) * 1e3, 2)


# ---------------------------------------------------------------------------
# The entry the Sol override calls
# ---------------------------------------------------------------------------

def run(*, dense_fn, sol_fn, stock_fn, heads: int, skip_output_reshape: bool, options, settings,
        block, tokens: int, batch: int, sink, device) -> torch.Tensor:
    """Run the fallback, then Sol at every tau, record the cell, return the
    fallback's output. `sol_fn(tau, counts)` is the node's own kernel call with
    only tau varied; `stock_fn()` is stock attention on the same tensors.

    Nothing here is caught: a kernel that fails mid-sweep stops the render,
    because a record with holes reads as a calibration."""
    obs = _obs()
    identity = obs._identity()
    _ensure_render(identity.get("prompt_id"))
    segments = obs._segments(options, tokens)
    if not segments:
        raise RuntimeError(f"{_LOG} block {block}: core published no segment table for this call, "
                           f"and a sweep without segments cannot tell video rows from conditioning")
    spans = [(a, b) for a, b, _kind in segments]
    qblocks = segment_query_blocks(spans)
    ntb = (int(tokens) + BLOCK - 1) // BLOCK
    sigmas = options.get("sigmas") if isinstance(options, dict) else None
    sigma = float(sigmas[0]) if sigmas is not None else None

    ms: dict = {"sol": {}}
    ref, ms["fallback"] = _timed(dense_fn)
    r = _bhnd(ref, heads, skip_output_reshape)
    den = segment_head_sums(r, None, spans)
    floor = None
    if _spec["floor"]:
        stock, ms["stock"] = _timed(stock_fn)
        s = _bhnd(stock, heads, skip_output_reshape)
        floor = {"den": _floats(segment_head_sums(s, None, spans)),
                 "num": _floats(segment_head_sums(r, s, spans))}
        del stock, s
    num, routed = {}, {}
    for tau in _spec["taus"]:
        counts = torch.empty((batch, heads, ntb), dtype=torch.int32, device=device)
        out, ms["sol"][str(tau)] = _timed(lambda: sol_fn(tau, counts))
        if out is None:
            raise RuntimeError(f"{_LOG} block {block}: the Sol call declined at tau {tau} after "
                               f"the override found the call eligible")
        num[str(tau)] = _floats(segment_head_sums(_bhnd(out, heads, skip_output_reshape), r, spans))
        routed[str(tau)] = segment_head_routed(counts, qblocks).tolist()
        del out, counts
    _write({"kind": "cell", "t_wall": time.time(), **identity, "capture": _spec.get("capture"),
            "config": _ensure_config(settings),
            "block": None if block is None else int(block), "sigma": sigma,
            "schedule": obs.schedule_index(sigma, options.get("sample_sigmas") if isinstance(options, dict) else None),
            "cond_or_uncond": obs._cond_or_uncond(options),
            "B": int(batch), "H": int(heads), "T": int(tokens), "NTB": ntb,
            "segments": segments, "query_blocks": [b1 - b0 for b0, b1 in qblocks],
            "sink_blocks": [int(sink[0]), int(sink[1])],
            "den": _floats(den), "num": num, "routed": routed, "floor": floor, "ms": ms})
    return ref
