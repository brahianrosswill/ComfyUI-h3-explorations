#!/usr/bin/env python3
"""Which model files and settings made each render, and which output files it wrote.

Written 2026-09-27 for the owner: "i have too many model files and i cant tell
what does what". Reads `run_graph_arms.py` rows (JSONL), resolves each row's
graph as it was when rendered (the graph at the commit whose file matches the
row's `graph_sha256`, else the file on disk, flagged) plus the row's patches,
and reports per row:

- the diffusion model (`UNETLoader.unet_name`), every LoRA loader's file,
  strength, head strength, block list and apply mode, the VAE;
- the schedule (sampler, scheduler and steps, or ManualSigmas), the shift;
- the attention wiring (Sol, VSA and its keep, sage, the dense backend);
- the files it wrote, by `run_graph_arms.py`'s `_<label>` naming: clips in
  `Video/`, final latents and per-step x0 latents in `latents/`. Basenames
  only; the output root comes from `bench/_paths.comfy_output`.

Rows are grouped by configuration (everything above except prompt, length and
seed). `--unclaimed` lists `latents/` files no row names.

    python bench/render_inventory.py bench/results/2026-09-26_*.jsonl [--md OUT] [--unclaimed]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workflows"))
from _paths import comfy_output  # noqa: E402
from h3_config import LORA_LOADER_CLASSES  # noqa: E402

_graph_cache: dict = {}


def graph_at(path: str, sha16: str):
    """The graph as rendered: the newest commit whose file hashes to sha16."""
    key = (path, sha16)
    if key in _graph_cache:
        return _graph_cache[key]
    found, note = None, "not found"
    p = REPO / path
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest().startswith(sha16 or "~"):
        found, note = json.loads(p.read_text()), "current file"
    else:
        revs = subprocess.run(["git", "log", "--format=%h", "--", path], cwd=REPO,
                              capture_output=True, text=True).stdout.split()
        for rev in revs:
            blob = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO, capture_output=True).stdout
            if blob and hashlib.sha256(blob).hexdigest().startswith(sha16 or "~"):
                found, note = json.loads(blob), f"git {rev}"
                break
        if found is None and p.exists():
            found, note = json.loads(p.read_text()), "current file, hash differs"
    _graph_cache[key] = (found, note)
    return found, note


def apply_patches(g: dict, patches: list) -> dict:
    g = json.loads(json.dumps(g))
    for p in patches:
        field = p.get("field", "")
        inp = field.split(".", 1)[1] if "." in field else field
        for nid in p.get("nodes", []):
            if str(nid) in g:
                g[str(nid)]["inputs"][inp] = p.get("value")
    return g


def describe(g: dict) -> dict:
    nodes = [n for n in g.values() if isinstance(n, dict)]
    by = defaultdict(list)
    for n in nodes:
        by[n.get("class_type")].append(n["inputs"])
    loras = []
    for cls in LORA_LOADER_CLASSES:
        for i in by.get(cls, []):
            d = {"file": i.get("lora_name")}
            for k in ("strength", "strength_model", "head_strength", "blocks", "modules", "backbone_apply"):
                if k in i and not isinstance(i[k], list):
                    d[k] = i[k]
            loras.append(d)
    sched = [f"{i.get('scheduler')} {i.get('steps')}" for i in by.get("BasicScheduler", [])]
    sched += [f"sigmas {i.get('sigmas')}" for i in by.get("ManualSigmas", [])]
    sched += [f"PDD node {i.get('steps')} steps" for i in by.get("MiniMaxH3PDDLoRA", [])
              if isinstance(i.get("steps"), int) and i.get("steps")]
    sched += [f"PDD node steps from {i.get('steps')[0]}" for i in by.get("MiniMaxH3PDDLoRA", [])
              if isinstance(i.get("steps"), list)]
    attn = []
    for i in by.get("BlockSparseAttention", []):
        attn.append(f"VSA keep {i.get('selection.keep_percent')} from {i.get('start_percent')}"
                    if i.get("selection") == "vsa" else f"sparse {i.get('selection')}")
    if by.get("MiniMaxH3SolAttn"):
        attn.append("Sol")
    if by.get("MiniMaxH3SageAttention"):
        attn.append("sage")
    for i in by.get("ModelAttentionBackend", []):
        attn.append(f"dense {i.get('attention')}")
    shift = [f"{i.get('shift_video')}/{i.get('shift_audio')}" for i in by.get("MiniMaxH3SigmaShift", [])]
    return {"unet": sorted({i.get("unet_name") for i in by.get("UNETLoader", [])}),
            "loras": loras,
            "vae": sorted({i.get("vae_name") for i in by.get("VAELoader", []) if "video" in str(i.get("vae_name"))}
                          | {i.get("vae_name") for i in by.get("VAELoader", []) if "tae" in str(i.get("vae_name"))}),
            "sampler": sorted({i.get("sampler_name") for i in by.get("KSamplerSelect", [])}),
            "schedule": sched, "shift": shift or ["checkpoint default"], "attention": attn or ["stock"],
            "x0_observer": bool(by.get("MiniMaxH3StepX0Observer")),
            "saves_latents": bool(by.get("SaveLatent"))}


def outputs(root: Path, label: str) -> dict:
    vid = sorted(p.name for p in (root / "Video").glob(f"*_{label}_0*") if p.suffix in (".mp4", ".png"))
    lat = sorted(p.name for p in (root / "latents").glob(f"*_{label}_0*_.latent"))
    x0 = sorted(p.name for p in (root / "latents").glob(f"*_x0_{label}_*_video.latent"))
    return {"video": vid, "latents": lat, "x0_steps": len(x0), "x0_example": x0[0] if x0 else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--unclaimed", action="store_true")
    args = ap.parse_args()
    root = comfy_output()
    recs, claimed = [], set()
    for f in args.rows:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            g, note = graph_at(r["graph"], r.get("graph_sha256", ""))
            d = describe(apply_patches(g, r.get("patches", []))) if g else {}
            o = outputs(root, r["label"])
            claimed.update(o["video"])
            claimed.update(o["latents"])
            claimed.update(p.name for p in (root / "latents").glob(f"*_x0_{r['label']}_*_video.latent"))
            recs.append({"record": f.name, "ts": r.get("ts"), "label": r["label"], "warmup": bool(r.get("warmup")),
                         "error": r.get("error"), "graph": Path(r["graph"]).name, "graph_source": note,
                         **d, "outputs": o})
    out = {"measured_by": "bench/render_inventory.py", "rows": recs}
    if args.unclaimed:
        out["unclaimed_latents"] = sorted(p.name for p in (root / "latents").iterdir() if p.name not in claimed)
    text = json.dumps(out, indent=1) + "\n"
    if args.json:
        args.json.write_text(text)
    print(f"{len(recs)} rows; graphs resolved: " +
          json.dumps({k: sum(1 for r in recs if r['graph_source'].startswith(k))
                      for k in ('current file', 'git', 'not found')}))
    if args.unclaimed:
        print(f"unclaimed latents: {len(out['unclaimed_latents'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
