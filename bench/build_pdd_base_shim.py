#!/usr/bin/env python3
"""A minimal `--base` for `convert_pdd_lora.py`, with any time embedder. CPU only.

Written 2026-09-26 to put PDD's adaln back onto a checkpoint whose timestep
conditioning is FastH3's (`bench/results/2026-09-26_fasth3_weights.md`:
FastH3's conditioning change is its time embedder). The converter reads
exactly four things from `--base` (`convert_pdd_lora.py`): the time embedder,
to derive `silu(time_embedder(t))` on the 1025-row grid PDD's adaln bake is
solved against; `final_layer.video_out.weight`, the partition fingerprint; and
one int8 backbone probe that must equal `--pruned`'s. This writes a file with
only those tensors:

- the time embedder from a diffusers-form transformer directory
  (`time_embedder.linear_{1,2}` renamed to core's `proj_in`/`proj_out`);
- the fingerprint and the probe copied from `--pruned`, the checkpoint the
  sidecar will load on.

The control is the same build from the MiniMaxAI fl2va transformer against
the shipped pruned fl2va: its bake must match the shipped sidecar's.

    CUDA_VISIBLE_DEVICES= python bench/build_pdd_base_shim.py --temb-dir DIR/transformer \\
        --pruned P.safetensors --out SHIM.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bake_pdd_checkpoint import git_commit  # noqa: E402
from convert_pdd_lora import BACKBONE_PROBE_KEY  # noqa: E402

RENAME = {"time_embedder.linear_1.weight": "time_embedder.proj_in.weight",
          "time_embedder.linear_1.bias": "time_embedder.proj_in.bias",
          "time_embedder.linear_2.weight": "time_embedder.proj_out.weight",
          "time_embedder.linear_2.bias": "time_embedder.proj_out.bias"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temb-dir", required=True, type=Path)
    ap.add_argument("--pruned", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    where = json.loads(next(args.temb_dir.glob("*.safetensors.index.json")).read_text())["weight_map"]
    out = {}
    for src, dst in RENAME.items():
        with safe_open(str(args.temb_dir / where[src]), "pt") as f:
            out[dst] = f.get_tensor(src).to(torch.float32).contiguous()
    probe_scale = BACKBONE_PROBE_KEY[: -len(".weight")] + ".weight_scale"
    with safe_open(str(args.pruned), "pt") as f:
        for k in ("final_layer.video_out.weight", BACKBONE_PROBE_KEY, probe_scale):
            out[k] = f.get_tensor(k).contiguous()
    meta = {"h3_shim_time_embedder_from": f"{args.temb_dir.parent.name}/{args.temb_dir.name}",
            "h3_shim_pruned": args.pruned.name,
            "h3_shim_by": f"bench/build_pdd_base_shim.py @ {git_commit()}"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out), metadata=meta)
    print(json.dumps({"out": args.out.name, "tensors": sorted(out), **meta}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
