#!/usr/bin/env python3
"""Where each distill changes H3, from full-precision weights. CPU only.

Written 2026-09-26 for the owner's plan (after "dont constrain yourself to
just thinking about loras"): map where FastH3, PDD and FlashGen each change
the base, then build strength dials, transplants and sums from that map
(`bench/results/2026-09-26_followup_looks.md`, the weights sections). The
int8 files cannot answer this: FastH3's change is about a twentieth of an
int8 step, so in int8 it is rounding flips (`2026-09-26_fasth3_lora_rank.json`).

Inputs are ComfyUI-layout pruned checkpoints in bf16 (FastH3 V2
`fastvideo_fasth3_8step_v2_pruned_bf16`, the base
`minimax_h3_fl2va_pruned_bf16`), their shipped int8 twins, and the PDD and
FlashGen LoRA files. Paths are arguments; nothing here names a machine path.

Two modes:

- `--identity`: can we rebuild the shipped int8 files from these bf16 files?
  For each sampled backbone module, quantise the bf16 weight through
  comfy-kitchen's round-to-nearest ConvRot quantiser (group size read from
  the int8 file's own `comfy_quant`) and compare codes and scales with the
  shipped int8, graded by `bake_pdd_checkpoint.TIES_CRITERION`. Run for the
  base pair and the FastH3 pair. Also reports whether the two bf16 files'
  non-backbone tensors (time table, AdaLN basis and coefficients, norms,
  embeddings) are identical. AdaLN can be diffed directly only if the time
  tables match (the pruned format stores a lookup table of the timestep
  embedding; FastVideo's survey, 2026-09-26).
- `--map`: per block and backbone kind, the relative change ||dW|| / ||W||
  for FastH3 (bf16 minus bf16), PDD and FlashGen (their LoRA deltas at
  strength 1, scaled as `comfy.lora` scales them), and the cosine between
  each pair of deltas on the same module. It also reports each non-backbone
  tensor's relative change for FastH3.

    CUDA_VISIBLE_DEVICES= python bench/checkpoint_delta_map.py --identity \\
        --base-bf16 B.safetensors --base-int8 B8.safetensors \\
        --other-bf16 F.safetensors --other-int8 F8.safetensors --json OUT
    CUDA_VISIBLE_DEVICES= python bench/checkpoint_delta_map.py --map \\
        --base-bf16 B.safetensors --other-bf16 F.safetensors \\
        --pdd P.safetensors --flashgen G.safetensors --json OUT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

from analyze_checkpoint_delta import header  # noqa: E402
from analyze_quant_delta import marker  # noqa: E402
from bake_pdd_checkpoint import (KINDS, delta_of, dequantise, load_deltas, quantise,  # noqa: E402
                                 stored_codes, ties_verdict)
import vendor_config  # noqa: E402


def mods(every: int) -> list[str]:
    depth, _ = vendor_config.transformer_depth()
    return [f"blocks.{i}.{k}" for i in range(0, depth, every) for k in KINDS]


def bf16(f, key) -> np.ndarray:
    return f.get_tensor(key).to(torch.float32).numpy()


def rel(a, b) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt((d * d).sum()) / max(np.sqrt((b.astype(np.float64) ** 2).sum()), 1e-30))


def cos(a, b) -> float | None:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float((a.ravel().astype(np.float64) @ b.ravel().astype(np.float64)) / (na * nb))


def identity_pair(bf16_path: str, int8_path: str, modules: list[str]) -> dict:
    hdr, base = header(int8_path)
    rows = []
    with safe_open(bf16_path, "pt") as f:
        for m in modules:
            conf = marker(int8_path, hdr, base, m) or {}
            gs = int(conf["convrot_groupsize"])
            w = bf16(f, m + ".weight")
            q, s = quantise(w, gs)
            q0, s0 = stored_codes(int8_path, hdr, base, m)
            diff = np.abs(q.astype(np.int16) - q0.astype(np.int16))
            srel = np.abs(s - s0) / np.maximum(np.abs(s0), 1e-30)
            wq, w0 = dequantise(q, s, gs), dequantise(q0, s0, gs)
            rows.append({"module": m, "codes_equal": bool((diff == 0).all()), "scales_equal": bool(np.array_equal(s, s0)),
                         "codes_max_abs_diff": int(diff.max()), "codes_differing_frac": float((diff > 0).mean()),
                         "scale_rel_max": float(srel.max()), "scale_rel_row_max": float(srel.max()),
                         "err_control": rel(wq, w), "err_shipped": rel(w0, w)})
            print(f"  {m:<26} max|dq| {rows[-1]['codes_max_abs_diff']}  differing {rows[-1]['codes_differing_frac']:.2e}  "
                  f"err ours {rows[-1]['err_control']:.5f} shipped {rows[-1]['err_shipped']:.5f}", flush=True)
    return {"rows": rows, "verdict": ties_verdict(rows)}


def non_backbone(base_bf16: str, other_bf16: str) -> dict:
    out = {}
    with safe_open(base_bf16, "pt") as a, safe_open(other_bf16, "pt") as b:
        backbone = {m + ".weight" for m in mods(1)}
        for k in sorted(set(a.keys()) & set(b.keys())):
            if k in backbone:
                continue
            ta, tb = a.get_tensor(k), b.get_tensor(k)
            same = bool(torch.equal(ta, tb))
            out[k] = {"identical": same} if same else {"identical": False,
                                                       "rel": rel(tb.float().numpy(), ta.float().numpy())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--base-bf16", required=True)
    ap.add_argument("--other-bf16", required=True)
    ap.add_argument("--base-int8")
    ap.add_argument("--other-int8")
    ap.add_argument("--pdd")
    ap.add_argument("--flashgen")
    ap.add_argument("--every", type=int, default=1, help="every Nth block (default all)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rec: dict[str, Any] = {"measured_by": "bench/checkpoint_delta_map.py", "base_bf16": Path(args.base_bf16).name,
           "other_bf16": Path(args.other_bf16).name}

    if args.identity:
        ms = mods(args.every)
        print("== identity: base bf16 against base int8")
        rec["identity_base"] = identity_pair(args.base_bf16, args.base_int8, ms)
        print("   ", rec["identity_base"]["verdict"]["same_regime"])
        print("== identity: other bf16 against other int8")
        rec["identity_other"] = identity_pair(args.other_bf16, args.other_int8, ms)
        print("   ", rec["identity_other"]["verdict"]["same_regime"])
        print("== non-backbone tensors, base bf16 against other bf16")
        nb = non_backbone(args.base_bf16, args.other_bf16)
        rec["non_backbone"] = nb
        changed = {k: v for k, v in nb.items() if not v["identical"]}
        print(f"   {len(nb)} shared, {len(changed)} changed")
        for k, v in list(changed.items())[:40]:
            print(f"   changed {k:<52} rel {v['rel']:.5f}")

    if args.map:
        depth, _ = vendor_config.transformer_depth()
        pdd = load_deltas(Path(args.pdd), depth) if args.pdd else {}
        fg = load_deltas(Path(args.flashgen), depth) if args.flashgen else {}
        rows = {}
        with safe_open(args.base_bf16, "pt") as a, safe_open(args.other_bf16, "pt") as b:
            for m in mods(args.every):
                wa = bf16(a, m + ".weight")
                d_f = bf16(b, m + ".weight") - wa
                row: dict[str, float | None] = {"fasth3": rel(wa + d_f, wa)}
                d_p = delta_of(pdd[m], 1.0) if m in pdd else None
                d_g = delta_of(fg[m], 1.0) if m in fg else None
                if d_p is not None:
                    row["pdd"] = float(np.linalg.norm(d_p) / np.linalg.norm(wa))
                    row["cos_fasth3_pdd"] = cos(d_f, d_p)
                if d_g is not None:
                    row["flashgen"] = float(np.linalg.norm(d_g) / np.linalg.norm(wa))
                    row["cos_fasth3_flashgen"] = cos(d_f, d_g)
                if d_p is not None and d_g is not None:
                    row["cos_pdd_flashgen"] = cos(d_p, d_g)
                rows[m] = row
                print(f"  {m:<26} " + "  ".join(f"{k} {v:+.4f}" if isinstance(v, float) else f"{k} -"
                                               for k, v in row.items()), flush=True)
        rec["map"] = rows

    if args.json:
        args.json.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
