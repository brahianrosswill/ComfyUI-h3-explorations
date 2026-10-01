#!/usr/bin/env python3
"""Convert a published PDMD LoRA to ComfyUI's H3 layout at its full rank.

`pdmd2026/pdmd_{4,2}NFE_lora` ship rank-128 pairs on the diffusers
`MiniMaxH3Transformer3DModel` (`transformer.<module>.lora_A/B.weight`, scale 1.0
in the header, no alpha tensor). kijai's ComfyUI files resize them by SVD; this
keeps every direction, so the published delta arrives unchanged.

## The mapping

`measure_pdmd_lora_conversion.published_factors`, imported so the rule has one
copy. The measurement there checks it on the base weights, with controls:

- `to_q`/`to_k`/`to_v` fuse into `attn.qkv_proj`: A concatenated over rank, B
  block-diagonal (ComfyUI's qkv is `cat([q; k; v])`). The fused rank is three
  times the published one.
- `ff.net.0.proj` becomes `mlp.fc1` with B's row halves swapped: diffusers
  stores SwiGLU as [value; gate], ComfyUI as [gate; value].
- `attn.to_out.0` and `ff.net.2` are renames to `attn.out_proj`, `mlp.fc2`.
- `transformer_blocks.N` and `token_refiner.refiner_blocks.N` become
  `blocks.N` and `token_refiner.blocks.N`.

One `.alpha` per module, equal to its rank, so ComfyUI's alpha / rank is the
publisher's `lora_scale` 1.0. ComfyUI reads alpha from a tensor, never from
metadata (`bench/check_lora_alpha.py`).

## The check it runs

Every output module's delta is compared with the published one, through the
same low-rank inner products the measurement uses. Anything short of an exact
copy is refused (`EXACT`).

    CUDA_VISIBLE_DEVICES="" <comfy venv python> bench/convert_pdmd_lora.py \\
        --lora <pdmd2026>/pdmd_4NFE_lora/lora_model_0.safetensors \\
        --steps 4 --out <dir>/minimax_h3_pdmd_4step_rank128_comfy.safetensors \\
        --record bench/results/<date>_pdmd_4step_rank128_conversion.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_pdmd_lora_conversion import compare, comfy_name, published_factors, sha256  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PREFIX = "diffusion_model."
#: How close to the published delta each module must land. The transform only
#: moves and repeats values, so anything below 1 is a bug. **Reasoned.**
EXACT = 1e-6
KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def blocks_of(lora: dict) -> list[str]:
    """ComfyUI block names present in the published file, in file order."""
    seen = []
    for k in lora:
        body = k[len("transformer."):]
        parts = body.split(".")
        blk = ".".join(parts[:2]) if parts[0] == "transformer_blocks" else ".".join(parts[:3])
        name = comfy_name(blk)
        if name not in seen:
            seen.append(name)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True, choices=(2, 4),
                    help="The step count this file was trained for; written into the header.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--record", type=Path)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    with safe_open(str(args.lora), "pt") as f:
        meta = f.metadata() or {}
        lora = {k: f.get_tensor(k) for k in f.keys()}
    if meta.get("fuse") != "W_base += lora_scale * (lora_B @ lora_A)" or meta.get("lora_scale") != "1.0":
        raise SystemExit(f"unexpected fuse rule or scale in {args.lora.name}: "
                         f"{meta.get('fuse')!r}, {meta.get('lora_scale')!r}")
    dtype = next(iter(lora.values())).dtype

    out, rows = {}, {}
    for blk in blocks_of(lora):
        for kind in KINDS:
            b, a = published_factors(lora, blk, kind)
            key = f"{PREFIX}{blk}.{kind}"
            out[f"{key}.lora_A.weight"] = a.to(dtype).contiguous()
            out[f"{key}.lora_B.weight"] = b.to(dtype).contiguous()
            out[f"{key}.alpha"] = torch.tensor(float(a.shape[0]))
    # Each block consumes six published pairs: q, k, v, to_out.0, ff.net.0.proj, ff.net.2.
    pairs = sum(1 for k in lora if k.endswith(".lora_A.weight"))
    consumed = 6 * len(blocks_of(lora))
    if consumed != pairs:
        raise SystemExit(f"{pairs} published pairs, {consumed} consumed: a module was missed")

    # The check: every module's delta against the published one.
    worst = 1.0
    for blk in blocks_of(lora):
        for kind in KINDS:
            b1, a1 = published_factors(lora, blk, kind)
            key = f"{PREFIX}{blk}.{kind}"
            r = compare(b1, a1, out[f"{key}.lora_B.weight"].float(), out[f"{key}.lora_A.weight"].float())
            rows[f"{blk}.{kind}"] = {"cosine": r["cosine"], "rel_err": r["rel_err"],
                                     "rank": int(out[f"{key}.lora_A.weight"].shape[0])}
            worst = min(worst, r["cosine"])
    if worst < 1 - EXACT:
        raise SystemExit(f"conversion is not exact: worst module cosine {worst}")

    header = {
        "format": "pt",
        "converted_layout": "comfyui_minimax_h3",
        "source": f"pdmd2026/pdmd_{args.steps}NFE_lora/{args.lora.name}",
        "source_sha256": sha256(args.lora),
        "source_tag": meta.get("tag", ""),
        "conversion": ("bench/convert_pdmd_lora.py: full rank; q/k/v fused with A concatenated and "
                       "B block-diagonal; ff.net.0.proj B halves swapped into mlp.fc1; alpha = rank "
                       "(publisher lora_scale 1.0)"),
        "sampler_steps": str(args.steps),
        "sampling": "euler, scheduler simple, shift 12 video / 3 audio, no CFG (trainer's contract)",
        "application": "at the call (MiniMaxH3LoRABranch); a merge into int8 loses most of it",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out), metadata=header)

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    record = {
        "script": "bench/convert_pdmd_lora.py",
        "commit": commit,
        "device": "cpu (stored weights only; no render)",
        "source": header["source"], "source_sha256": header["source_sha256"], "source_tag": header["source_tag"],
        "out": args.out.name, "out_sha256": sha256(args.out),
        "modules": len(rows), "published_pairs": pairs,
        "worst_cosine": worst, "exact_bound": EXACT,
        "rank_by_kind": {k: sorted({r["rank"] for m, r in rows.items() if m.endswith(k)}) for k in KINDS},
        "per_module": rows,
    }
    if args.record:
        args.record.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("out", "modules", "published_pairs", "worst_cosine", "rank_by_kind")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
