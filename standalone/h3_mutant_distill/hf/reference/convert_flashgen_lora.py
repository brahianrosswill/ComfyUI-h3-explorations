#!/usr/bin/env python3
"""Convert Beidouqixing's FlashGen 4-step LoRA, at full rank, for a pruned int8 H3 checkpoint.

A standalone copy of `bench/convert_flashgen_lora.py` in
https://github.com/fblissjr/ComfyUI-h3-explorations (commit e7d08d19), with
its two in-repo imports inlined and the comparison against kijai's resize
dropped. The arithmetic is unchanged.

Input: the PEFT LoRA from `Beidouqixing/minimax-h3-4step-lora-flashgen`,
whose own `merge_lora_ckpt.py` merges `W + lora_B @ lora_A` at scale 1.0 into
the release's transformer. That merge is what this reproduces, in ComfyUI's
layout:

- **Names.** `{module}.lora_A.default.weight` becomes
  `diffusion_model.{module}.lora_A.weight`, likewise B, plus an `.alpha`
  tensor equal to the rank, so ComfyUI's alpha / rank is the scale 1.0.
- **qkv rows.** The release interleaves q, k and v per head; ComfyUI's
  `qkv_proj` stores them in three bands. `lora_B` rows are permuted
  `[head, qkv, dim] -> [qkv, head, dim]`, and checked on the weights first:
  the release's `qkv_proj.weight`, permuted the same way, must match the
  checkpoint's dequantized one, and must not match it unpermuted.
- **AdaLN.** The pruned checkpoint's `adaln_proj.linear` reads an 8-column
  time basis (`adaln_t_table`), not the 2688-wide `silu(time_embedder(t))`.
  The LoRA's time curve over the release embedder's 1025-row grid is
  least-squares fitted onto `[adaln_t_table, 1]`: the basis part becomes the
  new `lora_A`, the constant part a `.diff_b`. The fit is refused above 1e-3
  relative. The table is the partition's own, so fl2va and ref2va differ.
- **fc1, fc2, out_proj.** Renamed only; fc1's order is checked.

Needs `torch`, `safetensors`, and `comfy_kitchen` (installed with ComfyUI) to
dequantize the int8 convrot weights for the layout check. CPU only.

    python convert_flashgen_lora.py \\
        --lora minimax_h3_4step_lora_flashgen_v1.0_768p_bf16.safetensors \\
        --release <MiniMaxAI/MiniMax-H3 download> \\
        --pruned minimax_h3_fl2va_pruned_int8_convrot.safetensors \\
        --partition FL2VA \\
        --out minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy.safetensors
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

A_SUF, B_SUF = ".lora_A.default.weight", ".lora_B.default.weight"
HEAD_DIM = 128          # the release config's attention_head_dim
GRID_ROWS = 1025        # adaln_t_table's first dimension
ADALN_FIT_BOUND = 1e-3
#: The publisher's base_schedule (merge_lora_ckpt.py DEFAULT_BASE_SCHEDULE),
#: and the sigmas it gives at H3's video shift 12, written to the metadata.
BASE_SCHEDULE = (1.0, 0.7, 0.4, 0.15, 0.0)
MANUAL_SIGMAS_SHIFT12 = "1.0, 0.965517, 0.888889, 0.679245, 0.0"


def shifted(shift, t):
    return shift * t / (1 + (shift - 1) * t)


def silu_temb_grid(proj_in_w, proj_in_b, proj_out_w, proj_out_b, rows):
    """`silu(TimeEmbedder(t))` over `linspace(0, 1, rows)`, from raw tensors.
    ComfyUI's `comfy/ldm/minimax/model.py::TimeEmbedder.forward`: cos before
    sin, fp32 throughout."""
    half = proj_in_w.shape[1] // 2
    t = torch.linspace(0.0, 1.0, rows, dtype=torch.float32)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    h = torch.nn.functional.linear(emb, proj_in_w.float(), proj_in_b.float())
    o = torch.nn.functional.linear(torch.nn.functional.silu(h), proj_out_w.float(), proj_out_b.float())
    return torch.nn.functional.silu(o)


def release_tensor(release: Path, partition: str, key: str) -> torch.Tensor:
    d = release / partition / "transformer"
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(d / idx[key]), "pt") as f:
        return f.get_tensor(key).float()


def dequantised(handle, keys, module) -> torch.Tensor:
    w = handle.get_tensor(f"{module}.weight")
    if f"{module}.weight_scale" not in keys:
        return w.float()
    cfg = json.loads(bytes(handle.get_tensor(f"{module}.comfy_quant").tolist()).decode())
    if cfg.get("format") != "int8_tensorwise" or not cfg.get("convrot"):
        raise SystemExit(f"{module}: quantised as {cfg}; this converter does not dequantise that")
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
    return dequantize_int8_convrot_weight(w, handle.get_tensor(f"{module}.weight_scale"),
                                          int(cfg["convrot_groupsize"])).float()


def qkv_to_bands(x: torch.Tensor, heads: int) -> torch.Tensor:
    """Rows [head, qkv, dim] -> [qkv, head, dim]."""
    rest = x.shape[1:]
    return x.reshape(heads, 3, HEAD_DIM, *rest).transpose(0, 1).reshape(3 * heads * HEAD_DIM, *rest)


def row_cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).median())


def main() -> int:
    ap = argparse.ArgumentParser(description="FlashGen, converted at full rank for a pruned int8 H3 checkpoint.")
    ap.add_argument("--lora", type=Path, required=True, help="the publisher's PEFT safetensors")
    ap.add_argument("--release", type=Path, required=True, help="the MiniMaxAI/MiniMax-H3 download")
    ap.add_argument("--pruned", type=Path, required=True, help="the pruned int8 convrot checkpoint")
    ap.add_argument("--partition", choices=("FL2VA", "Ref2VA"), default="FL2VA",
                    help="the release partition --pruned was cut from")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    part = args.partition

    with safe_open(str(args.lora), "pt") as f:
        src = {k: f.get_tensor(k) for k in f.keys()}
    modules = sorted({k[: -len(A_SUF)] for k in src if k.endswith(A_SUF)})
    stray = [k for k in src if not (k.endswith(A_SUF) or k.endswith(B_SUF))]
    if stray or any(m + B_SUF not in src for m in modules):
        raise SystemExit(f"unpaired or unexpected tensors: {stray[:4]}")
    ranks = {src[m + A_SUF].shape[0] for m in modules}
    if len(ranks) != 1:
        raise SystemExit(f"mixed ranks {sorted(ranks)}")
    rank = ranks.pop()

    pruned = safe_open(str(args.pruned), "pt")
    pkeys = set(pruned.keys())
    table = pruned.get_tensor("adaln_t_table").to(torch.float64)          # [1025, 8]
    heads = pruned.get_slice("blocks.0.attn.qkv_proj.weight").get_shape()[0] // (3 * HEAD_DIM)
    checks = {}

    # 1. layout, on the weights
    layout = {}
    for mod in ("blocks.0.attn.qkv_proj", "blocks.49.attn.qkv_proj", "token_refiner.blocks.0.attn.qkv_proj"):
        rel, ck = release_tensor(args.release, part, f"{mod}.weight"), dequantised(pruned, pkeys, mod)
        layout[mod] = {"permuted": row_cos(qkv_to_bands(rel, heads), ck), "as_stored": row_cos(rel, ck)}
        if not (layout[mod]["permuted"] > 0.99 and layout[mod]["as_stored"] < 0.9):
            raise SystemExit(f"{mod}: qkv layout is not the per-head interleave assumed: {layout[mod]}")
    rel = release_tensor(args.release, part, "blocks.0.mlp.fc1.weight")
    layout["blocks.0.mlp.fc1"] = {"as_stored": row_cos(rel, dequantised(pruned, pkeys, "blocks.0.mlp.fc1"))}
    if layout["blocks.0.mlp.fc1"]["as_stored"] < 0.99:
        raise SystemExit(f"fc1 is not stored in ComfyUI's order: {layout['blocks.0.mlp.fc1']}")
    checks["layout_median_row_cosine"] = layout

    # 2. the time grid from the release's own time embedder, and the adaln fit
    te = {k: release_tensor(args.release, part, f"time_embedder.{k}") for k in
          ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")}
    grid = silu_temb_grid(te["proj_in.weight"], te["proj_in.bias"], te["proj_out.weight"],
                          te["proj_out.bias"], rows=GRID_ROWS).to(torch.float64)
    design = torch.cat([table, torch.ones(table.shape[0], 1, dtype=torch.float64)], dim=1)

    out, worst_fit = {}, 0.0
    for m in modules:
        a, b = src[m + A_SUF], src[m + B_SUF]
        dst = f"diffusion_model.{m}"
        if m.endswith("attn.qkv_proj"):
            b = qkv_to_bands(b, heads)
        if m.endswith("adaln_proj.linear"):
            if f"{m}.weight" not in pkeys or pruned.get_slice(f"{m}.weight").get_shape()[1] != table.shape[1]:
                raise SystemExit(f"{m}: the checkpoint's adaln is not on the {table.shape[1]}-column basis")
            curve = grid @ a.to(torch.float64).T                                 # [rows, rank]
            coef = torch.linalg.lstsq(design, curve).solution                    # [9, rank]
            true = curve @ b.to(torch.float64).T
            got = (design @ coef) @ b.to(torch.float64).T
            err = float((got - true).norm() / true.norm())
            worst_fit = max(worst_fit, err)
            if err > ADALN_FIT_BOUND:
                raise SystemExit(f"{m}: adaln delta does not fit the basis (rel {err:.2e})")
            a = coef[:-1].T.to(torch.float32)                                    # [rank, 8]
            out[f"{dst}.diff_b"] = (b.to(torch.float64) @ coef[-1]).to(torch.float32)
        if f"{m}.weight" not in pkeys:
            raise SystemExit(f"the checkpoint has no {m}.weight")
        out[f"{dst}.lora_A.weight"] = a.contiguous()
        out[f"{dst}.lora_B.weight"] = b.contiguous()
        out[f"{dst}.alpha"] = torch.tensor(float(rank))
    checks["adaln_worst_relative_fit"] = worst_fit

    # 3. the emitted delta against the source delta
    recon = {}
    for m in ("blocks.7.attn.qkv_proj", "blocks.7.mlp.fc1", "blocks.7.attn.out_proj", "blocks.7.mlp.fc2"):
        s = src[m + B_SUF].float() @ src[m + A_SUF].float()
        if m.endswith("qkv_proj"):
            s = qkv_to_bands(s, heads)
        o = out[f"diffusion_model.{m}.lora_B.weight"].float() @ out[f"diffusion_model.{m}.lora_A.weight"].float()
        recon[m] = float((o - s).norm() / s.norm())
        if recon[m] > 1e-6:
            raise SystemExit(f"{m}: emitted delta differs from the source ({recon[m]:.2e})")
    checks["reconstruction_relative"] = recon

    sig = [shifted(12.0, t) for t in BASE_SCHEDULE]
    if any(abs(x - float(y)) > 1e-6 for x, y in zip(sig, MANUAL_SIGMAS_SHIFT12.split(","))):
        raise SystemExit(f"the publisher's schedule at shift 12 is {sig}, not {MANUAL_SIGMAS_SHIFT12}")
    meta = {
        "format": "pt",
        "source": f"Beidouqixing/minimax-h3-4step-lora-flashgen {args.lora.name}",
        "conversion": ("bench/convert_flashgen_lora.py: full rank; qkv lora_B rows [head,qkv,dim]->[qkv,head,dim]; "
                       f"adaln lora_A projected onto the pruned {part.lower()} adaln_t_table basis with the mean in diff_b; "
                       "alpha = rank (publisher scale 1.0)"),
        "adaln_curve_basis": part.lower(),
        "sampler_steps": "4",
        "manual_sigmas_shift12": MANUAL_SIGMAS_SHIFT12,
        "base_schedule": ",".join(str(x) for x in BASE_SCHEDULE),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out), metadata=meta)
    print(json.dumps(checks, indent=1))
    print(f"wrote {args.out.name}: {len(modules)} modules at rank {rank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
