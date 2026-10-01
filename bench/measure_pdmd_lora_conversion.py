#!/usr/bin/env python3
"""How faithful kijai's rank-reduced PDMD LoRA is to the published one, and whether its layout is right.

`pdmd2026/pdmd_{4,2}NFE_lora` ship `lora_model_0.safetensors`: rank-128 pairs
on the diffusers `MiniMaxH3Transformer3DModel` (`transformer.<module>.lora_A/B`,
scale 1.0, no alpha tensor). kijai's ComfyUI files
(`minimax_h3_pdmd_{4,2}step_lora_avg_rank_*_bf16`) fuse q/k/v block-diagonally
into `attn.qkv_proj`, swap the SwiGLU halves of `mlp.fc1`, and then resize
every module by SVD at `sv_fro 0.97`, capped at rank 128 (the file metadata).

Two measurements, CPU only:

1. **Layout, on the base weights.** The diffusers release (`transformer/`) mapped
   the way kijai describes -- `cat([to_q; to_k; to_v])`, `ff.net.0.proj` halves
   swapped -- must match the pruned fl2va checkpoint's dequantised weights row
   for row. Two controls must NOT match: `fc1` unswapped, and the ref2va
   checkpoint (whose backbone differs from fl2va, `docs/research/h3_partition_distance.md`).
   This is what says the mapping is right, independent of kijai.
   The release's own partitions settle which one `transformer/` is, exactly:
   its renamed modules against `FL2VA/transformer` and `Ref2VA/transformer`.
2. **Fidelity, every module.** The published delta, mapped as above, against
   kijai's delta (`B @ A * alpha / rank`): cosine, norm ratio, relative error.
   Computed from the low-rank factors (`<B1 A1, B2 A2> = sum((B1^T B2) * (A1 A2^T))`),
   so no full-size delta is formed.

    CUDA_VISIBLE_DEVICES="" <comfy venv python> bench/measure_pdmd_lora_conversion.py \\
        --lora <pdmd2026>/pdmd_4NFE_lora/lora_model_0.safetensors \\
        --kijai <models>/loras/h3/minimax_h3_pdmd_4step_lora_avg_rank_57_bf16.safetensors \\
        --release <MiniMaxAI_MiniMax-H3> \\
        --pruned <models>/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \\
        --pruned-control <models>/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \\
        --out bench/results/<date>_pdmd_lora_conversion.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]

#: Blocks the layout check reads. **Reasoned**: first, middle, last, and one
#: token-refiner block, as in `bench/probe_int8_lora_requant.py`.
LAYOUT_BLOCKS = ("blocks.0", "blocks.24", "blocks.49", "token_refiner.blocks.0")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def comfy_name(diffusers_block: str) -> str:
    """`transformer_blocks.N` / `token_refiner.refiner_blocks.N` -> ComfyUI's block name."""
    m = re.fullmatch(r"transformer_blocks\.(\d+)", diffusers_block)
    if m:
        return f"blocks.{m.group(1)}"
    m = re.fullmatch(r"token_refiner\.refiner_blocks\.(\d+)", diffusers_block)
    if m:
        return f"token_refiner.blocks.{m.group(1)}"
    raise SystemExit(f"unmapped block {diffusers_block}")


def diffusers_name(comfy_block: str) -> str:
    if comfy_block.startswith("blocks."):
        return "transformer_blocks." + comfy_block.split(".")[1]
    return "token_refiner.refiner_blocks." + comfy_block.split(".")[2]


def swap_halves(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[0] // 2
    return torch.cat([x[h:], x[:h]], dim=0)


def dequantised(handle, keys, module) -> torch.Tensor:
    """Same rule as `bench/convert_flashgen_lora.py::dequantised`."""
    w = handle.get_tensor(f"{module}.weight")
    if f"{module}.weight_scale" not in keys:
        return w.float()
    cfg = json.loads(bytes(handle.get_tensor(f"{module}.comfy_quant").tolist()).decode())
    if cfg.get("format") != "int8_tensorwise" or not cfg.get("convrot"):
        raise SystemExit(f"{module}: quantised as {cfg}; not dequantised here")
    sys.path.append(str(COMFY))
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
    return dequantize_int8_convrot_weight(w, handle.get_tensor(f"{module}.weight_scale"),
                                          int(cfg["convrot_groupsize"])).float()


def row_cos(a, b) -> float:
    return round(float(torch.nn.functional.cosine_similarity(a, b, dim=1).median()), 6)


class Release:
    def __init__(self, root: Path):
        self.d = root / "transformer"
        self.idx = json.loads((self.d / "diffusion_pytorch_model.safetensors.index.json")
                              .read_text())["weight_map"]

    def get(self, key):
        with safe_open(str(self.d / self.idx[key]), "pt") as f:
            return f.get_tensor(key).float()


def layout_check(release: Release, pruned: Path, control: Path | None) -> dict:
    out = {}
    handles = {"fl2va": pruned}
    if control:
        handles["ref2va_control"] = control
    for label, path in handles.items():
        with safe_open(str(path), "pt") as f:
            keys = set(f.keys())
            rows = {}
            for blk in LAYOUT_BLOCKS:
                src = diffusers_name(blk)
                q, k, v = (release.get(f"{src}.attn.to_{x}.weight") for x in "qkv")
                ck_qkv = dequantised(f, keys, f"{blk}.attn.qkv_proj")
                ff = release.get(f"{src}.ff.net.0.proj.weight")
                ck_fc1 = dequantised(f, keys, f"{blk}.mlp.fc1")
                rows[blk] = {
                    "qkv_cat": row_cos(torch.cat([q, k, v]), ck_qkv),
                    "fc1_swapped": row_cos(swap_halves(ff), ck_fc1),
                    "fc1_as_stored": row_cos(ff, ck_fc1),
                    "out_proj": row_cos(release.get(f"{src}.attn.to_out.0.weight"),
                                        dequantised(f, keys, f"{blk}.attn.out_proj")),
                    "fc2": row_cos(release.get(f"{src}.ff.net.2.weight"),
                                   dequantised(f, keys, f"{blk}.mlp.fc2")),
                }
            out[label] = {"checkpoint": path.name, "median_row_cosine": rows}
    return out


def partition_check(release: Release, root: Path) -> dict:
    """Max |diff| of diffusers `transformer/` against each native partition, on renamed modules.

    The PDMD README calls `transformer/` "the FL2VA/T2VA partition"; this says so
    on bits. `out_proj` and `fc2` need no reordering, so equality is exact.
    """
    out = {}
    for part in ("FL2VA", "Ref2VA"):
        d = root / part / "transformer"
        idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        rows = {}
        for blk in ("blocks.0", "blocks.49"):
            for native, diff in (("attn.out_proj", "attn.to_out.0"), ("mlp.fc2", "ff.net.2")):
                with safe_open(str(d / idx[f"{blk}.{native}.weight"]), "pt") as f:
                    w = f.get_tensor(f"{blk}.{native}.weight").float()
                rows[f"{blk}.{native}"] = float((w - release.get(f"{diffusers_name(blk)}.{diff}.weight"))
                                                .abs().max())
        out[part] = {"max_abs_diff": rows}
    return out


def published_factors(lora: dict, blk: str, kind: str, swap_fc1: bool = True):
    """(B, A) of the published delta in ComfyUI's layout, as low-rank factors."""
    p = f"transformer.{diffusers_name(blk)}"
    ab = lambda m: (lora[f"{p}.{m}.lora_B.weight"].float(), lora[f"{p}.{m}.lora_A.weight"].float())
    if kind == "attn.qkv_proj":
        bs, as_ = zip(*(ab(f"attn.to_{x}") for x in "qkv"))
        return torch.block_diag(*bs), torch.cat(as_)
    if kind == "mlp.fc1":
        b, a = ab("ff.net.0.proj")
        return (swap_halves(b) if swap_fc1 else b), a
    return ab({"attn.out_proj": "attn.to_out.0", "mlp.fc2": "ff.net.2"}[kind])


def inner(b1, a1, b2, a2) -> float:
    return float(((b1.T @ b2) * (a1 @ a2.T)).sum())


def compare(b1, a1, b2, a2) -> dict:
    n1, n2, x = inner(b1, a1, b1, a1), inner(b2, a2, b2, a2), inner(b1, a1, b2, a2)
    return {"cosine": round(x / (n1 * n2) ** 0.5, 6),
            "norm_ratio_kijai_over_published": round((n2 / n1) ** 0.5, 6),
            "rel_err": round(max(n1 + n2 - 2 * x, 0.0) ** 0.5 / n1 ** 0.5, 6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--kijai", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--pruned", type=Path, required=True)
    ap.add_argument("--pruned-control", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    with safe_open(str(args.lora), "pt") as f:
        lora_meta = f.metadata() or {}
        lora = {k: f.get_tensor(k) for k in f.keys()}
    if lora_meta.get("lora_scale") != "1.0":
        raise SystemExit(f"published lora_scale {lora_meta.get('lora_scale')!r}; this assumes 1.0")
    with safe_open(str(args.kijai), "pt") as f:
        kj_meta = f.metadata() or {}
        kj = {k: f.get_tensor(k) for k in f.keys()}

    release = Release(args.release)
    modules = sorted({k[len("diffusion_model."):-len(".alpha")] for k in kj if k.endswith(".alpha")})
    per_module, wrong_swap = {}, {}
    for mod in modules:
        m = re.fullmatch(r"(.*\.\d+)\.(attn\.\w+|mlp\.\w+)", mod)
        if m is None:
            raise SystemExit(f"unexpected module {mod}")
        blk, kind = m.groups()
        b1, a1 = published_factors(lora, blk, kind)
        a2 = kj[f"diffusion_model.{mod}.lora_A.weight"].float()
        b2 = kj[f"diffusion_model.{mod}.lora_B.weight"].float()
        rank = a2.shape[0]
        b2 = b2 * (float(kj[f"diffusion_model.{mod}.alpha"]) / rank)
        row = compare(b1, a1, b2, a2)
        row["kijai_rank"] = rank
        row["published_rank"] = a1.shape[0]
        per_module[mod] = row
        if kind == "mlp.fc1" and blk in LAYOUT_BLOCKS:
            wb, wa = published_factors(lora, blk, kind, swap_fc1=False)
            wrong_swap[mod] = compare(wb, wa, b2, a2)["cosine"]

    by_kind = {}
    for mod, row in per_module.items():
        kind = mod.split(".", 2 if mod.startswith("blocks") else 3)[-1]
        key = ("refiner." if mod.startswith("token_refiner") else "") + kind
        by_kind.setdefault(key, []).append(row)
    summary = {k: {"n": len(v),
                   "cosine_median": round(statistics.median(r["cosine"] for r in v), 4),
                   "cosine_min": round(min(r["cosine"] for r in v), 4),
                   "rel_err_median": round(statistics.median(r["rel_err"] for r in v), 4),
                   "rel_err_max": round(max(r["rel_err"] for r in v), 4),
                   "rank_mean": round(statistics.mean(r["kijai_rank"] for r in v), 1),
                   "at_rank_cap": sum(r["kijai_rank"] >= 128 for r in v)}
               for k, v in sorted(by_kind.items())}
    worst = sorted(per_module.items(), key=lambda kv: kv[1]["cosine"])[:8]

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    record = {
        "script": "bench/measure_pdmd_lora_conversion.py",
        "commit": commit,
        "device": "cpu (stored weights only; no render, no cache state)",
        # `source` is the trainer's cluster path, which names a person's home
        # directory; `tag` already names the run and step.
        "published": {"file": str(args.lora.name), "parent": args.lora.parent.name,
                      "sha256": sha256(args.lora),
                      "metadata": {k: v for k, v in lora_meta.items() if k != "source"}},
        "kijai": {"file": args.kijai.name, "sha256": sha256(args.kijai), "metadata": kj_meta},
        "release": args.release.name,
        "pruned": args.pruned.name,
        "partition_of_diffusers_transformer": partition_check(release, args.release),
        "layout_on_base_weights": layout_check(release, args.pruned, args.pruned_control),
        "fc1_unswapped_control_cosine": wrong_swap,
        "fidelity_by_kind": summary,
        "fidelity_worst_modules": dict(worst),
        "fidelity_per_module": per_module,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("partition_of_diffusers_transformer", "layout_on_base_weights", "fc1_unswapped_control_cosine",
                                             "fidelity_by_kind", "fidelity_worst_modules")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
