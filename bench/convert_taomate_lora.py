#!/usr/bin/env python3
"""Convert the TaoMate-H3 adapter to a ComfyUI LoRA, checking every mapping it can on the files.

TaoLiveAIGC/TaoMate-H3 ships a LoRA over MiniMax H3's merged linears:
`adapter_model.safetensors` holding `lora_a`/`lora_b`, rank and alpha in
`adapter_config.json`. The ComfyUI copies in circulation are either a straight
rename or an SVD resize with no published recipe. This converter writes the
authors' adapter at its own rank, says why each part of the mapping is right,
asserts what the files can show, and with `--compare` measures how far another
ComfyUI file's deltas sit from it.

## The mapping, and the evidence for each part

A rename: `{target}.lora_a` -> `diffusion_model.{target}.lora_A.weight`,
`lora_b` -> `lora_B.weight`, and one `.alpha` F32 scalar per module, because
ComfyUI reads alpha from a tensor and never from `__metadata__`
(`bench/check_lora_alpha.py` carries the reason). The key map is built from the
model's side: `comfy/lora.py::model_lora_keys_unet` strips `.weight` from each
model key `diffusion_model.<module>.weight` to get a LoRA prefix, and
`comfy/weight_adapter/lora.py::LoRAAdapter.load` then looks up
`<prefix>.lora_B.weight`, `.lora_A.weight` and `.alpha` under it. So the
target names must be the checkpoint's own names;
`checkpoint_inventory` asserts they are, same set and same shapes.

Three transforms `bench/convert_pdd_lora.py` needs are deliberately absent:

- **No q/k/v fuse.** The adapter already targets the merged `attn.qkv_proj`;
  its `lora_b` spans the whole merged output.
Source paths below are inside the upstream tree at
`h3_config.TAOMATE_UPSTREAM`, the revision they were read at. Nothing here
reads that tree: the values this file needs from it (the distilled grid it
writes into the output's metadata) are copied into `h3_config` beside that
pointer.

- **No `qkv_proj` row reorder.** Two legs, and neither proves the other. The
  adapter's: TaoMate reorders the release's per-head grouped qkv weight into
  q|k|v row bands at load
  (`src/taomate_h3/model/weight_loading.py::reorder_grouped_qkv_to_qkv`) and
  installs the LoRA on that reordered module
  (`src/taomate_h3/inference/lora_checkpoint.py::apply_h3_lora_checkpoint`),
  so `lora_b` rows are bands. A source read. Core's:
  `comfy/ldm/minimax/model.py::Attention.forward` splits the qkv output into
  bands, and `anchor_qkv_bands` asserts it on a file that renders correctly
  in ComfyUI on this box, the fl2va PDD LoRA (`h3_config.PDD_FL2VA_LORA`),
  whose fused `lora_B` must be block-diagonal over the three row bands. The
  anchor shows how core reads a fused LoRA; it says nothing about TaoMate's
  rows. `--release` adds the leg that joins them, on weights (below).
- **No SwiGLU half swap.** TaoMate reads `fc1` as `[gate; up]` on both of its
  paths (`src/taomate_h3/model/layers.py::MiniMaxH3MLP.forward` chunks
  `gate, up`; the triton `_swiglu_kernel` in
  `src/taomate_h3/inference/fused_kernels.py` loads the gate from the first
  half), and core's `comfy/ops.py::_swiglu_eager` chunks the same way. The PDD
  swap exists because diffusers stores `[value; gate]`; this adapter was never
  in diffusers naming. Source reads, and on weights with `--release` (below).

**The delta's statistics cannot settle either order; the base weights can.**
Row norms, within-band correlations and row directions of the delta against
the base weights were each tried on 2026-09-15 and read at chance: a rank-128
delta's rows carry no trace of which base row they sit on. Comparing base to
base does settle it. With `--release` pointing at the `MiniMaxAI/MiniMax-H3`
download, `release_layout` dequantises the checkpoint's own qkv and fc1 at
`RELEASE_PROBE_MODULES`. It then asserts that each matches the release row
for row under exactly one layout: qkv under TaoMate's reorder of the per-head
interleaved release into bands, and fc1 as stored. That shows the tensor core
loads is the tensor TaoMate installed its LoRA on. Both engines run that base
correctly, so they read its rows the same way, and so the LoRA's rows line up
too. A render with the fc1 halves deliberately swapped was meant as a
functional control and cannot be one. fc1's delta is too small against the
base weight for misplaced rows to break the model, and its render came out
coherent (`h3_config.TAOMATE_SWAPPED_CONTROL_LORA` has the pointer).
`--swap-fc1-halves`
writes that file (`h3_config.TAOMATE_SWAPPED_CONTROL_LORA`) and stamps it as a
control in its metadata.

## What `--compare` measures

Per module, the delta `scale * B @ A` of each file, compared without
materialising either matrix: `<B1 A1, B2 A2>_F = sum((B1^T B2) * (A1 A2^T))`.
It reports the relative Frobenius error against this adapter, the cosine, and
the other delta's energy as a fraction of this one's. Because both files come
from the same adapter, the same factored comparison can also score the other
file as if its qkv rows were grouped or its fc1 halves swapped: a conversion
that took a different layout from this one scores better under that
hypothesis. Diagnostics, not gates. The cast to `--dtype` is measured the same
way.

A delta distance is a weight-space statement. It says nothing about a render,
which a person judges (`docs/eval_comparison.md`).

Needs torch and safetensors, CPU only. Exit codes: 0 converted or verified,
1 a verification failed (nothing is written).

    <comfy venv python> bench/convert_taomate_lora.py \\
        --adapter <TaoMate-H3 adapter dir> \\
        --out <ComfyUI models>/loras/h3/<name>.safetensors \\
        [--compare <another ComfyUI conversion>] [--record bench/results/<date>_<name>.json]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# custom_nodes/<this pack>/bench -> the ComfyUI root, derived from where this
# file sits, as `bench/check_distill_grid.py` does.
COMFY = REPO.parent.parent
sys.path.insert(0, str(REPO / "workflows"))

import h3_config  # noqa: E402

CONVERTER_VERSION = 3
KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
# Reasoned: a block-diagonal fusion leaves only dtype rounding off the three
# diagonal blocks, while a grouped or band-mixed layout puts about two thirds
# of the energy there. Any value far between the two separates them.
ANCHOR_OFF_BLOCK_MAX = 1e-3
# Reasoned: an int8_convrot row dequantises to within rounding of its source,
# so its cosine with the right release row sits just under one, and against an
# unrelated row it sits near zero. Bounds far inside that gap separate layouts.
RELEASE_MATCH_MIN = 0.999
RELEASE_OTHER_MAX = 0.1
# Reasoned: the layout comes from one conversion applied to every module, so
# both ends of the stack and the text refiner catch a per-block path; checking
# every module costs a dequantisation each and adds nothing a layout can hide.
RELEASE_PROBE_MODULES = ("blocks.0", "blocks.25", "blocks.49", "token_refiner.blocks.0")


def distilled_grid() -> dict:
    """The adapter's sampling contract, written into the output's metadata so
    the file says how to run it. From the inherited copy in `h3_config`, whose
    pointer names the upstream source; nothing here reads that source."""
    shift_v = h3_config.TAOMATE_SHIFT["shift_video"]
    shift_a = h3_config.TAOMATE_SHIFT["shift_audio"]
    return {
        "upstream": h3_config.TAOMATE_UPSTREAM,
        "grid_points": h3_config.TAOMATE_GRID_POINTS,
        "state_indices": list(h3_config.TAOMATE_STATE_INDICES),
        "shift": [shift_v, shift_a],
        "sigmas_video": h3_config.taomate_sigmas(shift_v),
        "sigmas_audio": h3_config.taomate_sigmas(shift_a),
        "sampler": h3_config.TAOMATE_SAMPLER,
        "strength": h3_config.TAOMATE_STRENGTH,
    }


class VerificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summarise(values: list[float]) -> dict:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


# --------------------------------------------------------------------------
# the adapter
# --------------------------------------------------------------------------

def read_adapter(adapter_dir: Path):
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        config_path = adapter_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank, alpha = config.get("rank"), config.get("alpha")
    if type(rank) is not int or rank <= 0 or not isinstance(alpha, (int, float)):
        raise VerificationError(f"{config_path.name}: rank/alpha are not usable: {config}")
    weights = adapter_dir / "adapter_model.safetensors"
    with safe_open(str(weights), "pt") as handle:
        file_meta = dict(handle.metadata() or {})
    state = load_file(str(weights), device="cpu")
    return config, rank, float(alpha), weights, file_meta, state


def adapter_inventory(state: dict, rank: int) -> list[str]:
    """Every target, each with both factors at the declared rank, nothing else."""
    targets = sorted({key.rsplit(".", 1)[0] for key in state})
    problems = []
    expected_keys = set()
    for target in targets:
        if not target.endswith(KINDS):
            problems.append(f"{target}: not one of {KINDS}")
            continue
        a, b = state.get(f"{target}.lora_a"), state.get(f"{target}.lora_b")
        expected_keys.update((f"{target}.lora_a", f"{target}.lora_b"))
        if a is None or b is None:
            problems.append(f"{target}: needs both lora_a and lora_b")
        elif a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            problems.append(f"{target}: factor shapes {tuple(a.shape)}, "
                            f"{tuple(b.shape)} do not carry rank {rank}")
    leftover = sorted(set(state) - expected_keys)
    if leftover:
        problems.append(f"{len(leftover)} tensors are not LoRA factors, e.g. {leftover[:3]}")
    if problems:
        raise VerificationError("adapter inventory: " + "; ".join(problems))
    return targets


# --------------------------------------------------------------------------
# the checkpoint it loads on, and the file that shows core's qkv layout
# --------------------------------------------------------------------------

def checkpoint_inventory(checkpoint: Path, state: dict, targets: list[str]) -> dict:
    """The checkpoint's own linears of these kinds are exactly the targets, shape for shape."""
    with safe_open(str(checkpoint), "pt") as handle:
        keys = set(handle.keys())
        linears = sorted(
            key[: -len(".weight")] for key in keys
            if key.endswith(".weight")
            and key[: -len(".weight")].endswith(KINDS)
            and key.startswith(("blocks.", "token_refiner."))
        )
        problems = []
        missing = sorted(set(linears) - set(targets))
        extra = sorted(set(targets) - set(linears))
        if missing:
            problems.append(f"{len(missing)} checkpoint linears have no adapter module, e.g. {missing[:3]}")
        if extra:
            problems.append(f"{len(extra)} adapter modules have no checkpoint linear, e.g. {extra[:3]}")
        for target in sorted(set(targets) & set(linears)):
            have = list(handle.get_slice(f"{target}.weight").get_shape())
            want = [state[f"{target}.lora_b"].shape[0], state[f"{target}.lora_a"].shape[1]]
            if have != want:
                problems.append(f"{target}: checkpoint weight {have}, adapter delta {want}")
        if problems:
            raise VerificationError("checkpoint inventory: " + "; ".join(problems))
        head_dim = int(handle.get_slice("blocks.0.attn.q_norm.weight").get_shape()[0])
        qkv_rows = int(handle.get_slice("blocks.0.attn.qkv_proj.weight").get_shape()[0])
    if qkv_rows % (3 * head_dim):
        raise VerificationError(f"qkv rows {qkv_rows} are not 3 * heads * {head_dim}")
    return {"linears": len(linears), "head_dim": head_dim, "heads": qkv_rows // (3 * head_dim)}


def anchor_qkv_bands(anchor: Path) -> dict:
    """Core's qkv layout, shown by a fused LoRA that renders: B block-diagonal over row bands."""
    fractions = {}
    with safe_open(str(anchor), "pt") as handle:
        for key in handle.keys():
            if not key.endswith("attn.qkv_proj.lora_B.weight"):
                continue
            b = handle.get_tensor(key).double()
            rows, cols = b.shape
            if rows % 3 or cols % 3:
                raise VerificationError(f"anchor {key} {tuple(b.shape)} is not a three-band fusion")
            band, rank = rows // 3, cols // 3
            total = float(b.pow(2).sum())
            on = sum(float(b[p * band:(p + 1) * band, p * rank:(p + 1) * rank].pow(2).sum())
                     for p in range(3))
            fractions[key] = (total - on) / total
    if not fractions:
        raise VerificationError(f"anchor {anchor.name} has no fused qkv_proj lora_B")
    worst = max(fractions, key=lambda name: fractions[name])
    if fractions[worst] > ANCHOR_OFF_BLOCK_MAX:
        raise VerificationError(
            f"anchor {anchor.name}: {worst} carries {fractions[worst]:.3g} of its energy off "
            f"the band blocks, so it does not show core reading qkv as row bands")
    return {"file": anchor.name, "modules": len(fractions),
            "off_block_energy_fraction_max": fractions[worst],
            "off_block_energy_fraction_bound": ANCHOR_OFF_BLOCK_MAX}


def _dequantised(handle, keys: set, module: str) -> torch.Tensor:
    """A checkpoint linear as float32, dequantised when it is int8_convrot.

    The group size is the module's own `comfy_quant` config, not a constant,
    so a checkpoint built at another size still reads correctly."""
    weight = handle.get_tensor(f"{module}.weight")
    if f"{module}.weight_scale" not in keys:
        return weight.float()
    config = json.loads(bytes(handle.get_tensor(f"{module}.comfy_quant").tolist()).decode())
    if config.get("format") != "int8_tensorwise" or not config.get("convrot"):
        raise VerificationError(f"{module}: quantised as {config}, which this check does not dequantise")
    if str(COMFY) not in sys.path:
        sys.path.append(str(COMFY))
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
    return dequantize_int8_convrot_weight(weight, handle.get_tensor(f"{module}.weight_scale"),
                                          int(config["convrot_groupsize"])).float()


def _row_cosine_median(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).median())


def release_layout(release: Path, checkpoint: Path, heads: int, head_dim: int) -> dict:
    """The checkpoint's qkv and fc1 rows against the release's, under each candidate layout.

    The release interleaves q, k and v per head; TaoMate reorders that into
    row bands before installing its LoRA
    (`src/taomate_h3/model/weight_loading.py::_local_tensor`, one query group
    per head) and leaves fc1 as stored. Each module must match the release
    under exactly that layout and under no other."""
    transformer = release / "FL2VA" / "transformer"
    index = transformer / "model.safetensors.index.json"
    if not index.is_file():
        raise VerificationError(f"--release has no FL2VA/transformer/{index.name}")
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]

    def released(key: str) -> torch.Tensor:
        if key not in weight_map:
            raise VerificationError(f"release has no {key}")
        with safe_open(str(transformer / weight_map[key]), "pt") as shard:
            return shard.get_tensor(key).float()

    modules, problems = {}, []
    with safe_open(str(checkpoint), "pt") as handle:
        keys = set(handle.keys())
        for prefix in RELEASE_PROBE_MODULES:
            qkv, fc1 = f"{prefix}.attn.qkv_proj", f"{prefix}.mlp.fc1"
            source, stored = released(f"{qkv}.weight"), _dequantised(handle, keys, qkv)
            if source.shape != stored.shape or source.shape[0] != 3 * heads * head_dim:
                problems.append(f"{qkv}: release {tuple(source.shape)}, checkpoint "
                                f"{tuple(stored.shape)}, not one q, k and v per head")
                continue
            q, k, v = torch.split(source.reshape(heads, 3 * head_dim, -1), [head_dim] * 3, dim=1)
            bands = torch.cat([t.reshape(heads * head_dim, -1) for t in (q, k, v)])
            modules[qkv] = ("grouped_to_bands",
                            {"as_stored": _row_cosine_median(source, stored),
                             "grouped_to_bands": _row_cosine_median(bands, stored)})
            source, stored = released(f"{fc1}.weight"), _dequantised(handle, keys, fc1)
            half = source.shape[0] // 2
            modules[fc1] = ("as_stored",
                            {"as_stored": _row_cosine_median(source, stored),
                             "halves_swapped": _row_cosine_median(
                                 torch.cat([source[half:], source[:half]]), stored)})
    for module, (want, scores) in modules.items():
        others = [value for name, value in scores.items() if name != want]
        if scores[want] < RELEASE_MATCH_MIN or max(others) > RELEASE_OTHER_MAX:
            problems.append(f"{module}: row cosine medians {scores} do not single out {want}")
    if problems:
        raise VerificationError("release layout: " + "; ".join(problems))
    return {"release": "MiniMaxAI/MiniMax-H3 FL2VA/transformer",
            "release_index_sha256": sha256(index),
            "match_min": RELEASE_MATCH_MIN, "other_max": RELEASE_OTHER_MAX,
            "modules": {module: {"expected": want, "row_cosine_median": scores}
                        for module, (want, scores) in modules.items()}}


# --------------------------------------------------------------------------
# deltas, compared in factored form
# --------------------------------------------------------------------------

def delta_stats(b1, a1, s1, b2, a2, s2) -> dict:
    b1, a1, b2, a2 = (t.double() for t in (b1, a1, b2, a2))
    n1 = s1 * s1 * float(((b1.T @ b1) * (a1 @ a1.T)).sum())
    n2 = s2 * s2 * float(((b2.T @ b2) * (a2 @ a2.T)).sum())
    inner = s1 * s2 * float(((b1.T @ b2) * (a1 @ a2.T)).sum())
    return {"rel_err": math.sqrt(max(n1 + n2 - 2.0 * inner, 0.0)) / math.sqrt(n1),
            "cosine": inner / math.sqrt(n1 * n2),
            "energy_ratio": n2 / n1}


def grouped_to_block_index(heads: int, head_dim: int) -> torch.Tensor:
    """Row index taking a grouped-stored B to q|k|v band order."""
    per_head = torch.arange(heads)[:, None] * 3 * head_dim
    offsets = torch.arange(head_dim)[None, :]
    return torch.cat([(per_head + part * head_dim + offsets).reshape(-1) for part in range(3)])


def compare(state: dict, targets: list[str], rank: int, alpha: float,
            other_path: Path, heads: int, head_dim: int) -> dict:
    other = load_file(str(other_path), device="cpu")
    scale = alpha / rank
    grouped_index = grouped_to_block_index(heads, head_dim)
    rows, consumed = [], set()
    for target in targets:
        key_a = f"diffusion_model.{target}.lora_A.weight"
        key_b = f"diffusion_model.{target}.lora_B.weight"
        key_alpha = f"diffusion_model.{target}.alpha"
        if key_a not in other or key_b not in other:
            rows.append({"target": target, "missing": True})
            continue
        consumed.update((key_a, key_b, key_alpha))
        a2, b2 = other[key_a], other[key_b]
        other_rank = int(a2.shape[0])
        other_scale = float(other[key_alpha]) / other_rank if key_alpha in other else 1.0
        a1, b1 = state[f"{target}.lora_a"], state[f"{target}.lora_b"]
        row = {"target": target, "other_rank": other_rank, "other_scale": other_scale,
               **delta_stats(b1, a1, scale, b2, a2, other_scale)}
        if target.endswith("attn.qkv_proj"):
            row["if_other_grouped"] = delta_stats(b1, a1, scale, b2[grouped_index], a2, other_scale)
        if target.endswith("mlp.fc1"):
            half = b2.shape[0] // 2
            swapped = torch.cat((b2[half:], b2[:half]))
            row["if_other_swapped"] = delta_stats(b1, a1, scale, swapped, a2, other_scale)
        rows.append(row)
    present = [r for r in rows if not r.get("missing")]
    by_kind = {}
    for kind in KINDS:
        kind_rows = [r for r in present if r["target"].endswith(kind)]
        if not kind_rows:
            continue
        entry = {"modules": len(kind_rows),
                 "other_rank": summarise([r["other_rank"] for r in kind_rows]),
                 "cosine": summarise([r["cosine"] for r in kind_rows]),
                 "rel_err": summarise([r["rel_err"] for r in kind_rows]),
                 "energy_ratio": summarise([r["energy_ratio"] for r in kind_rows])}
        for hypothesis in ("if_other_grouped", "if_other_swapped"):
            scored = [r[hypothesis]["cosine"] for r in kind_rows if hypothesis in r]
            if scored:
                entry[f"cosine_{hypothesis}"] = summarise(scored)
        by_kind[kind] = entry
    unconsumed = sorted(set(other) - consumed)
    return {"file": other_path.name, "sha256": sha256(other_path),
            "missing_modules": [r["target"] for r in rows if r.get("missing")],
            "unconsumed_count": len(unconsumed), "unconsumed_examples": unconsumed[:8],
            "by_kind": by_kind, "modules": rows}


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert the TaoMate-H3 adapter to a ComfyUI LoRA, checking every mapping it can on the files.")
    ap.add_argument("--adapter", required=True, type=Path,
                    help="TaoMate-H3 adapter directory (adapter_config.json, adapter_model.safetensors)")
    ap.add_argument("--out", type=Path, help="ComfyUI LoRA to write; omit to verify and compare only")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--checkpoint", type=Path,
                    default=COMFY / "models" / "diffusion_models" / h3_config.MODELS["unet_fl2va"],
                    help="the ComfyUI H3 checkpoint the LoRA loads on (default: h3_config.MODELS['unet_fl2va'])")
    ap.add_argument("--anchor", type=Path,
                    default=COMFY / "models" / "loras" / h3_config.PDD_FL2VA_LORA,
                    help="a fused-qkv ComfyUI LoRA known to render (default: h3_config.PDD_FL2VA_LORA)")
    ap.add_argument("--release", type=Path,
                    help="the MiniMaxAI/MiniMax-H3 download (containing FL2VA/transformer): check the "
                         "checkpoint's qkv and fc1 rows against the weights TaoMate trained on")
    ap.add_argument("--swap-fc1-halves", action="store_true",
                    help="write the functional control: every mlp.fc1 lora_B with its gate and up "
                         "halves exchanged (h3_config.TAOMATE_SWAPPED_CONTROL_LORA). Never a usable LoRA")
    ap.add_argument("--compare", type=Path, help="another ComfyUI-format conversion of this adapter")
    ap.add_argument("--record", type=Path, help="write the report as JSON (bench/results/...)")
    args = ap.parse_args(argv)

    try:
        config, rank, alpha, weights, file_meta, state = read_adapter(args.adapter)
        targets = adapter_inventory(state, rank)
        geometry = checkpoint_inventory(args.checkpoint, state, targets)
        anchor = anchor_qkv_bands(args.anchor)
        release = (release_layout(args.release, args.checkpoint, geometry["heads"], geometry["head_dim"])
                   if args.release is not None else None)
    except VerificationError as exc:
        print(f"FAIL  {exc}")
        return 1
    print(f"ok    adapter: {len(targets)} modules at rank {rank}, alpha {alpha}")
    print(f"ok    checkpoint {args.checkpoint.name}: {geometry['linears']} linears, "
          f"every one matched by name and shape")
    print(f"ok    anchor {anchor['file']}: {anchor['modules']} fused qkv_proj lora_B are "
          f"block-diagonal over row bands (worst off-block energy "
          f"{anchor['off_block_energy_fraction_max']:.2e})")
    if release is not None:
        print(f"ok    release: the checkpoint's qkv is the release reordered to q|k|v bands and its "
              f"fc1 is the release as stored, at {', '.join(RELEASE_PROBE_MODULES)}")
    else:
        print("note  no --release: the qkv and fc1 layouts rest on source reads and the anchor")

    source_digest = sha256(weights)
    dtype = DTYPES[args.dtype]
    scale = alpha / rank
    out, cast, swapped = {}, [], 0
    for target in targets:
        a, b = state[f"{target}.lora_a"], state[f"{target}.lora_b"]
        a_cast, b_cast = a.to(dtype).contiguous(), b.to(dtype).contiguous()
        cast.append(delta_stats(b, a, scale, b_cast, a_cast, scale)["rel_err"])
        if args.swap_fc1_halves and target.endswith(".mlp.fc1"):
            gate, up = b_cast.chunk(2, dim=0)
            b_cast = torch.cat([up, gate], dim=0).contiguous()
            swapped += 1
        out[f"diffusion_model.{target}.lora_A.weight"] = a_cast
        out[f"diffusion_model.{target}.lora_B.weight"] = b_cast
        out[f"diffusion_model.{target}.alpha"] = torch.tensor(alpha, dtype=torch.float32)
    print(f"ok    cast to {args.dtype}: delta relative error "
          f"median {statistics.median(cast):.2e}, max {max(cast):.2e}")
    if args.swap_fc1_halves:
        fc1 = sum(1 for t in targets if t.endswith(".mlp.fc1"))
        if swapped == 0 or swapped != fc1:
            print(f"FAIL  control: swapped {swapped} of {fc1} fc1 modules")
            return 1
        print(f"ok    CONTROL: gate and up halves exchanged in all {swapped} fc1 lora_B")

    metadata = {
        "source_format": "TaoMate-H3 adapter (lora_a/lora_b over MiniMax H3 merged linears)",
        "source_repo": "TaoLiveAIGC/TaoMate-H3",
        "source_file": weights.name,
        "source_sha256": source_digest,
        "source_weight_source": str(config.get("weight_source", file_meta.get("weight_source", ""))),
        "source_optimizer_step": str(config.get("optimizer_step", file_meta.get("optimizer_step", ""))),
        "target_format": "ComfyUI generic LoRA",
        "base_model": "MiniMaxAI/MiniMax-H3, FL2VA partition (the directory TaoMate's --model-root loads)",
        "target_checkpoint": args.checkpoint.name,
        "training_rank": str(rank),
        "training_alpha": repr(alpha),
        "training_scale": repr(scale),
        "qkv_fusion": "none: the adapter targets the merged qkv_proj",
        "qkv_layout": ("q|k|v row bands on both sides, no reorder: TaoMate trains on its "
                       "reorder_grouped_qkv_to_qkv module; core splits bands (anchored on a "
                       "fused LoRA that renders)"),
        "swi_glu_mapping": ("TaoMate [gate;up] -> ComfyUI [gate;up], no swap ("
                            + ("source reads; checkpoint fc1 == release as stored"
                               if release is not None else "source reads") + ")"),
        "release_layout": ("checkpoint qkv == release reordered to bands, fc1 == release as stored, at "
                           + ", ".join(RELEASE_PROBE_MODULES)) if release is not None else "not checked",
        "distilled_grid": json.dumps(distilled_grid()),
        "control": ("fc1 lora_B gate and up halves SWAPPED on purpose: a render control, "
                    "not a usable LoRA" if args.swap_fc1_halves else "none"),
        "dtype": args.dtype,
        "modules": str(len(targets)),
        "converter": "bench/convert_taomate_lora.py",
        "converter_version": str(CONVERTER_VERSION),
    }
    out_digest = None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_file(out, str(args.out), metadata=metadata)
        out_digest = sha256(args.out)
        print(f"wrote {args.out.name} ({len(out)} tensors), sha256 {out_digest[:16]}")

    comparison = None
    if args.compare is not None:
        comparison = compare(state, targets, rank, alpha, args.compare,
                             geometry["heads"], geometry["head_dim"])
        print(f"\ncompared against {comparison['file']}: "
              f"{len(comparison['missing_modules'])} modules missing, "
              f"{comparison['unconsumed_count']} tensors unconsumed")
        for kind, entry in comparison["by_kind"].items():
            line = (f"  {kind:14s} rank {entry['other_rank']['min']}-{entry['other_rank']['max']}"
                    f"  cosine median {entry['cosine']['median']:.4f} (min {entry['cosine']['min']:.4f})"
                    f"  energy median {entry['energy_ratio']['median']:.4f}"
                    f"  rel_err median {entry['rel_err']['median']:.4f}")
            for hypothesis in ("if_other_grouped", "if_other_swapped"):
                if f"cosine_{hypothesis}" in entry:
                    line += (f"  [{hypothesis}: cosine median "
                             f"{entry[f'cosine_{hypothesis}']['median']:.4f}]")
            print(line)

    if args.record is not None:
        record = {
            "date": _dt.date.today().isoformat(),
            "what": ("TaoMate-H3 adapter converted to a ComfyUI LoRA by rename; inventory "
                     "checked against the checkpoint, core's qkv band layout checked on the anchor"
                     + ("; deltas compared against another conversion" if comparison else "")),
            "provenance": {
                "converter": "bench/convert_taomate_lora.py",
                "converter_version": CONVERTER_VERSION,
                "repo_commit": git_head(REPO),
                "comfy_commit": git_head(COMFY),
                "torch": torch.__version__,
                "source_file": weights.name,
                "source_bytes": weights.stat().st_size,
                "source_sha256": source_digest,
                "adapter_config": config,
                "checkpoint": args.checkpoint.name,
            },
            "adapter": {"modules": len(targets), "rank": rank, "alpha": alpha},
            "checkpoint_geometry": geometry,
            "anchor": anchor,
            "release_layout": release,
            "control_fc1_swapped": bool(args.swap_fc1_halves),
            "distilled_grid": distilled_grid(),
            "cast": {"dtype": args.dtype, "delta_rel_err": summarise(cast)},
            "output": ({"file": args.out.name, "sha256": out_digest, "tensors": len(out),
                        "metadata": metadata} if args.out is not None else None),
            "comparison": comparison,
        }
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nrecord {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
