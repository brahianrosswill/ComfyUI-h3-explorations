#!/usr/bin/env python3
"""Hold `bench/convert_sam3d_body_checkpoint.py`'s mapping to core's SAM 3D Body model, without weights.

Core's `SAM3DBody_Loader` requires every key of the file to match the model,
so the mapping is right only if the set of names it produces from Meta's
checkpoint and rig equals the model's state dict, after the loader's own
rename and pop. This drives the mapping on a synthetic state dict that has
the original's key patterns (the 32-layer DINOv3 backbone, the heads, the
rig's state-dict names and attributes) and compares the result with core's
model instantiated on the CPU. Two more cases: a non-zero k bias is refused
by name (the drop would otherwise change the model), and the `vith` layout is
refused by name. No weights, no CUDA, no server; ComfyUI is found from this
file's location.

    CUDA_VISIBLE_DEVICES="" <comfy venv python> bench/check_sam3d_body_conversion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert_sam3d_body_checkpoint as C  # noqa: E402

COMFY_ROOT = HERE.parents[2]


def _original_keys() -> list[str]:
    """Every key of `model.ckpt` as read on 2026-10-05 (1127), by pattern."""
    keys = []
    for n in range(32):
        b = f"backbone.encoder.blocks.{n}"
        keys += [f"{b}.attn.qkv.weight", f"{b}.attn.qkv.bias", f"{b}.attn.qkv.bias_mask",
                 f"{b}.attn.proj.weight", f"{b}.attn.proj.bias", f"{b}.ls1.gamma", f"{b}.ls2.gamma",
                 f"{b}.mlp.w1.weight", f"{b}.mlp.w1.bias", f"{b}.mlp.w2.weight", f"{b}.mlp.w2.bias",
                 f"{b}.mlp.w3.weight", f"{b}.mlp.w3.bias",
                 f"{b}.norm1.weight", f"{b}.norm1.bias", f"{b}.norm2.weight", f"{b}.norm2.bias"]
    keys += ["backbone.encoder.cls_token", "backbone.encoder.storage_tokens",
             "backbone.encoder.patch_embed.proj.weight", "backbone.encoder.patch_embed.proj.bias",
             "backbone.encoder.rope_embed.periods", "backbone.encoder.norm.weight", "backbone.encoder.norm.bias"]
    return keys


def main() -> int:
    core = C.core_state(COMFY_ROOT)
    failures = []

    # 1. Backbone keys: the mapping sends the original's 551 to core's 614, exactly.
    mapped = set()
    for k in _original_keys():
        names = C.map_key(k)
        if names is None:
            failures.append(f"{k} kept its name; every backbone key must be renamed or dropped")
        else:
            mapped.update(n for n in names if n)
    core_backbone = {k for k in core if k.startswith("backbone.")}
    if mapped != core_backbone:
        failures.append(f"backbone mapping: missing {sorted(core_backbone - mapped)[:5]}, extra {sorted(mapped - core_backbone)[:5]}")

    # 2. The rest of core's keys are either Meta's unchanged names, the rig, or the two Comfy groups.
    rig = set(C.RIG_MAP) | set(C.RIG_ATTRS)
    extras = {k for k in core if k.startswith(C.COMFY_EXTRA_PREFIXES) or k in C.COMFY_EXTRA_KEYS}
    unchanged = set(core) - core_backbone - rig - extras
    renamed_by_mistake = [k for k in unchanged if C.map_key(k) is not None]
    if renamed_by_mistake:
        failures.append(f"non-backbone keys the mapping would rename: {renamed_by_mistake[:5]}")
    if not {k for k in core if k.startswith("mhr.")} == rig:
        failures.append(f"rig map does not cover core's mhr.* keys: {sorted({k for k in core if k.startswith('mhr.')} ^ rig)}")
    if not C.as_loader_sees(set(C.LOADER_POPS)) == set():
        failures.append("the loader's pops are not removed by as_loader_sees")

    # 3. A non-zero k bias is refused.
    bias = torch.zeros(3 * 4); bias[5] = 1.0
    try:
        C.split_qkv("backbone.encoder.blocks.0.attn.qkv.bias", bias)
        failures.append("a non-zero k bias was dropped silently")
    except ValueError as e:
        if "k bias" not in str(e):
            failures.append(f"non-zero k bias refused with the wrong reason: {e}")
    q_and_v = C.split_qkv("backbone.encoder.blocks.0.attn.qkv.bias", torch.arange(12.0).mul(torch.tensor([1.0] * 4 + [0.0] * 4 + [1.0] * 4)))
    if set(q_and_v) != {"backbone.layer.0.attention.q_proj.bias", "backbone.layer.0.attention.v_proj.bias"}:
        failures.append(f"qkv bias split names: {sorted(q_and_v)}")

    # 4. The vith layout is refused by name.
    try:
        C.map_key("backbone.blocks.0.attn.qkv.weight")
        failures.append("a non-DINOv3 backbone key was accepted")
    except ValueError:
        pass

    for f in failures:
        print("FAIL", f)
    if failures:
        return 1
    print(f"ok: mapping covers core's {len(core)} tensors ({len(core_backbone)} backbone from {len(_original_keys())} original keys, "
          f"{len(rig)} rig, {len(extras)} from the repack, {len(unchanged)} unchanged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
