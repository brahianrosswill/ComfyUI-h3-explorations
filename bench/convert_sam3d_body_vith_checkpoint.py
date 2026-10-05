#!/usr/bin/env python3
"""Repack Meta's ViT-H SAM 3D Body checkpoint and its MHR rig as the one safetensors file this pack's loader reads.

`facebook/sam-3d-body-vith` ships the same two files as the DINOv3 release:
`model.ckpt` (a flat pickled state dict) and `assets/mhr_model.pt` (the MHR
rig as a TorchScript module). `bench/convert_sam3d_body_checkpoint.py`
converts the DINOv3 release for core and refuses this one, because core's
model has no ViT backbone. `sam3d_body_vith.py` is that model here, and this
script writes its file.

**Nothing is renamed.** The ViT-H module uses Meta's own backbone names
(`backbone.patch_embed.proj`, `backbone.pos_embed`, `backbone.blocks.N.*`,
`backbone.last_norm`), and every tensor outside the backbone already has the
name core's model gives it; both were read off the checkpoint on 2026-10-05.
Tensors keep the dtypes Meta published: the backbone's linears in bf16, its
norms and everything else in fp32. `hand_cls_embed.*` is written as it is;
the loader pops it, as core's does.

**The rig and the two groups Meta did not publish are taken exactly as the
DINOv3 converter takes them**, by importing its `RIG_MAP`, `RIG_ATTRS`,
`mapped_rig` and `Repack`: the rig from this release's own
`assets/mhr_model.pt`, and `face_landmarker.*` and
`head_pose.face_region_rgb` by HTTP range request from Comfy-Org's repack of
the DINOv3 model, the only published source. Neither group depends on the
backbone: one is a port of MediaPipe's face landmarker and the other a
colour per mesh vertex. The metadata records the URL and the sha256 of the
bytes taken. `--comfy-extras zeros` writes zeros instead, with the same
consequence as there: the face-expression node and the face tint do not
work with that file.

**What is checked before the file is trusted.** The backbone keys are exactly
the ViT-H patterns (a DINOv3 checkpoint is refused by name); the written key
set and shapes equal `SAM3DBodyViTH`'s state dict after the loader's rename
and pop; and every tensor reads back `torch.equal` to what was mapped.
`bench/check_sam3d_body_vith.py` holds the key set without weights and the
backbone's forward against Meta's own file with them.

CPU only. The paths are typed in the shell and never written here.

    CUDA_VISIBLE_DEVICES="" <comfy venv python> bench/convert_sam3d_body_vith_checkpoint.py \\
        --src-dir <facebook_sam-3d-body-vith> \\
        --out <facebook_sam-3d-body-vith>/sam_3d_body_vith.safetensors \\
        [--comfy-extras repack|zeros]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import convert_sam3d_body_checkpoint as C  # noqa: E402
from _lib import REPO, bootstrap  # noqa: E402

#: Bumped when what this script writes changes; written into the metadata.
MAPPING_VERSION = 1

#: Every backbone key of the release, as (pattern, how many). read: the
#: checkpoint's own keys on 2026-10-05.
VITH_BACKBONE = (
    (re.compile(r"^backbone\.pos_embed$"), 1),
    (re.compile(r"^backbone\.patch_embed\.proj\.(weight|bias)$"), 2),
    (re.compile(r"^backbone\.blocks\.(\d+)\.(norm1|norm2)\.(weight|bias)$"), 128),
    (re.compile(r"^backbone\.blocks\.(\d+)\.attn\.(qkv|proj)\.(weight|bias)$"), 128),
    (re.compile(r"^backbone\.blocks\.(\d+)\.mlp\.(fc1|fc2)\.(weight|bias)$"), 128),
    (re.compile(r"^backbone\.last_norm\.(weight|bias)$"), 2),
)


def check_backbone_keys(keys) -> None:
    """Raise unless the backbone keys are exactly the ViT-H release's."""
    backbone = [k for k in keys if k.startswith("backbone.")]
    counts = [0] * len(VITH_BACKBONE)
    for key in backbone:
        for i, (pattern, _n) in enumerate(VITH_BACKBONE):
            if pattern.match(key):
                counts[i] += 1
                break
        else:
            raise ValueError(
                f"backbone key that is not the ViT-H layout: {key}. A DINOv3 checkpoint goes "
                "through bench/convert_sam3d_body_checkpoint.py.")
    short = [(p.pattern, got, want) for (p, want), got in zip(VITH_BACKBONE, counts) if got != want]
    if short:
        raise ValueError(f"the backbone is not the 32-block ViT-H: {short}")


def model_state() -> dict[str, tuple[int, ...]]:
    """`SAM3DBodyViTH`'s state-dict keys and shapes, built on the CPU without weights."""
    bootstrap(cpu=True)
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    import comfy.ops
    import sam3d_body_vith
    model = sam3d_body_vith.SAM3DBodyViTH(dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--src-dir", required=True,
                    help="the facebook/sam-3d-body-vith download (model.ckpt, assets/mhr_model.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy-extras", choices=("repack", "zeros"), default="repack",
                    help="face_landmarker.* and head_pose.face_region_rgb: from the published "
                         "DINOv3 repack (default) or zeros")
    args = ap.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(args.src_dir).expanduser()
    ckpt_path, rig_path = src / "model.ckpt", src / "assets" / "mhr_model.pt"
    for p in (ckpt_path, rig_path):
        if not p.is_file():
            print(f"missing: {p}")
            return 2
    out_path = Path(os.path.expanduser(args.out))

    print(f"reading {ckpt_path.name}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and not any(torch.is_tensor(v) for v in sd.values()):
        sd = sd["state_dict"]
    not_tensors = [k for k, v in sd.items() if not torch.is_tensor(v)]
    if not_tensors:
        print(f"not tensors: {not_tensors[:5]}")
        return 1
    check_backbone_keys(sd)
    out = dict(sd)
    print(f"  {len(sd)} tensors, {sum(k.startswith('backbone.') for k in sd)} of them the backbone, none renamed")
    print(f"reading assets/{rig_path.name}")
    out.update(C.mapped_rig(torch.jit.load(str(rig_path), map_location="cpu")))
    print(f"  {len(C.RIG_MAP) + len(C.RIG_ATTRS)} rig buffers")

    model = model_state()
    extra_keys = [k for k in model if k.startswith(C.COMFY_EXTRA_PREFIXES) or k in C.COMFY_EXTRA_KEYS]
    meta = {
        "converter": Path(__file__).name, "mapping_version": str(MAPPING_VERSION),
        "release": "facebook/sam-3d-body-vith", "backbone": "vit_hmr_512_384, Meta's names, not renamed",
        "source_ckpt": ckpt_path.name, "source_ckpt_sha256": C.sha256_file(ckpt_path),
        "source_rig": "assets/" + rig_path.name, "source_rig_sha256": C.sha256_file(rig_path),
        "comfy_extras": args.comfy_extras,
        "comfy_extras_keys": f"{len(extra_keys)} tensors: face_landmarker.* and head_pose.face_region_rgb",
    }
    if args.comfy_extras == "repack":
        print(f"  fetching the {len(extra_keys)} tensors Meta did not publish from the DINOv3 repack")
        got, digest = C.Repack().group(extra_keys)
        out.update(got)
        meta["comfy_extras_source"] = C.REPACK_URL + " (the DINOv3 model's repack; neither group depends on the backbone)"
        meta["comfy_extras_sha256"] = digest
    else:
        for k in extra_keys:
            out[k] = torch.zeros(model[k], dtype=torch.float32)
        meta["comfy_extras_source"] = "zeros: the face-expression node and the face tint are unusable with this file"

    seen = C.as_loader_sees(out.keys())
    missing, unexpected = sorted(set(model) - seen), sorted(seen - set(model))
    if missing or unexpected:
        print(f"key mismatch against SAM3DBodyViTH: missing {missing[:8]} ({len(missing)}), "
              f"unexpected {unexpected[:8]} ({len(unexpected)})")
        return 1
    bad = [(k, tuple(t.shape), model[k.replace(*C.LOADER_RENAME)]) for k, t in out.items()
           if k.replace(*C.LOADER_RENAME) in model and tuple(t.shape) != model[k.replace(*C.LOADER_RENAME)]]
    if bad:
        print(f"shape mismatch: {bad[:5]} ({len(bad)})")
        return 1
    print(f"  key set and shapes equal SAM3DBodyViTH's ({len(model)} tensors)")

    # Cloned for the reason the DINOv3 converter gives: Meta stores some head
    # buffers once and shares them, and safetensors refuses shared memory.
    out = {k: v.contiguous().clone() for k, v in out.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_path), metadata=meta)
    print(f"wrote {out_path.name} ({out_path.stat().st_size / 1e9:.2f} GB)")

    with safe_open(str(out_path), framework="pt") as f:
        if set(f.keys()) != set(out):
            print("readback: the key set differs from what was written")
            return 1
        for k in out:
            if not torch.equal(f.get_tensor(k), out[k]):
                print(f"readback: {k} differs")
                return 1
    print("readback: every tensor equals what was mapped")
    print(f"source sha256: ckpt {meta['source_ckpt_sha256']}, rig {meta['source_rig_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
