#!/usr/bin/env python3
"""Repack Meta's SAM 3D Body checkpoint and its MHR rig as the one safetensors file core loads, from the originals.

`facebook/sam-3d-body-dinov3` ships `model.ckpt` (a flat pickled state dict,
1127 tensors: a DINOv3 ViT-H+ backbone in bf16, the SAM-style decoders, the
prompt encoder and the pose, camera and hand heads in fp32) and
`assets/mhr_model.pt` (the MHR body rig as a TorchScript module). Core's
`SAM3DBody_Loader` (`comfy_extras/nodes_sam3d_body.py`) reads one safetensors
file from `models/detection/`, renames `.layers.0.0.` to `.layers.0.`, drops
`hand_cls_embed.*`, and requires every other key to match
`comfy/ldm/sam3d_body/model/model.py::SAM3DBody` exactly. This script writes
that file from the originals, so what it holds is what Meta published, at the
dtypes Meta published, plus two things Meta did not publish (below).

**What is renamed, and nothing else changes** (`BACKBONE_RENAMES`,
`split_qkv`): the backbone moves from DINOv3's own layout to the layout core
copied from transformers' `Dinov3ViT`. The fused `qkv` weight is split into
thirds; the `qkv` bias's first and last thirds become the q and v biases and
the middle third is dropped, because core's k projection has no bias. The
drop loses nothing: in this checkpoint every q, k and v bias is exactly zero,
and DINOv3's `LinearKMaskedBias` multiplies the bias by a mask that is all
zeros here (`coderef/dinov3/dinov3/layers/attention.py`, checkout 6876159),
so Meta's own forward sees no attention bias either. The script asserts the
dropped third is zero. `rope_embed.periods` is dropped too: core computes its
rotary periods and has no buffer for them. Every tensor outside the backbone
keeps its name, `hand_cls_embed.*` included (the loader pops it).

**The rig** (`RIG_MAP`): 15 of core's 17 `mhr.*` buffers are tensors of the
TorchScript module's state dict, renamed; the other two are attributes the
module holds as Python lists (`_pmi_buffer_sizes` on the skeleton,
`sparse_shape` on the pose-correctives sparse layer) and are written as the
int64 tensors core expects. The module's mesh faces, texcoords and parameter
limits are not part of core's rig and are not written.

**Two groups are not Meta's, and the file says so in its metadata.** Core's
model also carries `face_landmarker.*` (421 tensors, a pure-PyTorch port of
Google's MediaPipe `face_landmarker_v2_with_blendshapes.task`, used only by
the face-expression node) and `head_pose.face_region_rgb` (a per-vertex
colour map painted for the mesh renderer). Neither exists in Meta's files.
By default they are fetched by HTTP range request from Comfy-Org's repack of
this model, the only published source, and the metadata records that
file's URL and the sha256 of the bytes taken. `--comfy-extras zeros` writes
zeros instead: the file then loads and the mesh, silhouette and skeleton
renders work, and the face-expression node and the face tint do not.

**What is checked before the file is trusted.** The written key set, shapes
and dtypes equal core's model's (`SAM3DBody` instantiated on the CPU, after
the loader's own rename and pop); every tensor reads back `torch.equal` to
what was mapped; and with `--compare-repack N`, N tensors chosen at random
are fetched from the repack by range request and compared bit for bit, which
is how the mapping was derived in the first place (every backbone pattern on
layer 0 and the embeddings matched on 2026-10-05).
`bench/check_sam3d_body_conversion.py` holds the mapping without weights.

The `vith` release (`facebook/sam-3d-body-vith`) has a different backbone
layout and core's model is the DINOv3 one; this script refuses a checkpoint
whose backbone keys do not match the DINOv3 patterns.

CPU only; needs the ComfyUI checkout on the path for the key check (found
from this file's location, or `--comfy-root`).

    CUDA_VISIBLE_DEVICES="" <comfy venv python> bench/convert_sam3d_body_checkpoint.py \\
        --src-dir <facebook_sam-3d-body-dinov3> \\
        --out <facebook_sam-3d-body-dinov3>/sam_3d_body_dinov3.safetensors \\
        [--comfy-extras repack|zeros] [--compare-repack 24]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import struct
import sys
import urllib.request
from pathlib import Path

import torch

#: The published repack of this model, the only source of the two groups Meta
#: did not publish. Read by HTTP range request; never downloaded whole.
REPACK_URL = ("https://huggingface.co/Comfy-Org/sam-3d-body/resolve/main/"
              "detection/sam_3d_body_dinov3_bf16.safetensors")
#: The groups that come from the repack rather than from Meta's files.
COMFY_EXTRA_PREFIXES = ("face_landmarker.",)
COMFY_EXTRA_KEYS = ("head_pose.face_region_rgb",)
#: Bumped when the mapping changes; written into the file's metadata.
MAPPING_VERSION = 1

#: Backbone renames from DINOv3's layout to core's, as (regex, replacement).
#: `qkv` is handled by `split_qkv`; `bias_mask` and `rope_embed.periods` are
#: dropped (`BACKBONE_DROPS`).
BACKBONE_RENAMES = (
    (r"^backbone\.encoder\.blocks\.(\d+)\.attn\.proj\.(weight|bias)$", r"backbone.layer.\1.attention.o_proj.\2"),
    (r"^backbone\.encoder\.blocks\.(\d+)\.ls1\.gamma$", r"backbone.layer.\1.layer_scale1.lambda1"),
    (r"^backbone\.encoder\.blocks\.(\d+)\.ls2\.gamma$", r"backbone.layer.\1.layer_scale2.lambda1"),
    (r"^backbone\.encoder\.blocks\.(\d+)\.mlp\.w1\.(weight|bias)$", r"backbone.layer.\1.mlp.gate_proj.\2"),
    (r"^backbone\.encoder\.blocks\.(\d+)\.mlp\.w2\.(weight|bias)$", r"backbone.layer.\1.mlp.up_proj.\2"),
    (r"^backbone\.encoder\.blocks\.(\d+)\.mlp\.w3\.(weight|bias)$", r"backbone.layer.\1.mlp.down_proj.\2"),
    (r"^backbone\.encoder\.blocks\.(\d+)\.(norm1|norm2)\.(weight|bias)$", r"backbone.layer.\1.\2.\3"),
    (r"^backbone\.encoder\.cls_token$", r"backbone.embeddings.cls_token"),
    (r"^backbone\.encoder\.storage_tokens$", r"backbone.embeddings.register_tokens"),
    (r"^backbone\.encoder\.patch_embed\.proj\.(weight|bias)$", r"backbone.embeddings.patch_embeddings.\1"),
    (r"^backbone\.encoder\.norm\.(weight|bias)$", r"backbone.norm.\1"),
)
BACKBONE_QKV = re.compile(r"^backbone\.encoder\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)$")
BACKBONE_DROPS = (
    re.compile(r"^backbone\.encoder\.blocks\.\d+\.attn\.qkv\.bias_mask$"),
    re.compile(r"^backbone\.encoder\.rope_embed\.periods$"),
)

#: core's `mhr.*` buffer <- the TorchScript rig's state-dict tensor.
RIG_MAP = {
    "mhr.base_shape": "character_torch.blend_shape.base_shape",
    "mhr.identity_basis": "character_torch.blend_shape.shape_vectors",
    "mhr.expr_basis": "face_expressions_model.shape_vectors",
    "mhr.param_transform": "character_torch.parameter_transform.parameter_transform",
    "mhr.skel_joint_translation_offsets": "character_torch.skeleton.joint_translation_offsets",
    "mhr.skel_joint_prerotations": "character_torch.skeleton.joint_prerotations",
    "mhr.skel_joint_parents": "character_torch.skeleton.joint_parents",
    "mhr.skel_pmi": "character_torch.skeleton.pmi",
    "mhr.lbs_inverse_bind_pose": "character_torch.linear_blend_skinning.inverse_bind_pose",
    "mhr.lbs_skin_indices": "character_torch.linear_blend_skinning.skin_indices_flattened",
    "mhr.lbs_skin_weights": "character_torch.linear_blend_skinning.skin_weights_flattened",
    "mhr.lbs_vert_indices": "character_torch.linear_blend_skinning.vert_indices_flattened",
    "mhr.pose_corr_sparse_indices": "pose_correctives_model.pose_dirs_predictor.0.sparse_indices",
    "mhr.pose_corr_sparse_weight": "pose_correctives_model.pose_dirs_predictor.0.sparse_weight",
    "mhr.pose_corr_weight": "pose_correctives_model.pose_dirs_predictor.2.weight",
}
#: core's two derived rig buffers <- attributes of the TorchScript module.
RIG_ATTRS = {
    "mhr.skel_pmi_buffer_sizes": ("character_torch.skeleton", "_pmi_buffer_sizes"),
    "mhr.pose_corr_sparse_shape": ("pose_correctives_model.pose_dirs_predictor.0", "sparse_shape"),
}

#: What the loader does to a file before `load_state_dict`; applied to the
#: written keys when they are compared with the model, never to the file.
LOADER_RENAME = (".layers.0.0.", ".layers.0.")
LOADER_POPS = ("hand_cls_embed.weight", "hand_cls_embed.bias")


def map_key(key: str) -> list[str | None] | None:
    """Core's name(s) for one original key: a list (the qkv weight and bias
    fan out; `None` inside it marks a dropped third), `[]` for a dropped key,
    `None` when the key keeps its name."""
    m = BACKBONE_QKV.match(key)
    if m:
        n, kind = m.groups()
        heads = ["q_proj", "k_proj", "v_proj"] if kind == "weight" else ["q_proj", None, "v_proj"]
        return [f"backbone.layer.{n}.attention.{h}.{kind}" if h else None for h in heads]
    for pattern in BACKBONE_DROPS:
        if pattern.match(key):
            return []
    for pattern, repl in BACKBONE_RENAMES:
        new, count = re.subn(pattern, repl, key)
        if count:
            return [new]
    if key.startswith("backbone."):
        raise ValueError(f"backbone key with no mapping: {key} (not the DINOv3 layout this script knows)")
    return None


def split_qkv(key: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """The three thirds of a fused qkv tensor under core's names; a bias's
    k third is dropped after being checked to be zero."""
    names = map_key(key)
    third = tensor.shape[0] // 3
    assert third * 3 == tensor.shape[0], key
    parts = [tensor[i * third:(i + 1) * third].clone() for i in range(3)]
    out = {}
    for name, part in zip(names, parts):
        if name is None:
            if bool((part.float() != 0).any()):
                raise ValueError(f"{key}: the k bias is not zero, so dropping it would change the model")
            continue
        out[name] = part
    return out


def map_state_dict(sd: dict) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in sd.items():
        if not torch.is_tensor(value):
            raise ValueError(f"{key}: not a tensor ({type(value).__name__})")
        names = map_key(key)
        if names is None:
            out[key] = value
        elif names == []:
            continue
        elif BACKBONE_QKV.match(key):
            out.update(split_qkv(key, value))
        else:
            out[names[0]] = value
    return out


def mapped_rig(rig_module) -> dict[str, torch.Tensor]:
    sd = rig_module.state_dict()
    out = {core: sd[src].clone() for core, src in RIG_MAP.items()}
    for core, (path, attr) in RIG_ATTRS.items():
        obj = rig_module
        for part in path.split("."):
            obj = getattr(obj, part)
        out[core] = torch.tensor(list(getattr(obj, attr)), dtype=torch.int64)
    return out


# ---- the repack, by range request --------------------------------------------
_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
       "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


class Repack:
    """The published safetensors file, read a tensor at a time over HTTP."""

    def __init__(self, url: str = REPACK_URL):
        self.url = url
        n = struct.unpack("<Q", self._get("bytes=0-7"))[0]
        self.header = json.loads(self._get(f"bytes=8-{8 + n - 1}"))
        self.base = 8 + n
        self.header.pop("__metadata__", None)

    def _get(self, rng: str) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": rng})
        return urllib.request.urlopen(req, timeout=120).read()

    def tensor(self, key: str) -> torch.Tensor:
        entry = self.header[key]
        a, b = entry["data_offsets"]
        buf = self._get(f"bytes={self.base + a}-{self.base + b - 1}")
        t = torch.frombuffer(bytearray(buf), dtype=_DT[entry["dtype"]])
        return t.reshape(entry["shape"]).clone()

    def group(self, keys: list[str]) -> tuple[dict[str, torch.Tensor], str]:
        """Several tensors, fetched as the fewest contiguous spans, and the
        sha256 of the bytes taken (in file order)."""
        keys = sorted(keys, key=lambda k: self.header[k]["data_offsets"][0])
        spans: list[list[str]] = []
        for k in keys:
            if spans and self.header[spans[-1][-1]]["data_offsets"][1] == self.header[k]["data_offsets"][0]:
                spans[-1].append(k)
            else:
                spans.append([k])
        out, h = {}, hashlib.sha256()
        for span in spans:
            a = self.header[span[0]]["data_offsets"][0]
            b = self.header[span[-1]]["data_offsets"][1]
            buf = self._get(f"bytes={self.base + a}-{self.base + b - 1}")
            h.update(buf)
            for k in span:
                ka, kb = self.header[k]["data_offsets"]
                t = torch.frombuffer(bytearray(buf[ka - a:kb - a]), dtype=_DT[self.header[k]["dtype"]])
                out[k] = t.reshape(self.header[k]["shape"]).clone()
        return out, h.hexdigest()


# ---- core's model, for the key check -------------------------------------------
def core_state(comfy_root: Path) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    sys.path.insert(0, str(comfy_root))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy.ops
    from comfy.ldm.sam3d_body.model.model import SAM3DBody
    model = SAM3DBody(dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    return {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}


def as_loader_sees(keys) -> set[str]:
    return {k.replace(*LOADER_RENAME) for k in keys if k not in LOADER_POPS}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--src-dir", required=True, help="the facebook/sam-3d-body-dinov3 download (model.ckpt, assets/mhr_model.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--comfy-extras", choices=("repack", "zeros"), default="repack",
                    help="face_landmarker.* and head_pose.face_region_rgb: from the published repack (default) or zeros")
    ap.add_argument("--compare-repack", type=int, default=0, metavar="N",
                    help="fetch N random tensors from the repack and compare them with the written ones")
    ap.add_argument("--comfy-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(args.src_dir).expanduser()
    ckpt_path, rig_path = src / "model.ckpt", src / "assets" / "mhr_model.pt"
    for p in (ckpt_path, rig_path):
        if not p.is_file():
            print(f"missing: {p}"); return 2
    comfy_root = Path(args.comfy_root).expanduser() if args.comfy_root else Path(__file__).resolve().parents[3]
    if not (comfy_root / "comfy" / "ldm" / "sam3d_body").is_dir():
        print(f"ComfyUI not found at {comfy_root}; pass --comfy-root"); return 2
    out_path = Path(os.path.expanduser(args.out))

    print(f"reading {ckpt_path.name}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and not any(torch.is_tensor(v) for v in sd.values()):
        sd = sd["state_dict"]
    out = map_state_dict(sd)
    print(f"  {len(sd)} tensors in, {len(out)} after the backbone renames")
    print(f"reading assets/{rig_path.name}")
    rig = torch.jit.load(str(rig_path), map_location="cpu")
    out.update(mapped_rig(rig))
    print(f"  {len(RIG_MAP) + len(RIG_ATTRS)} rig buffers")

    core = core_state(comfy_root)
    extra_keys = [k for k in core if k.startswith(COMFY_EXTRA_PREFIXES) or k in COMFY_EXTRA_KEYS]
    repack = None
    if args.comfy_extras == "repack" or args.compare_repack:
        print("reading the repack's header")
        repack = Repack()
    meta = {
        "converter": Path(__file__).name, "mapping_version": str(MAPPING_VERSION),
        "source_ckpt": ckpt_path.name, "source_ckpt_sha256": sha256_file(ckpt_path),
        "source_rig": "assets/" + rig_path.name, "source_rig_sha256": sha256_file(rig_path),
        "comfy_extras": args.comfy_extras,
        "comfy_extras_keys": f"{len(extra_keys)} tensors: face_landmarker.* and head_pose.face_region_rgb",
    }
    if args.comfy_extras == "repack":
        print(f"  fetching the {len(extra_keys)} tensors Meta did not publish from the repack")
        got, digest = repack.group(extra_keys)
        out.update(got)
        meta["comfy_extras_source"] = REPACK_URL
        meta["comfy_extras_sha256"] = digest
    else:
        for k in extra_keys:
            shape, dtype = core[k]
            out[k] = torch.zeros(shape, dtype=dtype)
        meta["comfy_extras_source"] = "zeros: the face-expression node and the face tint are unusable with this file"

    seen = as_loader_sees(out.keys())
    missing, unexpected = sorted(set(core) - seen), sorted(seen - set(core))
    if missing or unexpected:
        print(f"key mismatch against core's model: missing {missing[:8]} ({len(missing)}), "
              f"unexpected {unexpected[:8]} ({len(unexpected)})")
        return 1
    # Shapes against core's model; dtypes against the published file when it
    # is read, because core builds the model at the file's own dtype
    # (`comfy.utils.weight_dtype`), so the model instantiated here says
    # nothing about dtype.
    bad = []
    for k, t in out.items():
        kk = k.replace(*LOADER_RENAME)
        if kk in core and tuple(t.shape) != core[kk][0]:
            bad.append((k, tuple(t.shape), core[kk][0]))
        if repack is not None and k in repack.header and _DT[repack.header[k]["dtype"]] != t.dtype:
            bad.append((k, str(t.dtype), repack.header[k]["dtype"]))
    if bad:
        print(f"shape or dtype mismatch: {bad[:5]} ({len(bad)})"); return 1
    print(f"  key set and shapes equal core's model ({len(core)} tensors)"
          + ("; dtypes equal the published file's" if repack is not None else ""))

    # Cloned, not only contiguous: Meta's checkpoint stores eight head buffers
    # once and shares them between `head_pose` and `head_pose_hand`, and
    # safetensors refuses shared memory. Each key gets its own copy, as in
    # the published file.
    out = {k: v.contiguous().clone() for k, v in out.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_path), metadata=meta)
    print(f"wrote {out_path.name} ({out_path.stat().st_size / 1e9:.2f} GB)")

    with safe_open(str(out_path), framework="pt") as f:
        keys = set(f.keys())
        if keys != set(out):
            print("readback: the key set differs from what was written"); return 1
        for k in keys:
            if not torch.equal(f.get_tensor(k), out[k]):
                print(f"readback: {k} differs"); return 1
    print("readback: every tensor equals what was mapped")

    if args.compare_repack:
        rng = random.Random(args.seed)
        candidates = [k for k in out if k not in extra_keys and k in repack.header]
        sample = rng.sample(candidates, min(args.compare_repack, len(candidates)))
        differing = []
        for k in sample:
            theirs = repack.tensor(k)
            if theirs.shape != out[k].shape or theirs.dtype != out[k].dtype or not torch.equal(theirs, out[k]):
                differing.append(k)
                print(f"repack: {k} differs ({tuple(theirs.shape)} {theirs.dtype} vs {tuple(out[k].shape)} {out[k].dtype})")
        if differing:
            return 1
        print(f"repack: {len(sample)} randomly chosen tensors are bit-identical to the published file")
    print(f"source sha256: ckpt {meta['source_ckpt_sha256']}, rig {meta['source_rig_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
