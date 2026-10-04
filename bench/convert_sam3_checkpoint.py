#!/usr/bin/env python3
"""Repack Meta's SAM 3 / SAM 3.1 checkpoint as a safetensors file core loads, changing no weight.

`facebook/sam3.1` ships `sam3.1_multiplex.pt`, a pickled state dict. Core's
SAM3 nodes (`comfy_extras/nodes_sam3.py`) take the model from
CheckpointLoaderSimple, and core does every rename at load time
(`comfy/supported_models.py::SAM3.process_unet_state_dict`): `tracker.model.*`
to `tracker.*`, the fused `in_proj` split, the decoder MLP names. So the
conversion is a container change and nothing else, which is why it starts from
the original rather than from a repackage: what the file holds is what Meta
published, at the dtype asked for here.

What it drops, and why that is not a weight: the `*.attn.freqs_cis` buffers.
They are complex64, which safetensors cannot store, and core deletes them at
load and recomputes them (same function, "computed dynamically").

`--dtype fp32` (the default) is lossless and is checked to be: every kept
tensor is read back from the output and compared with `torch.equal`. `fp16`
halves the file and is a cast; the readback then reports the largest absolute
difference instead of asserting equality. Core picks its inference dtype
itself (`SAM3.supported_inference_dtypes`), so fp32 on disk costs disk only.

It also checks the two keys core detects the model by
(`comfy/model_detection.py`, the SAM3 / SAM3.1 branch), so a file this writes
is one core will recognise, and says which of the two it is.

CPU only.

    CUDA_VISIBLE_DEVICES="" python bench/convert_sam3_checkpoint.py \\
        --src <facebook_sam3.1>/sam3.1_multiplex.pt \\
        --out <facebook_sam3.1>/sam3.1_multiplex_fp32.safetensors [--dtype fp16]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

#: The keys core's detection reads; the second separates 3.1 from 3.
DETECT_KEY = "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight"
QUERY_KEY = "detector.transformer.decoder.query_embed.weight"
SAM31_KEY = "detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    args = ap.parse_args()

    sd = torch.load(args.src, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(sd.get("model"), dict):
        sd = sd["model"]
    if DETECT_KEY not in sd or QUERY_KEY not in sd:
        print(f"{args.src} is not a SAM 3 checkpoint core would detect: missing "
              f"{DETECT_KEY if DETECT_KEY not in sd else QUERY_KEY}")
        return 1
    kind = "SAM31" if SAM31_KEY in sd else "SAM3"

    dropped = [k for k, v in sd.items() if torch.is_complex(v)]
    not_buffers = [k for k in dropped if ".attn.freqs_cis" not in k]
    if not_buffers:
        print(f"complex tensors that are not freqs_cis buffers, refusing to drop them: {not_buffers[:5]}")
        return 1
    target = torch.float16 if args.dtype == "fp16" else torch.float32
    out = {}
    for k, v in sd.items():
        if k in dropped:
            continue
        out[k] = v.to(target).contiguous() if v.is_floating_point() else v.contiguous()

    meta = {"source": os.path.basename(args.src), "source_sha256": sha256(args.src),
            "converter": "bench/convert_sam3_checkpoint.py", "dtype": args.dtype,
            "core_image_model": kind,
            "dropped": f"{len(dropped)} complex64 *.attn.freqs_cis buffers, recomputed by core at load"}
    save_file(out, args.out, metadata=meta)

    worst, worst_key = 0.0, None
    with safe_open(args.out, framework="pt") as f:
        if set(f.keys()) != set(out):
            print("readback: the key set differs from what was written")
            return 1
        for k in out:
            got, want = f.get_tensor(k), sd[k]
            if args.dtype == "fp32":
                if not torch.equal(got, want):
                    print(f"readback: {k} differs from the source")
                    return 1
            elif want.is_floating_point():
                d = float((got.float() - want.float()).abs().max())
                if d > worst:
                    worst, worst_key = d, k
    print(f"{kind}: {len(out)} tensors written to {args.out} as {args.dtype}, {len(dropped)} freqs_cis buffers dropped")
    print("readback: every tensor equals the source" if args.dtype == "fp32"
          else f"readback: largest absolute difference from the fp32 source {worst:.3e} at {worst_key}")
    print(f"source sha256 {meta['source_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
