#!/usr/bin/env python3
"""Write netflix/void-model's checkpoint under the key names ComfyUI core loads, from the original. Header only.

Core's `UNETLoader` refuses the upstream file ("Could not detect model type"):
it is in diffusers' layout (`transformer_blocks.N.attn1.to_q...`), and core's
CogVideoX detection looks for `blocks.0.norm1.linear.weight`
(`comfy/model_detection.py`). Core's own VOID template names a repack,
`Comfy-Org/void-model` `diffusion_models/void_pass1.safetensors`. This script
makes the file core loads from the original, as
`bench/convert_sam3d_body_checkpoint.py` does for SAM 3D Body.

**The repack is a rename and nothing else** (seen 2026-10-05 by comparing
headers: the same tensors, the same shapes, the same dtype, and a file size
that differs from upstream's by exactly the difference in header length). So
nothing is fused, split or transposed, and this script reads no tensor: it
writes a safetensors file whose data section is the upstream file's, byte for
byte and in the same order, under a header with core's names (`rename`):

    transformer_blocks.N.attn1.to_q       -> blocks.N.q        (to_k, to_v alike)
    transformer_blocks.N.attn1.to_out.0   -> blocks.N.attn_out
    transformer_blocks.N.attn1.norm_q     -> blocks.N.norm_q   (norm_k alike)
    transformer_blocks.N.ff.net.0.proj    -> blocks.N.ff_proj
    transformer_blocks.N.ff.net.2         -> blocks.N.ff_out
    transformer_blocks.N.norm1 / norm2    -> blocks.N.norm1 / norm2
    time_embedding.linear_1 / linear_2    -> time_embedding_linear_1 / _2
    everything else                       unchanged

`--verify` compares the result with the Hub repack without downloading it:
every key, shape and dtype against the repack's header, and the bytes of a
sample of tensors by HTTP range request (`SAMPLED_BLOCKS`, `SAMPLE_BYTES`).
`bench/check_void_conversion.py` holds the rename to core's own model without
weights. The run on pass 1 is recorded in
`bench/results/2026-10-05_void_checkpoint_conversion.md`.

    <comfy venv python> bench/convert_void_checkpoint.py <upstream.safetensors> <out.safetensors> [--verify]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

REPACK_URL = "https://huggingface.co/Comfy-Org/void-model/resolve/main/diffusion_models/{name}"
#: diffusers' name for each part of a block, and core's (`comfy/ldm/cogvideo/model.py`). Seen in the two headers.
INNER = {"attn1.to_q": "q", "attn1.to_k": "k", "attn1.to_v": "v", "attn1.to_out.0": "attn_out",
         "attn1.norm_q": "norm_q", "attn1.norm_k": "norm_k", "ff.net.0.proj": "ff_proj", "ff.net.2": "ff_out"}
#: Bytes of each sampled tensor that `--verify` compares; a smaller tensor is compared whole. Reasoned: enough
#: to tell two weights apart without fetching the large ones.
SAMPLE_BYTES = 65536
#: How many blocks `--verify` samples, spread over the depth: the first, the middle, the last. Reasoned.
SAMPLED_BLOCKS = 3


def rename(key: str) -> str:
    """Core's name for an upstream key."""
    m = re.match(r"transformer_blocks\.(\d+)\.(.+)\.(weight|bias)$", key)
    if m:
        return f"blocks.{m[1]}.{INNER.get(m[2], m[2])}.{m[3]}"
    return key.replace("time_embedding.linear_", "time_embedding_linear_")


def read_header(path: Path) -> tuple[dict, int]:
    """A safetensors file's header without its metadata, and where its data section starts."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def convert(src: Path, dst: Path) -> None:
    header, data_at = read_header(src)
    names = [rename(k) for k in header]
    if len(set(names)) != len(names):
        raise SystemExit("the rename is not one to one: two upstream keys would share a name")
    out = {"__metadata__": {"format": "pt", "converted_from": f"netflix/void-model {src.name}",
                            "conversion": "keys renamed to ComfyUI core's CogVideoX names; data section unchanged",
                            "converter": "ComfyUI-h3-explorations bench/convert_void_checkpoint.py"}}
    out.update({rename(k): v for k, v in header.items()})
    raw = json.dumps(out, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    tmp = dst.with_suffix(".partial")
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        fout.write(struct.pack("<Q", len(raw)))
        fout.write(raw)
        fin.seek(data_at)
        while chunk := fin.read(64 << 20):
            fout.write(chunk)
    os.chmod(tmp, 0o644)   # the server may run under another account
    tmp.rename(dst)
    print(f"wrote {dst.name}: {len(names)} tensors, {dst.stat().st_size} bytes")


def sample_keys(keys) -> list[str]:
    """Every key outside the blocks, and every key of `SAMPLED_BLOCKS` blocks spread over the depth."""
    blocks = sorted({int(k.split(".")[1]) for k in keys if k.startswith("blocks.")})
    picked = {blocks[(i * (len(blocks) - 1)) // (SAMPLED_BLOCKS - 1)] for i in range(SAMPLED_BLOCKS)} if blocks else set()
    return [k for k in sorted(keys) if not k.startswith("blocks.") or int(k.split(".")[1]) in picked]


def verify(dst: Path, name: str) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from convert_sam3d_body_checkpoint import Repack  # the pack's own range reader for a published safetensors file
    theirs = Repack(REPACK_URL.format(name=name))
    ours, our_data = read_header(dst)
    problems = []
    if set(ours) != set(theirs.header):
        problems.append(f"key sets differ: {len(set(ours) - set(theirs.header))} only here, "
                        f"{len(set(theirs.header) - set(ours))} only in the repack")
    shared = set(ours) & set(theirs.header)
    for k in sorted(shared):
        if ours[k]["shape"] != theirs.header[k]["shape"] or ours[k]["dtype"] != theirs.header[k]["dtype"]:
            problems.append(f"{k}: {ours[k]['dtype']} {ours[k]['shape']} here, "
                            f"{theirs.header[k]['dtype']} {theirs.header[k]['shape']} in the repack")
    sample = sample_keys(shared)
    compared = whole = 0
    digest = hashlib.sha256()
    with open(dst, "rb") as f:
        for k in sample:
            a0, a1 = ours[k]["data_offsets"]
            b0 = theirs.header[k]["data_offsets"][0]
            size = min(a1 - a0, SAMPLE_BYTES)
            f.seek(our_data + a0)
            mine = f.read(size)
            digest.update(mine)
            if mine != theirs._get(f"bytes={theirs.base + b0}-{theirs.base + b0 + size - 1}"):
                problems.append(f"{k}: the first {size} bytes differ from the repack's")
            compared += size
            whole += size == a1 - a0
    for p in problems[:20]:
        print("FAIL ", p)
    print(f"{len(ours)} tensors against the repack's header; bytes of {len(sample)} tensors compared "
          f"({whole} whole, the rest their first {SAMPLE_BYTES} bytes, {compared} bytes in all, sha256 of them "
          f"{digest.hexdigest()[:16]}); {len(problems)} problem(s)")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="netflix/void-model's checkpoint under the key names ComfyUI core loads")
    ap.add_argument("src", type=Path, help="the upstream file, e.g. void_pass1.safetensors from netflix/void-model")
    ap.add_argument("dst", type=Path, help="the file to write")
    ap.add_argument("--verify", action="store_true", help="compare the result with the Hub repack by range request")
    ap.add_argument("--repack-name", default=None, help="the repack's file name on the Hub; the source's name by default")
    args = ap.parse_args()
    if not args.dst.exists():
        convert(args.src, args.dst)
    else:
        print(f"{args.dst.name} exists; not written again")
    return verify(args.dst, args.repack_name or args.src.name) if args.verify else 0


if __name__ == "__main__":
    sys.exit(main())
