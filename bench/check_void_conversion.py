#!/usr/bin/env python3
"""Hold `bench/convert_void_checkpoint.py`'s rename to core's CogVideoX model, without weights.

Core loads a VOID checkpoint only if its keys are the model's own, so the
rename is right only if the names it produces from upstream's keys are exactly
the state-dict keys of `comfy/ldm/cogvideo/model.py::CogVideoXTransformer3DModel`.
A wrong rename does not fail quietly here (core refuses the file), but a
converter that drifts from core after an update would write a file nobody can
load, and this says so without an eleven-gigabyte run.

1. **The rename lands on core's model.** Upstream's key names, by pattern as
   read from `void_pass1.safetensors` on 2026-10-05, renamed, equal the keys
   of core's model built at the checkpoint's layout and a small width. Names
   do not depend on the width.
2. **It is one to one**, and the converter refuses a header in which it is
   not.
3. **The written file is the source's data under new names.** On a small
   synthetic file: the data section is byte for byte the source's, every
   tensor reads back equal under its new name, and the metadata says where
   the file came from.
4. **The byte sample covers the model's ends and its middle**: every tensor
   outside the blocks, and the first, a middle and the last block.

No weights, no CUDA, no server, no network.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_void_conversion.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lib import bootstrap, case, finish  # noqa: E402

bootstrap(cpu=True)

import torch  # noqa: E402

import convert_void_checkpoint as C  # noqa: E402

#: The checkpoint's layout, as core's detection reads it from the converted file
#: (`comfy.model_detection`, seen 2026-10-05): the depth, and what decides which keys exist.
LAYERS = 42


def upstream_keys() -> list[str]:
    """Every key of upstream's `void_pass1.safetensors`, by pattern."""
    keys = []
    for n in range(LAYERS):
        b = f"transformer_blocks.{n}"
        for part in ("attn1.norm_k", "attn1.norm_q", "attn1.to_k", "attn1.to_out.0", "attn1.to_q", "attn1.to_v",
                     "ff.net.0.proj", "ff.net.2", "norm1.linear", "norm1.norm", "norm2.linear", "norm2.norm"):
            keys += [f"{b}.{part}.weight", f"{b}.{part}.bias"]
    for part in ("norm_final", "norm_out.linear", "norm_out.norm", "patch_embed.proj", "patch_embed.text_proj",
                 "proj_out", "time_embedding.linear_1", "time_embedding.linear_2"):
        keys += [f"{part}.weight", f"{part}.bias"]
    return keys


def core_keys() -> set[str]:
    import comfy.ops
    from comfy.ldm.cogvideo.model import CogVideoXTransformer3DModel
    model = CogVideoXTransformer3DModel(
        num_attention_heads=2, attention_head_dim=64, in_channels=48, time_embed_dim=512, text_embed_dim=4096,
        num_layers=LAYERS, patch_size=2, patch_size_t=2, use_learned_positional_embeddings=False,
        use_rotary_positional_embeddings=True, dtype=torch.float32, device="cpu",
        operations=comfy.ops.disable_weight_init)
    return set(model.state_dict().keys())


def lands_on_core():
    mapped, core = {C.rename(k) for k in upstream_keys()}, core_keys()
    assert mapped == core, (f"{len(mapped - core)} renamed key(s) core's model does not have, e.g. "
                            f"{sorted(mapped - core)[:3]}; {len(core - mapped)} of core's the rename does not produce, "
                            f"e.g. {sorted(core - mapped)[:3]}")
    return f"{len(core)} keys"


def one_to_one():
    keys = upstream_keys()
    assert len({C.rename(k) for k in keys}) == len(keys), "two upstream keys share a name after the rename"
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "clash.safetensors"
        save_file({"time_embedding.linear_1.weight": torch.zeros(2), "time_embedding_linear_1.weight": torch.ones(2)}, str(src))
        try:
            C.convert(src, Path(tmp) / "out.safetensors")
        except SystemExit as exc:
            assert "one to one" in str(exc), exc
        else:
            raise AssertionError("a header whose rename is not one to one was converted")


def file_is_the_source():
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    torch.manual_seed(0)
    original = {k: torch.randn(3, 2).to(torch.bfloat16) if k.endswith("weight") else torch.randn(3).to(torch.bfloat16)
                for k in upstream_keys() if k.startswith(("transformer_blocks.0.", "transformer_blocks.41.", "time_", "proj_out"))}
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "void_pass1.safetensors", Path(tmp) / "out.safetensors"
        save_file(original, str(src))
        C.convert(src, dst)
        _, src_at = C.read_header(src)
        _, dst_at = C.read_header(dst)
        assert src.read_bytes()[src_at:] == dst.read_bytes()[dst_at:], "the data section is not the source's, byte for byte"
        back = load_file(str(dst))
        assert set(back) == {C.rename(k) for k in original}, "the written keys are not the renamed keys"
        for k, v in original.items():
            assert torch.equal(back[C.rename(k)], v), f"{k} does not read back equal under {C.rename(k)}"
        with safe_open(str(dst), framework="pt") as f:
            meta = f.metadata() or {}
        assert "netflix/void-model void_pass1.safetensors" in meta.get("converted_from", ""), meta
        assert (dst.stat().st_mode & 0o044) == 0o044, "the written file is not readable by group and others"


def sample_covers():
    keys = [C.rename(k) for k in upstream_keys()]
    sample = C.sample_keys(keys)
    outside = [k for k in keys if not k.startswith("blocks.")]
    assert set(outside) <= set(sample), "a tensor outside the blocks is not in the byte sample"
    blocks = sorted({int(k.split(".")[1]) for k in sample if k.startswith("blocks.")})
    assert len(blocks) == C.SAMPLED_BLOCKS and blocks[0] == 0 and blocks[-1] == LAYERS - 1 and 0 < blocks[1] < LAYERS - 1, blocks
    per_block = len([k for k in keys if k.startswith("blocks.0.")])
    assert len(sample) == len(outside) + C.SAMPLED_BLOCKS * per_block, "a sampled block is not sampled whole"
    return f"blocks {blocks}"


def main() -> int:
    case("the rename lands on core's CogVideoX model", lands_on_core)
    case("the rename is one to one, and a clash is refused", one_to_one)
    case("the written file is the source's data under new names", file_is_the_source)
    case("the byte sample covers the ends and the middle", sample_covers)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
