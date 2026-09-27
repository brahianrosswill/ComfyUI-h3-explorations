#!/usr/bin/env python3
"""`core_sparse_capture.py` records core's sparse path and changes nothing.
CPU only, no server; stand-in blocks on the real `h3_capture` and core's real
`PRODUCER_CHUNK`. Cases:
  inert_passthrough  H3_CAPTURE unset: the block output is what core's patch
                     returns, and the projection is never computed
  output_unchanged   armed: the block output is bit-identical to unarmed
  core_attention     armed: the attention core's patch chose is the one that
                     runs, on the same input
  one_file           armed at block 1, step 1: exactly one qkvpre file, for
                     block 1, step 1, and none for any other call
  bytes_exact        its qkv equals the projection of that call's input built
                     in PRODUCER_CHUNK row chunks (several here), with the norm
                     weights. Chunked, not whole: a CPU matmul rounds by batch
                     size, and whether the real projection is row-independent
                     is what h3_capture's chunk_check measures on the card
  counts_once        each call advances its block's step by one: the file is
                     the SECOND call of block 1, not the first
  default_attention  with no core patch on a block, the block's own attention
                     runs
  control            planted: counting twice per call puts block 1 at step 2 by
                     its second call, so no step-1 file from that call exists
                     and one_file / counts_once would fail
    <comfy-venv-python> bench/check_core_sparse_capture.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parents[1]))
sys.path.insert(1, str(REPO))

import torch  # noqa: E402
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
import comfy_extras.nodes_sparse_attention as nsa  # noqa: E402
import h3_capture as cap  # noqa: E402
import core_sparse_capture as csc  # noqa: E402

HEADS, HEAD_DIM, HIDDEN = 2, 8, 16
results = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<18} {detail}")


class Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads, self.head_dim = HEADS, HEAD_DIM
        self.qkv_proj = torch.nn.Linear(HIDDEN, 3 * HEADS * HEAD_DIM, bias=False)
        self.q_norm = torch.nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.k_norm = torch.nn.RMSNorm(HEAD_DIM, eps=1e-6)
        self.proj_calls = 0
        orig = self.qkv_proj.forward

        def counted(x):
            self.proj_calls += 1
            return orig(x)
        self.qkv_proj.forward = counted

    def forward(self, h, rope_freqs=None, transformer_options={}):
        return h * 2.0


def core_like_patch(seen):
    """Core's block patch in miniature: it chooses an attention and calls original_block."""
    def sparse_attention(h, rope_freqs=None, transformer_options={}):
        seen.append(h)
        return h * 3.0

    def patch(args, extra):
        return extra["original_block"]({**args, "attention": sparse_attention})
    return patch


def original_block(args):
    return {"img": args["attention"](args["img"], rope_freqs=args["rope_freqs"],
                                     transformer_options=args["transformer_options"])}


def run(block_patches, inputs, rope):
    outs = []
    for step_inputs in inputs:                       # one sampling step
        for i, bp in enumerate(block_patches):       # blocks in order
            args = {"img": step_inputs[i], "rope_freqs": rope, "transformer_options": {"block_index": i}}
            outs.append(bp(args, {"original_block": original_block})["img"])
    return outs


def main() -> int:
    torch.manual_seed(0)
    nsa.PRODUCER_CHUNK = 5                           # several chunks over 12 rows
    blocks = [types.SimpleNamespace(attn=Attn()) for _ in range(2)]
    for i, b in enumerate(blocks):
        b.attn._h3_block_index = i
    rope = torch.randn(12, 4, 2, 2)
    inputs = [[torch.randn(12, HIDDEN) for _ in blocks] for _ in range(3)]

    os.environ.pop("H3_CAPTURE", None)
    seen = []
    plain = [core_like_patch(seen) for _ in blocks]
    ref = run(plain, inputs, rope)
    seen.clear()
    wrapped = [csc._wrap(b, core_like_patch(seen)) for b in blocks]
    got = run(wrapped, inputs, rope)
    calls = sum(b.attn.proj_calls for b in blocks)
    case("inert_passthrough", all(torch.equal(a, b) for a, b in zip(ref, got)) and calls == 0,
         f"{calls} projections unarmed")

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["H3_CAPTURE"] = f"dir={tmp},blocks=1,steps=1,pre=1"
        cap._sync_spec()
        seen.clear()
        got = run(wrapped, inputs, rope)
        case("output_unchanged", all(torch.equal(a, b) for a, b in zip(ref, got)))
        flat = [x for step in inputs for x in step]
        case("core_attention", len(seen) == len(flat) and all(s is x for s, x in zip(seen, flat)),
             f"{len(seen)} core attention calls")
        files = sorted(Path(tmp).glob("qkvpre_*.pt"))
        case("one_file", len(files) == 1 and "_b1_s1" in files[0].name, ", ".join(f.name for f in files))
        rec = torch.load(files[0], weights_only=True) if files else {}
        want = inputs[1][1]
        full = csc._host_projection(blocks[1].attn, want)
        ok = (bool(rec) and torch.equal(rec["qkv"], full)
              and torch.equal(rec["q_norm_weight"], blocks[1].attn.q_norm.weight.detach())
              and rec["block"] == 1 and rec["step"] == 1)
        case("bytes_exact", ok, f"chunk {nsa.PRODUCER_CHUNK} over {want.shape[0]} rows")
        wrong = inputs[0][1]
        first = csc._host_projection(blocks[1].attn, wrong)
        case("counts_once", bool(rec) and torch.equal(rec["qkv"], full) and not torch.equal(rec["qkv"], first),
             "step 1 is the second call")

        # default_attention: no core patch on the block
        os.environ.pop("H3_CAPTURE", None)
        cap._sync_spec()
        lone = csc._wrap(blocks[0], None)
        out = lone({"img": inputs[0][0], "rope_freqs": rope, "transformer_options": {}},
                   {"original_block": original_block})["img"]
        case("default_attention", torch.equal(out, inputs[0][0] * 2.0), "block.attn ran")

    # control: count twice per call, and the step-1 file comes from the first call's input
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["H3_CAPTURE"] = f"dir={tmp},blocks=1,steps=1,pre=1"
        cap._sync_spec()
        real = cap.count_call

        def twice(m):
            real(m)
            real(m)
        cap.count_call = twice
        try:
            fresh = [types.SimpleNamespace(attn=Attn()) for _ in range(2)]
            for i, b in enumerate(fresh):
                b.attn._h3_block_index = i
                b.attn.load_state_dict(blocks[i].attn.state_dict())
            cap._calls.clear()
            run([csc._wrap(b, core_like_patch([])) for b in fresh], inputs, rope)
            files = sorted(Path(tmp).glob("qkvpre_*.pt"))
            rec = torch.load(files[0], weights_only=True) if files else {}
            second = csc._host_projection(fresh[1].attn, inputs[1][1])
            planted_caught = not (bool(rec) and torch.equal(rec["qkv"], second))
        finally:
            cap.count_call = real
            os.environ.pop("H3_CAPTURE", None)
            cap._sync_spec()
        case("control", planted_caught, f"double counting: {len(files)} step-1 files, none from the second call")

    print("ok" if all(results) else "FAIL", "--", f"{sum(results)}/{len(results)} cases")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
