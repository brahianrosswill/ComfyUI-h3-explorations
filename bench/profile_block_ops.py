#!/usr/bin/env python3
"""Every non-attention op of one H3 DiT block, timed alone, at a ref2va length.

Replays core's `DiTBlock.forward` op by op (comfy/ldm/minimax/model.py) on block
0's real weights from the ref2va int8 convrot checkpoint, with seeded random bf16
activations, the card to itself. Attention itself is not run; its q/k/v are
produced and its output is a random tensor. Events around ITERS calls after a
warmup; x50 is one evaluation's worth.

What it cannot see: weight streaming, allocator pressure with half the DiT
resident, object-patch wrappers, Python. The live remainder minus this sum is
those.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

WARMUP, ITERS = 2, 5
HEADS, HEAD_DIM, HIDDEN = 56, 128, 5376


def timed(fn, iters=ITERS):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--m", type=int, default=119485)
    ap.add_argument("--segments", type=int, nargs="+", default=[791, 7360, 7360, 1150, 102816],
                    help="contiguous mod segments (rows); the remainder joins the last")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.comfy)

    import comfy_kitchen as ck
    from safetensors import safe_open
    pkg = Path(str(ck.__file__)).resolve().parent
    infos = sorted(pkg.parent.glob("comfy_kitchen-*.dist-info"))
    version = infos[0].name[len("comfy_kitchen-"):-len(".dist-info")] if infos else "unknown"

    w = {}
    with safe_open(args.ckpt, "pt", device="cpu") as f:
        for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            w[name] = (f.get_tensor(f"blocks.0.{name}.weight").cuda(),
                       f.get_tensor(f"blocks.0.{name}.weight_scale").cuda())
        for name in ("norm1", "norm2", "attn.q_norm", "attn.k_norm"):
            w[name] = f.get_tensor(f"blocks.0.{name}.weight").cuda()
        inv_freq = f.get_tensor("rope.inv_freq").cuda()

    M = args.m
    bf = torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(1234)
    x = torch.randn(M, HIDDEN, device="cuda", dtype=bf, generator=g)
    segs, a = [], 0
    for n in args.segments:
        segs.append((a, min(a + n, M)))
        a += n
    segs[-1] = (segs[-1][0], M)
    scale = [torch.randn(HIDDEN, device="cuda", dtype=bf, generator=g) * 0.1 for _ in segs]
    shift = [torch.randn(HIDDEN, device="cuda", dtype=bf, generator=g) * 0.1 for _ in segs]
    gate = [torch.randn(HIDDEN, device="cuda", dtype=bf, generator=g) * 0.1 for _ in segs]

    # rope table as core builds it: [1, S, 1, 48, 2, 2]
    pos = torch.rand(M, 3, device="cuda", generator=g) * 100.0
    per_axis = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    ang = torch.cat(per_axis.unbind(dim=1), dim=-1)            # [S, 48]
    c, s = torch.cos(ang), torch.sin(ang)
    rope = torch.stack([c, -s, s, c], dim=-1).reshape(1, M, 1, ang.shape[-1], 2, 2).to(bf)

    def lin(name, inp, **kw):
        q, sc = w[name]
        return ck.int8_linear(inp, q, sc, None, bf, convrot=True, convrot_groupsize=256, **kw)

    rows = {}

    def rec(name, fn, note=""):
        ms = timed(fn)
        rows[name] = round(ms, 3)
        print(f"{name:34s} {ms:9.3f} ms  {note}", flush=True)

    def norm(wn):
        return torch.nn.functional.rms_norm(x, (HIDDEN,), w[wn], 1e-5)

    rec("norm1 (F.rms_norm)", lambda: norm("norm1"))
    h = norm("norm1")

    def mod(t):
        for (a, b), sc, sh in zip(segs, scale, shift):
            t[a:b].mul_(1.0 + sc).add_(sh)
        return t

    rec("mod scale+shift (mul_, add_)", lambda: mod(h))
    rec("qkv int8_linear", lambda: lin("attn.qkv_proj", h))
    qkv = lin("attn.qkv_proj", h)
    inner = HEADS * HEAD_DIM
    qq, kk, vv = qkv.split(inner, dim=-1)
    q4 = qq.view(1, M, HEADS, HEAD_DIM)
    k4 = kk.view(1, M, HEADS, HEAD_DIM)
    rec("rms_rope_split_half_ (q,k)",
        lambda: ck.rms_rope_split_half_(q4, k4, rope, w["attn.q_norm"], w["attn.k_norm"],
                                        epsilon=1e-5, rot_dim=rope.shape[-3] * 2))
    del qkv, qq, kk, vv, q4, k4
    torch.cuda.empty_cache()

    attn_out = torch.randn(M, inner, device="cuda", dtype=bf, generator=g)
    rec("out_proj int8_linear", lambda: lin("attn.out_proj", attn_out))
    o = lin("attn.out_proj", attn_out)

    def gate_add(res, other):
        for (a, b), gt in zip(segs, gate):
            res[a:b].addcmul_(other[a:b], gt)
        return res

    rec("gate (addcmul_)", lambda: gate_add(x, o))
    del attn_out, o
    torch.cuda.empty_cache()

    rec("norm2 (F.rms_norm)", lambda: norm("norm2"))
    h = norm("norm2")
    rec("mod scale+shift #2", lambda: mod(h))
    rec("fc1 int8_linear", lambda: lin("mlp.fc1", h))
    h1 = lin("mlp.fc1", h)
    rec("fc2 int8_linear + swiglu", lambda: lin("mlp.fc2", h1, input_act="swiglu"))
    o = lin("mlp.fc2", h1, input_act="swiglu")
    rec("gate #2 (addcmul_)", lambda: gate_add(x, o))

    # what the LoRA branch's fc2 path recomputes
    def swiglu_chunks():
        for a in range(0, M, 16384):
            gt, up = h1[a:a + 16384].chunk(2, dim=-1)
            torch.nn.functional.silu(gt).mul_(up)
    rec("eager swiglu, 16k-row chunks", swiglu_chunks, "(LoRA fc2 branch only)")

    core = [k for k in rows if "eager" not in k]
    total = sum(rows[k] for k in core)
    gemm = sum(v for k, v in rows.items() if "int8_linear" in k)
    print(f"\nblock non-attention sum {total:.1f} ms  (GEMMs {gemm:.1f}, elementwise {total - gemm:.1f})")
    print(f"x50 blocks: {total * 50 / 1000:.2f} s per evaluation "
          f"(GEMMs {gemm * 50 / 1000:.2f}, elementwise {(total - gemm) * 50 / 1000:.2f})")
    Path(args.out).write_text(json.dumps({
        "kitchen": version, "torch": torch.__version__, "m": M, "segments": segs,
        "iters": ITERS, "ms": rows, "block_sum_ms": round(total, 2),
        "gemm_ms": round(gemm, 2), "per_eval_s": round(total * 50 / 1000, 3)}, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
