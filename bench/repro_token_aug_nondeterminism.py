#!/usr/bin/env python3
"""Standalone repro: comfy_kitchen `sol_attn` is not deterministic with `token_aug`.

Self-contained on purpose. It imports torch and comfy_kitchen and nothing from
this repo, so it can be attached to an upstream issue beside its input file and
run by someone who has neither.

    python repro_token_aug_nondeterminism.py --slice token_aug_slice.pt

WHAT IT SHOWS
    The same q/k/v, the same arguments, the same process: repeated `sol_attn`
    calls with `token_aug` on return different outputs. With `token_aug` off
    the same inputs are bitwise identical over the same number of launches, so
    the variation belongs to that code path rather than to attention generally.

    The deltas are far too large to be reduction order. Summing the same
    contributions in a different order cannot move an output by many times its
    own magnitude; a different set of contributions can.

WHAT IT DOES NOT SHOW
    Which set changed, or why. That is not observable from outside the kernel.

    The public docstring says the selection uses whole histogram bins "so the
    set never depends on scheduling". Reading `sol_attn_token.cu`, the property
    that has to hold for that to be true is that the number of tokens clearing
    pass 2's threshold never exceeds `n_tok` -- because a token that clears it
    takes a slot with `atomicAdd(&tok_cnt[...])` and is written only
    `if (slot < n_tok)`, so above the budget the survivors are decided by the
    order the atomics land in. This script does not verify that the count ever
    exceeds the budget; `tok_cnt` is not exposed to callers.

ANYTHING THAT SERIALISES THE LAUNCHES HIDES IT
    The race needs the launches to overlap. A synchronising call in the loop is
    enough to suppress it, and allocation counts: with an empty caching
    allocator the underlying cudaMalloc synchronises, and that alone was enough
    here to make 60 launches look perfectly stable on a slice that otherwise
    moves every time. This script therefore warms up before measuring, and
    reads nothing back to the host inside the loop. If you adapt it, keep both
    properties or you will get a false negative.

SENSITIVE TO OCCUPANCY, WHICH IS WHY THE INPUT IS SHAPED THE WAY IT IS
    Whether it reproduces depends on the head count and sequence length in a
    non-monotone way: some shapes reproduce and slightly larger ones do not.
    That is what an intermittent scheduling race looks like from outside, and
    it means a negative on ONE shape is not evidence of correctness. The
    shipped slice is a shape that reproduces here on an RTX 4090 (sm89).
"""
from __future__ import annotations

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", required=True, help="the .pt holding q, k, v")
    ap.add_argument("--launches", type=int, default=40)
    args = ap.parse_args()

    import torch
    import comfy_kitchen as ck
    try:
        import importlib.metadata
        build = importlib.metadata.version("comfy-kitchen")
    except Exception:
        build = "<unknown>"

    # weights_only: this file is meant to be downloaded from an issue and run
    # by someone who did not make it. It holds tensors and scalars only.
    d = torch.load(args.slice, map_location="cuda", weights_only=True)
    q, k, v = (d[n].to("cuda") for n in ("q", "k", "v"))
    token_aug, tau = int(d.get("token_aug", 64)), float(d.get("tau", 1.0))
    print(f"comfy-kitchen {build} on {torch.cuda.get_device_name()}")
    print(f"q/k/v {tuple(q.shape)} {q.dtype}, tau={tau}, token_aug={token_aug}, "
          f"{args.launches} launches\n")

    failed = False
    for label, aug in (("token_aug=0 (control)", 0), (f"token_aug={token_aug}", token_aug)):
        kw = {"tau": tau}
        if aug:
            kw["token_aug"] = aug
        for _ in range(2):        # warm the allocator; see the docstring
            del_me = ck.sol_attn(q, k, v, **kw)
            del del_me
        first = ck.sol_attn(q, k, v, **kw).clone()
        # Accumulators stay on the device: float()/int() here would synchronise
        # every iteration and that is enough to hide the race.
        peak_t = torch.zeros((), device=q.device)
        moved_t = torch.zeros(first.shape[1], dtype=torch.bool, device=q.device)
        for _ in range(args.launches - 1):
            out = ck.sol_attn(q, k, v, **kw)
            per_row = (out.float() - first.float()).abs().amax(dim=(0, 2, 3))
            peak_t = torch.maximum(peak_t, per_row.max())
            moved_t |= per_row > 0
            del out, per_row
        scale = float(first.abs().float().mean())
        peak, rows = float(peak_t), int(moved_t.sum())
        verdict = "NOT deterministic" if rows else "deterministic"
        print(f"  {label:22s} {verdict:18s} rows moved {rows:5d}/{first.shape[1]}  "
              f"max|delta| {peak:.4g} vs mean|out| {scale:.4g}")
        if rows and aug:
            failed = True
        if rows and not aug:
            print("       the control moved too, so this run says nothing about "
                  "token_aug specifically")
        del first
        torch.cuda.empty_cache()

    print()
    if failed:
        print("REPRODUCED: identical inputs, different outputs, with token_aug on.")
    else:
        print("Not reproduced in this many launches on this device. That is not "
              "proof of correctness: reproduction is occupancy-sensitive and "
              "non-monotone in shape (see the module docstring).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
