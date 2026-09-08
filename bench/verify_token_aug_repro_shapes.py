#!/usr/bin/env python3
"""Which shapes reproduce the `token_aug` nondeterminism IN A FRESH PROCESS?

`bench/minimise_token_aug_repro.py` searched for a small reproducing slice
inside one process and found one, and the artifact it saved then failed to
reproduce when `bench/repro_token_aug_nondeterminism.py` ran it cold. That is a
methodology error worth naming rather than quietly fixing: the phenomenon is an
intermittent, occupancy-sensitive race, so "this shape moved during a sweep"
does not carry to "this shape moves on a cold start". A repro is only a repro
if it works the way the person receiving it will run it.

So this drives the standalone script as a SUBPROCESS, once per candidate shape,
each with its own CUDA context. Nothing is inherited between candidates: no
warm allocator, no prior launches, no workspace already sized.

Two honest limits on the output:

  * A shape that fails here failed in ONE cold process at this launch count. It
    is "not seen", never "stable". The whole reason this file exists is that
    this effect hides.
  * A shape that passes here passed on this GPU, this driver and this build.
    Occupancy is the variable that decides, so another card can differ.

Ranking is by artifact size, because the point of the exercise is something
small enough to attach to an issue.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--scratch", required=True, help="where candidate slices are written")
    ap.add_argument("--launches", type=int, default=60)
    ap.add_argument("--token-aug", type=int, default=64)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--keep", help="save the smallest passing slice here")
    ap.add_argument("--out")
    args = ap.parse_args()

    import torch
    sys.path.insert(0, str(_HERE))
    from probe_token_aug_selection_structure import load_cell
    from minimise_token_aug_repro import unstable_heads, kernel_build

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    q, k, v, meta = load_cell(args.cell)
    H, T = meta["heads"], meta["seq_len"]
    heads = unstable_heads(q, k, v, args.token_aug, args.tau)
    print(f"cell: block {meta['block']} step {meta['step']} seq {T} heads {H}")
    print(f"heads that move at full size: {len(heads)} of {H}\n")

    # Candidates: head counts crossed with lengths, always including an
    # unstable head first. Ordered small-first so the first pass is the prize.
    keep0 = heads[0]
    others = [i for i in range(H) if i != keep0]
    cands = []
    for nh in (2, 4, 8, 16, 56):
        if nh > H:
            continue
        for L in (8192, 16384, 32768, 65536, T):
            if L > T:
                continue
            cands.append((nh, L))
    cands.sort(key=lambda c: c[0] * c[1])

    results, passing = [], []
    for nh, L in cands:
        idx = torch.tensor([keep0] + others[:nh - 1], device="cuda")
        sq, sk, sv = (t.index_select(2, idx)[:, :L, :, :].contiguous()
                      for t in (q, k, v))
        path = scratch / f"cand_h{nh}_l{L}.pt"
        torch.save({"q": sq.cpu(), "k": sk.cpu(), "v": sv.cpu(),
                    "token_aug": args.token_aug, "tau": args.tau,
                    "source_block": meta["block"], "source_step": meta["step"]}, path)
        del sq, sk, sv
        torch.cuda.empty_cache()
        size = path.stat().st_size
        # Fresh process: its own CUDA context, allocator and workspace.
        proc = subprocess.run(
            [sys.executable, str(_HERE / "repro_token_aug_nondeterminism.py"),
             "--slice", str(path), "--launches", str(args.launches)],
            capture_output=True, text=True)
        ok = proc.returncode == 1        # the script exits 1 when it reproduces
        row = {"heads": nh, "length": L, "bytes": size, "reproduces_cold": ok,
               "launches": args.launches}
        if not ok:
            row["reading"] = (f"not seen in {args.launches} cold launches; not "
                              f"a claim that this shape is stable")
        results.append(row)
        print(f"  heads {nh:2d} len {L:6d}  {size / 1e6:6.0f} MB  "
              f"{'REPRODUCES' if ok else 'not seen'}")
        if ok:
            passing.append((size, path, row))
        else:
            path.unlink(missing_ok=True)

    record = {
        "what": "which q/k/v shapes reproduce token_aug's nondeterminism in a "
                "cold process, which is the only way a repro is useful to "
                "someone else",
        "produced_by": "bench/verify_token_aug_repro_shapes.py",
        "kernel_build": kernel_build(),
        "gpu": torch.cuda.get_device_name(),
        "source_cell": Path(args.cell).name,
        "token_aug": args.token_aug, "tau": args.tau,
        "launches_per_candidate": args.launches,
        "unstable_heads_at_full_size": len(heads),
        "candidates": results,
        "smallest_reproducing_bytes": min((p[0] for p in passing), default=None),
        "cannot_settle": [
            "that a non-reproducing shape is correct. Reproduction is "
            "occupancy-sensitive and non-monotone in shape, so a negative is "
            "'not seen at this launch count on this device'.",
            "that a reproducing shape carries to another GPU or driver. "
            "Occupancy is the variable that decides.",
        ],
    }
    if passing and args.keep:
        passing.sort(key=lambda p: p[0])
        Path(args.keep).write_bytes(passing[0][1].read_bytes())
        record["kept"] = {"bytes": passing[0][0], **passing[0][2]}
        print(f"\nkept smallest passing shape: {passing[0][0] / 1e6:.0f} MB")
    elif not passing:
        print("\nNo candidate reproduced cold; the full cell remains the repro.")
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
