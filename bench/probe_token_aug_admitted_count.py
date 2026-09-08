#!/usr/bin/env python3
"""Does the admitted token count exceed the budget, and is that where the output moves?

This is open experiment 29's arm 5, and it turned out to need no kernel change.

The mechanism read out of `sol_attn_token.cu` is: pass 2 admits whole histogram
bins down to the lowest one that still fits `n_tok`, every token clearing that
threshold takes a slot with `atomicAdd(&tok_cnt[...])`, and a token is written
to the list only `if (slot < n_tok)`. So the selected set is independent of
scheduling exactly while the number of tokens clearing the threshold is at most
the budget. Above it, which tokens survive is decided by the order the atomics
land in.

`tok_cnt` is that count, and it counts every token that cleared the threshold,
including the ones that then lost their slot -- the atomic increments before the
bound is tested. The public `sol_attn` never returns it, which is why the entry
said an out-parameter was needed. But the Python wrapper allocates the kernel's
workspace itself and the plan already exports a `tokCnt` offset, so the count
can be read by calling the same C entry point with a workspace we keep. No
kernel build, no wheel swap.

## The three things it reports, in increasing order of what they settle

1. **Does anything overflow.** If no entry ever exceeds `n_tok`, the mechanism
   is REFUTED and the cause is elsewhere. This is the arm's main value: it can
   come back negative.
2. **Is the overflow set stable across launches.** The mechanism says which
   groups overflow is a deterministic function of the input, and only the
   winners inside them are racy.
3. **Do the overflowing groups coincide with the moving output rows.** This is
   the one that turns a fingerprint into a cause. Overflow elsewhere in the
   tensor, with the output moving somewhere else, would mean the two are
   unrelated however suggestive each looks alone.

## The control that has to pass first

The public wrapper is not called here -- its argument preparation is replicated
so the workspace can be kept. If that replication is wrong, everything below
describes a different computation. So the run first asserts that the replicated
path and `ck.sol_attn` return bitwise-identical output with `token_aug` off,
where both are deterministic. A failure there voids the run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Inherited from comfy_kitchen sol_layout.cuh: one token-routing centroid is
# shared by TOK_GROUP query blocks of BLOCK rows.
BLOCK = 64
TOK_GROUP = 2


def kernel_build():
    import importlib.metadata
    try:
        return importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        return "<no dist metadata>"


def _call_keeping_workspace(q, k, v, tau, token_aug):
    """`sol_attn` with its workspace kept, so the token stage can be read.

    Mirrors the wrapper's body for the simple case only: no sinks, no key bias,
    no top-k, no block_len, no coarse gate. Anything else must go through the
    public function, because the argument preparation this skips is what those
    options need.
    """
    import torch
    from comfy_kitchen.backends.cuda import _C, _wrap_for_dlpack
    batch, t, h, d = q.shape
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    p = _C.sol_attn_plan(batch, t, h, token_aug=int(token_aug))
    ws = torch.empty(p["total"], dtype=torch.uint8, device=q.device)
    _C.sol_attn(
        _wrap_for_dlpack(q), _wrap_for_dlpack(k), _wrap_for_dlpack(v),
        _wrap_for_dlpack(out), _wrap_for_dlpack(ws),
        batch, t, h, d, float(tau), float(d ** -0.5),
        0, 0, 0, 0,
        torch.cuda.current_stream(q.device).cuda_stream,
        key_bias=None, threshold=None, block_len=None,
        tail=True, token_aug=int(token_aug),
    )
    return out, ws, p


def read_tok_cnt(ws, p, batch, h):
    """The per-centroid admitted count, [B, H, NG].

    NG is derived from the plan's own slot arithmetic and cross-checked against
    the query-block count, because a wrong NG would silently reinterpret the
    slot as a different shape and every number below would be fiction.
    """
    import torch
    bh = batch * h
    span = (p["tokHist"] - p["tokCnt"]) // (4 * bh)     # the slot taken after tokCnt
    expect = (p["NQ"] + TOK_GROUP - 1) // TOK_GROUP
    if span != expect:
        raise RuntimeError(
            f"tokCnt slot holds {span} entries per (b,h) but NQ={p['NQ']} with "
            f"TOK_GROUP={TOK_GROUP} implies {expect}; the layout moved and this "
            f"probe would be reading the wrong bytes")
    n = bh * span
    return ws[p["tokCnt"]:p["tokCnt"] + 4 * n].view(torch.int32).view(batch, h, span).clone()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", required=True, help="a .pt with q, k, v")
    ap.add_argument("--launches", type=int, default=12)
    ap.add_argument("--token-aug", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out")
    args = ap.parse_args()

    import torch
    import comfy_kitchen as ck

    d = torch.load(args.slice, map_location="cuda", weights_only=True)
    q, k, v = (d[n].to("cuda").contiguous() for n in ("q", "k", "v"))
    n_tok = args.token_aug if args.token_aug is not None else int(d.get("token_aug", 64))
    tau = args.tau if args.tau is not None else float(d.get("tau", 1.0))
    batch, _T, H, _ = q.shape
    print(f"q/k/v {tuple(q.shape)}  tau={tau}  token_aug={n_tok}  "
          f"launches={args.launches}")

    # Control: the replicated path must equal the public one where both are
    # deterministic. Without this, nothing below is about `sol_attn`.
    mine, _, _ = _call_keeping_workspace(q, k, v, tau, 0)
    theirs = ck.sol_attn(q, k, v, tau=tau)
    same = bool(torch.equal(mine, theirs))
    print(f"  replication control (token_aug=0): "
          f"{'bitwise identical' if same else 'MISMATCH'}")
    del mine, theirs
    if not same:
        raise SystemExit("replication control failed; the probe is not calling "
                         "what sol_attn calls")

    # The loop stays on the device, and the allocator is primed before it.
    # Both matter, and finding out why cost a detour worth recording: anything
    # that serialises the launches suppresses the race. Reading a count back
    # with .tolist() or int() synchronises. So does allocation itself when the
    # caching allocator is empty, because the underlying cudaMalloc is a
    # synchronising call -- an `empty_cache()` before this loop was enough to
    # make 80 launches look perfectly stable while the same slice moved
    # reliably through a warm tight loop. That is a trap for anyone re-testing
    # this, and it is also evidence: a wrong-but-deterministic computation
    # would not care how the launches are spaced.
    for _ in range(2):                       # prime the allocator
        out, ws, p = _call_keeping_workspace(q, k, v, tau, n_tok)
        del out, ws
    first_out = None
    moved = None
    tcs = []
    for _ in range(args.launches):
        out, ws, p = _call_keeping_workspace(q, k, v, tau, n_tok)
        tcs.append(read_tok_cnt(ws, p, batch, H))
        if first_out is None:
            first_out = out.clone()
        else:
            m = (out.float() - first_out.float()).abs().amax(dim=-1)[0]  # [T, H]
            moved = m > 0 if moved is None else (moved | (m > 0))
            del m
        del out, ws

    over_sets, counts = [], []
    for tc in tcs:
        over = (tc > n_tok)
        over_sets.append(frozenset(map(tuple, over.nonzero().tolist())))
        counts.append({"max": int(tc.max()), "groups_over": int(over.sum()),
                       "groups": int(tc.numel()), "_tc": tc})
    del tcs
    torch.cuda.empty_cache()

    # How far over, not just whether: the two candidate explanations for an
    # overflow predict different magnitudes. Boundary leakage between pass 1's
    # binning and pass 2's threshold reconstruction predicts counts a little
    # above the budget -- UNLESS the flip happens at the bottom edge, where
    # pass 2's threshold collapses to the window's lower bound and it admits
    # the whole unbinned population below it. So a distribution hugging the
    # budget and one sitting at a large fraction of the sequence say different
    # things about where the derivation fails.
    import torch as _t
    last = _t.stack([c.pop("_tc") for c in counts]) if "_tc" in counts[0] else None
    mx = max(c["max"] for c in counts)
    tot_over = max(c["groups_over"] for c in counts)
    stable = len(set(over_sets)) == 1
    print(f"  admitted count: max {mx} against a budget of {n_tok}; "
          f"groups over budget {tot_over} of {counts[0]['groups']}")
    print(f"  overflow set identical across launches: {stable}")
    over_mag = None
    if last is not None and tot_over:
        vals = last[0][last[0] > n_tok].float()
        qs = [float(vals.quantile(x)) for x in (0.5, 0.9, 1.0)]
        over_mag = {"budget": n_tok, "seq_len": int(q.shape[1]),
                    "median": qs[0], "p90": qs[1], "max": qs[2],
                    "median_over_budget_ratio": qs[0] / n_tok,
                    "median_as_frac_of_seq": qs[0] / float(q.shape[1]),
                    "what": "how far over the budget the overflowing groups "
                            "sit. Near the budget points at leakage at a bin "
                            "edge; a large fraction of the sequence points at "
                            "the threshold collapsing to the window's lower "
                            "bound, where the whole unbinned population below "
                            "it is admitted."}
        print(f"  overflow magnitude: median {qs[0]:.0f}, p90 {qs[1]:.0f}, "
              f"max {qs[2]:.0f} against budget {n_tok} and seq {q.shape[1]}")

    # The correlation: are the groups that overflow the groups whose rows move?
    corr = None
    if moved is not None and bool(moved.any()) and tot_over:
        union_over = set().union(*over_sets)
        moved_groups = set()
        for h in range(H):
            idx = moved[:, h].nonzero(as_tuple=True)[0]
            for w in (idx // (BLOCK * TOK_GROUP)).tolist():
                moved_groups.add((0, h, w))
        inter = moved_groups & union_over
        corr = {
            "groups_that_moved": len(moved_groups),
            "groups_that_overflowed": len(union_over),
            "moved_and_overflowed": len(inter),
            "moved_but_never_overflowed": len(moved_groups - union_over),
            "frac_of_moving_groups_that_overflow":
                len(inter) / len(moved_groups) if moved_groups else None,
        }
        print(f"  groups whose rows moved: {corr['groups_that_moved']}; "
              f"of those, over budget: {corr['moved_and_overflowed']} "
              f"(never over: {corr['moved_but_never_overflowed']})")

    moved_any = moved is not None and bool(moved.any())
    if tot_over == 0:
        verdict = ("REFUTED: no group ever admitted more than the budget, so "
                   "the slot race cannot be what varies. The cause is "
                   "elsewhere in the token stage.")
    elif not moved_any:
        # Guard against a false green: with nothing moving, every "all moving
        # groups overflowed" test is vacuously true.
        verdict = ("INCONCLUSIVE on the correlation: the admitted count "
                   "exceeds the budget, but the output did not move in this "
                   "run, so there is nothing to correlate it with. Overflow "
                   "alone does not establish the link -- raise --launches "
                   "until the output moves.")
    elif corr and corr["moved_but_never_overflowed"] == 0:
        verdict = ("CONFIRMED as far as an external observer can: every group "
                   "whose output moved also admitted more tokens than the "
                   "budget, which is exactly the condition under which the "
                   "kernel's slot assignment stops being order-independent.")
    elif corr:
        verdict = ("PARTIAL: overflow happens and overlaps the moving groups, "
                   "but some groups move without ever overflowing, so overflow "
                   "is not the whole story.")
    else:
        verdict = ("overflow happens, but the output did not move in this run, "
                   "so nothing here ties the two together.")
    print(f"\n{verdict}")

    record = {
        "what": "whether the token stage ever admits more tokens than the "
                "budget, and whether those groups are the ones whose output "
                "varies between identical launches",
        "produced_by": "bench/probe_token_aug_admitted_count.py",
        "kernel_build": kernel_build(),
        "gpu": torch.cuda.get_device_name(),
        "slice": Path(args.slice).name,
        "shape": list(q.shape), "token_aug": n_tok, "tau": tau,
        "launches": args.launches,
        "replication_control_passed": same,
        "per_launch": counts,
        "max_admitted_count": mx,
        "budget": n_tok,
        "overflow_set_identical_across_launches": stable,
        "overflow_magnitude": over_mag,
        "correlation_with_moving_output": corr,
        "verdict": verdict,
        "cannot_settle": [
            "which tokens were dropped. tok_cnt is a count; the admitted set "
            "itself is still not observable from outside the kernel.",
            "that overflow is the only way this kernel could vary. It is the "
            "one the source names and the one measured here.",
        ],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
