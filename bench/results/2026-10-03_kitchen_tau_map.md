# comfy-kitchen: a tau per head and query block in `sol_attn` (2026-10-03)

The owner asked for the per-head table on 2026-10-03 ("love it. lets do it"),
and for finer control by segment (text, reference, audio and video rows
apart). Both need one thing from the kernel, and this is it.

## What changed

`h3-frontier` in the kitchen fork carries one more commit, `6272371`:
`sol_attn(..., tau_map=None)`. The routing threshold was already stored per
(batch times head, query block) as tau times a spread estimate
(`sol_attn_preprocess.cu`, `thr_of`). `tau_map`, a float32 tensor of shape
(heads, query blocks), supplies the tau for each of those products. Nothing
else in the kernel moved. CUDA and the eager reference take it; HIP refuses
it; the chunked producer does not take it; it cannot be combined with
`topk_ratio`.

The build is `0.2.37+sol.6272371.up.be003b7`, installed by
`vendor/rebuild_kernel.sh`, whose `--check` and `bench/check_sol_kernel.py
--require` both pass. The build record beside the venv lists the carried
commits. Not pushed.

## Checks, on a real cell

Block 24, step 2 of the 2026-09-27 two-reference ref2va capture (120,582
tokens, 56 heads), the node's shipped call (rotated quantizer, pooled tail,
conditioning prefix as exact keys and exact rows):

- **A call without the map is unchanged.** The output's bit digest equals the
  one taken on the build before, on this cell and on blocks 40 and 0.
- **A map holding one value is the scalar call**, `torch.equal`.
- **A head under a per-head map is that head in a scalar call** at its value,
  `torch.equal`, routed counts included. So the tau sweep in
  `2026-10-03_per_head_tau_ref2va.json` prices exactly what the map does.
- **A large negative entry makes that head and query block dense**, and with
  the conditioning prefix set that way and `sink_q` off, the output is
  `torch.equal` to today's `sink_q` over the same range. The map can therefore
  express any set of exact query rows, not one contiguous range.
- **No cost in time.** Scalar, filled map and per-head map are within a
  millisecond of each other on a call of about half a second.

## Tests in the fork

Five added to `tests/test_sol_attn.py` (filled map, per-head equality with
the eager reference as a cross-check, the dense entry, argument checks), run
from a copy outside the clone against the built wheel. The file's other tests
pass as before, with `test_bindings_*` skipped as in
`2026-10-01_kitchen_merge_aade8d5.md` and `test_topk_ties_over_select`
failing on this build and on the one before it.

## What it does not do yet

Nothing in this pack passes `tau_map`. The node has no table input and there
is no calibrated table: the capture covers ten blocks of fifty.
