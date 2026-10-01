# comfy-kitchen: h3-frontier merged to upstream main, 2026-10-01

The owner asked whether upstream had anything to merge into the kitchen fork's build branch, and then
to do it. This records what came in, what was run to check it, and what the build is.

**Result: merged and installed. Every output checked is bit-identical to the build before it, and the
sampler is faster on the two graphs timed.**

## The build

- **Before:** `h3-frontier` at `a4e0dd8`, on upstream main `888b13e` (v0.2.36), build
  `0.2.36+sol.a4e0dd8.up.888b13e`.
- **After:** merge commit `aade8d5` (a merge of `upstream/main`, nothing rewritten) on upstream main
  `3f7210f`, build `0.2.36+sol.aade8d5.up.3f7210f`, CUDA arch 89. The merge had no conflicts. The
  14 commits we carry (blk_cnt, qk_balance, rotate) are all still on top.
- **Built by** `vendor/rebuild_kernel.sh`, whose gate passed (the source contains ComfyUI's pinned tag,
  the submodules are at the source's pins) and whose own `bench/check_sol_kernel.py --require` passed.
  The build record is `comfy_kitchen_build.json` beside the venv. nvcc was the shell's default, CUDA
  13.4; the earlier build's toolkit was not recorded, so this is not known to be the same.
- **Not pushed.** The local `origin/h3-frontier` and `nas/h3-frontier` refs still sit at `a4e0dd8`; pushing
  is the owner's say.
- **Rollback:** the old wheel is still in the clone's `dist/`. Reinstall it with
  `uv pip install --python <comfy venv python> --force-reinstall --no-deps <wheel>`, then rebuild the
  record from a worktree of `a4e0dd8` (`SRC=<worktree> vendor/rebuild_kernel.sh`) so the record and the
  install agree.

## The seven upstream commits

| commit | what | for this box (RTX 4090, `int8_convrot` checkpoints) |
|---|---|---|
| `178050f` (#215) | int8 GEMM: banded stream-K tile walk keeps weights in L2; a new `cutlass_int8_selected_config` binding | the one that matters; below |
| `b176d46` (#216), `12389a3`, `4133803` | fused ConvRot requant and a faster requant for W4A8 and W6A8 | only for w4a8 and w6a8 checkpoints, which are not the default |
| `19ea55b`, `a62ac9f`, `3f7210f` | HIP (AMD) fixes and ports | none |

## 1. The GEMM, layer by layer

`bench/bench_kitchen_int8_linear.py` runs `int8_linear` as ComfyUI does, with the checkpoint's real
block-0 weights, at two token counts: H3's packed length at 1344x768 and 345 frames
(`M=104103`), and with two 2048x2048 references (`M=120666`). Rows:
`2026-10-01_kitchen_merge_int8_linear.json`. Old is the previous wheel installed beside the venv with
`--target`; new is the venv's.

| layer (N x K) | old ms | new ms | tile config, new |
|---|---|---|---|
| qkv, 21504 x 5376 | 74.0 | 42.7 | 13 (banded) |
| fc1, 28672 x 5376 | 77.5 | 56.2 | 13 |
| out, 5376 x 7168 | 15.1 | 16.3 | 13 |
| fc2, 5376 x 14336 | 33.9 | 33.5 | 13 |

(`M=104103`; the other count scales the same way.) qkv and fc1 are about 1.7x and 1.4x faster; out and
fc2 are unchanged within the run-to-run spread of about 5%. Over the 50 blocks that is about 2.6 s per
forward at the shorter count and 3.0 s at the longer. **All 8 output digests are equal between the
builds**: the output is bit-identical. The new heuristic picks the banded config for all four layers at
both counts, as the reroute threshold (the larger of three quarters of L2 and 48 MB, against the
activation's bytes) predicts for the 4090's L2.

## 2. The kitchen's tests

Run on both wheels, on the files for Sol, int8 linear, int8 attention, the input-activation and residual
epilogues, and W4A8 and W6A8. **Run them from a copy of `tests/` outside the clone.** Run in the clone,
or with the clone on `sys.path`, pytest imports the clone's own `comfy_kitchen/`, and its gitignored
`_C.abi3.so` is a build from 2026-09-28: my first two runs tested that file under both labels, and
returned identical counts. A run is valid when `comfy_kitchen.__file__` is the wheel's.

- **`test_bindings_*` skipped (`-k "not test_bindings"`).** Those tests check the dtype and range
  validation of the HIP bindings (`workspace must be a uint8`); the CUDA bindings do not raise, so on
  CUDA the call reaches the kernel with a bad buffer, and the illegal address it triggers is sticky for
  the rest of the process (hundreds of later tests fail). Same on the old build.
- **Old: 42 failed. New: 3 failed.** No test fails on new only. The 39 that fail on old only are the
  tests that came with the merged commits (the banded config's, W4A8's and W6A8's).
- **The 3 that fail on both**, so not from this merge:
  - `test_int8_attention_long_sequence_and_partial_tile` at an explicit softmax `scale` of `0.0` and of
    `-(128**-0.5)`: NaN output on the CUDA backend. The default scale passes, and H3 uses the default.
    The test arrived with the HIP NaN fix (`19ea55b`).
  - `test_topk_ties_over_select` in the Sol tests: cosine against the eager reference below its 0.99
    bar. It fails on the build before the merge too; not looked into here.

## 3. End to end

The two pack graphs of `2026-10-01_mutant_parity_flashgen_i2v_r2v_finish.md` (FlashGen from a first
frame, and the ref2va PDD8-then-FlashGen finish), written by `check_mutant_parity.py graphs`, seed
730451892. The old-build latents are that morning's, on the previous wheel. The new ones were rendered
by `bench/run_graph_arms.py` on a server started fresh for this, unarmed (`H3_CAPTURE_ROOT` was its
only `H3_*` key), rows `2026-10-01_kitchen_merge_ab.jsonl`. No row has an error or a cache hit.

- **All four final latents (video and audio, both graphs) are `torch.equal` to the old build's.** So the
  merge changes no output on these two graphs, through Sol attention, kitchen int8 attention, and
  every linear.
- **Sampler seconds, old to new:** the i2v graph 65.5 to 56.0, and the ref2va finish 361.9 to 335.1.
  The GEMM rows predict about 10 s and 24 s from the evaluation counts (4 and 8); the measured
  differences are 9.5 s and 26.8 s.
- **Conditions:** one render each. The i2v graphs were each the first render after a server start. The
  old ref2va render was the third of its server's session and the new one the second. The old server
  had run an `ours` arm of the same recipe between; those rows (60.4 s and 358.3 s) are in the parity
  record.

## Open

- Push `h3-frontier` to `origin` and `nas`: the owner's say.
- The clone's in-tree `_C.abi3.so` (gitignored, 2026-09-28) is stale. Anything run from the clone that
  imports `comfy_kitchen` gets it, not the installed wheel. Deleting it is the owner's call.
- Two upstream edge-case tests fail on the CUDA backend (above) and one of ours fails before and after.
