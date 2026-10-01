# h3-mutant-distill's keyframe canvas node, checked end to end (2026-10-01)

The owner approved giving the mutant i2v examples the release's canvas rule instead of core's
stretch (the stretched square keyframe that drifted is in
`2026-10-01_mutant_parity_flashgen_i2v_r2v_finish.md`). `H3KeyframeCanvas`
(`standalone/h3_mutant_distill/keyframe_canvas.py`) is a trimmed port of this pack's retired
`MiniMaxH3KeyframeCanvas` (recovered from `d68c697d~1`): core's `adapt_canvas` picks the
canvas from the first frame, the image is scaled to it, and an aspect outside 1:4 to 4:1 is
refused. `bench/build_mutant_examples.py` wires it between `LoadImage` and core's
`MiniMaxH3ImageToVideo` in `h3_i2v_pdd8` and `h3_i2v_flashgen`.

**Result: on the square keyframe, the node picks 768x768, and the clip is frame-for-frame
identical to the earlier run where the canvas was set to 768x768 by hand.**

- **Build:** `build_mutant_examples.py` regenerated all nine files through the live
  frontend and `--check` round-trips them green. Only the two i2v files changed.
- **Render:** the builder's `h3_i2v_flashgen` API graph as shipped, with the placeholder image
  swapped for `1-man.png` (1024x1024) and the LoRA name given its `h3/` folder, seed 730451892:
  `2026-10-01_mutant_keyframe_canvas_smoke.jsonl`. The server log line
  `[h3-mutant] canvas 768x768 from a 1024x1024 first frame` is the node's.
- **Compare:** `ffmpeg -f framemd5` over the video stream of this clip and of the earlier
  hand-set 768x768 smoke (`smoke_i2v_flashgen_768`,
  `2026-10-01_mutant_smoke_flashgen_i2v_r2v_finish.jsonl`) gives the same digest over all 345
  frames. The earlier clip ran on kitchen `a4e0dd8`, this one on `aade8d5`, which also agrees with
  `2026-10-01_kitchen_merge_aade8d5.md` that the new build is bit-identical.
- **After a fix:** `bench/check_mutant_parity.py static` failed at import once the package gained
  the node, because it imports core's H3 module at load and that harness puts this repo's
  `nodes.py` first on `sys.path`. The import moved into `execute()`; on a restarted server the same
  render (`smoke_kc_i2v_flashgen_after_lazy_import`, same JSONL) gives the same frame digest, and
  static parity is green.
- Not run: `h3_i2v_pdd8` through the node. It shares the wiring and the node, not the sampler.
