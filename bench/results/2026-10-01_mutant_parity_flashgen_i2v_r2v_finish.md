# Two more h3-mutant-distill recipes against the pack's graphs (2026-10-01)

The owner approved adding recipes the mutant repos did not ship (`2026-10-01_r2v_i2v_graph_coverage.md`,
"Owner's calls"). Two have positive evidence and a pack graph: FlashGen alone from a first frame
(`h3_i2v_flashgen`) and PDD8 then a FlashGen finish on ref2va (`h3_r2v_pdd8_flashgen_finish`). This holds
each to its pack graph the way `2026-09-28_mutant_parity.md` held the first six.

**Result: on both recipes the final video and audio latents are `torch.equal` to the pack's.**

## How

- **Static, CPU:** `bench/check_mutant_parity.py static` is green. It already covered both
  partitions' PDD files, FlashGen files, block selections and the finish's first-pass schedule
  (`pdd8_to_0.8`), so the new recipes needed no new static case.
- **Graphs:** `check_mutant_parity.py graphs --only i2v_flashgen r2v_pdd8_flashgen_finish` writes each
  recipe's pack graph twice, as shipped (`pack`) and with `MiniMaxH3PDDLoRA` and `MiniMaxH3LoRABranch`
  replaced by `H3ExactLoRA` (`ours`), both saving the final AV latent split into video and audio.
  Everything else is the shipped graph, Sol-Attn and kitchen attention included, at seed 730451892.
  The pack graphs are `distill_experiments/h3_probe_i2v_flashgen_4step` and
  `distill_experiments/h3_probe_r2v_step_switch_pdd8_flashgen_h080`.
- **Renders:** `bench/run_graph_arms.py`, one run per arm in the order pack, ours, i2v then ref2va, with
  no warmup: `2026-10-01_mutant_parity_flashgen_i2v_r2v_finish.jsonl`. No row carries an error or a
  `suspect_cache_hit`, and the i2v pair ran at one commit and the ref2va pair at another; the commit
  moved because peer sessions committed in between. No commit in that range touched `pdd_lora.py`,
  `lora_branch.py`, `pdd_math.py` or `standalone/h3_mutant_distill/exact_lora.py` (`git log` over
  those paths from `de9743db` to `29e96f55` is empty).
- **Compare:** `check_mutant_parity.py compare` on the saved `.latent` files, both streams.

## Controls

- A copy of the i2v pack video latent with one element moved by one ulp reads RED against the
  original under the same `compare`, so the comparison is sensitive to a single bit.
- `static` carries its own controls from 2026-09-28 (a block selection one grid point off, a strength
  scaled by 1.0001), both red.

## Caveats

- **The ref2va pack graph is the market scene with one reference image; the example workflow carries the
  other ref2va examples' prompt and two references.** Prompt and images are not part of a recipe, so
  the equality holds for the recipe and not for the example's exact inputs. The example takes the
  shared prompt so the three ref2va recipes can be compared on one input.
- **The parity arms ran Sol-Attn; the examples run kitchen attention,** as every mutant example does
  (`bench/build_mutant_examples.py` `HEADER`). Parity is between the LoRA nodes inside one graph.
- **Parity does not cover the example graphs themselves**, which use core nodes and kitchen attention;
  the section "The examples run end to end" below does.
- Both recipes are untrained transfers: FlashGen was trained for text to video only. One render each
  held up by eye; neither is compared against PDD8.

## The examples run end to end

Parity holds the LoRA nodes to each other inside the pack's graph. The example workflows are different
graphs (core conditioning nodes, kitchen attention only), so each recipe's own builder output
(`bench/build_mutant_examples.py`, the same API graph its UI file round-trips to) ran once through
`run_graph_arms.py`, with the placeholder images swapped for the pack's input images and each LoRA
name given its `h3/` folder: `2026-10-01_mutant_smoke_flashgen_i2v_r2v_finish.jsonl`. No row carries an
error or a cache hit, and each clip is 345 frames with a soundtrack.

- **`h3_r2v_pdd8_flashgen_finish`:** the clip shows both references, the man and the lake, cut
  between across the clip, by eye. One render, the shared generic prompt, not compared against PDD8 or
  FlashGen alone.
- **`h3_i2v_flashgen`, first run:** the keyframe is a 1024x1024 image, the example's canvas was its
  default 1344x768, so the image was stretched. The man held through the push-in, but a pair of glasses
  appeared from mid-clip that neither the keyframe nor the prompt has (five frames across the clip
  read).
- **`h3_i2v_flashgen`, second run:** the same seed at 768x768, the keyframe's own aspect, which is
  the canvas the 2026-09-26 clip of the pack graph used. No glasses, and the five frames read alike to
  that clip's. So the drift came with the stretch, as the example's note already says to avoid; the
  note now carries this case. One render each, and the two runs differ only in canvas.
