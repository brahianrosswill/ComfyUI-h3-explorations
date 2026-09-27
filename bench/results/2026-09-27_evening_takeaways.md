# The 2026-09-26 evening: takeaways and what to carry forward

The VAE session's own ranking, written 2026-09-27 at the owner's request.
Each item points to the record that holds its numbers. Everything rests on
one seed and a few scenes; these are judgements, not measurements. The data
for all of it is `2026-09-27_render_dataset/` (DuckDB), and the readable
inventory is `2026-09-27_inventory.md`.

## Findings, most important first

1. **Weight change did not predict what does the work.**
   - FastH3's time embedder moved about 5.6% and its backbone linears about
     1e-4, and the swap renders showed the conditioning to be near-inert:
     speed and look live in the gates and the small backbone drift
     (`2026-09-27_fasth3_swap.md`, retracting part of
     `2026-09-26_fasth3_weights.md`).
   - FlashGen agrees from the other side (fastdude's FT1,
     `2026-09-27_inventory_fastdude.md`). Its small near-rank-2 late-block
     change does the 4-step finishing, while its larger early change mostly
     adds haze.
   - Method lesson: a weight map says where a distill moved, not what the move
     does. Swaps and transplants are the test, and both are now cheap
     (`bench/build_adaln_swap.py`, `MiniMaxH3LoRABranch.blocks`).
2. **The dark splotchy blocking is the save format, not the models.** A
   lossless decode shows none; the 8-bit 4:2:0 h264 at crf 19 adds it
   (`2026-09-27_o1_lossless.md`, one clip). One house setting affects every
   render.
3. **PDD8's dim highlights partly come from its coarse final step.** A
   FlashGen finish from sigma 0.8 lifts the white point to the other
   distills' level without changing the take (`2026-09-27_reverse_switch.md`,
   one scene).
4. **Each distill has a consistent signature across 13 scenes.** FastH3 has
   the most detail and colour; PDD8 is the flattest and moves about half as
   much; FlashGen is hazy and cool, with the brightest audio
   (`2026-09-26_distill_signatures.md`). It is a map for choosing a distill
   per scene, and the motion gap backs routing still shots to PDD.
5. **The subway "clone" is the prompt's two people, decided at step 1, and
   the base draws the same.** CPU previews of per-step x0 latents were enough
   to see it (`2026-09-27_clone_base_control.md`), and the lane closed
   without more card time.
6. **Final-latent distance is divergence, not effect size** (fastdude). A
   0.1% conditioning change lands 0.57-0.71 away. Look and event reads carry
   the weight.

## Techniques and recipes, most promising first

1. **FlashGen with `blocks="34-49"`.** One widget: 4-step speed with about
   half the haze. The strongest practical candidate, pending the owner's eye
   (fastdude's FT1).
2. **The reverse switch: PDD8, then FlashGen finishing from 0.8.** PDD8's
   composition and naturalness with its highlights fixed (one scene).
3. **PDD6.** 90-95% of PDD8's fine detail at three quarters of the steps
   (`2026-09-27_ladder.md`).
4. **FastH3 on FastVideo's contract settings**, not ComfyUI's template: its
   own shift, rungs and VSA. It still over-polishes.
5. **Infrastructure that held up:** LoRAs on int8 through the exact branch
   (FlashGen merged into int8 loses most of its delta), and the int8 video
   VAE (about 12 s saved per render, not visible to the owner).

**Next, and untried:** PDD8 first, finished by FlashGen on blocks 34-49
instead of full FlashGen. That puts PDD's composition and FlashGen's clean
finish in one render. It needs one render per scene and no new nodes;
start with the lamp-lit interiors where PDD8 is flattest.

**The open question:** gates or backbone drift, for FastH3. Two CPU builds
answer it (fl2va plus FastH3's gates, and FastH3 without them), and the answer
decides whether a lighter FastH3 is possible.

## Not worth pursuing

- PDD at strength below 1 (`2026-09-27_pdd_strength.md`);
- PDD4;
- FlashGen's early blocks alone;
- the time-embedder recipes: the dial checkpoints, PDD re-baked onto FastH3's
  time embedder, and both swap hybrids. The swap refuted the idea behind
  them (`2026-09-27_inventory.md`, "Safe to delete").
