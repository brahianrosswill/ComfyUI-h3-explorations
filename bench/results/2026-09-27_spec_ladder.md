# Specificity ladder (the owner's O2): read (2026-09-27)

`t2va_spec_typical`, `_specific` and `_unusual`, at 124 frames and seed
730451892, on FlashGen, PDD8 and FastH3 (`bench/followup_spec_ladder_arms.json`).
Records: `2026-09-26_followup_spec_{tone,resolution,temporal}.json`. One clip
per rung and model.

| rung | FlashGen contrast / detail / moved | PDD8 | FastH3 |
|---|---|---|---|
| typical | .313 / .0525 / .188 | .220 / .0375 / .103 | .295 / .1321 / .388 |
| specific | .256 / .0425 / .120 | .235 / .0366 / .090 | .258 / .0569 / .145 |
| unusual | .291 / .0505 / .165 | .240 / .0421 / .091 | .253 / .0657 / .168 |

- **The measures do not show FlashGen degrading more on the unusual rung.**
  Its unusual clip sits between its typical and specific ones on every column.
  O2's claim is about "weird shit" at the edge of the guided path, which is
  semantic, and these measures cannot see it. The verdict is the owner's eye
  on the three FlashGen clips against PDD8's and FastH3's.
- **FastH3 changes most across the rungs.** On the typical rung it has the
  most detail and motion of any clip in the batch; both fall by half or more
  on the specific and unusual rungs.
- **PDD8 is the steadiest**, and the lowest in motion on every rung.
