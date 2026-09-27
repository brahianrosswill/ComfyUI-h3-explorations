# The PDD8 finisher grid, #46 and telemetry on one batch (2026-09-27)

The owner's consolidation (board direction `pdd8-finisher-grid`), fastdude's
batch. Manifest `bench/pdd8_finisher_grid_arms.json` (`7228fc2c`), 24 renders
at seed 730451892, each scene at its written length, all on `_savelat` twins.
Server at `af2e5261`, kitchen `0.2.35+sol.fc32da2.up.c8c7825`, `MiniMaxH3Sol`
with its shipped `balanced` quantizer, `H3_TELEMETRY` armed.

- Rows: `2026-09-27_finisher_grid.jsonl`.
- Measures: `bench/analyze_followup.py --group grid --group fg46` (`3da366d6`),
  in `2026-09-27_grid_<scene>_*.json` and `2026-09-27_fg46_<scene>_*.json`.
- Telemetry summary: `2026-09-27_finisher_grid_telemetry.json`. The raw
  records stay under the capture root.

Everything here is measured at one seed and one clip per arm. None of it is
judged by eye yet: the "For your eye" tab on the board carries the pairs.

## The control

`spec_typical__flashgen_rerun`'s final latent is `torch.equal` to the
2026-09-27 01:15 `spec_typical__flashgen` latent, video and audio. So FlashGen
reproduces bit for bit across the Sol redesign (`MiniMaxH3SolAttn` to
`MiniMaxH3Sol`), the kitchen move to `fc32da2` and a new server process. The
existing full-FlashGen rows are valid comparisons. PDD8 has the same property
from the redesign's own check (`2026-09-27_sol_redesign_bitid.md`).

## Part A: how to finish PDD8

PDD8 runs to sigma 0.8, then no finish, full FlashGen (`rev_h080`), FlashGen on
blocks 34-49 (`rev_late_h080`), or the base on Euler over the 32-point grid,
8 evaluations (`rev_base_h080`, #37, the VAE session's `a3f06741`).

| scene | arm | white | rms_contrast | haze | detail | hf | motion_detail |
|---|---|---|---|---|---|---|---|
| noodle_bar | PDD8 | .882 | .191 | .070 | .0663 | .0225 | 2.061 |
| | full FlashGen | .873 | .197 | .067 | .0696 | .0259 | 2.267 |
| | late FlashGen | .862 | .195 | .066 | .0684 | .0256 | 2.228 |
| | base | .831 | .184 | .071 | .0644 | .0249 | 2.414 |
| radio_drama | PDD8 | .798 | .190 | .049 | .0451 | .0164 | 1.256 |
| | full FlashGen | .848 | .203 | .051 | .0483 | .0178 | 1.322 |
| | late FlashGen | .842 | .201 | .050 | .0478 | .0177 | 1.353 |
| | base | .792 | .190 | .049 | .0454 | .0175 | 1.371 |
| courtroom_verdict | PDD8 | .907 | .220 | .099 | .0446 | .0122 | 2.293 |
| | full FlashGen | .930 | .225 | .101 | .0474 | .0138 | 2.650 |
| | late FlashGen | .923 | .224 | .101 | .0470 | .0139 | 2.585 |
| | base | .908 | .215 | .101 | .0440 | .0125 | 2.394 |
| subway_chase | PDD8 | .918 | .243 | .220 | .0396 | .0073 | 2.485 |
| | full FlashGen | .958 | .250 | .224 | .0399 | .0077 | 2.739 |
| | late FlashGen | .956 | .249 | .222 | .0403 | .0079 | 2.694 |
| | base | .919 | .243 | .220 | .0389 | .0074 | 2.579 |
| subway_chase_short | PDD8 | .870 | .202 | .153 | .0292 | .0068 | - |
| | full FlashGen | .890 | .213 | .162 | .0320 | .0077 | - |
| | late FlashGen | .895 | .211 | .160 | .0322 | .0079 | - |
| | base | .871 | .202 | .154 | .0297 | .0072 | - |
| spec_unusual | PDD8 | .857 | .240 | .233 | .0421 | .0092 | 3.518 |
| | full FlashGen | .859 | .238 | .233 | .0447 | .0105 | 4.080 |
| | late FlashGen | .860 | .240 | .234 | .0434 | .0099 | 4.002 |
| | base | .859 | .238 | .233 | .0424 | .0094 | 3.540 |
| offpath_tortoise | PDD8 | .906 | .229 | .158 | .0732 | .0173 | 0.997 |
| | full FlashGen | .912 | .234 | .161 | .0795 | .0216 | 0.852 |
| | late FlashGen | .910 | .232 | .159 | .0788 | .0213 | 0.858 |
| | base | .906 | .225 | .159 | .0745 | .0183 | 0.854 |
| i2v (1-man.png) | PDD8 | .793 | .202 | .045 | .0268 | .0093 | 4.628 |
| | full FlashGen | .817 | .208 | .050 | .0266 | .0088 | 4.364 |

(`subway_chase_short` is on the temporal exclusion list; `motion_detail`
there is not read.)

**Reading:**
- **The full and late-only FlashGen finishes are nearly the same on all seven
  t2v scenes.** #34 (`2026-09-27_late_switch.md`) holds on three more scenes,
  the new off-path one included.
- **A FlashGen finish lifts PDD8's highlights where they are dim.** That covers
  radio_drama, courtroom, subway and subway_chase_short, plus i2v. It does not
  on noodle_bar, where PDD8 is already brightest, nor on spec_unusual or
  offpath, where PDD8 is already near the finish's level. It raises detail and
  `hf` on every t2v scene.
- **The base finish (#37) is gentle.** Highlights, contrast, haze and detail
  stay at PDD8's level, and on noodle_bar the highlights drop. Motion detail
  rises on four of the six scenes read, less than under the FlashGen finish.
  On this batch's measures it does not repair PDD8's dim highlights, which is
  what the reverse switch was for.
  - By #37's own rule (the VAE session's reading), a base finish that looks
    like PDD8 means PDD8's dim highlights are not its coarse tail's doing. The
    lift from the reverse switch is FlashGen's own.
- **Inference, not measured:** the #43 pilot found the teacher refines
  high-change frames late and PDD8 does not. A base finish from 0.8 gives back
  some motion detail, which fits, but not the highlight lift, which is
  FlashGen's.
- **i2v:** the FlashGen finish lifts highlights and haze slightly and lowers
  motion detail. It is one image, so the owner's eye decides.

## Part B: #46, FlashGen alone, full against blocks 34-49

| scene | arm | white | rms_contrast | haze | detail | hf | motion_detail |
|---|---|---|---|---|---|---|---|
| spec_typical | full | .950 | .313 | .277 | .0525 | .0071 | 5.338 |
| | late | .929 | .237 | .366 | .0295 | .0044 | 4.029 |
| spec_specific | full | .960 | .256 | .327 | .0425 | .0091 | 3.732 |
| | late | .912 | .222 | .306 | .0386 | .0106 | 4.587 |
| spec_unusual | full | .980 | .291 | .294 | .0505 | .0105 | 4.009 |
| | late | .948 | .269 | .276 | .0425 | .0086 | 3.374 |
| subway_chase_short | full | .904 | .221 | .171 | .0371 | .0110 | - |
| | late | .942 | .243 | .193 | .0306 | .0072 | - |
| offpath_tortoise | full | .922 | .221 | .151 | .0894 | .0316 | 1.835 |
| | late | .868 | .209 | .137 | .0869 | .0353 | 1.402 |

**Reading:**
- **FT1's "half the haze" does not generalise.** At 124 frames, late-only has
  less haze on three scenes and more on two. On spec_typical it is much
  hazier, with about half the detail and `hf`: nearer an unfinished render.
  FT1 was two scenes at 345 frames
  (`2026-09-26_flashgen_weights_predictions.md`), and its reading of late-only
  as a less hazy FlashGen needs this caveat.
- **Contrast drops with late-only on four of five scenes.** Highlights drop on
  four of five.
- **The question #46 asks is adherence.** Do the late blocks alone follow the
  unusual rungs, subway's roles and the off-path scene's beats better? The
  measures cannot say. The owner's eye against `bench/adherence_checklists.json`
  can (the ladder and `t2va_offpath_tortoise` have checklists since
  `03c30f94`). The pairs are on the board's review tab.

## Telemetry (#39)

Per render, from `2026-09-27_finisher_grid_telemetry.json`:
- **The card is full.** Peak GPU use is 24.3-24.5 GB in every render.
- **The text encoder is loaded to the card for each new prompt and evicted
  before sampling.** Its node costs 8.5-10 s and 22-31 GiB of PCIe traffic to
  the GPU. It costs nothing when a render reuses the previous prompt, which
  ComfyUI caches.
- **The DiT is about 90% resident during sampling** (`blocks_resident` 0.90 at
  the sampler's end), and the rest streams every forward. So the sampler's
  PCIe traffic scales with evaluations and length: 74-168 GiB on the 345-frame
  base finishes, 5-15 GiB on a two-step finish.
- **About 19.5 GB was read from disk on most renders that encode a new
  prompt.** The page cache was not holding a file that size, most likely the
  encoder. A CPU check sweep and this analysis ran alongside part of the
  batch, so eviction pressure was not typical. This is a hypothesis, not a
  finding. A batch with nothing else running would settle it.
- **Ordering:** renders that repeat the previous prompt skip the encoder and
  its PCIe cost. Batching arms by scene, as this manifest did, saves one
  encoder load per extra arm.
