# The 2026-09-26 follow-up night: what we learned

Written for the owner's morning. All at one seed (730451892), on the
0.154.8 code, with no base renders (the owner deferred them). Everything
here is preliminary in that sense. Each line points at the record that
holds it, and the predictions made before any render are in
`2026-09-26_distill_run_predictions.md`, each with its verdict. Other
people's records are cited, not restated.

## What changed in the code

- **Every LoRA on an int8 checkpoint is applied at the call** (0.154.0). This
  includes PDD's backbone (`MiniMaxH3PDDLoRA.backbone_apply`).
- **A bug in that change, found by the owner's eye and fixed in 0.154.8.**
  - With several branch-loaded models in one process, each wrapped the
    previous model's still-applied forward.
  - FlashGen rendered as base + Turbo + PDD + FlashGen ("blocky as hell").
  - The contaminated rows, and which other renders the bug could reach:
    `2026-09-26_followup_contamination.md`.
  - `bench/check_lora_branch.py` now reproduces it.
- **Prompts made specific for their tests** (0.154.6), from a read-only
  fitness audit.
  - Head counts are stated, and kpop's mirror wall is gone.
  - samurai's four defects are fixed.
  - radio_drama now states that the listeners are silent, and is a deadpan
    comedy (0.154.7).

## Findings

- **The distills' signatures, 13 matched scenes**
  (`2026-09-26_distill_signatures.md`, the VAE session):
  - FlashGen is hazy, cool and the least saturated.
  - PDD8 has the lowest contrast and the dimmest highlights. The owner reads
    that as "naturally so", not as a defect.
  - FastH3 has the most detail and colour.
- **The looks** (`2026-09-26_followup_looks.md`):
  - FlashGen lifts the blacks on every look.
  - FastH3 leaks the most colour into black-and-white and pulls toward warm
    orange.
  - PDD8 is the most colourful on neon.
- **FastH3's "over-polish" comes with its attention path.** With VSA off,
  fine detail roughly halves (the F6 verdict). That arm also drops the
  learned coarse branch, so the VAE session's swap is what separates
  sparsity, gates and the time embedder.
- **A faster PDD:** PDD6 keeps detail within 5 to 10% of PDD8, at three
  quarters of the steps (`2026-09-27_ladder.md`, the VAE session). PDD4
  loses more.
- **The subway "clone" is the prompt's two people** (the VAE session's
  `2026-09-27_clone_base_control.md`).
  - The base draws the same suspect and agent at 1.1-1.3 s, from the same
    noise.
  - PDD8 decides that composition at step 1, at every step count, merged or
    exact.
  - What can read as a clone is PDD8's overlap at 1.1 s, one figure passing
    in front of the other: the owner's call.
  - The clone lane closes.
- **FastH3's timestep conditioning is near-inert** (`2026-09-27_fasth3_swap.md`,
  the VAE session). Swapping it either way leaves each model as it was, so
  FastH3's look and speed live in its VSA gates and backbone. That fits its
  VSA-off result below, and falsifies the earlier reading that its time
  embedder carried the look.
- **PDD strength** (`2026-09-27_pdd_strength.md`, the VAE session): 0.85 and
  0.7 lose fine detail and some motion, and contrast does not move. 1.0
  stays. The reverse switch is the fix for PDD8's dim highlights.
- **PDD first, FlashGen finishing (the owner's idea) repairs PDD8's dim
  highlights without changing the scene** (`2026-09-27_reverse_switch.md`,
  the VAE session).
  - Both handoffs keep PDD8's composition, motion and saturation.
  - Highlights lift to FlashGen's and FastH3's level, and fine detail rises.
  - The two-figure composition survives, as predicted before it rendered.
    It is the prompt's two people, decided at step 1 (below).
  - So part of PDD8's dim-highlight signature is its coarse tail.
  - A candidate route for PDD's lamp-lit interiors. The owner's call.
- **Specificity ladder (the owner's O2)** (`2026-09-27_spec_ladder.md`):
  - The measures do not show FlashGen degrading more on the unusual prompt.
    "Weird" is semantic, so this one is the owner's eye.
  - FastH3 varies most across the rungs, and PDD8 is the steadiest.
- **FlashGen's late blocks alone make a finished, less hazy render** (FT1,
  `2026-09-26_flashgen_weights_predictions.md`).
  - `blocks="34-49"`: a developed 4-step clip with deeper blacks, half the
    haze and full FlashGen's motion detail.
  - `blocks="0-33"`: broken, dark and undeveloped.
  - So the tiny late change does the 4-step work, and the early high-rank
    change adds the hazy, lifted-black look.
  - Worth the owner's eye: late-only FlashGen on the adherence scenes. If the
    early blocks are where it is "overfit", late-only might follow prompts
    better.
- **FlashGen's weights** (`2026-09-26_flashgen_weights_predictions.md`,
  verdicts):
  - The effective rank is about 18 of 64.
  - The change is concentrated in the early blocks.
  - It barely touches timestep conditioning, where FastH3 retrains its time
    embedder (`2026-09-26_fasth3_weights.md`, the VAE session).
- **The blocky, splotchy darks are our save setting, not the distill** (O1
  and F8, `2026-09-27_o1_lossless.md`, the VAE session).
  - Decoded losslessly from PDD8's anchor latent, the darks show no blocking.
  - The mp4 adds it, in the darks only. Every graph saves 8-bit 4:2:0 h264 at
    CRF 19 through `VHS_VideoCombine`.
  - One clip. A FlashGen look would confirm it.
  - A lower CRF or a 10-bit format is the owner's call.
- **Method:** final-latent distance at one seed measures divergence, not
  effect size. A 0.1% modulation change still lands 0.57 away. Verdicts on
  magnitude use tone, temporal and the owner's eye.

## Server

Stopped by the VAE session after the last runner. The substrate record is
`2026-09-27_followup_substrate.json` (dd7e71ea), and the raw history is in
`internal/history/`.

## For the owner's eye

- radio_drama, v1 against v2 on the three distills: do the lips stay closed
  until each actor speaks, and is it funny?
- FlashGen's clean looks, to confirm the blockiness is gone.
- The specificity ladder, typical against unusual, per distill.
- PDD8's subway overlap at 1.1 s (`subway_chase__pdd8`): a motion artefact,
  or two people passing?
- FlashGen late-only against full, on look_anchor and slapstick
  (`*_flashgen_blk34_49` against `*__flashgen`).
