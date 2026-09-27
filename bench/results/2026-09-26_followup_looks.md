# The look family on the three distills: first measures, 2026-09-26

**Preliminary.** One seed (730451892), the look family's anchor (low-key
chiaroscuro night interior) and noir (black-and-white 35 mm) on PDD8, FlashGen
and FastH3, from the follow-up rerun on 0.154.8 (the look arms whose earlier
launch was contaminated are excluded). No base render: the owner dropped base
arms tonight. So nothing here says what the base does with the same look.
Records: `2026-09-26_followup_looks_tone.json`,
`2026-09-26_followup_looks_temporal.json`, from `bench/analyze_followup.py
--group looks`. Noir asks for silver film grain, so its texture numbers are
read apart from the anchor's.

## What the numbers say

- **Black-and-white: FastH3 leaks the most colour, FlashGen the least.**
  `chroma_p95` on noir: FastH3 0.224, PDD8 0.102, FlashGen 0.063. FastH3's
  noir also keeps an orange-heavy palette (orange share 0.27, hue spread 45
  degrees). For F7 (distills pull an unusual look toward a typical one), that
  holds for FastH3 and not for FlashGen on this variant.
- **Low-key: FlashGen lifts the blacks.** Shadow share (luma under 0.10) on the
  anchor: PDD8 0.49, FastH3 0.44, FlashGen 0.33. FlashGen has the highest
  midtones and haze on both looks, and about half the crushed-black share of
  the other two.
- **O1, dark blocking: a small signal, not yet attributable.** On the anchor,
  gradient strength across 8-pixel boundaries is 6 to 8% higher in dark
  regions than in bright ones on all three (PDD8 1.079, FastH3 1.062, FlashGen
  1.062 against about 1.0 bright); on noir it is 0 to 3%. 8 pixels is also an
  H.264 block, so the mp4 cannot separate the codec from the model. O1's own
  test is a lossless decode of the saved latent against the mp4.
- **FastH3 keeps the most detail in motion:** `motion_detail` 5.6 (anchor) and
  6.5 (noir), against about 3.2 to 3.7 for PDD8 and FlashGen. That fits the
  owner's read of FastH3's anchor, "super high detail like almost way too
  much causing it to look a bit ai generated in polish". Its texture boils
  slightly more than PDD8's (0.66 against 0.54 on the anchor).

## The owner's follow-ups on weights (2026-09-26, late)

- **Is FlashGen least because its weights move least?** Median relative
  weight change from the base on the same 12 layers: FlashGen 0.044%, FastH3
  0.177%, PDD8 0.534% (FlashGen and PDD from
  `2026-09-26_int8_lora_requant.json`; FastH3 measured against the base int8
  checkpoint in `2026-09-26_fasth3_lora_rank.json`). Against noir's colour
  leak (FlashGen 0.063 < PDD8 0.102 < FastH3 0.224), the idea holds for
  FlashGen but is not monotone: PDD moves three times FastH3's weights and
  leaks half as much. FastH3 also differs in VSA's sparse attention, which a
  weight delta cannot see, and in its DMD2 objective.
- **Can a LoRA be extracted from FastH3?** Not from the int8 files. Rank 256
  holds a median 12% of the delta's energy
  (`2026-09-26_fasth3_lora_rank.json`, `bench/measure_checkpoint_lora_rank.py`),
  but the delta is about a twentieth of an int8 step, so in int8 it is mostly
  scattered one-step rounding flips, which are full-rank whatever the real
  change is. A real extraction, or its rank, needs bf16 FastH3 and bf16 base
  weights; neither is on disk. The AdaLN projection would be carried whole,
  as the owner suggested, because its curve form cannot be compared
  coefficient by coefficient.
- **Is PDD too strong per module?** One outlier: `blocks.49.mlp.fc2` moves 4.4%
  (0.86 of an int8 step), against 0.3 to 1.3% elsewhere (`blocks.49.mlp.fc1`
  1.3%). Tonight's exact-branch arms apply it exactly for the first time; the
  merged path applied it through heavy rounding noise. Size alone is not
  proof of excess: PDD's head bank was distilled on the features these
  weights produce.
- **Match a factor of FastH3?** Not by magnitude. It does not transfer between
  distillation methods: FlashGen reaches 4 steps with a twelfth of PDD's
  change. The test for over-strength is a dose-response on the clone scene:
  PDD at strength 0.7, 0.85 and 1.0, and the block-49 MLP alone at 0.5
  through the branch's per-module controls. Over-strength shows as clones
  fading while detail holds.


## All four looks, added 2026-09-27 (this session)

Neon and anime landed after the section above was written. The records were
re-run over all four looks, and the numbers are in the same two JSONs. One
seed, no base, so still preliminary.

- **FlashGen lifts the blacks on every look.** It has the lowest shadow share
  (luma under 0.10) on all four, about half the crushed-black share of the
  others, and the highest midtones: anchor 0.33 against 0.44 to 0.49, noir
  0.31 against 0.45 to 0.47, neon 0.22 against 0.34 to 0.39, anime 0.21
  against 0.27 to 0.29. Haze is highest on anchor, noir and neon. This is the
  "lifted, hazy" signature of `2026-09-26_distill_signatures.md`, and it holds
  whatever look the prompt asks for.
- **Colour is not PDD8's weak point when the prompt asks for it.** On neon,
  PDD8 has the most chroma (p95 0.616, against 0.475 FastH3 and 0.427
  FlashGen) and near-top saturation. That fits the owner's answer that PDD8's
  usual grade reads flat and dim, not colourless.
- **FastH3 pulls toward warm and orange.** It is the warmest on anchor,
  noir, neon and anime, with an orange share of 0.68 on anime (against
  0.49 to 0.56), and it has the most fine detail on all four (the owner's
  "over-polish"). With noir's colour leak (above), FastH3 is the distill that
  bends a requested look toward its own palette, which is F7's pull for
  FastH3.
- **FlashGen bends the look the other way, toward haze, not toward colour.**
  Its noir leaks the least colour. F7's "toward a typical look" does not
  describe it: its shift is a lifted black level applied to every look.
- **Temporal (the temporal JSON):**
  - PDD8 boils least on the anchor and anime and most on neon (0.69), whose
    strobing practicals may drive it.
  - Dark-over-bright 8-pixel blocking is largest for PDD8 on neon (1.19) and
    anime (1.10). That is still unattributable between the model and the
    H.264 encode, as above.
