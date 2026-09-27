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
