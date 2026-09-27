# PDD8 strength dose-response: read (2026-09-27)

`bench/pdd_strength_arms.json` (predictions before rendering; widened to five
scenes, bb73cfc0). The exact branch ran at strength 0.85 and 0.7 with the
fused heads held at 1.0, and at 0.7 on every surface (`s07all`), at seed
730451892. References are the followup rerun's strength-1.0 `<scene>__pdd8`
rows. Records: `2026-09-27_pdd_strength_{tone,resolution,temporal,audio}.json`
and `2026-09-26_pdd_strength.jsonl`. One clip per arm per scene; each arm is a
different sample from the same noise.

| scene | contrast 1.0 / .85 / .7 | hf 1.0 / .85 / .7 | moved 1.0 / .85 / .7 |
|---|---|---|---|
| look_anchor | .221 / .217 / .210 | .0064 / .0058 / .0054 | .049 / .050 / .048 |
| subway_chase | .243 / .250 / .255 | .0073 / .0063 / .0057 | .242 / .231 / .223 |
| slapstick_moving_piano | .221 / .223 / .219 | .0185 / .0157 / .0158 | .246 / .222 / .198 |
| radio_drama | .190 / .190 / .188 | .0164 / .0159 / .0157 | .072 / .069 / .072 |
| samurai_bamboo_duel | .215 / .218 / .218 | .0161 / .0142 / .0138 | .223 / .206 / .173 |

## Verdicts

- **D1, turning the delta down moves PDD8 toward the base's grade and contrast
  with modestly softer detail: not supported.**
  - Contrast is flat within ±0.01 on every scene.
  - Fine detail falls 10-20% at 0.7.
  - Motion falls on the two action scenes (slapstick .246 to .198, samurai
    .223 to .173).
  - Only subway_chase's highlights rise (white .918 to .97), which is one
    sample.
- **D2, weakening the heads too is visibly softer than s07: not supported.**
  `s07all` matches `s07` on look_anchor and slapstick.
- **D3, the effect is scene-dependent: not supported.** Every scene moves the
  same way, slightly worse.
- **Audio.** Loudness is level. The spectral centroid rises at lower
  strength on four of five scenes (slapstick 583 to 675 Hz, samurai 1043 to
  about 1190-1350 Hz), so PDD's full delta darkens its audio a little.

## So what

- **Strength 1.0 stays PDD8's default.** PDD's outsized delta is doing work:
  less of it costs detail and motion, and does not touch the flat grade.
- **The grade has a better fix.** The FlashGen finish
  (`2026-09-27_reverse_switch.md`) lifts PDD8's highlights while keeping its
  take.
