# What separates PDD8, FlashGen and FastH3 (2026-09-26)

`bench/distill_signatures.py` over the followup batch's matched clips: same
prompt, length and seed per scene, PDD8 and FlashGen from the post-0.154.8
rerun, FastH3 on its contract graph. 13 scenes; temporal on 11, because
`kpop_dance_studio` and `silent_film` ask for brightness change. One clip per
arm per scene, so "on n of 13" counts shared signs, not an effect size.
Medians and counts are in `2026-09-26_distill_signatures.json`; the per-clip
records are `2026-09-26_distill_sig_{tone,resolution,temporal,audio}.json`.
Base is absent: its clips predate the 0.154.6 bank prompts.

| arm | consistent signature (share of scenes) |
|---|---|
| FastH3 | most fine detail (`hf`, `loss2`, `loss4`, `detail`: 13/13); most chroma (13/13); most saturated and warmest; least haze (11-12/13) |
| PDD8 | lowest contrast (12/13); dimmest highlights and narrowest range (11/13); least motion (`moved_share` 11/13, `motion_detail` 7/11); least fine detail (11/13); quietest by median LUFS, loudest on 1/13 |
| FlashGen | most haze, lifted blacks (11/13); coolest and bluest (11/13); least saturated, least warm (11/13) and lowest chroma (10/13); brightest audio, spectral centroid highest on 11/13; highest `block16` (12/13, a latent-grid statistic that does not track the owner's eye) |

Saturation order by median: FastH3 0.387 > PDD8 0.364 > FlashGen 0.305
(`tone_sat`). The owner's by-eye order, as fastdude relays it, was "fasth3 more
saturated, fastgen slightly less, pdd least"; the measure agrees on FastH3 and
swaps the other two. PDD8's low contrast and dim highlights may read as
washed out by eye. **Owner, 2026-09-27, asked whether PDD8 reads less
colourful or flatter and dimmer: "The latter usually - but usually naturally
so."** So the by-eye "pdd least" was the grade's contrast and level, not its
colour. The owner, same day: "its a defect in contexts... it depends on the
scene". Per scene, PDD8's `rms_contrast` over the mean of FlashGen's and
FastH3's (from `2026-09-26_distill_sig_tone.json`):
- largest gap, 0.82-0.84: noodle_bar, courtroom_verdict,
  slapstick_moving_piano, radio_drama (interior, practically lit);
- smallest, 0.93-0.95: samurai_bamboo_duel, look_noir, look_anchor,
  box_office, kpop_dance_studio.
The look scenes differ in kind: contrast is close, but PDD8's midtones sit at
0.66 of the others', so they read darker. Which contexts are the defect is
the owner's call; the table only says where the gap is largest. Corrected 2026-09-27: this table first said FlashGen was
"least saturated and warmest", which read as a contradiction; it is the least
warm.

Leads, each with the test that would settle it:
- **FastH3's detail sits in its time embedder, or not.** The swap arms'
  prediction S4 (`bench/fasth3_swap_arms.json`); if it does, the dial files
  `fastvideo_fasth3_8step_v2_pruned_int8_convrot_temb_a{05,075}` are a
  detail control.
- **PDD8 moves half as much.** This backs routing PDD to still shots. The
  strength arms (`bench/pdd_strength_arms.json`) show whether turning its
  delta down restores motion.
- **FlashGen's haze and cool grade** belong to its late blocks, or not:
  fastdude's block transplant (`bench/flashgen_transplant_arms.json`).
