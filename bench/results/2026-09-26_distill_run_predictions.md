# Predictions for the 2026-09-26 distill run and its follow-up

Registered before any distill arm of the run was looked at. When this was
committed the run was on its base arms (`bench/distill_run_arms.json`, rows in
`2026-09-26_distill_run.jsonl`). A prediction is kept as written. When results
land, each gets a dated verdict line under it: held, failed, or could not tell.
The reasoning behind each is `../../docs/h3_distills.md`, "Why they differ: a
working model".

Two facts constrain what the run can say:
- **The base and the distills do not share starting noise at one seed**
  (`er_sde` replaces it at step 1; `h3_distills.md`). A tone or layout
  difference between the base and a distill in this run is two unrelated
  samples, not a controlled pair.
- **The three distills do share starting noise** (all Euler, no injected
  noise), so comparisons among them are the fairest this run offers. They
  still differ in weights, so each is its own scene.

## Claude (this session)

### Tonight's run

- **P1. FlashGen follows fewer checklist beats than the base.**
  - The gap is largest on beats about who does what to whom and which way
    things move: slapstick, kpop, samurai, courtroom.
  - It is smallest on scenes with one subject or little motion: radio_drama,
    post_office, noodle_bar.
  - Why: a distribution-matching distill favours likely outputs, and role and
    direction instructions are where "likely" and "asked for" part.
  - Fails if FlashGen matches the base on the multi-person beats across most
    of those scenes.
- **P2. FlashGen dense does not fix its adherence.**
  - Dense and Sol follow about the same number of beats on the four motion
    scenes.
  - Why: the adherence loss is in the weights, not in the attention sparsity.
  - Fails if dense follows clearly more beats on three or more of the four.
- **P3. FlashGen's grade moves with its adaln update and its strength, and
  its adherence does not.**
  - Without adaln: less contrast and brightness than full FlashGen, and less
    coherent detail. The few-step dynamics live partly in the timestep
    modulation.
  - At strength 0.8: grade slightly toward the base, detail softer.
  - At 1.2: more contrast and saturation, possibly artifacts.
  - Beats followed stay about the same across the three.
  - Fails if removing adaln leaves the grade unchanged.
- **P4. PDD8 clones or doubles people on the motion scenes, not on the still
  ones.**
  - Motion scenes: slapstick, kpop, samurai, rooftop_pov.
  - Its artifact severity ranks with each scene's inter-frame delta
    (`measure_clip_delta.py`).
  - Its detail and colour are best on radio_drama, post_office, courtroom.
- **P5. Among the distills, FastH3 is the most saturated and warmest on most
  scenes, FlashGen next, and PDD8 the least.**
  - Measured by `measure_clip_tone.py`.
  - Not claimed against the base (see above).
- **P6. FastH3 shows more temporal texture instability than FlashGen on the
  same scene.** This is the owner's "low bitrate streaming video". It is
  graded once the motion-compensated measure exists. The per-frame
  resolution measures do not separate them.
- **P7. Step-switch (FlashGen layout, then PDD detail) clones less than PDD8
  on its four scenes, and keeps more detail than FlashGen.** A visible
  handoff artifact is possible and would count against the route.
- **P8. On the five-second subway twin, FlashGen follows the roles better
  than on the 14-second prompt, and still worse than the base.**

### The follow-up batch (after the run and a restart)

- **F1. The base on Euler at 16 steps is a different scene from the shipped
  base at the same seed.**
  - Its tone is punchier than the `er_sde` base.
  - If its tone sits where the distills do, most of "every distill shifts
    the grade" is the base's sampler, not distillation.
  - Fails if base-Euler's tone matches base-`er_sde`'s.
- **F2. PDD8 lands closer to the base on Euler at 32 steps (the teacher's
  grid) than FlashGen or FastH3 do.** Measured per latent frame, with no
  decode.
- **F3. PDD clones follow the coarseness of the last step: 4 steps worst,
  6 and 8 alike, 16 rarest.**
  - Six has 8's last step at fewer evaluations.
  - Fails if 6 sits halfway between 4 and 8, which would mean the count, not
    the tail.
- **F4. PDD8 exact vs merged moves the output less than FlashGen's switch
  did.** PDD's delta is a larger fraction of an int8 step, so the merge lost
  less of it.
- **F5. In the per-step capture, the double image first appears at PDD8's
  last two steps, not in the early ones.**
- **F6. FastH3 with VSA off shows less temporal texture instability and the
  same grade.** Sparsity would explain the texture, and the weights the
  grade.
- **F7. The look family (`t2va_look_*`).**
  - The base follows each requested look.
  - The distills do not shift tone by a constant offset. They pull the
    unusual looks toward typical ones: more saturation into the muted
    variant, deeper blacks into the faded-16mm variant's lifted blacks, and
    colour leaking into black-and-white noir.
  - The pull is largest for FastH3, then FlashGen, then PDD8.
  - Fails, and points to a fixed tone bias instead (adaln, a learned curve),
    if each distill's shift from the base is the same whatever the look.

- **F8. Part of the "low bitrate" look is our own encode, not the model.**
  - The owner, 2026-09-26: "its always in shows like the witcher on netflix
    where dark/night scenes looked so blocky and splotchy".
  - Every graph saves through `VHS_VideoCombine` as 8-bit 4:2:0 h264 at
    CRF 19 (the generator's save node). x264's default adaptive quantisation
    starves dark flat regions of bits.
  - Prediction: decoding a `_savelat` latent losslessly shows less blocking
    and banding in the shadows than the mp4 of the same render. The gap is
    largest on grainy output (FastH3), where grain takes the bits.
  - Fails if the lossless decode is as blocky in the shadows as the mp4:
    then the blocking is the model's.

## vaedude

To be added by that session, signed and dated, before it looks at any
distill arm.
