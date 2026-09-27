# Reverse step switch, PDD8 first and FlashGen finishing: read (2026-09-27)

The owner's idea: fix PDD's weaknesses with FlashGen's complement. The runs are
`bench/followup_reverse_switch_arms.json`, on subway_chase at 730451892, with
handoffs at sigma 0.8 (`rev_h080`) and 0.632 (`rev_h063`), against the rerun's
exact PDD8. There is no full-length FlashGen subway_chase render, so the read
is against PDD8 alone. Records: `2026-09-27_reverse_{tone,resolution,temporal}.json`,
and divergence in `2026-09-26_followup_reverse_switch.json`. One clip per arm.

| arm | white | contrast | hf | moved | boil |
|---|---|---|---|---|---|
| PDD8 | .918 | .243 | .0073 | .242 | .894 |
| rev_h080 | .958 | .250 | .0077 | .244 | .892 |
| rev_h063 | .939 | .247 | .0079 | .243 | .918 |

- **The handoffs keep PDD8's take.** Motion, saturation, haze and the
  composition are unchanged.
- **The handoffs lift the highlights.** White point rises from .918 to .958 at
  the 0.8 handoff and .939 at 0.632, which is where FlashGen and FastH3 sit
  across the 13 signature scenes. Fine detail rises 5-8%. So part of PDD8's
  dim-highlight signature (`2026-09-26_distill_signatures.md`) comes from its
  coarse final steps, and a FlashGen finish repairs it without changing the
  scene.
- **The clone survives both handoffs.** Latents 8-9 still show two figures
  (`bench/x0_step_frames.py --preview`). This confirms the prediction
  registered before these rendered (`2026-09-26_distill_run_predictions.md`,
  the x0 entry): a late handoff cannot remove a step-1 composition decision.

So what: the reverse switch is a candidate fix for PDD8's dim look on scenes
where the owner calls it a defect, at one scene and one seed so far. The
lamp-lit interiors where PDD8's gap is largest (noodle_bar, radio_drama,
courtroom_verdict) are where it would be tested next.
