# The motion metric against the eye: the same thirteen renders, every joint (2026-10-05)

lane: masked
verdict: separates the clips that follow the source from those that do not on the one clip tried; not yet tried on a motion that is not a turn

**Result: the three renders judged to move as the source moves score 0.34
to 0.51 on `followed` (0.51 to 0.54 at their best time shift), and the ten
judged not to score within 0.04 of zero. Nothing sits between.** Thirteen
of thirteen against the eye. The catch is the calibration set: one clip
whose only motion is a half turn, so this says the metric agrees with the
turn metric where they must agree, and nothing yet about a walk, a profile
or a hand to the mouth. Numbers and every joint's curve:
`2026-10-05_subject_motion_calibration.json`. Poses:
`2026-10-05_subject_pose_band_turn.json`. Scripts:
`bench/measure_subject_motion.py` over the output of
`bench/measure_subject_yaw.py`.

**This record was rewritten the evening it was first committed** (0.197.0),
because the metric's definition changed twice that evening and the first
numbers no longer describe it. As first committed it compared joint
positions directly against "the source's first pose held", and said 0.57
to 0.68 for the three and -0.05 to 0 for the ten. A made-up body showed two
faults in that form: a constant difference in stance was charged to every
frame, and a render frozen in the middle of the source's range scored
above zero. The poses are the same file; only the arithmetic over them
changed.

## What is measured

The pose pass of `2026-10-05_subject_yaw_calibration.md`, run again with
seventeen body joints kept per sampled frame (its yaw readings came out
identical to the first pass). Each joint is taken from its own mean
position over the shot, on the source and on the render separately, and the
two displacements are compared frame by frame: that is the curve. Dividing
its mean by how far the source's joint travels from its own mean gives
`followed`: 1 moving as the source's joint does in every frame, exactly 0
for a joint that never moves, whatever pose it is frozen in. A clip's
`followed` is the same ratio over the joints its source moves, weighted by
how far each moves. 3D, in units of the hips-to-shoulders distance. CPU,
masked, no card.

## Against the eye

| by-eye verdict | clips | `followed` | at the best time shift | the shift |
|---|---|---|---|---|
| moves as the source | `ref2va_base12_s1`, `ref2va_base_s1`, `ref2va_base_s2` | 0.34, 0.36, 0.51 | 0.52, 0.51, 0.54 | 6, 6 and 2 frames late |
| does not | the ten others | -0.04 to 0.01 | -0.01 to 0.04 | none that helps |

The by-eye verdict used is the turn's, since on this shot the turn is the
motion.

- **A frozen subject reads zero, and ten do.** The ten that do not turn sit
  within four hundredths of zero, which is what the definition says a
  subject who does not move must score. That is the metric's floor on real
  renders: about 0.04.
- **The three that follow are late, and lateness is most of what they
  lose.** Shifted by their own lag the two seed-one renders go from about a
  third to about a half. The turn takes about eighteen frames, so six
  frames late is a third of it.
- **They are not on the source even when shifted.** About half the source's
  motion is unaccounted for at the best shift. By part, on the seed-two
  render: hips and elbows 0.66, shoulders 0.55, knees 0.48, head 0.46,
  hands 0.45, feet 0.33. The feet are out of frame for part of the shot.
- **The lean does not count.** `ship_ref2va_motion_s1`, which leans 34
  degrees and comes back, scores -0.04.
- **The gesture does not show.** `ref2va_pdd8_s1` opens its hands where the
  source does not; its hands score -0.06, inside the floor. In a turn every
  joint moves, so nothing is "held" and `stray` has nothing to report here.

## The verdict line was set from these clips

`FOLLOWS` is 0.25: between the ten at zero and the three at 0.34 and up.
It was chosen after seeing them, which is the weakness of it, and it has
met no other motion. The curves are the output; the line is one reading.

## The on-screen reading

Joints on screen, in units of the subject's box height, give the same
order and nearly the same numbers once each joint is centred on its own
mean: 0.31 to 0.46 for the three, -0.04 to 0.02 for the ten. Before the
centring that reading was thrown by the new subject's different build. The
verdict stays on the 3D reading; the two now agree.

## Limits

- **One clip, one motion.** Until a clip with another motion has by-eye
  verdicts, this is a metric that works on a turn.
- A render that performs the source's motion with the other arm reads as
  not following, with the motion showing as `stray`.
- Legs are partly out of frame on this shot; their joints are the body
  model's guess on both sides.
- Fingers, the face and the mouth are not read.
- The pose pass took about twelve minutes on the CPU with other work on
  the machine; a statement about that state.
