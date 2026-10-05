# The motion metric against the eye: the same thirteen renders, the whole body (2026-10-05)

lane: masked
verdict: separates the clips that follow the source from those that do not on the one clip tried; not yet tried on a motion that is not a turn

**Result: the three renders judged to move as the source moves score 0.57
to 0.68 on `followed`, and the ten judged not to score between -0.05 and 0.
Nothing sits between.** Thirteen of thirteen against the eye. The catch is
the calibration set: one clip whose only motion is a half turn, so this
says the metric agrees with the turn metric where they should agree, and
nothing yet about a walk, a profile or a hand to the mouth. Numbers:
`2026-10-05_subject_motion_calibration.json`. Poses:
`2026-10-05_subject_pose_band_turn.json`. Scripts:
`bench/measure_subject_motion.py` over the output of
`bench/measure_subject_yaw.py`.

## What was measured

The pose pass of `2026-10-05_subject_yaw_calibration.md` was run again with
seventeen body joints kept per sampled frame (the yaw readings came out
identical to the first pass, to the last digit). For each render the mean
distance between its joints and the source's, frame by frame, in 3D from
the middle of the hips in units of the hips-to-shoulders distance; and the
same distance for a subject who holds the source's first pose and never
moves. `followed` is one minus the ratio: 1 on the source's pose in every
frame, 0 no closer than standing still. CPU, masked, no card.

## Against the eye

| by-eye verdict | clips | `followed`, whole body | at the best time shift | the shift |
|---|---|---|---|---|
| moves as the source | `ref2va_base_s1`, `ref2va_base_s2`, `ref2va_base12_s1` | 0.57 to 0.68 | 0.63 to 0.69 | 2 to 6 frames late |
| does not | the ten others | -0.05 to 0.00 | -0.02 to 0.02 | none that helps |

The by-eye verdict used is the turn's, since on this shot the turn is the
motion.

- **The three that follow do not reach 1.** About a third of the distance a
  still subject would be from the source is left, and shifting them in time
  recovers only a few hundredths of it. They follow late (two to six frames,
  which agrees with where each passes side-on in the yaw curves) and not
  exactly: the legs follow least (0.39 to 0.60), the head and torso most.
- **The lean does not count.** `ship_ref2va_motion_s1`, the render that
  leans 34 degrees and comes back, scores -0.04, with the other clips that
  do not turn.
- **The gesture barely shows.** `ref2va_pdd8_s1`, which opens its hands
  where the source does not, has the lowest arms score of the ten (-0.09
  against -0.05 to -0.07), a difference too small to read as a finding.

## The on-screen reading is not usable as a verdict

The second reading, joints on screen in units of the subject's box height,
orders the clips the same way and its numbers are lower throughout: 0.35 to
0.39 for the three that follow, -0.16 to -0.27 for the rest. The new
subject is not the old one's shape and does not stand in the box the same
way, and on screen that reads as distance. It is recorded; the verdict is
on the 3D reading.

## Limits

- **One clip, one motion.** Until a clip with another motion has by-eye
  verdicts, this is a metric that works on a turn.
- `FOLLOWS` is 0.5, reasoned before the run; on these clips anything from
  0.01 to 0.57 gives the same verdicts, so the line itself is untested.
- Legs are partly out of frame on this shot; their joints are the body
  model's guess on both sides.
- Fingers, the face and the mouth are not read.
- The pass took about twelve minutes on the CPU with other work on the
  machine; a statement about that state.
