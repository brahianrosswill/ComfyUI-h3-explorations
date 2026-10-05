# The turn metric against the eye: thirteen renders of the band clip's turn shot (2026-10-05)

lane: masked
verdict: accepted for "holds the turn or not"; it contradicts the one by-eye "partial", and the frames side with the metric

**Result: the metric puts every clip the eye called a turn within sixteen
degrees of the source at the end of the shot, and every clip the eye called
no turn about 160 degrees from it. The one clip the eye called a partial
turn measures as a 34-degree lean that comes back to the camera, and its
frames show that, not the three-quarter back view the eye recorded.**
Numbers: `2026-10-05_subject_yaw_calibration.json`. Script:
`bench/measure_subject_yaw.py`. Eye verdicts as transcribed:
`bench/turn_metric_eye_verdicts.json`, from
`2026-10-05_masked_v2v_motion_arms.md`.

## What was measured

The band clip's window from 112 s, the turn shot, frames 237 to 279, every
second frame. On each sampled frame core's SAM 3D Body predictor (the ViT-H
release, no hand refinement) ran on the subject's box, taken from the
window's kept mask and fitted to the render's canvas. The yaw is the
direction of the shoulder line in the camera's horizontal plane: 0 facing
the camera, 180 with the back to it. The source itself and thirteen renders
under `Video/mryellow/` on the output share. On the CPU, float32, masked;
no card was used.

The source turns from 6 degrees to about 170 away across frames 243 to 261
and stays there. Its largest turn is the full half circle.

## Against the eye

| eye verdict | clips | end difference from the source (degrees) | largest turn (degrees) | metric's verdict at 45 |
|---|---|---|---|---|
| yes | `ref2va_base_s1`, `ref2va_base_s2`, `ref2va_base12_s1` | 15 to 16 | 179 to 180 | holds, all three |
| partial | `ship_ref2va_motion_s1` | 152 | 34 | fails |
| no | the eight others judged | 158 to 164 | 1 to 7 | fails, all eight |
| not judged | `ref2va_pdd8mid_s1` | 160 | 1 | fails |

The end differences keep the eye's order (every yes below the partial below
every no), which is the card's acceptance. The order is not the finding.
The gap between yes and everything else is about 135 degrees; the gap
between the partial and the no clips is six, which is nothing.

**Where it disagrees with the eye: `ship_ref2va_motion_s1`.** The record
says "side-on by frame 257, three-quarter back by 267, the face still
partly visible at 277". The metric says he leans to 38 degrees around frame
259 and is back within 15 degrees of the camera by 273. I looked at the
render's own frames 257, 267 and 277: his chest and face are toward the
camera in all three, the body angled in the first and square to the camera
in the last. So on this clip the metric is right and the by-eye sentence
does not describe the file. It is the only render of that graph on the
share (one file, written 16:39). The consequence is for the record that
judged it: the shipped ref2va motion graph's own render did not carry the
turn on this seed, where the arm with mrhf's base prompt did.

**The three that hold arrive late.** They reach the source's facing but
trail it through the turn: the source passes side-on at frame 251, the two
seed-one renders at 257 and the seed-two render at 253, and their mean
difference over the shot is 15 to 29 degrees. None parts from the source
for good, so the verdict is "holds"; the lag is in the curves.

## What the metric adds to the end difference

The largest turn separates three things the end difference folds into two:
a full turn (about 180), a part-turn that comes back (34 on the one clip
that has it), and no turn (under 8). The verdict stays on the end
difference, at a tolerance set before the run.

## Limits

- One clip, one shot, one kind of motion: a half turn about the vertical.
- The body model reads a generated person. On these frames its readings are
  smooth frame to frame, the shoulders and the hips agree to within ten
  degrees on the renders and fifteen on the source, and the one surprising
  curve was checked against its frames. A front and back confusion would
  show as a jump of 180 between neighbouring samples; the largest jump in
  the fourteen curves is 33, in the middle of a turn.
- The card names fifteen clips; thirteen are under `Video/mryellow/`. The
  two not measured are the generic-prompt controls under another session's
  folder, which have no row in the eye file.
- The tolerance of 45 degrees was reasoned, not fitted. Anything from 16 to
  150 gives the same verdicts on these clips.
- The whole pass took thirteen minutes on the CPU with other work running
  on the machine; it is a statement about that state and not a rate.
