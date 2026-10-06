# The turn and motion metrics on a second clip: where they agree with the eye and where they do not (2026-10-05)

lane: masked
verdict: the two verdict lines set on the band clip say nothing useful here; the figures under them separate the renders, and which render is closer depends on whether the head or the shoulders are read

**Not judged by the owner.** The owner's words on these renders are about
flicker, the edge and the composite
(`2026-10-04_masked_v2v_first_run.md`); no verdict on turn or motion exists
from the owner for this clip. The by-eye column is this session's, read
from eight stills a shot at thumbnail size and written down before any
number was read. Stills cannot show a motion over time, so the motion
column is the weakest thing here.

**Result.** On `idontwannatalk.mp4`, first window, five renders, three
shots:

- **Turn, the verdict:** every render holds on every shot. The clip has no
  turn the size of the band's, and the line that was set on a half turn
  passes everything.
- **Turn, the figures:** they do separate the renders, and the head and
  the shoulders give opposite orders on two shots of three. The eye
  agreed with the head on one and with the shoulders on the other.
- **Motion, the verdict:** on the one shot judged for motion the metric
  says no render follows and the eye said four of five do: none of four
  matches.
- **Motion, the figures:** the amplitude-free reading added here orders
  those renders as the eye did, at a scale far below the band's.

Scoreboard, both clips: `2026-10-05_lane_scoreboard.md`. Poses:
`2026-10-05_subject_pose_solo_first_window.json`. Motion by shot and joint:
`2026-10-05_subject_motion_solo_first_window.json`. Verdicts and what is
known of each clip's provenance: `bench/turn_metric_eye_verdicts.json`.

## What was read

The window's first 345 frames at 24 a second, every second frame, on the
CPU. The source cuts at frames 164 and 303 (the Subject Track's cut
finder), so three shots: he walks in at a medium shot, sings at full
length in profile, and a close-up. No kept mask exists for this window, so
the subject's box came from a mask video an earlier Subject Track run left
on the share; it is used for the box and nothing else. The singer has not
entered for about the first second and a half: no side has a reading
before frame 38.

Five renders: `generic_w1`, `specific_w1`, and the first window of
`generic_30s`, `generic_paint_out` and `generic_changed`. **Four of the
five are one prompt and one seed**, and `generic_30s` read identically to
`generic_w1` in every figure: the first-run record says its first window
came back from the server's cache. So this is two prompts, and three
variants of the composite on one of them.

## Turn: head against shoulders

End of each shot, distance from the source in degrees.

| shot | render | shoulders | head | by eye (stills) |
|---|---|---|---|---|
| walk-in, 0 to 163 | `generic_w1` | 3.8 | 29.0 | partial |
| | `specific_w1` | 20.4 | 6.1 | yes |
| full length, 164 to 302 | `generic_w1` | 12.0 | 2.9 | yes |
| | `specific_w1` | 4.4 | 27.9 | yes |
| close-up, 303 to 344 | `generic_w1` | 22.7 | 8.6 | partial |
| | `specific_w1` | 5.4 | 24.7 | yes |

- **Walk-in.** The source ends with the body at three quarters and the
  head turned to the microphone. `generic_w1` matches the body and not the
  head; `specific_w1` turns body and head together into profile. The eye
  called `specific_w1` the closer: it read the head.
- **Full length.** The eye saw five renders in profile and said yes to
  all. The head reading puts `specific_w1` 28 degrees from the source,
  which thumbnails of a small figure did not show.
- **Close-up.** The eye, and the first-run record's reading of the same
  shot, called `specific_w1` the closer: in profile to the microphone,
  chin lifted. By the ears its head is 25 degrees past the source's at the
  end and 33 on average over the shot, against 6 for `generic_w1`: the
  source is at three quarters and `specific_w1` is in full profile. The
  shoulders agree with the eye here and the head does not.
- **The chin.** The source sings with the chin lifted: 18, 10 and 19
  degrees at the end of the three shots. `specific_w1` lifts it (16, 12,
  22) and `generic_w1` mostly does not (4, 1, 13). This is the one thing
  the eye's "follows the blocking" and a number agree on in all three
  shots, and neither yaw reading carries it.

So "does he face where the source faces" is three numbers on this clip,
and the band clip could not show it: there the turn is the whole body's
and the three agree (head and shoulders both thirteen of thirteen).

## Motion: the full-length shot

The source's forearms go out, come down and go out again while it sways.
By eye the four generic renders do the same and `specific_w1` keeps its
arms nearer its sides.

| render | followed | in step | hands, in step | right wrist, in step | by eye (stills) |
|---|---|---|---|---|---|
| `generic_w1` | -0.04 | 0.19 | 0.26 | 0.44 | yes |
| `generic_paint_out` | -0.05 | 0.23 | 0.30 | 0.53 | yes |
| `generic_changed` | -0.06 | 0.12 | 0.16 | 0.23 | yes |
| `specific_w1` | -0.09 | 0.04 | 0.03 | 0.03 | partial |

- **`followed` puts all of them at zero**, inside its own floor. It asks
  for the source's trajectory at the source's size in 3D, and on a
  full-length profile nothing passes: the left wrist is unrelated to the
  source's on every render (in step at 0.02 to 0.10) and the right wrist
  is only partly with it.
- **`in_step`, the same motion at any size, orders them as the eye did**,
  and almost all of it is one wrist. It is 0.75 to 0.86 on the band
  clip's three renders that follow and 0.12 to 0.23 here on four the eye
  called yes, so it has no line that serves both.
- **Every render's wrists travel about two thirds as far as the source's,
  `specific_w1`'s included** (`render_travel` in the motion record); its
  are unrelated to the source's in time. The eye read that as arms held
  near the sides, so the stills fell between its movements. A caution
  about the eye column, not about the metric.

On the walk-in only the elbows, shoulders and head are in frame, and the
metric is scored on those: `followed` 0.20 to 0.38, with three of five
over the band's line. Nobody judged that shot for motion. On the close-up
only the head and shoulders are in frame and nothing follows by either
reading.

## What changed in the tools because of this clip

- The pose pass takes a mask video or finds the box itself when no kept
  mask exists, finds the source's cuts, and compares per shot.
- It reports the head's facing and the chin's lift beside the shoulders'.
- The motion metric reports `in_step` beside `followed`, and leaves out
  joints that are outside the picture: a close-up was being scored on
  knees and feet the body model invented.
- `bench/run_lane_benchmark.py` writes the scoreboard for every window
  the verdicts file names.

## Not measured, and what would settle the rest

- **`friday_chorus_14s.mp4` has no render on the share**, only tracking
  tiles, so there was nothing to score. It needs renders, and a kept mask
  for the box: several people are in frame and the body model cannot pick
  the subject.
- **Which reading the turn verdict should use** is the owner's to say:
  when a render "faces where the source faces", is that the head or the
  body. The band clip does not distinguish them and this one does.
- **A line for motion on a gesture.** It needs the owner's playback
  verdict on the full-length shot for these five, and a second clip with a
  gesture; four renders of one prompt and seed cannot set one.
- Profile at full length is where a body model is weakest: the far arm is
  hidden and left and right can be confused. Part of the left wrist's
  disagreement may be that and not the render.
- The mask video is from a different tracker run than the renders' own
  mask; a box a little off is the same box for source and render.
- The pose pass's wall time is in the pose record (`seconds`), on the CPU
  with other work on the machine.
