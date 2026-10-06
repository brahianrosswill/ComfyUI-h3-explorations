# The shipped motion graph with "a person" and with "a man" in its text: two renders on the band window (2026-10-06)

lane: masked
verdict: one seed: with "a person" he still turns and ends where the source ends, and starts later than with "a man"; the shipped graph on the measured text reproduces the 2026-10-05 arm frame for frame; neither clip watched

**How to read it, fixed before the second render existed.** The first
render is the shipped ref2va motion graph with the text the prompt node
now writes, "the person" and "their" where the measured arms of 2026-10-05
said "the man" and "his". Set beside that day's arm `ref2va_base12_s1`
alone it was one-sided: the arm was an arm graph, not the shipped graph,
a day of code earlier, so a difference could be the noun, the graph, the
day, or one seed's spread. The second render is the control that separates
them: the same queued graph with only the prompt node's `subject` set to
"a man". Three readings were written down by the lane's lead before its
result:

- (a) the control starts the turn as late as the first render: the delay
  is the shipped graph or the day's code, not the noun;
- (b) the control starts as early as the arm: the noun costs timing on
  this seed;
- (c) in between: one seed's spread is as large as the effect, and nothing
  follows.

The rule for telling them apart, written in the session's notes before
the result: by the frame at which the shoulders' facing passes side-on,
with less than a frame and a half counted as no difference, since the
samples are two frames apart.

**Result: (b), and exactly. The control is the arm, frame for frame. With
"a person" the turn completes and ends with the source's, and starts
about five and a half frames later than with "a man".** One seed, one
window. Nobody has watched either clip, this session included: the look,
the lips and everything a number here does not name are not judged.

**Second result: on the measured text the shipped graph is the arm.** The
prompt node, the generated graph, a day of commits and the tracker's new
version change nothing in the output on this window and seed.

## What rendered

`workflows/daily/h3_mask_ref2va_motion_api.json` as generated at
9ae35f54, through `bench/run_graph_arms.py`, patched only where the graph
names its placeholder media: the still, the source clip with its start
and frame cap (window `band_112s` of
`bench/turn_metric_eye_verdicts.json`), the song node's extent and its
output prefix. The seed, the step count (`h3_config.MASKED_MOTION_STEPS`),
the Masked Source and the prompt node are the graph's own. The control
adds one patch, `MiniMaxH3MaskedPrompt.subject`. Both rows are in
`2026-10-06_masked_v2v_person_text.jsonl`. The runner recorded the
prompts as bank entries `ref2va_masked_person_motion` and
`ref2va_masked_subject_motion`; the control's prompt hash is the one on
the arm's row of `2026-10-05_masked_v2v_motion_arms.jsonl`. The Subject
Track ran at `MASK_VERSION` 7, whose mask on this window is frame for
frame the earlier one (`2026-10-06_subject_track_defaults.md`).

## The shipped graph reproduces the arm

The control's file and the arm's file decode to the same video: one hash
over every frame. So on this window and seed nothing between the arm
graph and the shipped graph changes the output: not the generator's
graph, not the prompt node writing the text in place of a typed string,
not the tracker's new version, not a day of commits. The render the
2026-10-05 notes left owed, the shipped graph on the measured text, is
this one.

That leaves the four changed words as the only difference between the
first render and the arm.

## The three rows

Shot 237 to 279, the shot where the group turns its back. All three were
read in one pass of `bench/measure_subject_yaw.py` off today's kept mask,
on the CPU, and `bench/measure_subject_motion.py` over the same poses.

| | `person_text_s1` | `man_text_s1` | `ref2va_base12_s1` (2026-10-05) |
|---|---|---|---|
| text | "a person" | "a man" | "a man", typed |
| shoulders at the shot's end, degrees from the source | 18.4 | 14.6 | 14.6 |
| turn verdict (tolerance `TOLERANCE_DEGREES`) | holds | holds | holds |
| mean difference over the shot, degrees | 49.0 | 29.1 | 29.1 |
| frame the facing passes side-on (source: 250.8) | 262.4 | 256.8 | 256.8 |
| motion `followed`, zero shift | 0.07 | 0.38 | 0.38 |
| motion verdict (at `FOLLOWS`) | does not follow | follows | follows |
| best shift, frames | 10 | 6 | 6 |
| `followed` at the best shift | 0.54 | 0.55 | 0.55 |

Same direction, same end, a longer delay. At its own delay the first
render matches the source's motion as well as the other two do at theirs.
The facing curves are in `2026-10-06_masked_v2v_person_text.json`.

## What it does not say

- That "a person" is later in general. A change of wording moves the
  sample, and this is one sample of that movement on one seed. A second
  seed could put it either side.
- Anything about a second window or another clip.
- Anything about how either clip looks. The metrics read the pose of
  whoever is in the kept mask's box; a subject who turns on time and has
  lost the still's look would score the same.
- Whether a delay of this size matters to the eye. That is the owner's on
  playback, and until then the default is held on these clips, not
  verified.
