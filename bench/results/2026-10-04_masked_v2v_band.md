# Masked video to video: a subject among other people

last updated: 2026-10-04

**Status: rendered, not judged.** Read from a one-frame-per-second sheet and
three single frames; lip sync, edges at full size and the audio are the
owner's to judge on playback.

## What ran

`bench/masked_v2v_band_arms.json`: 1:35 to 2:07 of a six-person band video
with coloured lighting changes and cutaways, the lead singer replaced from
the same reference still as `2026-10-04_masked_v2v_first_run.md`, on the
PDD8 song chain with the audio frozen hard. Prompt
`ref2va_masked_lead_swap`: the clip-agnostic prompt with the claim that the
subject is alone removed. Rows: `2026-10-04_masked_v2v_band_arms.jsonl`.
The clip is under `Video/mrpink/` on the output share.

## Isolating one person among several

Tracker probes on the span, each read from an overlay sheet or a per-object
table (presence, size and horizontal position per shot):

| phrase | what it marked |
|---|---|
| "person", object cap raised | everyone, and the cap was spent before the main shot: the lead is untracked from the neon take on |
| "man with long hair" | the whole front row and the men in the cutaways |
| "lead singer" | only the centre singer in the group shots, and the most prominent person in each cutaway |

"lead singer" broken into its objects gives the lead as three of them, one
per shot he is in, with the whole neon take a single object present in every
frame. Selecting those three by index is what the arm does. Another member
is re-identified across a cut as one object, so identity across cuts is not
simply lost; it is unreliable.

**The indices are bookkeeping for one clip.** They hold for this loader
start, frame cap and phrase, and a different clip needs the probe again.

## What the frames show

- The lead is replaced in the warm group shot and through the neon take, at
  the original's place in the frame, and the others are untouched where he
  does not cover them.
- **The light follows.** Under the green and purple washes his skin and
  shirt take the room's colour.
- In the push-in he stands with hands on hips as the two beside him do.
- The cutaways and the opening two-shot are untouched: nothing was masked
  there.
- **The shot from behind is wrong.** The group turns its back; he faces the
  camera. The prompt names no action, and nothing told him to turn.
- He wears the still's T-shirt and cap among five green sweatshirts. The
  prompt asked for that; it is a choice, not a failure.

## Not established

- Whether the regenerated tokens beside him alter the neighbours they
  overlap. A peer's design for protecting them is in
  `internal/claude/2026-10-04_mrhf_notes.md`.
- Whether the faint remnant of the original subject a peer measured in the
  first clip's flicker shot is here too.
