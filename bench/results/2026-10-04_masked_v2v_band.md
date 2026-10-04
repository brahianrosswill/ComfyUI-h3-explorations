# Masked video to video: a subject among other people

last updated: 2026-10-04 (later arms added the same day)

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

## Later the same day: the turn, the clothes, and what `replace` can do

All on the cleaned still (`f_img_outside_nomark.png`; the first still carried
a watermark sparkle that the model painted on the shirt as a bright dot, and
the owner withdrew it, so `band_generic_32s` cannot be re-run as written).
Read from sheets and single frames; the owner's words are quoted.

| arm | what changed | what the frames show |
|---|---|---|
| a third-window prompt block (`ref2va_masked_lead_turns_away`), windows 1 and 2 reused; run from `internal/`, not in the manifest | the prompt names the turn | he turns away with the group, a little after the original does. It works and it is specific prompting, which the owner does not want as the answer |
| head and hair, outline only (interrupted after one window) | `replace` = head and hair, the region the head-and-hair outline | sweatshirt, its print and the neighbours intact, the hair gone from the chest; the head about half as wide again as the original's, on the collar with no neck. Owner: "a giant bobblehead. doesnt work" |
| `band_head_w1` | the region extended down to where the hair ends, so neck and shoulders are drawn too | the same oversized head. The part detection also cost several minutes before sampling (`per_node_s` in the rows: the Masked Source node) |
| `band_follows_32s` | whole subject; the prompt ties his facing, movement and clothes to the people beside him (`ref2va_masked_lead_follows`) | he turns with the group at the right moment, and also turns his back twice when nobody else does; a plain grey-green T-shirt for most of the clip and a green long-sleeved top only from behind. No dot on the shirt |

**What this says.** The model frames what it paints to the hole: a
whole-body hole gives a person in proportion, a head-and-chest hole gives a
head-and-shoulders portrait that fills it. Nothing in the per-token-timestep
method anchors scale, so head-only replacement on a long-haired original is
not reachable by shaping the region. And a prompt cannot reliably carry the
original's movement: naming the shot works, a generic clause over-applies.
Not tried: the source as a motion reference, which the model is trained for
and which pays the reference's rows on every step.

## The margin, the pose and the member nobody can see

`band_changed_w1` (the composite that keeps only what changed, the mask grown
twice as far, the cleaned still) against `band_generic_32s` (whole region,
the default margin, the old still), stacked with the source as
`compare_band_w1_stacked.mp4` on the output share. The owner's reading, from
their annotated frames:

- Both renders keep the visible band members "mostly the same", with changes
  too small to notice unless the clips are stacked.
- At the wider margin the replaced lead stands differently, and that uncovers
  the member behind him on the left, whom the source hides behind the
  original's height and long hair. The model invents him. At the default
  margin a sliver of an invented face shows in the same place.
- "The invented person on the left in the back keeps the same identity
  throughout."

**Why the pose differs** (inferred, endorsed by the owner, not yet isolated):
the only thing telling the model where the original stood is the shape of the hole, so a wider hole lets him stand somewhere else. Two things differ between those renders, the margin and the
still, and either changes the sample; `band_changed32_w1` puts only the
margin back and says which. The composite cannot be the cause: it runs after
sampling. The still is a close selfie and carries no height or build, so the
subject's size and place come from the hole and the scene.

**What it means for the method.** Replacing a subject with a smaller outline
uncovers space the source never shows, whatever the margin. The margin sets
how much: tight keeps the new subject in the original's footprint, wide gives
a differently shaped subject room and exposes more. Nothing is built for it.

## Not established

- Whether the regenerated tokens beside him alter the neighbours they
  overlap. A peer's design for protecting them is in
  `internal/claude/2026-10-04_mrhf_notes.md`.
- Whether the faint remnant of the original subject a peer measured in the
  first clip's flicker shot is here too.
