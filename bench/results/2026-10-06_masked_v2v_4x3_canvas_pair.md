# A 4:3 source on the 4:3 canvas and padded onto the shipped one: thirty seconds each way (2026-10-06)

lane: masked
verdict: owner, on playback: the 1024x768 render is good through its first two windows and loses the subject's look near the end; the padded 1344x768 render loses the source's movement and the reference's clothing; why the padded one fails is not known

**What was asked.** The lane's fourth clip is 4:3 and every masked graph is
built for 1344x768. The owner chose to try the trained 4:3 canvas,
1024x768, and to compare it with the same picture inside black bars on the
shipped canvas. Two renders of one thirty-second stretch.

**The verdict is the owner's, on playback, relayed by the lane's lead; in
tool terms where the owner's words would say what the clip shows.** A, the
original at 1024x768: "looks good". Later, of the same render: from about 25 s the replaced
subject is no longer the still's person, and by the end is dressed
differently from earlier in the clip. B, the padded file at 1344x768: it
does not keep the source's movement and it changes the clothing. Their
question, which this record cannot answer: the two differ only in the
padding, so why would one fail.

**What is established about why, and what is not.** The mask is not the
cause: a fresh track equals the stored mask on A's stretch (the numbers
are under `mask_check.stretch` in
`2026-10-06_masked_v2v_body_window_arms.json`), and
the regenerated share of each window is the same inside the picture on A
and B. The day's code changes are not the cause: a rerun of a short arm on later code decodes to the
same bytes (`code_neutral` in the same file). Nothing else is established. Padding
changes four things at once (the sequence length, what the model sees as
kept context, the motion reference's frame, and the noise) and no arm here
separates them.

## What rendered

`thrill_2160.mkv` from 56.0 s, thirty seconds, on the ref2va motion graph
at `h3_config.MASKED_MOTION_STEPS`, the graph's own seed, the prompt node's
subject set to a noun. A: the three canvas values patched to 1024x768
(the loader's width, the song node's width and height). B:
`thrill_2160_pad1344.mkv`, the same picture scaled to 1024x768 and centred
on 1344x768 with black at each side, no canvas patch. Three windows each.
The seconds, by node and by stage, and everything about the server they
ran on are in `2026-10-06_masked_render_time_breakdown.md`; in short the
4:3 canvas rendered the stretch in about seven tenths of the padded one's
time, with the caveats that record gives.

## The tracker on this stretch

- At defaults the tracker does not carry the subject across this clip's
  cuts: with the pick named on the opening shot it takes one more shot on
  its own and leaves three empty that hold the subject. Three typed
  corrections, read off the tile by the lane's lead, fix them; two shots
  without the subject stay empty. A no-sampling pass before each render
  confirmed which shots were masked.
- The padded file lost every cut at defaults; that is its own record,
  `2026-10-06_bordered_source_cuts.md`. B rendered on the old code with
  the cut threshold named, which gave the original's cuts.
- Checked after the owner asked that the mask be looked at and not only
  reused: a fresh track on later code equals the mask A used on all of its
  frames; its area changes sharply only at cuts and it is empty only on
  the two shots without the subject. The lead read sheets of every eighth frame of
  the stretch's second half and found the mask on the person to replace on
  every tile; the first half rests on the numbers.
- The node printed, per window, the share of video tokens that
  regenerate: 49.4%, 34.3% and 25.6% on A and 37.7%, 26.1% and 19.5% on B.
  B's are A's counts over the wider canvas, which is what one mask inside
  the same picture gives.

## What this does not say

- Why B fails. See the top.
- Anything about a second seed, or about the shipped canvas with a source
  that is not padded.
- That 1024x768 is good in general: one stretch, and the owner's later
  note about its last seconds stands beside the first verdict.
