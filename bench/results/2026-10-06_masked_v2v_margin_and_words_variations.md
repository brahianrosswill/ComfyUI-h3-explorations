# Three mask margins and a wordier subject on two short windows, with the owner's verdicts (2026-10-06)

lane: masked
verdict: owner, on playback: on the close window only the widest margin keeps the still's headwear, and at every margin the mouth opens where the track has no voice; the body window's base arm is "broken", so its four rows say nothing about margins or words. Sampling took the same seconds at every margin

**What this is.** Eight renders the owner asked for so that two choices can
be made by eye and not in the abstract: how wide the mask's margin should
be, and whether more words about the still help. This record holds what
the song node printed for each, so the choice has its numbers beside it,
and the owner's verdicts as they came back the same day.

**How to read the seconds.** They are soft. Other sessions' check sweeps
were running on the CPU while these rendered, and the composite, the write
and the source encode are CPU-side. The lane's timing record is
`2026-10-06_masked_render_time_breakdown.md`, on the day's two 30-second
renders, not this set.

**Why these are comparable with the day's two 30-second renders.** They ran
on the same server process, so the same loaded code: the tracker at
`MASK_VERSION` 7 and the song node as of bbe80949. The server was armed
with `H3_TELEMETRY` and nothing else.

## What rendered

`thrill_2160.mkv` at the 1024x768 canvas, through the ref2va motion graph
at `h3_config.MASKED_MOTION_STEPS` and the graph's own seed, one window of
141 frames per render, the pick named on frame 12, no correction needed.
Two windows: `close`, from 56.0 s, and `body`, from 67.0 s, both inside one
uncut shot. Four arms a window, everything else equal:

- `base`: `grow_pixels` 64 and the subject as a noun, the day's defaults
  with the noun set;
- `grow32` and `grow16`: the margin at 32 and at 16, `feather_pixels` left
  at 8;
- `words`: the margin at 64 and the subject as the noun plus the still's
  clothing in words. The text itself is not kept in this repo.

The figures are in `2026-10-06_masked_v2v_margin_and_words_variations.json`.

## What the node printed

| window | arm | margin | video tokens that regenerate | regenerated pixels the composite kept | song node, s | of which sampling, s |
|---|---|---|---|---|---|---|
| close | base | 64 | 58.6% | 83% | 133 | 99.9 |
| close | grow32 | 32 | 49.2% | 87% | 132 | 98.6 |
| close | grow16 | 16 | 44.5% | 90% | 134 | 98.8 |
| close | words | 64 | 58.6% | 83% | 132 | 98.4 |
| body | base | 64 | 31.8% | 84% | 132 | 98.6 |
| body | grow32 | 32 | 22.6% | 86% | 132 | 98.7 |
| body | grow16 | 16 | 17.9% | 88% | 134 | 98.7 |
| body | words | 64 | 31.8% | 84% | 132 | 98.7 |

Two things the table says without anyone watching a clip:

- **The margin is a large share of what regenerates.** From 64 to 16 the
  regenerating share falls by about a quarter on the close window and by
  nearly half on the body window, where the subject is smaller in frame.
- **Sampling costs the same at every margin.** Every arm sampled in the
  same seconds to within a second and a half, whether a sixth of the
  window's video tokens regenerate or three fifths. The stock sampler runs
  every row either way. That is the case for the frozen-row cache (masking
  board, `use-frozen-row-cache`), stated by the node's own lines.

The six margin figures were reproduced to the decimal from the two kept
masks by mrsun's offline copy of the node's mask arithmetic. The same
script gives one figure the node does not print: with no margin at all
the two windows would regenerate 39.9% and 13.4%, the floor a smaller
margin approaches. That is an offline figure, not a render.

Each window was tracked once and its other arms reused the kept mask, as
the Masked Source's log says for seven of the eight; the margin and the
subject text are both downstream of the mask.

## The owner's verdicts

From playback, relayed by the lane's lead the same day; in tool terms
where the owner's words would say what the clip shows.

- **The close window, by margin.** Only margin 64 keeps the still's
  headwear on the subject; at 32 and at 16 it is gone. At every margin the
  mouth opens while the track has no voice: about a second in at margins
  64 and 16, about two seconds in at 32. So on this window a narrower
  margin costs the still's look and buys nothing the owner named.
- **The close window, the wordier subject.** No verdict was relayed for it.
- **The body window.** The base arm is "broken": the still's own framing
  is drawn into the region, a head at the size of a region that holds a
  whole figure in the source. The other three body rows were not judged
  one by one, and they do not need to be: they sit on that failure. It is
  the window's and not this set's, since the same arm at the shipped
  canvas, a rerun on later code, and a run of typed prompts all show it
  (`2026-10-06_masked_v2v_body_window_arms.md`). So the body rows are
  evidence about the node's figures and about nothing else.

## What it does not say

- Which subject text is better: the wordier arm has no verdict on the
  close window and sits on a broken base on the body window. The files
  are under the output folder's `Video/mrteal_jr/choose/`: the eight
  singles, and per window a margin row and a words pair with each panel
  labelled.
- Which margin is right in general. One window with a verdict, at one
  size of subject in the frame.
- Anything about a second seed. One render per arm.
- Anything at the shipped 1344x768 canvas. These are all at 1024x768.
