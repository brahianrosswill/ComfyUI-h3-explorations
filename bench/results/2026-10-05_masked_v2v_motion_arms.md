# 2026-10-05: the turn arms on the masked lane (encoder-only motion reference)

Session mryellow. Rows: `2026-10-05_masked_v2v_motion_arms.jsonl` (one per
arm, wall and per-node seconds, from `bench/run_graph_arms.py`). Graphs and
contact sheets under `internal/claude/2026-10-05_mryellow/arms/` (not
tracked). The research note is `docs/research/masking/2026-10-05_mryellow.md`;
the question it answers is section 1's: why the replaced subject does not
turn with the band when the original does.

## Setup, shared by every arm

The band clip from 112 s, 360 source frames, one window of 345, the
Masked Source with the subject's SAM 3.1 mask, the generic male prompt
(`internal/internal_prompt_bank/mask/swap_male.txt`, mrhf) and the still. The
motion arms add the Masked Source's `motion_reference = subject only` at
`motion_short_edge = 384` with `motion_vae` off (the subject's own frames on
grey reach the encoder only, no DiT rows; 0.190.7) and the prompt's
relationship sentence on `<Video 1>` (`swap_male_motion.txt`). Seed one is
mrhf's seed; seed two is one above it. Judged on contact sheets of frames
237, 247, 257, 267 and 277 (the turn shot) against mrhf's generic-prompt
control at seed one and `control_s2` at seed two. One window, so the
multi-window carry is not exercised.

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `motion_s1` | fl2va PDD8 bake (the shipped graph) | 8 | subject frames, encoder only | no | kept |
| `motion_s2` | same, seed two | 8 | same | no | drifted (no cap, longer hair) |
| `control_s2` | fl2va PDD8 bake, generic prompt, no reference | 8 | none | no | same drift as `motion_s2` |
| `ref2va_base_s1` | ref2va undistilled | 16 | same as motion | **yes**, back to camera from about frame 267 | kept |
| `fl2va_base_s1` | fl2va undistilled | 16 | same as motion | no | kept |

Pending at the time of writing (appended below as they land):
`ref2va_pdd8_s1` (ref2va with its PDD8 bake, 8 steps), `ref2va_pdd6_s1` (the
same bake on `h3_config.PDD_MANUAL_SIGMAS`, six evaluations; a `steps`
request of 6 is refused by the node and the knot list is the sanctioned
form, `docs/h3_pdd.md`), `ref2va_base_s2` (the base at seed two).

## What the five rows say

- The encoder-only video reference carries the turn on ref2va and nothing
  on fl2va. fl2va ignores it as the PDD8 bake (two seeds) and undistilled at
  sixteen steps (one seed), so the checkpoint is the variable, not the
  distill or the step count.
- The seed-two look drift on the bake is the seed's, not the reference's:
  the control at that seed drifts the same way with no reference wired.
- A same-clip reference of the subject does not confuse ref2va: the look
  stays the prompt's (cap, grey T-shirt) while the pose follows the
  reference. This answers the board's `q-same-clip-reference` for one seed.
- Wall times in the rows are queue-inclusive where arms shared the server
  (`control_s2` sat behind the ref2va base); the song node's own seconds
  are the per-node field.

## Caveats

One seed on each base arm; the bake arms have two. The turn is read by eye
on five frames; no motion metric. The band clip's turn is the only motion
tested. Cache state: every arm ran on one server started at 14:02 with the
models staged once, the encoder reloaded per prompt change.

## Appended 14:55: the ref2va PDD8 bake

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_pdd8_s1` | ref2va with its PDD8 bake (`PDD_REF2VA_LORA`) | 8 | same as motion | no; he faces the camera and from frame 257 opens his hands in a gesture the source does not make | kept |

The bake on ref2va loses the turn the base had at the same seed, so the
reference's hold on the pose does not survive the distill at eight steps, or
does not survive eight steps. Those two are separated by `ref2va_base8_s1`,
queued: the undistilled base at eight steps. If it turns him, the distill is
what loses the reference; if it does not, it is the step count. The gesture
is new on this arm and may be the bake's own habit under a video reference;
one seed.

## Appended 15:10: the ref2va base at a second seed

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_base_s2` | ref2va undistilled, seed two | 16 | same as motion | **yes**, back to camera from about frame 257 | kept (cap, short hair, dark T-shirt; the control at this seed has long hair) |

Both seeds on the base turn him. The seed-two drift seen on the fl2va bake
does not appear here with the reference wired.

## Appended 15:15: the ref2va bake on six evaluations

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_pdd6_s1` | ref2va with its PDD8 bake on `PDD_MANUAL_SIGMAS` | 6 evaluations | same as motion | no; arms at his sides, no gesture | kept |

Same as eight on the turn (none) and cleaner on this seed: the open-hands
gesture the eight-step bake added is absent. The song-node seconds are in the
rows for the cost difference. One seed.

## Appended 15:22: the ref2va base at eight steps

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_base8_s1` | ref2va undistilled | 8 | same as motion | no; faces the camera, hands lifted a little from frame 257 | kept |

The undistilled base at eight steps loses the turn the same base carried at
sixteen, so it is the step count, not the distill, that loses the reference:
the PDD8 bake at eight behaves as the base at eight does, gesture included.
The reference's hold on the pose needs more evaluations than eight on this
seed. `ref2va_base12_s1` is queued to narrow where between eight and sixteen
the turn appears.

## Appended 15:35: the ref2va base at twelve steps, and the set complete

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_base12_s1` | ref2va undistilled | 12 | same as motion | **yes**, back to camera from about frame 257, as at sixteen | kept |

Twelve carries the turn as sixteen does; eight does not. The threshold on
this seed is between eight and twelve evaluations. The set is complete:
eleven rows, the server stopped at 15:35.

## Verdict

- fl2va does not take an encoder-only video reference of the subject here,
  distilled or not, at eight or sixteen steps.
- ref2va takes it at twelve and sixteen steps on the base and loses it at
  eight with or without its PDD8 bake, and at six evaluations. The step
  count is the variable, not the distill.
- The route therefore exists today on ref2va at twelve steps, at that
  cost; the song-node seconds in the rows price it against the shipped
  eight-step fl2va graph. Whether the lane moves its moving shots there is
  the owner's call on the board.

## Appended 16:45: the control with no reference, and the shipped graph's own render

| arm | chain | steps | reference | turn | look |
|---|---|---|---|---|---|
| `ref2va_base12_noref_s1` | ref2va undistilled, the generic prompt as mrhf wrote it, `motion_reference` none | 12 | **none** | no; faces the camera through the shot | kept |
| `ship_ref2va_motion_s1` | `workflows/h3_video_to_video_masked_song_ref2va_motion_api.json` as generated, the band clip, seed one, one window | 12 | same as motion | yes, a partial one: side-on by frame 257, three-quarter back by 267, the face still partly visible at 277 where the arm shows his back | kept |

The control is the question the owner asked ("if you just use the ref2va
model and dont pass in the video reference"): on ref2va at twelve steps with
nothing but the plate, the still and the generic prompt he does not turn. So
the reference is what carries the turn, and ref2va above eight steps is what
lets it through. The shipped graph's own render turns him less fully than the
arm at the same seed; the two differ only in the prompt's base text (the
shipped `ref2va_masked_subject_swap` against mrhf's generic male prompt,
each with the same three `<Video 1>` lines). One seed each; the base text is
the next knob if a fuller turn is wanted from the shipped graph.

## Verdict, revised

- fl2va does not take an encoder-only video reference of the subject here,
  distilled or not, at eight or sixteen steps.
- ref2va takes it at twelve and sixteen steps and loses it at eight with or
  without its PDD8 bake, and at six evaluations. Without the reference,
  ref2va at twelve does not turn him. The step count and the reference are
  both needed; the distill is not the variable.
- The route ships as `h3_video_to_video_masked_song_ref2va_motion_api.json`
  (and `daily/h3_mask_ref2va_motion_api.json`), verified on one render of
  the generated graph; the fl2va PDD8 graph stays the default for a shot
  that needs no movement from the source.
