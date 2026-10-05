# Masked video to video: how it works, what it cannot do, where to go next

last updated: 2026-10-05 (the Sapiens2 nodes named); 2026-10-04 (first written, after the Subject Track's third clip)

Written by hand. This is the lane's map for a reader who has not followed
it: the pieces in the order a render meets them, the limits each one has
shown, and the directions worth trying. It carries no measurement. A value
is named by the constant that holds it, a result by the dated record that
holds it, and where this page and the code disagree the code is right.

The dated account, with the owner's words, is
[`../h3_audio_freeze.md`](../h3_audio_freeze.md) section 4. What was
decided is in [`decisions.md`](decisions.md) under 2026-10-04. Each session's
working notes are in [`../research/masking/`](../research/masking/README.md).

## What it is

Take a video with its own audio, keep everything outside one person, and
replace that person from a reference still. The shipped graph is
`workflows/h3_video_to_video_masked_song_pdd8_api.json`: a video loader, the
SAM 3 checkpoint, two pack nodes, and the PDD8 song chain with Sol-Attn.

It is not a trained task. The release trains t2va, fl2va and ref2va; a
spatial mask on a base checkpoint is an inference-time method. The mechanism
is core's: a latent noise mask on H3 is a per-token timestep
(`comfy/ldm/minimax/model.py::mask_row_values`), and the clean latent is put
back into the kept tokens every step.

## The pieces, in the order a render meets them

### 1. `MiniMaxH3SubjectTrack` (`subject_track.py`): which pixels are the person

One mask per frame, empty where the person is not on screen. The module
docstring is the authority and lists the steps; `follow` is the function.

- **Cuts.** `cut_scores` is one minus the correlation of consecutive frames'
  gradient maps at `CUT_SIZE`: a cut moves the edges, a lighting change does
  not. The threshold is the user's value, or `auto_cuts`: the middle of the
  widest gap in the clip's own scores above `CUT_FLOOR`.
- **People.** Each shot is looked at `PROBE_OFFSET` frames in. Core's
  detector is asked for `subject_phrase` with a count (`counted`), because
  core returns one detection per phrase without one.
- **The pick.** `pick` is a rule (largest, most central, best match for the
  phrase). On a named frame it is applied there. Left automatic,
  `main_subject` takes the person the rule favours for most of the clip's
  frames.
- **The match.** A person in another shot is compared with the subject in
  two places, and the lower counts: the trunk's features under the top third
  of their mask (`top_third`, `signature`), and under the head SAM 3 finds
  for `head_phrase` inside their mask (`head_of`). When the pick frame shows
  other people, their average is subtracted first (`relative`). A person
  with no head is no match. The line is the user's value or `auto_match`.
- **A clip with one person.** With nobody else on the pick frame and the
  match automatic, a shot under the line is probed to its end and its best
  frame showing one person with a head is taken, whatever it scores.
- **Tracking.** Each shot is tracked forward and backward from its seed with
  core's `SAM3_VideoTrack` on its `initial_mask` path, no text prompt.
- **What it shows.** One labelled tile per shot and a text report: who was
  taken, who was the best candidate where nobody was, every score with the
  line marked, every phrase and threshold used.

Why not core's tracker with a text prompt: across a cut it starts new
objects, it has an object cap that detection stops at for good, and a
confident detection overwrites the tracked mask. The module docstring cites
where in core.

### 2. `MiniMaxH3MaskedSource` (`video_mask.py`): what happens to those pixels

- **The region.** The mask is grown by `grow_pixels` and turned into a token
  mask (`grow`, `token_mask`). A token is regenerated or kept whole, in
  space and in time: the video VAE packs frames in runs (`run_lengths`), so
  a token covers several frames.
- **`replace`.** `whole subject`, or `head and hair`, which keeps the body's
  pixels and finds the part with SAM 3 from `part_phrases`.
- **What is encoded.** The source frames themselves; with `paint_out`, a copy
  with the subject filled in from its surroundings (`fill_subject`).
- **The composite.** After the decode the source's pixels come back outside
  the region. `only what changed` (`changed_alpha`) keeps the render where
  it differs from the source or where the old subject stood, and restores
  the source in the rest of the margin, so the margin can be generous.
- **The kept mask.** `mask_store.py` keeps the finished mask on disk, keyed
  on the upstream graph, a fingerprint of the frames, the input files'
  stats and every upstream node's `MASK_VERSION` (`mask_key`). On a hit the
  Masked Source never asks
  for its `mask` input, so the tracker does not run. `MASK_KEY_SKIP` names
  the settings that do not change the mask.

### 3. The song node (`audio_freeze_song.py`): the render

Each window starts from the source's frames over its span
(`video_mask.window`), regenerates the masked tokens with the track's audio
frozen, and composites. A window whose mask is empty is written from the
source without sampling. One sampler per window is what carries the mask.

## Known limits

Each line names where the evidence is. "Seen" means on a render or a tile.

**Finding the person**

- **It is not an identity model.** SAM 3's trunk says what a thing is, not
  who. People who look alike score alike: on the car clip one shot of the
  lead scores the same as another woman.
  `../../bench/results/2026-10-04_subject_track_three_clips.md`.
- **The rule was chosen on the three clips it passes.** No clip it has not
  seen has been tried. Same record.
- **A miss leaves the original in the shot; a wrong take replaces somebody
  else.** There is no way to correct one shot by hand short of naming a
  frame or a value for the whole clip.
- **Seen from behind the subject is not found.** A shot in which they never
  face the camera is left alone. `subject_track.py`, "How far the matching
  can be trusted".
- **A change of framing lowers the score of the same person.** The report
  says when a shot is framed much closer or wider than the pick. A crop of
  the head run through the trunk again holds across the one-person clip and
  fails on the car clip; it is not in the node. Same record.
- **In a one-person clip a cutaway to a different lone person is taken.**
  Naming a value for `match` turns that rule off.
- **Cuts.** A dissolve or a jump cut inside a cutaway can score under the
  threshold. The report prints the highest steps with the threshold marked.
- **Cost.** Two detector passes on every frame looked at, and a tracker pass
  per shot. The kept mask removes it from every later run.

**Replacing the person**

- **Movement is not carried.** On the generic prompt the new subject faces
  the camera when the original turns. Starting the sampler part-way into its
  schedule carries the turn and the original's hair and clothes with it;
  doing so from a softened copy of the original turns him and still changes
  his clothes. Neither is a fix.
  `../../bench/results/2026-10-04_masked_v2v_band.md`, "The turn without a
  prompt", and `../../bench/results/2026-10-04_masked_v2v_turn_soft_arms.md`.
- **The hole decides size and place.** The reference still carries no height
  or build. A tight margin keeps the new subject in the original's
  footprint; a wide one lets a differently shaped subject fit and lets him
  stand somewhere else. `../h3_audio_freeze.md` section 4.
- **`head and hair` gives an oversized head on a long-haired original.**
  Kept as an option, not a recommendation. Same section.
- **`paint_out` was judged worse** and stays off. Same section.
- **The mask is the body, not what the body does to the room.** The
  original's shadow and reflections stay.
- **Two-sampler graphs do not carry the mask as wired.**
  `MiniMaxH3RestorePlate` (`plate_restore.py`) exists for it and is in no
  shipped graph.
- **No control adapter.** The owner ruled it out for this lane
  (`decisions.md`, 2026-10-04).
- **The first clip's flicker was set aside**, its dark lines being seams in
  the source's own backdrop. One part of it has no explanation yet.
  `../../bench/results/2026-10-04_masked_v2v_first_run.md` and
  `../research/masking/2026-10-04_mrhf.md`, section 4.

**Using it**

- **Inputs marked advanced are not folded away** in the owner's frontend, so
  both nodes show every input.
- **Core's SAM 3 checkpoint does not load in a plain process outside the
  server**: the text projection's shape is refused (seen by two sessions).
  mrblue's probes leave that one key out of the load in their own process,
  the projection not being used for the conditioning; the scripts are under
  `internal/claude/`, which is not tracked, so a new probe has to repeat it.

## Directions

[`next_steps.md`](next_steps.md) holds the list that is acted on. This is the
wider set, with what each would buy and what is known about it.

**Telling people apart**

- **A model trained for identity**, used where SAM's features are used now:
  on the head, at the seed frames only. It is the real answer to people who
  look alike and to a change of framing. None is wired; a licence check
  comes first.
- **Correcting one shot by hand**: take or leave a shot, or click the person,
  from the preview. It needs a frontend widget, which nothing in this node
  has so far.
- **A fourth clip**, chosen before the rule is looked at again.
- **Upstream SAM 3's caller-mask request** (`coderef/sam3`) would replace the
  per-shot seeding with the tracker's own. Not served by the multiplex
  checkpoint and not in core's port.

**The turn, and movement in general**

- What the owner has left to choose: a clause narrowed to facing, which is
  prompt text; or accepting different clothes on the new subject.
- Untried on the late start: a start between the schedule's first two knots,
  more seeds, another clip. The record lists them.
- **The scene's clothes on the new subject**: a crop of the original's torso
  as a second reference. Untried.

**What the body does to the room**

- **Removing the original's shadow and reflections.** Core ships VOID
  (`comfy_extras/nodes_void.py`); it needs the affected area marked and is a
  second model pass. Read, not run on this clip.
  `../research/masking/2026-10-04_mrhf.md`, "Effects outside the silhouette"
  and "VOID for the plate with nobody in it".
- **A soft edge instead of a hard one**: a matting model on the subject.
  The nodes for it exist (`sapiens2_parts.py`, `MiniMaxH3SubjectParts`) and
  are in no shipped graph. What was read of the model:
  `../research/masking/2026-10-04_mrhf.md`, "Matting".

**The render**

- The mask through two samplers, with the plate restored between them.
- The kept mask across a server restart, which no run has exercised.

## Where to look

| question | where |
|---|---|
| what a node input does | the node's `define_schema` tooltip |
| why a default is what it is | the comment beside the constant, and `../../workflows/h3_config.py` (`SUBJECT_TRACK`, `MASKED_SOURCE`) |
| what would go red | `bench/check_subject_track.py`, `bench/check_video_mask.py`, `bench/check_mask_store.py`, `bench/check_plate_restore.py`; [`../checks.md`](../checks.md) |
| what the owner said of a render | `../../bench/results/2026-10-04_masked_v2v_first_run.md`, `../../bench/results/2026-10-04_masked_v2v_band.md` |
| how the prompts are written | `../../prompt_bank/` (`ref2va_masked_*`), [`../prompting.md`](../prompting.md) |
| what was measured on which clip | `../../bench/results/2026-10-04_subject_track_three_clips.md` |
