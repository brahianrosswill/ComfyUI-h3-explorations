# A clean plate from VOID on the band clip's turn window: a first probe (2026-10-05)

lane: masking
verdict: judged by the owner, all four clips: neither arm is a plate on either shot; the empty prompt is the less bad, "still really bad"

Session mrorange, the board's `build-void-plate`. **Status: judged by the
owner, all four clips (below).** What is described under "What the frames
show" was read by me from still frames at VOID's size, three per shot and a
ten-frame sheet, before the judgment; **where it and the owner's judgment
differ, the judgment is right**, and for shot 1 they differ. One clip, one
window, one seed, two prompts.

## Judged by the owner, 2026-10-05

On playback, in the owner's words, as relayed by mryellow and as the board's
`build-void-plate` card quotes them:

- the turn shot, empty prompt: "some ghosting at the start and end and
  shadows are still there but gone... but you can see glitchy at the very
  last frames";
- the turn shot, scene prompt: "looks horrible. deformed / morphing
  faces/bodies everywhere";
- the whole window, scene prompt: "has people disappearing and appearing and
  morphing";
- the whole window, empty prompt: "does too but not as much but... still
  really bad".

corrected: my reading of shot 1 below, from three stills, was that the plate
looks plausible there. On playback it is not: people disappear, appear and
morph over time, which a still cannot show. Stills were the wrong instrument
for a plate that has to hold over time.

mryellow, from contact sheets of every sixth frame, agrees, and adds for the
empty arm: a flat translucent panel exactly the size of the subject's cell
region with the neon tube drawn across it, the man behind him smeared beside
the woman on the right, grey streaks in the last frames, and **the cast
shadow outside the one-cell grey ring**. So the affected region did not hold
the shadow: one cell beyond the cells he touches covers the lane's hole and
not what he casts on the wall. The ring's width is `GRID_GROW` in the probe.

The numbers are in `2026-10-05_void_plate_turn.json`. The script and its
graph are `internal/claude/2026-10-05_mrorange/void_plate_turn.py`, not
tracked: the graph was built by hand from core's nodes and is in no
generator.

## For the owner to look at

On the output share under `Video/mrorange/`, each the source, the plate and
the quadmask side by side at 672x384:

- `void_plate_turn_empty_turnshot.mp4` and
  `void_plate_turn_scene_turnshot.mp4`: the shot with the turn in it, frames
  237 to 279 of the window.
- `void_plate_turn_empty.mp4` and `void_plate_turn_scene.mp4`: the whole
  window. The plate is VOID's where the lead is on screen and the source
  elsewhere.

## What ran

The window `turn_control` loads: `thinkaboutthings_compressed.mp4` from 112 s,
345 frames at 24 fps, loaded at width 1344.

- **Who to remove**: `MiniMaxH3SubjectTrack`, everything automatic. Four
  shots, cuts at 186, 237 and 280. The lead is taken in shot 1 (frames 0 to
  185) and shot 3 (237 to 279, where the group turns its back); shots 2 and 4
  are cutaways without him. The node's report is in the record.
- **The model**: core's VOID nodes, pass 1 only, at the settings of core's own
  template (`utility_void_video_inpainting.json`): 672x384, cfg 6, 30
  `simple` steps, `VOIDSampler`, the T5 encoder and the CogVideoX VAE. Pass 2
  and the optical-flow file were left out
  (`docs/research/masking/2026-10-04_mrhf.md`, "VOID for the plate"). The
  checkpoint is upstream's, converted:
  `2026-10-05_void_checkpoint_conversion.md`.
- **The quadmask**, built from core's mask nodes at VOID's size so its levels
  are exact. The subject's mask, any pixel it touches, grown by 6 pixels, is
  the object to remove. The 32-pixel grid cells the subject touches, grown by
  one cell, are the affected region. One cell at this size is the lane's
  `grow_pixels` at 1344 wide (`h3_config.MASKED_SOURCE`), so the affected
  region covers the hole the lane cuts. Core's template wires a plain mask
  straight into `VOIDInpaintConditioning`, so 1.0 there is "remove"; the
  affected level is the one `VOIDQuadmaskPreprocess` produces.
- **The windows**: one per shot the lead is in. VOID takes lengths of 8k + 5
  frames (`comfy_extras/nodes_void.py::_valid_void_length`), so each window is
  the shot rounded up and runs past its cut: frames 0 to 188, **dropping 186,
  187 and 188**, and frames 237 to 281, **dropping 280 and 281**. Neither shot
  needed splitting.
- **Two arms on seed 43**: an empty prompt, and one sentence describing what
  remains: "A small group of young people in matching green sweatshirts stand
  and dance in a living room with framed pictures on the wall and table
  lamps, under coloured light."

## What the frames show

**Shot 1, 189 frames** (stills; see the correction above). The lead is gone in both arms. The wall, the pictures
and the neon tube behind him are filled in, and the people he stood in front
of are completed where he covered them. The two arms give nearly the same
plate. Around frame 170 both arms appear to put a second copy of the woman
with glasses where he stood, and the man who stood behind him is not there.
If that reading holds on playback, the plate under the hole holds an invented
person in that part of the shot.

**The turn shot, 45 frames.** Not clean in either arm.

- Empty prompt: where he stood there is a tall, door-like panel on the wall
  that the room does not have, the neon tube is drawn through it, and there
  are smeared dark figures beside the woman on the right.
- The scene sentence: two new people and a brown sofa are painted into the
  hole.

So on this shot the prompt decides what fills the hole, and a sentence that
mentions people puts people there. On shot 1 it barely matters. In the
record, `arms_compared`: the two arms differ inside the quadmask's region
about two and a half times as much on the turn shot as on shot 1.

**The quadmask** is as designed on the frames looked at: his silhouette, and
a blocky ring one cell wide. A stray speck of tracker mask at frame 237
became a small square of "remove".

## Three things about core's VOID nodes, found on the way

1. **The output is three frames longer than the input, and shifted.** Each
   window came back with three frames more than it was given. Matching the
   kept pixels of the source against the output at each offset puts input
   frame j at output frame j + 3 on both windows (`output_offset` in the
   record). The assembly here applies that. Core's template does not, so its
   pass 1 clip is three frames late against its source.
2. **The kept area is not the source's pixels.** VOID decodes the whole frame,
   and outside the quadmask its output differs from the source by a few
   levels (`arms_compared`). A plate has to be composited under the hole; it
   cannot replace the frame.
3. **Core cannot load the upstream checkpoint.** See the conversion record.

## The quadmask's polarity, read

Asked by mryellow after the judgment: a flat panel in the shape of the
subject's region is what an inverted or self-repainting mask would give. Read
in core and in upstream's own inference code (`netflix/void-model`, `main` at
e3914f8f551d, fetched 2026-10-05; copies under
`internal/claude/2026-10-05_mrorange/upstream_void/`). **The polarity is
right.**

- Upstream (`get_video_mask_input`, in its own utils module): the quadmask
  on disk is 0 on the object, 63 on the overlap, 127 on the affected area and
  255 where the frame is kept. It is quantised to those four values and then
  inverted, `255 - mask`, and divided by 255. So the mask the pipeline
  receives is 1.0 on the object, about 0.5 on the affected area and 0.0 on
  what is kept.
- Core's `VOIDQuadmaskPreprocess` is the same quantisation and inversion, and
  its docstring says 1.0 is "remove". Core's template skips that node and
  wires a plain mask, 1 on the subject, into the conditioning node.
- The probe's quadmask is 1.0 on the object, 0.5 on the ring and 0.0
  elsewhere: the same convention.
- Both then encode `1 - mask` with the VAE as the mask channels (core:
  `VOIDInpaintConditioning`, `vae.encode(inverted_mask_3ch)`; upstream:
  `1 - mask_condition_tile` with `use_vae_mask`), which is the paper's
  picture: black object, grey affected area, white kept.
- seen, and independent of the reading: in shot 1, on the same nodes, the lead
  is gone and the wall behind him is drawn. An inverted mask would have kept
  him.

## Where core's port differs from upstream's own inference

Found by the same reading. None of this was tested; each is a candidate for
the panel, and the first is the one that fits mryellow's description.

1. **What the model is shown under the mask.** Upstream's shipped config sets
   `zero_out_mask_region = False`, and the pipeline then conditions on the
   whole, unmasked video (`masked_video = init_video`). Core's node always
   multiplies the video by `1 - mask`: the object is blacked out and **the
   affected ring is shown at half brightness**. Even upstream's other path
   (`zero_out_mask_region = True`) blacks only the object and leaves the
   affected area as it is. So in core the ring is a dimmed copy of the source,
   which neither upstream path produces.
2. **Guidance.** Upstream's config has `guidance_scale = 1.0`, with a negative
   prompt about quality; core's template uses cfg 6 with an empty negative.
   The probe used the template's. Both use 30 steps (upstream's script passes
   30 and ignores its config's 50).
3. **Windows.** Upstream's script calls the pipeline with `num_frames` set to
   its `temporal_window_size`, 85, and a multidiffusion stride in the config.
   Core's node samples the whole window at once; the probe gave it 189 and 45
   frames. How upstream stitches its windows was not read.
4. Upstream quantises the mask to three levels inside the pipeline whatever
   the config says (`use_trimask = True` is passed literally), so its overlap
   level becomes "object". The probe has no overlap level, so nothing follows
   for it.

An arm at cfg 1 needs only a changed number in the probe's graph. Showing the
model the unmasked video needs a conditioning node that does not multiply,
which core's does not offer; nothing in this pack patches core, so that would
be a node here.

## Time

Sampling took about four minutes for the 189-frame window and about forty
seconds for the 45-frame one, the same in both arms, on the 4090
(`per_node_s`). A whole arm, with its loads and decodes, was between five and
six minutes.

## What this does not say

- Whether the plate is good enough under the hole in shot 1: nobody has
  watched it play, and the hole is about a fifth of the frame.
- Anything about removing a shadow. corrected: this line first said the lead
  casts no clear shadow at this size. He does in the turn shot, on the wall,
  and it lies outside the ring (the owner's judgment above), so the probe
  never asked VOID to remove it.
- Whether the turn shot fails because it is short, because the subject is
  close and moving fast, because of the prompt, or because of how core's
  node conditions (the section above). The paper names close subjects as a
  weak spot. Untried: upstream's guidance; a ring wide enough to hold the
  shadow; a sentence that describes the room and no people; a window that
  carries the turn shot together with more context; pass 2; a second seed.
- Whether a window that runs past its cut harms the shot's last frames. The
  sheet's last two columns for each window are the shot's last frames and
  show nothing the earlier ones do not; that is five frames, by eye.
- Anything at the lane's size. The plate is 672x384 and would be enlarged
  twofold to sit under a hole at 1344 wide.
