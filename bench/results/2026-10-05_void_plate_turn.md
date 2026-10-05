# A clean plate from VOID on the band clip's turn window: a first probe (2026-10-05)

lane: masking
verdict: rendered, not judged; the lead is removed in both shots, and the turn shot's plate is not clean under either prompt

Session mrorange, the board's `build-void-plate`. **Status: rendered, not
judged.** What follows was read from still frames at VOID's size, three per
shot and a ten-frame sheet; playback is the owner's to judge. One clip, one
window, one seed, two prompts.

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

**Shot 1, 189 frames.** The lead is gone in both arms. The wall, the pictures
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

## Time

Sampling took about four minutes for the 189-frame window and about forty
seconds for the 45-frame one, the same in both arms, on the 4090
(`per_node_s`). A whole arm, with its loads and decodes, was between five and
six minutes.

## What this does not say

- Whether the plate is good enough under the hole in shot 1: nobody has
  watched it play, and the hole is about a fifth of the frame.
- Anything about the shadow or reflections. The lead casts no clear shadow in
  these shots at this size; the affected ring was regenerated, and what it
  removed was not looked for.
- Whether the turn shot fails because it is short, because the subject is
  close and moving fast, or because of the prompt. The paper names close
  subjects as a weak spot. Untried: a sentence that describes the room and
  no people; a window that carries the turn shot together with more context;
  pass 2; a wider or a tighter affected region; a second seed.
- Whether a window that runs past its cut harms the shot's last frames. The
  sheet's last two columns for each window are the shot's last frames and
  show nothing the earlier ones do not; that is five frames, by eye.
- Anything at the lane's size. The plate is 672x384 and would be enlarged
  twofold to sit under a hole at 1344 wide.
