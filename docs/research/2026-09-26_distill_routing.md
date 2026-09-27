# Routing distills within a clip: six routes, three of them H3's own

last updated: 2026-09-26

> **Provenance.** Written by the VAE session on the owner's request ("flesh
> out" routes 1 and 2 of the idea in [`../wiki/next_steps.md`](../wiki/next_steps.md),
> "Idea saved for later, 2026-09-26", and "something unique to h3" beyond
> position or step). Everything below is read from code on 2026-09-26.
> Nothing was rendered or built. Route 3's design is in that entry, agreed
> with the fastdude session, and is not repeated here.

**The premise.** The owner's one-seed looks: PDD8 is strong on close-ups,
medium shots and low motion (detail, colour), and weak on large motion.
FlashGen and FastH3 hold motion better
(`../../bench/results/2026-09-26_distill_compare_s1.md`, `../h3_distills.md`).
So the aim is to put each distill where it is strong within one clip. H3
generates every frame of a clip in one joint pass, which rules out the
obvious per-shot switch.

## What H3 offers that most video models do not

Three facts, read from core, make routes 4 to 6 possible:

1. **Per-row noise levels in one forward.** A denoise mask sets each row's
   timestep separately. A row is one 2x2 patch of one latent frame. Mask
   value m puts the row at `m * sigma` of its stream, and a row at 0 is
   pinned at the conditioning timestep and returned bit-identical by the
   sampler's final blend (`comfy/ldm/minimax/model.py`, `_forward` and
   `mask_row_values`; `comfy/model_base.py::MiniMaxH3.scale_latent_inpaint`).
   The pack's audio-refine pass is built on it (`audio_refine.py`).
2. **Two streams with separate masks.** Video and audio are one packed
   sequence, but each takes its own denoise mask (`denoise_mask`,
   `audio_denoise_mask`), and audio runs on its own shift, derived from the
   same base time (`time_shift_sigma`).
3. **A step-indexed output head bank.** PDD is a backbone LoRA plus 32 output
   heads, and the final layer fuses the heads each step spans
   (`FinalLayer.forward`, `_pdd_head`; `../h3_pdd.md`). FlashGen and FastH3 are
   backbone deltas only.

## Route 1: render each shot on its distill and join them

**Mechanism.** The pack already builds a two-window seam, and the second
window takes the first's sampled latent as a frozen prefix (`freeze_windows`
in `build_api`, `audio_freeze.py::MiniMaxH3FreezeAudioWindow`;
[`../h3_audio_freeze.md`](../h3_audio_freeze.md) section 4 step 6). Route 1 is
that seam with a different model chain per window: window 1 on, say, PDD8
and its schedule, and window 2 on FlashGen and its own. Each window runs its
distill on its own trained schedule, and that is this route's advantage.

**What it needs, beyond what exists:**
- **A model chain per window.** `build_api` builds one chain for both
  windows today.
- **Per-shot prompts.** A multi-shot prompt becomes one prompt per window,
  each timed to its shot ([`../prompting.md`](../prompting.md) §5.10). The
  model no longer plans the cut; each window is its own scene.
- **The join.** Three designs are compared in
  [`2026-09-25_continuation_guide_rows.md`](2026-09-25_continuation_guide_rows.md):
  our masked prefix (arm A), guide rows (arm B, #32), or the last frame into
  i2va, which loses the audio tail and the motion state. At a hard cut the
  prefix carries scene identity and audio continuity, and a cut needs no
  visual continuity. Mid-shot, the seam is the whole risk.
- **Audio across the cut.** Ours carries the prefix in video only, with the
  audio locked to a known track. With generated dialogue, window 2's audio
  is a fresh sample, so it needs the audio tail as guide rows (design B) or
  a locked track.

**Costs.** One render per window, with prefix frames re-attended in each
window after the first. Attention grows with the square of the sequence, so
two half-length windows can cost less than one full clip, before the
overlap. Not priced here; `bench/preflight_graph.py` prices a graph
statically.

**Risk.** The seam where the cut is not a cut, and a prompt split that the
model would have staged differently as one scene.

## Route 2: mask an adapter to some rows, within one pass

**Mechanism.** `lora_branch.py` adds `s * B(A x)` at the call for every row
of the packed sequence. A per-row scale is one multiply: build it from each
row's latent frame. Latent frames map to video frames in the VAE's
`1, 4, 4, 4, 4` pattern per 17 frames (`FRAME_PER_TOKEN`), so a mask aligns
to latent frames, not video frames. Video rows sit after text and references
in the packed layout (`PackedLayout`), with one row per 2x2 patch.

**Hard constraints, in order of severity:**
1. **One schedule for every row.** All rows share the sampler's sigmas, so
   whichever distill's schedule runs, the masked-in other distill runs off
   its trained trajectory. FlashGen is four steps and PDD8 eight. This is
   worse than the attention leak and was not written down before.
2. **PDD is not only a LoRA.** Its output heads are chosen per step for the
   whole final layer. Per-row heads would mean running both head sets and
   blending per row in the final layer. The pack already wraps it
   (`pdd_lora.py::_make_final_layer_forward`), but that is untrained. In
   practice only FlashGen or FastH3 can be masked in, over a PDD run.
3. **Attention mixes the adapters.** Every row attends to every other, so
   adapter-A rows feed adapter-B rows within each forward. This is the
   owner's concern, and a low strength or a ramped mask at the boundary only
   softens it.
4. **Audio rows** span the whole clip. They can follow the frame mask by time,
   or stay on one adapter; either is a choice with no evidence behind it.

**Verdict.** This is the least trained route, and route 4 does the same job
without constraints 1 and 3.

## Route 4 (H3's own): route by noise level per row, not by adapter

**The idea.** Render the whole clip on the motion distill. Then run a short
second pass on PDD that reopens only the rows that should be PDD's, at a
partial sigma, with every other row frozen at mask 0. Each forward runs one
adapter on one schedule. What differs between rows is their noise level,
which H3 supports natively (fact 1 above). Frozen rows are exact conditioning
and are returned bit-identical. This is the audio-refine pass with a
per-frame or per-region video mask in place of a per-stream constant.

**What decides the mask, H3-specifically.** The owner's observation is that
PDD fails on large inter-frame change, and an older finding agrees that PDD's
artifact severity tracks inter-frame delta (`bench/measure_clip_delta.py`,
docstring). Pass 1's latent measures that directly: per-latent-frame delta,
with no decode. Low-delta latent frames get reopened for PDD's detail, and
high-delta frames keep FlashGen's motion. The mask can be soft, with m in
(0, 1) putting a row at a fraction of the pass's sigma, so a boundary can
ramp. Shot changes in a multi-shot render are cuts, where the delta spikes,
so a delta-driven mask tends to change value exactly where continuity is not
needed. The mask is spatial as well as temporal (`[T, H, W]`), so a region,
such as a face in a close-up, is expressible too.

**How far to reopen.** The pass starts at a PDD8 knot so its heads are on
grid:
- 0.632 → 0: one evaluation, PDD's last step, the most conservative;
- 0.8 → 0.632 → 0: two evaluations, which let PDD move more.

Those are knots 24 and 28 of 32 (`pdd_math.schedule_knots`). A higher start
hands PDD more of the composition, and with it the motion PDD is bad at.

**What it needs.**
- A mask writer for the video stream that is per latent frame or per row,
  computed from pass 1's latent. `MiniMaxH3AudioRefineMask` writes per-stream
  constants only.
- Two model chains, as in route 3.
- **Time the model switch** under `H3_TELEMETRY`.

**Risks.** A reopened frame beside a frozen one may flicker at the boundary
when it is not a cut. The sampler adds noise to reopened rows at the start
sigma, so the pass is also a re-sample of those rows, not a pure polish. How
much of pass 1's motion a start at 0.8 keeps is the first thing to measure.

## Route 5 (H3's own): route by stream

**The idea.** Video from one distill and audio from another. The joint model
takes separate masks per stream (fact 2), and the pack already runs the
video-frozen, audio-reopened pass. `h3_probe_t2v_pdd8_audio_refine` and
`h3_probe_t2v_flashgen_4step_audio_refine` re-sample the audio on the
undistilled model with the distill's video frozen. The motivation is PDD's
own: its audio loses energy at coarse partitions
(`pdd/audio_under_pdd.md`). Generalised, the audio pass can take any chain,
FlashGen's included, and the inverse (keep a distill's audio, reopen the
video) is the same node with the masks swapped.

**What exists and what is owed.** The graphs exist. The owner listened only
to cached against uncached refine
(`../../bench/results/2026-09-25_frozen_cache_s1.md`). No owner verdict on
refine against no refine was found in the records. That listen is the
cheapest thing on this page.

## Route 6 (H3's own): route by component

**The idea.** PDD's contribution splits into a backbone LoRA and a
step-indexed head bank (fact 3). H3's final layer makes a hybrid
expressible: FlashGen's backbone through `MiniMaxH3LoRABranch`, with PDD's
heads selected by the schedule.

**Why it is last.** The heads were distilled against PDD's own backbone
features, so pairing them with another backbone is off-distribution. The
repo's evidence on the head half so far concerns keeping it with PDD's own
backbone ([`../wiki/next_steps.md`](../wiki/next_steps.md), "Does our PDD node
keep its head half?"). Worth one probe only if routes 3 to 5 leave colour
and detail unexplained.

## Order, cheapest evidence first

| route | built? | first step | trained regime per forward |
|---|---|---|---|
| 5 stream | graphs exist | owner listen: refine against no refine | yes |
| 3 step | new arm | one render under `H3_TELEMETRY` | yes, apart from the handoff |
| 4 noise per row | new mask node + arm | one render, a start at 0.632 then 0.8 | yes; frozen rows are exact |
| 1 per shot | new per-window chain | two windows at a hard cut | yes; the seam is the risk |
| 6 component | new arm | one probe, only if 3 to 5 leave a gap | no |
| 2 masked adapter | new branch mask | not recommended | no |

The shared prerequisite is the second seed of the distill comparison, which
fastdude has queued. None of these routes is worth building until the
pattern it serves holds on more than one seed.
