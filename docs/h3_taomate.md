# TaoMate-H3 in ComfyUI

Opened 2026-09-15. What the TaoMate-H3 adapter is on this box, how a graph
runs it, how our conversion differs from kijai's, and the test plan, including
the audio-freeze arm. What the authors' runtime is, and what their release is
not evidence of, is [`wiki/references.md`](wiki/references.md), "The streaming
references: TaoMate". The probe and freeze arms rendered on 2026-09-15; their
runner rows are `bench/results/2026-09-15_taomate_probe_arms.jsonl` and
`bench/results/2026-09-15_taomate_audio_freeze_arms.jsonl`. **The owner's
verdict is `bench/results/2026-09-15_taomate_verdicts.json`, and the lane is
parked (section 6).**

---

## 1. The file

- **Base.** The adapter was trained on the FL2VA partition of
  `MiniMaxAI/MiniMax-H3`, the bf16 release: the directory the authors'
  `--model-root` has to contain. It loads here on
  `h3_config.MODELS["unet_fl2va"]`, the pruned int8_convrot build of that
  partition. Pruning touches only the adaln time embedder
  ([`pdd_artifacts.md`](pdd_artifacts.md)), and the adapter does not target
  it. It must not load on `MODELS["unet_fl2va_pdd8_baked"]`. That backbone
  already carries PDD's delta, so the stack would be two distills and neither
  was trained on top of the other. `bench/check_distill_settings.py` fails a
  graph that tries.
- **Conversion.** `bench/convert_taomate_lora.py` renames the authors' factors
  and keeps their rank. Its docstring gives the evidence for each part of the
  mapping. With `--release` it also checks the ComfyUI checkpoint's qkv and
  fc1 row for row against the bf16 release TaoMate trained on. That shows the
  layout on the weights themselves, not only by reading source. The records
  are
  `bench/results/2026-09-15_taomate_lora_conversion.json` (the file, plus the
  comparison with kijai's) and
  `bench/results/2026-09-15_taomate_lora_control_fc1_swapped.json` (the
  control).
- **The contract** is `taomate_streaming.py` at the repo root, the one module
  the sampler node, the generator, the converter and the checks all read; the
  LoRA filenames are `workflows/h3_config.py`'s `TAOMATE_*_LORA`:
  - `STRENGTH`
  - the grid: `STATE_INDICES` over `GRID_POINTS` at `SHIFT_VIDEO` and `SHIFT_AUDIO`
  - `SAMPLER`
  - `UPSTREAM`, the upstream revision every value was read at

  The converter writes the same contract into the output file's metadata as
  `distilled_grid`. Converting and building graphs need no sister checkout.
  `bench/check_distill_grid.py` grades each graph's sigma vector against the
  grid. It also checks that core's derived audio clock lands on the adapter's
  audio list.

## 2. Ours and kijai's

Both files come from the same adapter and use the same qkv and SwiGLU layout.
In the record, the grouped-qkv and swapped-halves hypotheses score far below
the direct comparison. They differ in four ways:

1. **Rank.** Ours keeps the trained rank in every module. Kijai's is a
   per-module truncated SVD, so its rank varies by module. Per module and per
   kind, the record's `comparison` gives the cosine, the energy kept and the
   error against ours. The two deltas point the same way but are not equal.
2. **Provenance.** Ours carries in its metadata:
   - the source hash, rank and alpha
   - each layout decision and the evidence for it
   - the sampling contract

   The converter reproduces the file from the authors' download. Kijai's has
   no metadata and no published recipe.
3. **What reaches the weights on this checkpoint.** On an int8_convrot base
   under dynamic VRAM, a LoRA is not added once. Each cast dequantises the
   weight, adds the delta and requantises with stochastic rounding
   ([`research/merge_requantisation.md`](research/merge_requantisation.md)
   section 7). The noise that adds depends on how large the delta is against
   the quantisation step. So on the weights actually used, the two deltas can
   end up further apart than the gap between them, or closer.
   `bench/measure_merge_noise.py --only taomate` measures both on stored
   weights, CPU only. Its record is
   `bench/results/2026-09-15_merge_noise_taomate.json`; read it beside the
   conversion record's per-module `rel_err`. **The result, 2026-09-15:** in
   every module, and for both files, the noise one cast adds is larger than
   the whole gap between ours and kijai's. The two files carry the same
   noise. So on this checkpoint the resize's loss is below what each load
   perturbs anyway, and ours' fidelity advantage is unlikely to survive into
   the weights a render uses. As the record's own caveat says, this is a
   statement about stored weights. PDD carries the same noise and renders
   well.
4. **Cost.** Ours is larger on disk and in each cast, since every module keeps
   full rank (`ls -l` the two files).

Nobody knows yet which renders better, and one matched pair will not settle
it. A distance between deltas is a statement about weights, not about the
picture.

## 3. What a ComfyUI graph changes from the authors' runtime

The graphs run the adapter outside the regime it was distilled in. That is
deliberate: it is the only way this box can run it. Each difference below is a
possible cause of a bad render, to rule out before blaming the adapter.

- **Whole clip, no cache.** The authors' runtime denoises causal chunks. Each
  chunk attends to a clean K/V cache of earlier chunks, with a sink. Core runs
  one unmasked attention over the whole packed sequence. The adapter was
  trained the first way.
- **Who owns the audio.** Upstream overwrites the audio rows after every step
  with a pass of the base model without the adapter, so the adapter never owns
  audio. In the plain probe graph it does. The freeze graph holds a known
  track instead, which is closer to upstream.
- **Attention.** Upstream is FlashAttention-3, dense, in bf16. The graphs
  carry sage and Sol, as every shipped video graph does
  (`bench/check_attention_defaults.py`).
- **Weights.**
  - Upstream quantises the interior blocks' qkv and fc1 linears to W8A8 and
    keeps blocks 0, 1, 47, 48 and 49 in bf16
    (`src/taomate_h3/inference/w8a8.py` at `taomate_streaming.UPSTREAM`).
  - This box's checkpoint is int8_convrot in every block.
  - The adapter carries no norm tensors, so it inherits the block-49 attention
    sensitivity every LoRA here has
    ([`h3_block49_quant_error.md`](h3_block49_quant_error.md)).
- **Renormalisation.** Upstream matches each chunk's video to the first
  chunk's per-channel statistics. Nothing in a graph does.

## 4. The test plan

Rigour here is proportional, per CLAUDE.md's "A tinkering repo": the owner
judges matched pairs in free text, and nothing here is a published finding.
Each arms file's `run` field holds its command.

0. **One throwaway render.** Run `workflows/h3_probe_taomate_3step_api.json`
   once on its default prompt and read it end to end:
   - the log shows no skipped LoRA keys
   - the sampler runs three steps
   - the clip is a picture at all

   Denoising the whole clip at once, outside the chunked regime, could fail
   outright. This render finds out before any arm is spent.
1. **Probe arms, `bench/taomate_probe_arms.json`.** Three unrelated scenes,
   one matched seed each, and three arms per scene:
   - ours
   - kijai's
   - the PDD8 candidate, the chain the owner already renders with

   The swapped-fc1 control ran on one scene. It was written expecting a
   broken clip, and that expectation was wrong: its render is coherent
   (2026-09-15). fc1's delta is too small against the base weight for
   misplaced rows to break a working model, so the control cannot test the
   mapping. `--release` tests it, on weights. Do not spend another slot on it.
2. **Freeze arms, `bench/taomate_audio_freeze_arms.json`.** This is
   [`h3_audio_freeze.md`](h3_audio_freeze.md) section 5, idea 9. It uses that
   lane's dancer and voice scenes, seeds and tracks: TaoMate with the track
   frozen, beside the PDD8 freeze candidate. The dancer also gets a loose mask
   (idea 2). Rows a little below clean sit nearer the noisy teacher audio the
   adapter trained beside than a fully clean track does.
3. **The trajectory arm, not built.** Upstream's audio rows are the base
   model's intermediate states, noisy at each step's audio sigma. Core cannot
   express that with a mask. `MiniMaxH3.scale_latent_inpaint` re-injects the
   clean latent into masked rows, and `MiniMaxH3Model._forward` pins their
   timestep at clean ([`h3_audio_freeze.md`](h3_audio_freeze.md) section 1).
   Holding the track re-noised to the stream's audio sigma, and labelled at
   that sigma, needs a new model patch in this pack and a server restart.
   Build it only if step 2 shows the adapter follows a clean track at all.
4. **Levers, after a verdict.** Change one variable at a time against the
   step-1 arm the owner prefers:
   - Sol off, since the adapter was distilled under dense attention
   - the block-49 levers from that page
   - kijai's against ours again, if the merge-noise record in section 2 says
     requantisation noise swamps the gap between them

## 5. What would count

A dated record under `bench/results/` holding:

- the owner's free-text verdict on each pair
- the render paths
- the rows the runner stamps

Until one exists, this page is a plan.

## 6. Verdict, 2026-09-15: parked

Steps 1 and 2 rendered, and the owner judged them in free text
(`bench/results/2026-09-15_taomate_verdicts.json`):

- **PDD8 is far better than TaoMate in every scene**, by the owner and by
  everyone the owner showed the clips to.
- **Ours and kijai's are indistinguishable**, which is consistent with the
  merge-noise record in section 2.
- **The TaoMate renders fail on consistency.** The diner arms are washed out,
  with morphing and cloning. The dancer morphs into two people, frozen and
  loose alike.
- **The frozen voice arm is the only good TaoMate render.**

No cause is isolated. Of the differences in section 3, denoising the whole
clip without the chunked, cached regime the adapter was trained for is the
most direct candidate for morphing and cloning. That is a reasoned guess, not
a measurement.

Steps 3 and 4 do not run, and nothing here gets another render slot unless
the owner reopens the lane. The converter, the contract in `h3_config`, the
probe graphs and their checks stay as they are, so a reopening starts from a
rebuild rather than a re-derivation.

## 7. Reopened 2026-09-15: porting the streaming runtime

**The owner reopened the lane the same day** ("why not try building the
streaming pipeline? isnt that the point of it?"). Section 6's verdict judged
TaoMate's weights run the only way a stock graph can: over the whole clip at
once. A port asks a different question: whether TaoMate is good when run as
its authors run it.

### 7.1 What the port must reproduce

The source is `src/taomate_h3/streaming/` at `taomate_streaming.UPSTREAM`. Of what the
runtime does, these parts are the regime the adapter was distilled in:

- **Chunk plan.** A request is 37 latent frames, split into chunks of 12, 10,
  10 and 5. A continuation request is 35, split 10, 10, 10, 5. Audio chunk
  boundaries are rounded from the frame timeline; they are not the freeze
  lane's exact 39 + 51k grid.
- **Steps per chunk.** Three steps at the adapter's grid with an Euler update,
  on video only.
- **Audio.** The adapter's audio velocity is ignored. After each step the
  chunk's audio is replaced by a base-model teacher's state at 3, 6 and 9 of
  a 10-point schedule, so the commit pass sees clean audio.
- **Colour matching.** Each chunk's patchified video is matched, feature by
  feature, to the mean and standard deviation of the first chunk.
- **Commit pass.** A clean forward at t = 1, adapter on, records every main
  block's post-norm, post-RoPE K and its V for the chunk's audio and video
  rows. Text is never cached.
- **Cache.** The first chunk's video is kept as a sink, plus the two most
  recent commits with their audio. Audio history is dropped every 12
  requests.
- **Attention.** Text attends only to text. Audio and video attend to the
  text, the cache and the current chunk, with no causal mask.
- **Positions.** One global timeline in audio-latent units, with its origin
  at the first request's prompt length. Each prompt is right-aligned to the
  start of its request's media.

These are engineering and are not reproduced: tensor and sequence
parallelism, FlashAttention-3, the Triton kernels, W8A8, and the ffmpeg
retiming to exact five-second requests. ComfyUI decodes at native length.

### 7.2 How it fits this box

- **Hook points, all in this pack, none in core.**
  - A `comfy.samplers.Sampler` behind a SAMPLER node drives the chunk loop
    through `apply_model`.
  - `patches_replace["dit"][("double_block", i)]` swaps in an attention
    function per block that concatenates the cache.
  - Positions are sliced from one full-length `PackedLayout`. A chunk-local
    layout would restart the frame-span pattern at zero.
- **The cache lives in pinned host memory**, streamed to the card one block
  at a time. Its size per cached row is 50 blocks × K and V × 56 heads × 128
  dims in bf16. At every TaoMate canvas the worst-case history is larger than
  what the card has left after the staged model, so this is the only shape
  that fits, not a fallback.
- **Canvas.** Build and verify at 864x480, where the host cache is smallest.
  A render for the owner's verdict runs at 1344x768, the trained canvas
  (CLAUDE.md), which TaoMate also accepts.
- **Length.** A run is 124 frames plus 119 per further request: 124, 243 and
  362 are all on ComfyUI's 17k + 5 grid, so the stitched latent decodes in one
  pass. 345 is not a TaoMate length.
- **Audio from a frozen track instead of the teacher.** In a loop this pack
  owns, the track is mixed with noise at the teacher's audio sigma for states
  3 and 6 and used clean for state 9. No core patch is needed. This makes
  section 4 step 3 buildable. A plain text-to-audio-video run would need the
  base-model teacher, which core cannot run audio-only; that is the last
  milestone, if any.
- **Attention stack.** Sol and sage are refused on the model this sampler
  drives. The adapter was trained under dense bf16 attention. The block hook
  bypasses sage's patch, and Sol's Morton reordering and sink find their spans
  by the identity of `position_ids`, which sliced positions do not carry. The
  streaming graph takes a `SOL_EXEMPT_STEMS` entry that says so.

### 7.3 Milestones, each gated on the one before

The pieces:
- `taomate_streaming.py`: constants and torch-only parts;
- `taomate_stream_sampler.py`: the `MiniMaxH3TaoMateStreamSampler` node;
- `bench/check_taomate_streaming.py`: step 1;
- `bench/verify_taomate_stream.py`: steps 2, 4 and 5 on the server.

1. **Unit checks on the CPU, no model:**
   - the chunk plan and audio boundaries against the upstream tables;
   - the split text/media attention against the same attention with an
     explicit mask;
   - the cache's retention and eviction bookkeeping.
2. **The degenerate case equals the probe.** Run one chunk covering the whole
   clip, with an empty cache, full attention for text, the adapter's own
   audio and no colour matching. It must reproduce the sampled latent of
   `h3_probe_taomate_3step` at the same seed, within dtype tolerance. It
   exercises the sliced positions, the layout, the per-segment timesteps, the
   audio carry and the bypassed inpaint path against a graph that already
   renders. If it does not pass, stop.
3. **The cache is exact where it can be.** A two-chunk run's first chunk has
   no history, so it must equal the matching rows of a single-chunk run with
   the same masks.
4. **Cost at the target canvas.** One chunk forward with the largest history
   at 1344x768: peak VRAM, pinned host memory and wall time. Record it under
   `bench/results/`.
5. **One request, one throwaway render** at 124 frames with a frozen track,
   read end to end. Then 243 frames, two requests, beside the PDD8 freeze
   chain at the same scene, track and seed, for the owner.

**Status, 2026-09-15.**
- **Step 1 passes** (`bench/check_taomate_streaming.py`).
- **Step 2 passes bit-exact**
  (`bench/results/2026-09-15_taomate_verify_whole_clip.json`). Its control
  shows the match came through the hook: `control_text_only` ran the hook on
  every block at every step and departed from core by orders of magnitude
  more than the bound
  (`bench/results/2026-09-15_taomate_control_text_only.json`).
- **Step 3 was replaced.** A first chunk with no history only equals a
  single-chunk run of the same length, and no such run exists to compare
  against.
- **Step 5's throwaway ran at 864x480**, a canvas that proves the harness and
  not a verdict. All four chunks committed, and the cache sizes matched
  upstream's retention arithmetic.
  - Frames either side of each chunk boundary show one person, a stable room,
    and no morph or duplicate.
  - Its peak-memory log field was unusable under the server's async allocator
    and now reads the driver.
- **Step 4 and the 1344x768 renders wait on a card window.** The sage fork has
  priority.

**Status, later on 2026-09-15.**
- **Steps 4 and 5 ran at 1344x768**, 243 frames, on `t2va_dancer_stream_243`
  against the drum-machine track.
  - **Stream.** All eight chunks committed across both requests, with the
    cache at upstream's sizes. Wall time and card use per chunk are in
    `bench/results/2026-09-15_taomate_stream_dancer_243.json`.
  - **Host memory.** The cache lives in pageable host memory, which left
    little headroom on this box. Watch it before a longer run.
- **Two controls beside the stream:**
  - a whole-clip render with every attention patch stripped
    (`bench/results/2026-09-15_taomate_whole_clip_dense_dancer_243.json`);
  - PDD at 5 and 8 steps on the shipped default chain
    (`bench/results/2026-09-15_taomate_stream_vs_pdd_arms.jsonl`).
- **The owner's verdict on the four is pending.** Read from stills only, not
  judged: the stream holds one stable dancer with a clean push-in but little
  dance motion; PDD dances.
- **Owner's verdict on the two TaoMate arms, the same day**
  (`bench/results/2026-09-15_taomate_stream_verdict.json`). Both look "a lot,
  lot better" than the earlier TaoMate renders. The streamed clip "looks like
  a distill lora but it doesnt duplicate ppl and no obvious artifacts. just
  the usual distill stuff".
  - **Reading (reasoned, not measured).** Both arms dropped sage and Sol, so
    that quantized attention stack was a large part of the section 6 failure.
    The streaming runtime adds stability on top.
  - **Still to judge:** TaoMate against PDD at 5 and 8 steps on the same scene
    and seed.
- **A fidelity review** of the port against upstream and core found no
  fidelity bug. Its runtime findings (host cache memory, allocation scope,
  silent masks) are fixed; see CHANGELOG 0.116.2.
