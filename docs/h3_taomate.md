# TaoMate-H3 in ComfyUI

Opened 2026-09-15. What the TaoMate-H3 adapter is on this box, how a graph
runs it, how our conversion differs from kijai's, and the test plan, including
the audio-freeze arm. What the authors' runtime is, and what their release is
not evidence of, is [`wiki/references.md`](wiki/references.md), "The streaming
references: TaoMate". Nothing here has been rendered.

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
- **The contract** is the TaoMate block in `workflows/h3_config.py`:
  - `TAOMATE_LORA` and `TAOMATE_STRENGTH`
  - the grid: `TAOMATE_STATE_INDICES` over `TAOMATE_GRID_POINTS` at `TAOMATE_SHIFT`
  - `TAOMATE_SAMPLER`
  - `TAOMATE_UPSTREAM`, the upstream revision every value was read at

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
    (`src/taomate_h3/inference/w8a8.py` at `TAOMATE_UPSTREAM`).
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
