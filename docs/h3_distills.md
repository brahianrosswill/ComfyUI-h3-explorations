# Which distill for which shot

last updated: 2026-09-26 (written from one seed, two scenes, unblinded)

> **Seed-matched clips are different scenes.** The same seed through each
> distill gives a different scene: different camera angle and framing,
> different subjects, a different palette and mood. Each distill maps noise to
> video its own way, through its training data and objective. Nothing in its
> training keeps the teacher's seed-to-scene mapping, and the distribution-
> matching distills (FlashGen, FastH3) settle on their own likeliest scenes. So
> a seed-matched clip from two distills is two samples, not a controlled pair
> (`eval_comparison.md`, step 1). A claim that distill A beats distill B holds
> only when it is judged on something independent of content, across many
> scenes, and blind: artifacts, clones, identity across cuts, following the
> shot plan, the grade against the base. The owner, 2026-09-26: "seed-matched
> clips are different scenes for each lora distill adapter due to training
> data and all sorts of other things."

What each few-step distill of H3 is good and bad at, and why, as far as the
records show. **This rests on one seed and two scenes (diner, subway), judged
unblinded by the owner, plus measurements of those clips.** It is a working
guide, not a verdict. A second seed is queued
(`../bench/flashgen_adherence_seed2_arms.json`), and this page is revised when
it lands. Anything under "why" that is not measured is marked as inference.

Each distill is compared against the undistilled base model at the same seed
and prompt, which is the reference for how H3 itself renders the scene.

The numbers live in the records, not here:
- `../bench/results/2026-09-26_distill_compare_s1.md`: the three-way look, old
  subway prompt;
- `../bench/results/2026-09-26_subway_v2_s1.md`: four models on the rewritten
  subway prompt, with the colour measurement;
- `../bench/results/2026-09-26_distill_tone.md`: brightness and contrast
  against the base.

## At a glance

| distill | good at | bad at | speed at 345 frames |
|---|---|---|---|
| **PDD8** | close-ups and medium shots: detail and colour | fast motion: artifacts, a cloned extra person, mangled text | slowest of the three distills |
| **FlashGen** | motion that stays coherent (zoom-outs); detail; colour closest to the base | following who does what in a multi-person action scene | fastest |
| **FastH3** (contract) | motion; fewer artifacts than PDD8 on the chase; audio | warm, oversaturated, contrasty grade; some coarse texture | between the two |
| base (reference) | lighting and prompt adherence | nothing here; it is the reference | the slowest by far |

The sampler times per model are in the two subway records' `sampler_s`.

## Why they differ: a working model, open for annotation

Written 2026-09-26 as a set of hypotheses, not findings. Each claim is
labelled **measured** (with its record), **code** (read from the source) or
**inference**. A peer session annotates in place, never by rewriting: add a
dated, signed blockquote under the claim (`> vaedude, 2026-09-26: ...`). A
claim that a test refutes keeps its text and gains an annotation saying so.

### Why the same seed gives a different scene

- **Code: the base and the distills do not share starting noise, even at the
  same seed.**
  - Both draw the same starting tensor from the seed (`Noise_RandomNoise`).
  - Every distill steps with Euler, which adds no noise, so that tensor
    carries through the whole render.
  - The base samples with `er_sde` (`comfy/k_diffusion/sampling.py::sample_er_sde`),
    which adds fresh noise after every step but the last. The fresh noise
    comes from a second generator seeded with the same number, so the base
    still reproduces run to run.
  - At `er_sde`'s first step the carry factor on the starting tensor
    (`r_alpha * r`, with the first sigma offset just below 1) is vanishingly
    small, and the fresh draw takes its place. Drive that function's
    arithmetic at the base schedule to see the factors.
  - So a seed-matched base clip and distill clip start from unrelated noise.
    Distills at one seed do share their start with each other.
- **Inference: the first steps, at noise near 1, settle the layout**: camera,
  who stands where, the overall grade.
  - Between the base and a distill, the start differs, so the scene would
    differ even with identical weights.
  - Between two distills, the start is the same, and the weights alone move
    the scene. Every distill changes what the model predicts from that noise.

### PDD8: the teacher's path in coarse averaged steps

- **Code** (`h3_pdd.md` "What it is"):
  - PDD keeps a 32-point grid and one output head per interval of it.
  - Each sampling step fuses a contiguous block of those heads into one
    output. That output is the block's mean velocity, weighted by each
    interval's step size.
  - PDD8 fuses 4 heads per step, so each step covers an eighth of the grid by
    index. In noise terms the steps are very uneven: under shift 12 the last
    step is by far the widest, and it is wider at 4 steps than at 8 or 16. The
    schedules come from `pdd_lora.emit_sigmas(12, 32, width)`, and
    `h3_pdd.md` lists them.
- **Code** (`h3_pdd.md`, the section on sub-steps): every head in a block reads
  the same hidden state. A fused step is therefore exactly one Euler step at
  the mean velocity, and nothing inside the block reacts to what the frame
  becomes.
- **Measured** (`evidence.md`, "Settled about H3"): PDD quality is governed by
  how coarse the schedule's tail is, not by the evaluation count.
- **Measured, with a caveat** (`h3_pdd.md`, the partition table dated
  2026-08-28): PDD8 already sits far from PDD's own 32-step path (width 1, no
  fusion). The table was taken at 39 frames with Sol inert, so treat it as a
  direction, not a size.
- **Inference: the averaging explains PDD's motion failures.**
  - For a still or slow scene the velocity barely changes across a block, so
    the mean is a good stand-in. Hence the good close-ups, detail and colour.
  - When something moves, the velocity inside a block changes a lot, and the
    mean of two positions can put a person in both. The clone at 1 s on subway
    would be that: an averaging artifact, not a prompt problem. Mangled text
    on moving signage would be the same thing at fine scale.
  - The widest blocks come last, so fine detail on moving things is decided in
    one or two coarse averaged steps. That fits artifact severity tracking
    inter-frame delta (the PDD8 "Why" below).
- **Inference, not yet read in the paper:** PDD is the only one of the three
  that tries to reproduce the teacher's path. It was the closest to the base
  in every scene measured (`2026-09-26_distill_compare_s1.md`), but that
  comparison started from unrelated noise (above). So the closeness is
  shared prompt and grade, not a shared path, and the shipped base is not
  PDD's target.

### FlashGen and FastH3: matching the distribution, not the path

- **Code / release:** both are distribution-matching distills. FlashGen is
  VSD (`research/2026-09-26_flashgen.md`) and FastH3 is DMD2 with VSA sparse
  attention (`h3_config.FASTH3_CONTRACT_VSA`). They are trained so their few
  steps land on outputs the teacher would plausibly produce. They are not
  trained to land where the teacher would from the same noise.
- **Inference: what that would explain.**
  - **Motion stays coherent:** no averaged velocities, so no ghosting from
    that cause.
  - **The look gets bolder:** likely outputs are high-contrast, saturated and
    conventionally framed. That would give the grade in
    `2026-09-26_distill_tone.md` and "a very different scene".
  - **Adherence weakens:** unusual instructions (who chases whom, which way
    they run) are exactly what "likeliest" tends to override.
  - **FastH3's grainy texture:** VSA lets each video cube attend in full to
    only the fraction of cubes `FASTH3_CONTRACT_VSA` keeps, with a coarse
    summary covering the rest. Fine texture is then partly invented locally.

### Tests that would move this section

- **PDD's schedule, on motion scenes** (the owner, 2026-09-26: "we should
  absolutely do the faster path"). The same file runs at several schedules:
  - 4 steps, width 8, the coarsest tail;
  - 6 steps (`PDD_MANUAL_SIGMAS`): 8's tail at fewer evaluations;
  - 8 steps, width 4;
  - 16 steps, width 2, the finest tail.

  If the clones are averaging artifacts, they get worse as the tail gets
  coarser. If the tail matters rather than the count (`evidence.md`), 6 steps
  should match 8, not sit between 4 and 8. Judged blind across scenes, as the
  box at the top requires. The unjudged `C2_pdd4_nosol` / `C2_pdd8_nosol`
  pair (`research/pdd/queued_arms.md`, three seeds each) is a zero-card-time
  first look.
- **Base on Euler over PDD's grid, same seeds.** This is the only base render
  that shares the distills' starting noise, so it is the controlled pair for
  every distill, not only PDD. If PDD8 lands much closer to it than FlashGen
  or FastH3 do, PDD is following its teacher. If all three stay far, the
  scene difference is the weights.
- **FlashGen strength and module arms** in the 2026-09-26 run
  (`../bench/distill_run_arms.json`): whether the grade and adherence move
  with the LoRA's strength, or without its adaln.

## PDD8

**Good.** The owner, 2026-09-26: "looks great when its closeups or medium
shots - lots of detail and good color", "seems to do great at low
movement/deltas". On diner, fine apart from the zoom-out.

**Bad.** Motion. Subway on the old prompt: "super artifacty... text is
mangled. people disappear", and the pursuer cloned. On the rewritten prompt:
"clones a guy out of nowhere right at the 1s mark... way less artifacty than
before". Diner's zoom-out: "a little artifacty".

**Why.**
- **Measured, 2026-08-28:** artifact severity under PDD tracks inter-frame
  delta (+0.676; `../bench/measure_clip_delta.py`, docstring). Big motion means
  big frame-to-frame change, and that is where PDD breaks.
- **Inference:** PDD decodes each step's whole interval from one forward pass
  through its head bank (`h3_pdd.md`), which is more to get right when the
  frame changes a lot between steps.
- **Measured tone:** PDD8 brightens and warms the picture against the base, on
  subway as warm as FastH3 in places (`2026-09-26_subway_v2_s1.md`).

## FlashGen

Shipped as `h3_text_to_video_flashgen`: rank 64, applied at the call. What it
is, and why it is built that way: `research/2026-09-26_flashgen.md`.

**Good.** Diner's zoom-out "looks good, keeps coherency". It "complements pdd
well". Detail "looks good". Its colour stays the closest of the three to the
base's (`2026-09-26_subway_v2_s1.md`).

**Bad.**
- **Following instructions in a multi-person action scene.** On the rewritten
  subway prompt, "the suspect runs right past him. then it becomes unclear who
  is chasing who as they run down the escalator", while the base followed the
  same prompt at the same seed.
- **Its own composition:** "a very different scene from all the others".
- **More contrast than the base.** Less so than FastH3
  (`2026-09-26_distill_tone.md`).

**Why.**
- **Not our sampling setup.** Sigmas, Euler, 4 steps, no CFG and exact LoRA
  application all match vllm-omni, the engine FlashGen ships for
  (`research/2026-09-26_flashgen.md`).
- **Candidates, each with an arm queued** (`../bench/flashgen_adherence_arms.json`):
  - its trained length: 5.2 s, where we render 14.4 s in three shots;
  - dense attention in its recipe, where we run Sol;
  - the one seed.
- **Inference:** distribution-matching distillation (FlashGen's VSD) favours
  the most likely outputs, which can mean a different composition from the
  base and weaker adherence to unusual instructions.

## FastH3 V2

Run to FastVideo's own contract (`h3_probe_t2v_fasth3_8step_contract`), not
ComfyUI's template: see `../bench/results/2026-09-26_fasth3_contract_s1.md`.

**Good.** "much better audio and less artifacty than pdd". Motion holds. The
owner: "a tossup between this and pdd8", and it is faster than PDD8 at this
length.

**Bad.**
- **The grade:** "more washed out / contrasty than flashgen", "contrasty-color
  grading issues (look at the stairs at the 9s mark - orangeish)... maybe its
  over saturated".
- **Texture:** "stairs at the 10s mark look like low res ps2 polygons".
- **Clones:** a cloned pursuer around 5 s on the old subway prompt.

**Why.**
- **Measured:** the brightest highlights and steepest contrast of the three,
  and on subway the warmest and most saturated overall
  (`2026-09-26_distill_tone.md`, `2026-09-26_subway_v2_s1.md`). "Washed out"
  here is bright, near-clipped highlights on a steep curve, not faded colour.
- **Inference:** DMD2, like FlashGen's VSD, favours the most likely outputs,
  which pushes toward a punchier grade.
- **Inference:** the "ps2 polygons" texture may come from VSA's sparse
  attention. It keeps 20% of video cubes, as trained, and a coarse branch
  covers the rest.

## Rules of thumb, provisional

- **Close-ups, medium shots, little motion:** PDD8.
- **Fast motion or a moving camera:**
  - FlashGen, if the action is simple enough that roles cannot swap;
  - FastH3, if the colour grade is acceptable or will be corrected.
- **A multi-person action scene where who-does-what matters:** the base model,
  until the FlashGen adherence arms say otherwise.
- **Speed first:** FlashGen.

## What would change this page

- The second seed (queued): whether each pattern holds on another draw.
- The FlashGen adherence arms (queued): which of length, attention or seed
  explains its adherence.
- A tone correction toward the base, and switching distills by denoising step
  (`wiki/next_steps.md`, the idea saved 2026-09-26): either could move a
  distill out of its "bad at" column.
