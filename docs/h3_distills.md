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
