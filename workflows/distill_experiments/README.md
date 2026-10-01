# distill_experiments: the distill research graphs

last updated: 2026-10-01 (the first-frame and reference map)

Generated like every graph here by `workflows/build_workflows.py`; never
hand-edit the JSON. Moved here 2026-09-27 at the owner's request. The routing
rule is `build_workflows._is_distill_experiment`:
- every `_savelat` (saves its latents) and `_x0` (saves each step's x0
  prediction) twin;
- every `h3_probe_*` graph that runs a distill (PDD, FlashGen, FastH3, a step
  switch, an audio refine pass);
- the entries marked `distill_experiment=True`.

The everyday distill graphs stay at `workflows/`: `h3_text_to_video_pdd`,
`h3_text_to_video_pdd8_flashgen_finish` (PDD8 then a FlashGen finish, the
owner's t2v pick on 2026-09-27), `h3_text_to_video_flashgen`,
`h3_text_to_video_pdd_manual_sigmas` (PDD6), and the PDD ref and
first/last-frame graphs.

## Try these first (2026-09-27, one seed; `bench/results/2026-09-27_evening_takeaways.md`)

The owner judged these on 2026-09-27 (the board's "For your eye" review; the
verdicts are in the render dataset's findings). The last column says what
held.

| graph | what | why |
|---|---|---|
| `h3_probe_t2v_step_switch_pdd8_flashgen_h080_api.json` | t2v: PDD8 to sigma 0.8, then FlashGen finishing | **Promoted** to `workflows/h3_text_to_video_pdd8_flashgen_finish_api.json`: better than PDD8 alone on three scenes, never worse, same wall time. This probe stays for the bench manifests |
| `h3_text_to_video_flashgen_late_blocks_api.json` | t2v: FlashGen on blocks 34-49 only | Scene-dependent: more natural motion on four of seven scenes, but it added a person on subway_chase_short and lost the piano on slapstick |
| `h3_probe_t2v_fasth3_8step_contract_api.json` | t2v: FastH3 V2 on FastVideo's own settings | the most detail of any distill; it over-polishes |
| `h3_first_frame_to_video_pdd_api.json` | i2v: PDD8 from a first frame | The owner's i2v pick over the FlashGen finish |
| `h3_probe_i2v_step_switch_pdd8_flashgen_h080_api.json` | i2v: PDD8 then a FlashGen finish from 0.8 | Not for i2v: the finish brightens the whole frame at once, away from the input image |
| `h3_probe_i2v_flashgen_4step_api.json` / `h3_probe_r2v_flashgen_4step_api.json` | FlashGen on i2v / ref2va | held up by eye in one render each; untrained tasks |

`h3_decode_saved_latent_api.json` decodes any `_savelat` twin's latents
again without sampling.

## First frame and reference images: what each graph is

Every approach rendered on i2v or ref2va in 2026-09-24 to 2026-09-29 has a graph
here or at the root. The row for each is below; the evidence is
[`bench/results/2026-10-01_r2v_i2v_graph_coverage.md`](../../bench/results/2026-10-01_r2v_i2v_graph_coverage.md),
which also lists what was never built. "Mutants" says whether h3-mutant-distill
ships the recipe. The graphs are API format: drag the file onto ComfyUI.

| task | approach | graph | where it stands | mutants |
|---|---|---|---|---|
| i2v | PDD8 alone | `h3_first_frame_to_video_pdd_api.json` | the owner's i2v pick | yes |
| i2v | FlashGen, 4 steps | `h3_probe_i2v_flashgen_4step_api.json` | one first look, held up by eye | no |
| i2v | PDD8, then FlashGen from 0.8 | `h3_probe_i2v_step_switch_pdd8_flashgen_h080_api.json` | not for i2v: the frame brightens at once | no |
| ref2va | PDD8 alone | `../h3_image_ref_plus_text_to_video_pdd_api.json`, `../h3_ref2v_market_pdd_api.json` | the ref2va PDD8 we run | yes |
| ref2va | FlashGen, 4 steps | `h3_probe_r2v_flashgen_4step_api.json` | one first look; both references held | yes |
| ref2va | PDD8, then FlashGen from 0.8 | `h3_probe_r2v_step_switch_pdd8_flashgen_h080_api.json` | the control of the blind finisher session; market scene, one reference | no |
| ref2va | PDD8, then FastH3 at 12/3 | `h3_probe_r2v_step_switch_pdd8_fasth3_s12_api.json` | same as the FlashGen finish by eye; parked | no |
| ref2va | PDD8, then FastH3 at 10/3 | `h3_probe_r2v_step_switch_pdd8_fasth3_s10_api.json` | grainy by eye; the arm that picks `s12` | no |

To find which graph made a clip, or whether a new experiment has one at all, run
`bench/clip_recipe_coverage.py` over the renders (its docstring has the command).
It reports a recipe with no graph instead of passing over it.
