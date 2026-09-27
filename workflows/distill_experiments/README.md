# distill_experiments: the distill research graphs

Generated like every graph here by `workflows/build_workflows.py`; never
hand-edit the JSON. Moved here 2026-09-27 at the owner's request. The routing
rule is `build_workflows._is_distill_experiment`:
- every `_savelat` (saves its latents) and `_x0` (saves each step's x0
  prediction) twin;
- every `h3_probe_*` graph that runs a distill (PDD, FlashGen, FastH3, a step
  switch, an audio refine pass);
- the entries marked `distill_experiment=True`.

The everyday distill graphs stay at `workflows/`: `h3_text_to_video_pdd`,
`h3_text_to_video_flashgen`, `h3_text_to_video_pdd_manual_sigmas` (PDD6),
and the PDD ref and first/last-frame graphs.

## Try these first (2026-09-27, one seed; `bench/results/2026-09-27_evening_takeaways.md`)

| graph | what | why |
|---|---|---|
| `h3_probe_t2v_step_switch_pdd8_flashgen_h080_api.json` | t2v: PDD8 to sigma 0.8, then FlashGen finishing | PDD8's take with its dim highlights lifted (one scene) |
| `h3_text_to_video_flashgen_late_blocks_api.json` | t2v: FlashGen on blocks 34-49 only | 4-step speed with about half the haze (fastdude's FT1; not yet judged by the owner) |
| `h3_probe_t2v_fasth3_8step_contract_api.json` | t2v: FastH3 V2 on FastVideo's own settings | the most detail of any distill; it over-polishes |
| `h3_first_frame_to_video_pdd_api.json` | i2v: PDD8 from a first frame | new 2026-09-27; not yet rendered |
| `h3_probe_i2v_step_switch_pdd8_flashgen_h080_api.json` | i2v: PDD8 then a FlashGen finish from 0.8 | new 2026-09-27; not yet rendered |
| `h3_probe_i2v_flashgen_4step_api.json` / `h3_probe_r2v_flashgen_4step_api.json` | FlashGen on i2v / ref2va | held up by eye in one render each; untrained tasks |

`h3_decode_saved_latent_api.json` decodes any `_savelat` twin's latents
again without sampling.
