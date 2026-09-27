# The 2026-09-26 render dataset

Built 2026-09-27 by the VAE session and fastdude, for the owner: "put it into
a json ... so I can analyze it as data ... into a duckdb database". It covers
every render queued on 2026-09-26/27: 196 run rows over 19 records. The
readable companion is `../2026-09-27_inventory.md`.

**Rebuild:** `python bench/build_render_dataset.py --out bench/results/2026-09-27_render_dataset`
(needs the output share: `H3_COMFY_OUTPUT` or a live server).

**Load:** `duckdb h3_renders.duckdb < load.sql`, or in Python
`duckdb.connect().execute(open('load.sql').read())`. That creates the four
tables and a `looks` view.

Everything is one seed, with one clip per arm per scene. A pattern across
renders is a count of shared signs, not an effect size. Final-latent
distances measure divergence, not effect, which is why they are left out of
`measures`.

## Tables, joined on `render_id` = `"<record stem>:<label>:<n>"`

`n` counts that label's rows within the record, from 1. The follow-up
batch's seven relabelled arms have a contaminated first launch (`n=1`) and a
clean rerun (`n=2`).

### `renders.jsonl`: one row per render

| field | meaning |
|---|---|
| `render_id`, `record`, `ts`, `label`, `scene`, `arm` | identity; `scene`/`arm` split from `<scene>__<arm>` |
| `seed`, `length`, `prompt_bank_id`, `prompt_sha256` | the prompt as sent. The bank id is null when the text is not in today's bank: composed at build time, or older bank text |
| `model_family` | base, pdd, flashgen, fasth3, hybrid (the swap and blend checkpoints), route (PDD then FlashGen), turbo, decode_only |
| `graph`, `graph_source` | the graph file, and where its as-rendered copy came from (`current file` or `git <rev>`) |
| `unet`, `loras[]`, `vae[]`, `sampler[]`, `schedule[]`, `shift[]`, `attention[]` | the resolved configuration. Each `loras` entry: `file`, `strength` or `strength_model`, `head_strength`, `blocks`, `modules`, `backbone_apply` |
| `x0_observer`, `saves_latents` | whether per-step x0 latents or final latents exist |
| `git_commit`, `git_dirty`, `code_version` | the tree at render time, from the row's substrate, and the CHANGELOG version at that commit. `git_dirty` is common, because the tree is shared |
| `gpu`, `torch`, `comfy_kitchen` | the environment |
| `total_s`, `sampler_s`, `decode_s`, `wall_s`, `per_node_s[]` | timings; `per_node_s` is a list of `{node_id, class_type, seconds}` |
| `session_position`, `cache_hit_fraction` | from the substrate record, for the last server session only (111 renders); null elsewhere |
| `warmup`, `error`, `suspect_cache_hit`, `contaminated`, `interleaved` | flags |
| `valid_look`, `valid_timing` | filter on these. `valid_look`: not contaminated (the 0.154.8 stacking bug) and no error. `valid_timing`: also not a warmup, not interleaved with another runner, and not a suspected cache hit |
| `outputs` | `{video[], latents[], x0_steps, x0_example}`: basenames in `Video/` and `latents/` |

### `measures.jsonl`: long format, one row per (render, tool, metric)

`render_id`, `clip` (basename), `tool`, `metric`, `value`, and `records[]`,
the result files it came from. A clip measured twice by the same tool is
kept once. The metrics by tool:
- `measure_clip_tone.py`: `black`, `white`, `range`, `mid`, `rms_contrast`,
  `crushed`, `clipped`, `chroma`, `chroma_p95`, `haze`, `detail`, `warmth`,
  `sat`, `r`/`g`/`b`, `orange`, `blue`, `flicker`, `shadow`, `hue_spread`;
- `measure_clip_resolution.py`: `hf`, `loss2`, `loss4`, `flat`,
  `moved_share`, `motion_sharp`, `poster`, `eff_res`, `block16`/`32`/`128`
  (the `block*` columns are latent-grid statistics that do not track the
  owner's eye);
- `measure_clip_temporal.py` (medians over frames): `boil`,
  `motion_detail`, `dark_block8`/`16`, `bright_block8`/`16`, `band`;
- `compare_audio_pairs.py`: `lufs`, `lra`, `true_peak`, `centroid_hz`,
  `band_*_db`.

### `models.jsonl`: one row per model file used or built

`file`, `kind` (diffusion_model, lora, vae), `renders` (how many used it),
`size_bytes`, `family`, `what`, `hybrid_of`, `made_by`, `commit`, `status`.
Descriptions come from `bench/render_dataset_models.json`.

### `findings.jsonl`: one row per claim

| field | meaning |
|---|---|
| `finding_id` | `fd-*` (fastdude), `vd-*` (VAE session), `ow-*` (the owner, as recorded by either) |
| `author`, `date`, `kind` | kind is one of owner_read, measured, prediction, verdict, retraction |
| `text` | the claim; owner quotes are verbatim |
| `render_ids[]` | renders it rests on; may be empty for cross-scene claims |
| `scope`, `model_family` | free text, plus a family that joins to `renders` |
| `prediction_id`, `verdict` | verdict is one of held, falsified, not_supported, open, or null |
| `record` | the file holding the evidence |
| `supersedes` | the finding this one corrects, so retraction chains are queryable |
| `basis` | strength of evidence, e.g. "13 matched scenes, 1 seed" |
| `tags[]` | fixed vocabulary: grade, haze, blacks, saturation, warmth, detail, motion, adherence, clone, audio, speed, weights, int8, encode, method, bug, prompt, vae, schedule, attention |

The sources are `findings_fastdude.jsonl` and `findings_vaedude.jsonl`; the
builder validates both and concatenates them.

## Queries to start from

```sql
-- look by family, valid renders only
SELECT model_family, count(*), median(hf) hf, median(contrast) contrast, median(haze) haze, median(moved) moved
FROM looks WHERE hf IS NOT NULL GROUP BY 1 ORDER BY hf DESC;

-- sampling time at full length, valid timings only
SELECT model_family, median(sampler_s) FROM renders WHERE valid_timing AND length = 345 GROUP BY 1 ORDER BY 2;

-- what we believed and what held, by family
SELECT model_family, verdict, count(*) FROM findings WHERE verdict IS NOT NULL GROUP BY ALL ORDER BY 1;

-- retraction chains
SELECT f.finding_id, f.text, s.finding_id AS corrects, s.text AS corrected
FROM findings f JOIN findings s ON f.supersedes = s.finding_id;

-- every LoRA setting against a measure
SELECT l.file, l.strength, l.head_strength, l.blocks, median(m.value) AS hf
FROM (SELECT render_id, unnest(loras) AS l FROM renders WHERE valid_look) r
JOIN measures m USING (render_id)
WHERE m.metric = 'hf' GROUP BY ALL ORDER BY 1, 2;

-- where the time goes, per node class
SELECT n.class_type, median(n.seconds) AS s
FROM (SELECT unnest(per_node_s) AS n FROM renders WHERE valid_timing) GROUP BY 1 ORDER BY 2 DESC;
```
