# Sol start_percent 0.2 against 0.0 on the t2v finish: the owner's blind pairs (2026-10-01)

**Result: no difference the owner could see or hear, and 0.0 is markedly faster.** All five
pairs were scored "can't tell" with both halves tagged good, judged blind with audio. Removing
the two dense steps before Sol's window cut the sampler by roughly a fifth on every scene
(rows below).

## How

- **Arms:** `workflows/h3_text_to_video_pdd8_flashgen_finish_api.json` as shipped
  (`start_percent` 0.2: sigma 1.0 and 0.988 run dense) against the same graph with both Sol nodes
  at 0.0 (every step sparse). `bench/start_percent_panel_arms.json`.
- **Scenes:** `t2va_marching_band`, `t2va_samurai_bamboo_duel`, `t2va_nature_doc_arctic`,
  `t2va_police_interrogation`, `t2va_cartoon_vaudeville`; 1344x768, 345 frames, seed 730451892
  held, one render per arm per scene after a warmup, kitchen `0.2.36+sol.aade8d5.up.3f7210f`.
  Rows: `2026-10-01_start_percent_panel.jsonl`.
- **Blind:** `blind_batch.py --pairs` per scene, session `start_percent_2026-10-01`, the key
  sealed until the scores existed; the owner scored pairs with each half's audio
  (`bench/briefs/2026-10-01_start_percent_panel.md`). Verdict:
  `2026-10-01_start_percent_2026-10-01_verdict.json`.

## The owner's notes, unblinded against the prompts

- **Nature doc:** the biologist speaking to camera was the 0.0 half. The prompt asks for it ("turns
  toward the camera, and says"), so a small adherence point to 0.0.
- **Interrogation:** a figure sitting down beside the suspect was the 0.0 half. The prompt has the
  detective pull out a chair and sit directly opposite; the 0.2 half left the sit-down out. A
  partial miss on each side.
- **Cartoon:** a slight preference for the 0.2 half's colours (red curtains against blue),
  "hard to tell".
- **Marching band, samurai duel:** different takes, both equally good.

## What it covers

One seed per scene on one graph, the t2v PDD8-to-FlashGen finish, so a preference over five
samples, not a distribution. Not covered: the 16-step base graphs (0.2 leaves three or four of
16 steps dense there), i2v and ref2va, and the FlashGen-alone graphs, where 0.2 already leaves
only sigma 1.0 dense. `start_percent` lives in `h3_config.SOL_RECOMMENDED_CUDA`, shared by every
Sol graph, so a change there moves all of them.
