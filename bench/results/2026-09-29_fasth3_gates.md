# #35: FastH3's gates against its backbone drift, read (2026-09-29)

`bench/fasth3_gates_arms.json` (predictions written 2026-09-27, before any
render, by the VAE session). FastH3's contract harness (10/3, its rungs,
euler, core VSA keep 20% from the first step), seed 730451892, three scenes
at their written lengths, one clip per arm:

- `fl2va_gates`: fl2va's weights and conditioning plus FastH3's gate tensors
  (`hybrid__fl2va-weights__plus-fasth3v2-gates__int8`);
- `fasth3_nogates`: FastH3 V2 without its gates
  (`hybrid__fasth3v2-weights__no-gates__int8`; core warns and runs VSA's fine
  stage alone);
- `fasth3_rerun` (look_anchor only): FastH3 as released, rendered today;
- the other two cells of the 2x2 are the earlier rows: `<scene>__fasth3` (the
  followup batch) and `<scene>__fl2va_contract` (the swap batch).

Rows: `2026-09-29_fasth3_gates.jsonl`. Measures:
`bench/analyze_followup.py --group gates`, in
`2026-09-29_gates_<scene>_{tone,temporal,resolution,divergence}.json`. The
arms loaded the two hybrid files, not the overlay loader.

## Read

| scene | arm | hf | detail | chroma | contrast | moved |
|---|---|---|---|---|---|---|
| look_anchor | FastH3 | .0118 | .0494 | .131 | .244 | .094 |
| | FastH3 rerun today | .0113 | .0488 | .129 | .243 | .097 |
| | fl2va + FastH3 gates | .0113 | .0449 | .122 | .247 | .085 |
| | FastH3 without gates | .0076 | .0238 | .068 | .185 | .027 |
| | fl2va | .0081 | .0211 | .055 | .157 | .016 |
| slapstick | FastH3 | .0257 | .1020 | .203 | .266 | .345 |
| | fl2va + FastH3 gates | .0281 | .1012 | .207 | .255 | .320 |
| | FastH3 without gates | .0210 | .0693 | .151 | .211 | .170 |
| | fl2va | .0265 | .0753 | .152 | .183 | .131 |
| radio_drama | FastH3 | .0267 | .0821 | .158 | .239 | .157 |
| | fl2va + FastH3 gates | .0250 | .0762 | .156 | .229 | .150 |
| | FastH3 without gates | .0225 | .0560 | .100 | .183 | .072 |
| | fl2va | .0242 | .0507 | .093 | .159 | .042 |

1. **The gates carry the look.** On every scene, fl2va with FastH3's gates
   sits next to FastH3 on detail, chroma, contrast and moved share, and FastH3
   without its gates sits next to plain fl2va. By the manifest's own
   falsifier (hf within 10% of FastH3 on all three scenes) fl2va plus gates
   passes, and FastH3 without gates does not on look_anchor.
2. **hf alone does not separate the arms on slapstick**: plain fl2va is within
   10% of FastH3 there. The reading above rests on detail, chroma, contrast and
   moved share, where the pairing holds on all three scenes.
3. **The rerun is close on tone, not identical.** Today's FastH3 differs from
   yesterday's clip (`2026-09-29_gates_look_anchor_divergence.json`), while its
   tone measures agree to a few percent. The kitchen was rebuilt between them
   (upstream's int8 attention changes came in with the merge); that is a
   candidate, not a finding, and nothing here tests it. Every arm above
   except the rerun is read against yesterday's rows, so a shift of the
   rerun's size is inside every difference the table calls small.
4. **What this cannot say.** The no-gates arm also drops VSA's coarse branch
   (core warns), so its failure could be the missing branch and not FastH3's
   gate values. The arm that would separate them is fl2va with FastH3's gates
   scaled down (the loader's `gate_scale`, direction `fasth3-gate-dial`), not
   run. One seed, one clip per arm, three scenes, the tone and resolution
   measures only; nothing here is judged by eye.

## The predictions, graded

- **G1** (`fl2va_gates` stays near the floor): **falsified.** It lands at
  FastH3 on every scene.
- **G2** (`fasth3_nogates` keeps FastH3's convergence and loses only detail):
  **falsified.** It lands at the fl2va floor on detail, chroma, contrast and
  moved share.
- **G3** (the rerun reproduces yesterday's latent exactly, or within the floor
  if today's changes touched the path): **not exact**; within a few percent on
  tone. Whether that is the floor or a change on this path is not known.
- **Falsifier**: the gates arm meets its branch (hf within 10% on all three
  scenes) and the no-gates arm does not on look_anchor. Read with point 2.

## What it decides

FastH3's look and its few-step behaviour travel with the 50 gate tensors under
core's VSA, so a lighter FastH3 is the gates on a base. The gates are exact
overlay pieces (`2026-09-29_fasth3_overlay_exact.md`). The open items are the
gate dial (#36), whether the gates work on ref2va and PDD-applied bases
(untested; the overlay is exact only on fl2va), and the owner's eye on the
clips.
