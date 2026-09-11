# Decisions and reversals

Written by hand. One dated line per decision the owner made, or per claim in
the prose that was corrected: what changed, what it used to say, where it
lives now, and the commit. Newest first. `CLAUDE.md` and `VISION.md` carry no
history notes; this page is where those go. Another document may also keep a
dated note in place, where a reader would otherwise trust the stale text.

Older history lives elsewhere and is not copied here:

- [`CHANGELOG.md`](../../CHANGELOG.md): every change, by version.
- [`docs/rules_history.md`](../rules_history.md): `CLAUDE.md` as it stood
  before the 2026-09-03 cut, frozen.
- [`docs/roadmap.md`](../roadmap.md): "Closed lanes", and the dated "Owner
  decisions" and forward-plan sections.
- `bench/results/`: the verdict records, each with its conditions.

## 2026-09-11

- **`CLAUDE.md` routes to the wiki instead of carrying its tables** (owner;
  CHANGELOG 0.99.72). The "What is where" tables moved into
  [`index.md`](index.md), which is now written by hand.
  `bench/build_wiki_index.py` used to generate that page from `CLAUDE.md`; it
  now only reports documents no link reaches.
- **History notes leave `CLAUDE.md`** (owner; CHANGELOG 0.99.72). Its rule
  "when you find prose that lost, correct it and say what it used to claim"
  now logs the old claim on this page.
- **"A perceptual claim needs a distribution of seeds judged blind, never a
  pair" is withdrawn** (owner; CHANGELOG 0.99.72) from `VISION.md`,
  `CLAUDE.md`, the `h3-experiment` skill, `docs/open_experiments.md` and
  [`prompting.md`](prompting.md), under the tinkering-repo rule.
  `docs/eval_comparison.md` still describes the blind process for when one
  is wanted.
- **Where the Sol-Attn kernel comes from.** `CLAUDE.md` said it was
  "installed from comfy-kitchen main". It is built from the owner's fork by
  `vendor/rebuild_kernel.sh`: the tag ComfyUI pins plus the `blk_cnt`
  commits, enforced since `d378479`.
- **The tinkering-repo rule** (owner). `CLAUDE.md` opens with it: not
  research-grade, rigour proportional to the claim, and a default that
  sglang and ComfyUI's own node or comfy-kitchen agree on is adopted without
  waiting on our evals.
- **No check has to be shown red before it is trusted** (owner, `48e1219`).
  `docs/checks.md` "The standard" says what it used to require.
- **Closed lanes moved into the repo** (`48e1219`). They were recorded only
  in agent memory; `docs/roadmap.md` "Closed lanes" is their home.
- **The probe canvas** (`48e1219`). The `h3-experiment` skill said to bench
  canvases cheaper than 16:9 by default; probes run at 1152x768 or 1344x768
  with 345 frames, the owner's rule of 2026-08-30.
- **Sol runs through the last step** (owner, `07b903c`). `end_percent` is 1.0
  on every graph, adopting sglang's and core's default. It used to stop short
  so the last step ran dense; the retired values are in the comments beside
  `workflows/h3_config.py::SOL_END_PERCENT_BY_STEPS` and `SOL_PDD_OVERRIDES`.
