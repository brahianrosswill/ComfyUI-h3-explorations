# The Sol node redesign: plan, reasons, and the tests that decide it

Opened 2026-09-27. The owner approved the approach the same day ("I'm good
with this approach"). Nothing here is built yet.

This page owns the plan and the reasons for it. The evidence is in
[`bench/results/2026-09-27_sol_node_compound_audit.md`](../../bench/results/2026-09-27_sol_node_compound_audit.md),
cited here as "the audit", by section. The node-against-node comparison is
[`bench/results/2026-09-27_attention_parity.md`](../../bench/results/2026-09-27_attention_parity.md).
Numbers live in those records, not on this page.

## Why redesign at all

The owner asked for an account of `MiniMaxH3SolAttn` as a compound system:
every knob, how the knobs combine, how they differ from core's
`BlockSparseAttention`, and which ones do nothing. The audit found five
problems:

1. **Several knobs do nothing, or nothing any more.**
   - `tau_profile` is wired in no graph.
   - The `2d_frame` and `hilbert` curves are measured worse and already
     deprecated.
   - Top-k selection is used in no graph, and its LoRA arm retired.
   - `pooled_tail` is True everywhere.
   - `dense_blocks` switches every other Sol knob off on its blocks (audit
     §4.1).
2. **Two knobs overlap.** `qk_balance` and `rotate` compose, and rotation
   does most of balance's work (audit §4.4). The shipped `qk_balance=True` is
   probably a no-op behind the dense tail, because its gate is recorded as
   opening only on the blocks we send dense.
3. **One default is wrong by the owner's rule.** The node defaults
   `qk_balance` off and 129 of 131 graphs turn it on, which breaks "best
   defaults end to end".
4. **The plumbing is fragile.** The node installs its override once, so any
   attention node placed after it silently takes over. It also patches core
   process-wide (`PackedLayout.__init__`, `model._forward`), even though
   core now publishes the layout itself (audit §4.8, §7). Both shapes caused
   trouble before: the second is the one that got the VSA node parked.
5. **One kernel bug was found in passing** (audit §5.1; below).

## What is certain without a test

Each of these follows from reading the code or counting the shipped graphs,
so no measurement can move it:
- the balance/token-routing kernel bug;
- `dense_blocks` taking precedence over every other knob;
- top-k switching off `tau` and `tau_profile`;
- the zero-use knobs;
- install-once fragility;
- the process-wide patching;
- the default mismatch;
- sage's mode never reaching Sol's own calls (audit §9b);
- the prose that lost to code (audit §5.4);
- removing widgets in place would re-point values in every editor-saved
  graph.

## The design

**Ship a new node, appended to the node list, and freeze the old one.**
Editor-saved graphs map widget values by position (`sol_attn_h3.py:1678-1681`),
so editing `MiniMaxH3SolAttn` in place would corrupt them. The old node keeps
working for saved graphs and is marked deprecated. The generated graphs move
to the new one.

**Inputs.** Each input is a decision, not a mechanism:

| Input | Form | Reason |
|---|---|---|
| `method` | DynamicCombo: adaptive tau (`tau`) / top-k (`keep_percent`, `pooled_tail`) | The tail only means something under top-k. Top-k and its two sub-inputs go entirely if the owner closes the SLA lane. |
| `quantizer` | plain / balanced / rotated / balanced+rotated | Replaces two booleans that interact. The default is chosen by test 2. |
| `dense_blocks` | text | Kept. Its default is chosen by test 2 for each dense chain, not carried over from the kitchen chain. |
| `sink_conditioning` | as today, including `exact_kv_and_all_rows` | Kept. Its default flips if pair C of `bench/sol_core_ab_arms.json` says so. Upstream makes every conditioning row dense (audit §9). |
| `token_routing` | off / measured blocks / all / custom (the list as a sub-input) | The list stops being a top-level string. The requirement "all" makes of the quantizer is re-derived after the kernel fix (test 3). |
| `start_percent`, `end_percent`, `min_tokens`, `verbose` | advanced | They mean the same as core's. |
| `reorder` | none / 3d, advanced | Replaces `morton` + `morton_curve`. See "Morton" below. |

**Retired from the node:** `tau_profile`, the `2d_frame` and `hilbert`
curves, and, if the SLA lane closes, top-k with `keep_percent` and
`pooled_tail`.

**Plumbing:**
- **Re-install the override on top at every step,** as core does
  (ON_PREPARE_STATE). That closes the silent takeover.
- **Read core's `minimax_h3_layout` and `block_index`** instead of patching
  `PackedLayout` and `model._forward`. The sink needs only the layout.
  Morton's hooks move to `add_object_patch`, and install only when `reorder`
  is on. A shipped graph then patches nothing in core.
- **Write the dense kernel into provenance.** The node stays chainable, so
  both dense chains keep working. It detects which dense kernel sits under it
  (kitchen int8, a sage mode, bf16) and records that in each render's
  provenance and log. The dense steps are where composition is decided (the
  coin beat in `bench/results/2026-09-15_block49_community_chain.md`), and
  nothing records that kernel today except by inference from the graph.
- **Raise on a kernel exception.** Today the node logs the exception and runs
  the dense kernel instead: a render that succeeds and is not a Sol render,
  which this repo refuses everywhere else.
- **Accept fp16 on the direct entry.** The kernel already takes it, and the
  bf16-only prose is wrong (audit §5.4).
- **Keep the direct entry, with per-call statistics.** Core's chunked entry
  carries statistics from the previous step. That measured further from the
  floor (`bench/results/2026-09-10_sol_impl_capture_grade.json`), and the
  chunked entry takes no `qk_balance`. `MiniMaxH3SolChunked` stays the
  memory option; it engages only in the sage chain.

**Kitchen fork (`h3-frontier`):** fix §5.1 and add a test that tracks eager
for qk_balance with token_aug. Merge, rebuild, and record the new build
(`vendor/rebuild_kernel.sh`).

## Retiring code (owner, 2026-09-27)

> "if you retire any parts of the node, be sure to do the same to the
> underlying code that it uses as well, since its all git tracked anyway."

For each retired input, the code only it reaches goes too: helpers, hooks,
the checks that pin them, and the prose that describes them. Records that
cite a deleted script keep citing it by the commit where it last existed.
That commit goes in the `docs/wiki/decisions.md` line for the retirement.

Starting map. `grep -rlw` over the pack, excluding `coderef/`, `internal/`
and `vendor/`. This is a starting point, not the final list. Each item is
re-traced when it is removed.

| Input | Code that reaches it today |
|---|---|
| `tau_profile` | `sol_attn_h3.py` (`parse_tau_profile`, the `block_tau` path), `sol_chunked_h3.py`, `provenance.py`, `bench/check_sol_kernel.py`, `bench/check_sol_observe.py` |
| `2d_frame`, `hilbert` | `sol_curves.py`, `sol_attn_h3.py`, `sol_block_probe.py`, `workflows/h3_config.py`; the analysis scripts `bench/sweep_sol_orderings_on_capture.py`, `analyze_routing.py`, `analyze_capture.py`, `analyze_morton.py`, `analyze_canvas_geometry.py`, `gen_figures.py`, `_live_sol.py`, `bench_e2e_h3.py` |
| top-k, `keep_percent`, `pooled_tail` (if the SLA lane closes) | the node path in `sol_attn_h3.py`, `build_workflows.py`, `h3_config.py`, `render_inventory.py`, `check_widget_deviations.py`, `check_sol_node_equivalence.py`. The kernel's `topk_ratio` stays: VSA and core use it. The capture instruments that sweep it are reviewed one by one. |

`vendor/sol_attn_minimax.py` is not ours. It is the pristine upstream
reference and stays byte-identical.

## Morton

Morton gets its own section because the case is mixed.
- **The captures favour it.** `3d` has lower Sol error at equal routed
  density on every captured cell (`bench/results/2026-09-17_sol_orderings.md`).
- **The eye doesn't, at the same tau.** The blind panel found no defect
  removed on full-length clips (`bench/results/2026-09-18_sol_reorder_panel.md`).
- **One short clip is the exception:** a single on-length short scene lost a
  ghost figure with `3d`, and the short-clip panel is unscored.
- **Upstream runs no reordering on H3**, and core's merged node never had it
  (audit §9).

The plan:
1. **Fold it into one `reorder` input,** none or 3d, advanced. Retire the
   other curves and their code.
2. **Decouple the sink from Morton's machinery,** as above, so the default
   path patches nothing.
3. **Give it one chance as a speed lever,** the use nobody has tested. The
   captures say `3d` reaches plain order's error with fewer routed blocks,
   and Sol's time follows routed density
   (`bench/results/2026-09-17_sol_stage_profile.md`). The gates run in order:
   1. **Bound the prize.** From the existing 2026-09-17 sweep data, find the
      tau at which `3d` matches plain order's error at tau 1.0, and read off
      the density saving. That saving applies to Sol's share of the render
      only. If it is small, stop.
   2. **Time a pair:** plain order at tau 1.0 against `3d` at the matched
      tau, same scene, comparing sampler time.
   3. **Score a blind panel** at the prompts' written length, short clips
      included. It passes if the owner calls it a tie or better.
4. **Remove Morton entirely** (mechanism, checks, prose) if neither the
   short-clip panel nor the speed gates favour it. The capture result stays
   a recorded property of `3d`: closed, not refuted.

## Tests, in order

The test material (scenes, seeds, and graphs for t2v, a distill, i2va and
ref2va) comes from fastdude, per the owner. The owner wants i2va and ref2va
included to see whether shapes and activations differ by mode. The card is
free only after fastdude's int8-attention after-runs, and ComfyUI is
restarted on the new kitchen before anything runs.

0. **Determinism control.** Render the same graph twice and check the
   latents are bit-identical. `bench/results/2026-09-18_sol_reorder_panel.md`
   records renders repeating bit for bit, but on the old kitchen; this
   confirms it on `0.2.35+sol.863e953.up.c8c7825`. Every "does this knob do
   anything" answer rests on it.
1. **`qk_balance` on against off**, shipped graph, same seed, latents
   compared, in each mode.
   - **If bit-identical:** it is inert behind the dense tail, so the shipped
     graphs need nothing fork-only from Sol, and `quantizer` matters only for
     arms that change `dense_blocks`.
   - **If not:** some block below 45 opens the gate, and the node default
     becomes on.
   - Bit-identity is a yes/no answer for the prompts run. The gate depends on
     activations, so a second scene is what generalizes it.
2. **One capture session.**
   - Blocks 40 and 44 to 49 (45 to 48 were never captured), at steps 4 and
     15, in each mode.
   - Graded offline for Sol under the four quantizers, against kitchen int8
     and sage rotated as the dense kernel.
   - It decides the `quantizer` default, whether the dense tail pays under
     each chain, which blocks belong in `dense_blocks`, and kitchen against
     sage on matched cells, which has never been graded (audit §9b).
   - Captures, not renders, because these are numerical questions
     (`CLAUDE.md`: a rendered clip cannot A/B a numerical change).
3. **The kernel fix, then a re-grade of token routing** per quantizer on the
   same captures. This decides the token-routing gate and whether routing
   becomes a default anywhere.
4. **Renders, last,** only where captures cannot decide: the chain choice
   (the coin beat), sink pair C, and the Morton panel. Each is blind, with
   matched seeds, at least two seeds and two scenes
   (`docs/eval_comparison.md`).

**Test material** (fastdude, 2026-09-27; checked before use):

| Mode | Graph | Before rows |
|---|---|---|
| t2v base | `workflows/h3_text_to_video_api.json` | `bench/results/2026-09-26_distill_run.jsonl`, base arms (most scenes on bank text older than 0.154.6) |
| distill | `workflows/distill_experiments/h3_text_to_video_pdd_savelat_api.json` (PDD8, saves latents); `h3_text_to_video_flashgen_savelat` has the same coverage | `2026-09-26_followup.jsonl`; after on this kitchen: `2026-09-27_int8attn_after.jsonl` |
| i2va | `workflows/h3_first_frame_to_video_api.json` (canvas follows the keyframe; no Resolution node). Distill: `distill_experiments/h3_first_frame_to_video_pdd_api.json`, never rendered | none |
| ref2va | `workflows/h3_image_ref_plus_text_to_video_api.json` | none on this kitchen |

- **Scenes** (seed 730451892, on-length at 345 frames): slapstick piano,
  courtroom verdict, samurai duel, subway chase, and the `t2va_look_*`
  family.
- **Short on-length clips:** noodle bar (107), and `subway_chase_short` and
  the spec ladder (124).
- **Avoid for temporal measures:** scenes that ask for frame-to-frame
  brightness change (kpop, silent film, subway chase short).
- **Warmup only:** swimming lesson.
- **Prompt setup:** prompts are patched with `@bank:<id>`, and
  `MiniMaxH3Resolution.length` is set to the bank's frames, as
  `bench/make_followup_manifest.py` does.

**Capture tooling:**
- Captures are enabled by `H3_CAPTURE="dir=...,blocks=a:b:c,steps=x:y"` in
  the server's environment (`h3_capture.py` docstring), with `pre=1` to add
  the pre-norm qkv.
- The file tag `_ksol`/`_ksage` names the kernel that took the call. Sol's
  seam captures what Sol's kernel receives
  (`bench/results/2026-09-19_sol_seam_capture.md`).
- **Unverified:** whether the dense-tail blocks on the kitchen chain are
  captured at all, since they bypass Sol and go to core's backend. One short
  render checks that before the session.
- A cell at the trained canvas and full length is several GB; budget disk
  before a session.

**Gotchas that shape the runs:**
- **Waiting on a run:** wait on the runner's pid, never a `pgrep -f` match.
  Two runs interleaved on 2026-09-26 that way.
- **Determinism across processes:**
  - FastH3: confirmed.
  - FlashGen: only with the same load order.
  - PDD on the exact branch: unchecked. Test 0 covers it.
- **FastH3's contract graph never calls the dense backend,** so it is
  useless for tests of the backend or the quantizer.
- **Balanced arms keep token routing off** until the fix in bug #1.

What each result can claim:
- Tests 0 and 1 are exact for what they ran.
- Test 2 is exact on the captured cells and inference elsewhere.
- Test 4 is a verdict with a sample size, never certainty.

## Bugs and issues found, by reach into `workflows/`

Every entry is from the audit. The workflow census is the one in audit §1:
131 Sol nodes in 119 graphs, read with json over `h3_config.graph_paths`.

| # | Issue | Reach in shipped workflows | Status |
|---|---|---|---|
| 1 | **Kernel:** `qk_balance` + `token_aug` score in different spaces (the token stage's centroid is unbalanced, the keys are balanced) | None today: token routing is off in all 131. Live for the "all blocks" preset, custom routing under the shipped `qk_balance=True`, and every balanced token arm in `2026-09-15_sol_token_aug_x_options_b49_s15.json`. | open, fix planned |
| 2 | **Kernel:** `sol_attn_chunked` with `rotate` and `topk_ratio` gets a threshold mixing two spaces | None: core never passes `rotate`, and ours never uses the chunked entry | open, fix with #1 |
| 3 | **Kernel (possible):** on the chunked entry, exact K is centred on the last step's mean and the tail on this call's | Core's H3 path only, and the VSA probe graphs that run core's node | unmeasured |
| 4 | **Node:** installs its override once, so an attention node placed after it silently takes over | None: every shipped graph puts Sol last. Any hand-built graph can break it. | fix in the redesign |
| 5 | **Node:** patches `PackedLayout.__init__` and `model._forward` process-wide whenever the sink is on | Every shipped Sol graph | fix in the redesign |
| 6 | **Node:** a kernel exception becomes a logged dense render | Any graph, whenever the kernel fails | fix in the redesign |
| 7 | **Node:** the "all blocks" gate requires `rotate` as well as `qk_balance`, but the record supports needing balance only | None (off everywhere) | re-derive after #1 |
| 8 | **Node:** default `qk_balance=False` against True shipped | 129 graphs deviate from the node default | test 1 decides |
| 9 | **Core:** `block_index` is never cleared, so refiner calls see a stale index | Output-neutral while `min_tokens` exceeds the refiner length | recorded, not ours to fix |
| 10 | **Prose** that lost to code (audit §5.4) | n/a | correct with the redesign |

Anything the tests turn up is added here with its reach.
