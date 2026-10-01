# PDMD: what kijai's two LoRAs are, how they were made, and their closest sibling

2026-10-01. Kijai published two rank-reduced H3 distill LoRAs,
`minimax_h3_pdmd_4step_lora_avg_rank_57_bf16` and
`minimax_h3_pdmd_2step_lora_avg_rank_38_bf16` (HF
`Kijai/MiniMax-H3-experimental`), "converted and rank reduced from"
`pdmd2026/pdmd_2NFE_lora` and `pdmd2026/pdmd_4NFE_lora`. The owner asked what
they are, how they were made and which distill here they are closest to.

Sources read:
- the paper, `internal/refs/pdmd/2609.35768v1.pdf` (arXiv 2609.35768v1);
- the trainer's repo, `coderef/pdmd` at `03ee66b`;
- the three HF model cards and the `lora_model_0.safetensors.json` sidecars;
- the published LoRA files themselves, downloaded and checked against HF's
  LFS sha256.

Nothing was rendered. Every number lives in a record under `bench/results/`
or in the paper, and is pointed at, not copied. The visual version,
[`pdmd_on_h3.html`](pdmd_on_h3.html), is published as a claude.ai artifact and
charts the records directly. The board's `pdmd-lora` direction and findings
`mo-01` to `mo-07` track the work.

## In one paragraph

PDMD is a few-step student of the **base H3 `transformer/`, the FL2VA/T2VA
partition**, trained as a LoRA with the DMD objective plus a single projection
step. It samples with **plain Euler at the base model's shifts and ComfyUI's
own `simple` schedule**, with no CFG. Kijai's files have the right layout and
keep most of each delta, uniformly across modules. Like FlashGen, they must be
applied **at the call, not merged** into our int8 checkpoint. The closest
sibling here is **FlashGen**.

## What the method is

The paper's citations, checked by a subagent reading the whole PDF:
- **The DMD part (Sec 3.1, Eq. 1-2).** The student endpoint is re-noised. The
  update is `d = s_critic − s_teacher`, backpropagated through the student. The
  critic is an online fake-score model, trained on re-noised student endpoints
  with a denoising loss.
- **The projection (Sec 3.2, Eq. 3-4; Algorithm 1).** Let
  `r = x0_critic − x0_student`, the residual between the critic's and the
  student's endpoints. The update becomes `d ← d − (⟨d,r⟩/‖r‖²)·r`. That line
  is the whole change from DMD. On H3 it is applied per sample and separately
  to the video and audio latents (App. C.2).
- **What it does not add:** no GAN, no regression loss, no extra network, no
  extra model pass (p.4, p.10). The paper's DMD† baseline is PDMD without the
  projection line.
- **Two time scales on H3:** several critic updates per student update
  (App. C.2, Table C1).

## How the H3 students were trained

- **Teacher:** frozen H3. Because it is guidance-distilled, it is sampled
  without CFG (App. C.2; Table C1 lists teacher guidance "none").
- **Student and critic:** both LoRAs, on the attention projections and both
  FFN layers of every block and token-refiner block (App. C.2).
  - No adaLN, no `proj_in`/`proj_out`. The sidecar key lists confirm this.
  - Rank, alpha and scale are in the sidecar headers (`lora_rank`,
    `lora_alpha`, `lora_scale`).
- **Data:** prompts only, from VidProM, with no real video (Table C1).
- **Training clip:** 5 s at 544p (Table C1). The paper's eval renders are also
  544p (App. C.2).
- **Which checkpoints shipped:**
  - **4-step:** `global_step_3500` of run `pdmd-zwang000` (sidecar `source` and
    `tag`). This is the checkpoint the paper reports (App. C.2).
  - **2-step:** step 4000 of a "3-role fp32 ckpt", extracted by
    `extract_lora_dcp.py`. Its `base_check` says the base weights were
    untouched, so this is a trained LoRA, not one extracted from a full
    finetune.
- **`pdmd_4NFE_full`:** a 66 GB drop-in `transformer/` that its card calls the
  same model as the 4-step LoRA. Not downloaded.

## The sampling contract

`coderef/pdmd/worker/run_a10.py` and `run_a100.py` run diffusers'
`ModularPipeline`:
- `VIDEO_SHIFT`/`AUDIO_SHIFT` at the base model's values;
- `num_inference_steps = steps + 1`, because the scheduler counts the terminal
  zero as a grid point;
- no guidance;
- t2va only. `load_jobs` refuses other tasks, because image or reference
  conditioning "would be silently dropped".

The scheduler is `MiniMaxH3Scheduler` at the diffusers commit the README pins
(`coderef/diffusers`, `e0abab83`, `src/diffusers/schedulers/scheduling_minimax_h3.py`):
- the grid is `shift(linspace(1, 0, N + 1))`;
- the step is Euler at eta 0, with no re-noising.

**In ComfyUI terms** this is exactly:
- `DISTILL_SAMPLING` (euler, `simple`);
- `SIGMA_SHIFT` (`workflows/h3_config.py`);
- 4 or 2 steps.

ComfyUI's `simple` scheduler at shift 12 gives the same sigmas bit for bit at
both step counts. To re-derive it, compare
`comfy.samplers.calculate_sigmas(ModelSamplingAV+CONST at 12/3, "simple", n)`
with the scheduler's formula for n = 2 and n = 4. The 4-step grid is also
PDD4's block grid, the uniform 4-evaluation schedule that
`h3_config.PDD_MANUAL_SIGMAS` was built to improve on. So PDMD needs no
`ManualSigmas`, unlike FlashGen (`FLASHGEN_MANUAL_SIGMAS`).

## Kijai's conversion

From the file metadata:
- **Layout:** q/k/v fused block-diagonally into `attn.qkv_proj`, `mlp.fc1`
  SwiGLU halves swapped, alpha folded into `lora_B`, scale 1.0.
- **Resize:** dynamic SVD, `ss_training_comment` "sv_fro: 0.97 from 384",
  capped at rank 128. 384 is the rank of the fused q/k/v.

`bench/measure_pdmd_lora_conversion.py` checks both parts against the
published files. The records are
`bench/results/2026-10-01_pdmd_4step_lora_conversion.json` and
`bench/results/2026-10-01_pdmd_2step_lora_conversion.json`.

- **`transformer/` is the fl2va partition, exactly.** Its renamed modules are
  bit-identical to the release's `FL2VA/transformer` and differ from
  `Ref2VA/transformer` (`partition_of_diffusers_transformer`). This settles
  the README's "FL2VA/T2VA partition" claim on bits.
- **The layout is right.**
  - The release's diffusers weights, mapped kijai's way, match the pruned
    fl2va checkpoint row for row, at int8 dequantisation noise.
  - `fc1` *unswapped* does not match, and the unswapped mapping of the LoRA
    delta does not match kijai's either (`layout_on_base_weights`,
    `fc1_unswapped_control_cosine`).
  - The ref2va checkpoint is a weak control here: the partitions are close, so
    its row cosines sit only slightly below fl2va's.
  - This is the same q/k/v and `fc1` convention `bench/convert_pdd_lora.py`
    uses for PDD's diffusers-side files.
- **The resize keeps a fixed fraction of every delta, and loses the tail.**
  - Kijai's delta is a strict sub-delta of the published one: its norm ratio
    equals its cosine. So what is lost is the tail directions, not a rotation.
  - Away from the rank cap, every module sits at the fraction `sv_fro` asks
    for.
  - The q/k/v modules that hit the rank-128 cap lose more. The 4-step file has
    more of them than the 2-step, all in the first two-thirds of the stack
    (`fidelity_by_kind`, `fidelity_worst_modules`, `at_rank_cap`).
  - Whether that is visible in a render is untested. A full-rank conversion is
    the arm that would say (below).

## Loading on our checkpoint: at the call, not merged

`bench/probe_int8_lora_requant.py` ran on kijai's two files and FlashGen's,
against `minimax_h3_fl2va_pruned_int8_convrot`
(`bench/results/2026-10-01_pdmd_int8_lora_requant.json`). It ran on CPU, over
the probe's own `LAYERS`.

- PDMD's per-layer deltas are larger than FlashGen's, but below one int8 step
  in every layer sampled (`delta_over_step`). The last block comes closest,
  and only there does much of the delta survive a merge.
- So a `LoraLoaderModelOnly` merge would lose most of PDMD to requantisation
  noise, as it does FlashGen (`docs/research/2026-09-26_flashgen.md`, "What we
  found").
- **PDMD needs `MiniMaxH3LoRABranch`**, the route `h3_text_to_video_flashgen`
  already uses.
- **Both at-the-call loaders take the files as they are.** `lora_branch.py`'s
  `MiniMaxH3LoRABranch` and the mutant pack's `H3ExactLoRA`
  (`standalone/h3_mutant_distill/exact_lora.py`) were each run on both files,
  on CPU. Each loader's `parse_lora` places every module on a module of the
  pruned fl2va checkpoint, at scale alpha / rank = 1, with no `diff_b`. `fc2`
  takes the MLP path, as FlashGen's does. This is a parse and placement check,
  not a render.
- `MiniMaxH3OverlayLoader` (`overlay_loader.py`) is not a LoRA loader. It
  loads a full research checkpoint stored as per-piece diffs on the released
  one, which is how FastH3 V2 ships here. It would matter for PDMD only if
  `pdmd_4NFE_full` became an overlay. Its backbone diff is stored as int8
  codes, so the same requantisation question would apply to it.

The probe measured kijai's reduced deltas, which are slightly smaller than the
published ones, so the full-rank files would fare no better.

**No adaLN in the LoRA** means the pruned checkpoint's adaLN time-basis trap
(`docs/h3_pdd.md`; the FlashGen converter's adaLN refit) does not arise.
Nothing needs re-expressing in `adaln_t_table`'s basis.

## Closest sibling: FlashGen

| | PDMD | FlashGen | FastH3 V2 | PDD |
|---|---|---|---|---|
| objective | DMD + one-line projection, no GAN | VSD (its merge script says "DMD2 student"), no GAN | DMD2, with VSA sparse attention | trajectory: fused per-interval heads |
| teacher target | distribution | distribution | distribution | the teacher's path |
| data | prompts only | data-free | per FastVideo | per alibaba-pai |
| form | LoRA on attn + FFN | LoRA on attn + FFN + adaLN | full weights | LoRA + head bank + adaLN |
| base | `transformer/` = fl2va | fl2va (and a ref2va file) | own checkpoint | fl2va and ref2va files |
| schedule | `simple` at `SIGMA_SHIFT` | `FLASHGEN_MANUAL_SIGMAS` at `SIGMA_SHIFT` | `FASTH3_CONTRACT_SIGMAS` at `FASTH3_SHIFT` | the 32-point grid, fused |
| sampler | Euler | Euler | Euler (contract) | Euler |
| training clip | 544p, 5 s | 1344x768, 5.2 s | per FastVideo | per alibaba-pai |
| loading here | at the call | at the call | checkpoint | `MiniMaxH3PDDLoRA` |

FlashGen matches PDMD on every axis that decides how it is run and loaded:
- a distribution-matching objective with a learned critic and no GAN;
- trained without real data;
- a 4-step LoRA on the same fl2va base;
- Euler, no CFG, the base shifts;
- a short training clip;
- kijai's dynamic-resize lineage;
- applied at the call.

It differs on three things: the step schedule (a trained list against plain
`simple`), adaLN (FlashGen trains it, PDMD does not), and the training
resolution.

FastH3 V2 shares the DMD family name but differs on GAN, sparse attention,
full weights, step count and shift. PDD is a different kind of distill
altogether.

Inside the paper, the nearest relative is the **DMD† baseline**: the same
recipe without the projection. Its Table 2 also scores the larryvrh Turbo LoRA
(v4 step-600 EMA), which is this repo's retired
`minimax_h3_turbo_v4_step600_ema` (`docs/evidence.md`). It is history only;
the turbo lane is closed (`docs/roadmap.md`, "Closed lanes"). The paper does
not compare PDD, FlashGen or FastH3.

## What the paper claims, by pointer

- 4-step against the teacher and the baselines on VideoGen-Eval, video and
  audio: Table 2.
- The user study: Table 2 and Table C4. The 50-step teacher is still
  preferred overall, which the paper names as a metric limit (Sec 5.3, App. F).
- Saturation: Table E1. Lower than the DMD and DMD2 baselines, which matters
  here given the warm, contrasty grade FastH3 shows in `docs/h3_distills.md`.
  E.2 warns that lower saturation alone is not quality.
- 2-step against its DMD twin: Table D2. The paper says the 2-step student is
  slightly worse on motion and texture than the 4-step one (Fig. D5).
- Stated limits: 1 step smears, the projection can remove useful signal
  aligned with `r`, only two backbones were tested, and diversity is
  unmeasured (Sec 7, App. F).

## What does not line up

- **The released 2-step is not the paper's scored 2-step.** Table D2 reports
  PDMD at an earlier iteration of the 2-step run than the released file's
  `tag` (step 4000).
- **Critic:student ratio.** The 2-step card says "6:1"; the paper's H3 setup
  (App. C.2, Table C1) gives a different count. Possibly the same schedule
  counted two ways; unverified.
- **Partition.** The paper never names one. The README says FL2VA/T2VA, and
  `partition_of_diffusers_transformer` confirms it on bits.
- **Training canvas and length against ours.** Trained at 544p for 5 s; we
  render at 1344x768 with 345 frames (`docs/h3_resolutions.md`). The README's
  own example jobs render at our canvas and length, so the trainer does run it
  there, but the paper's scores do not cover it. FlashGen has the same
  length gap.

## Open, in order

1. **The graphs: built the same day, and rendered once for the first look (item 3).**
   `workflows/h3_text_to_video_pdmd_api.json` (4-step, full rank) ships at
   the root. `distill_experiments/h3_probe_t2v_pdmd_2step` and the two
   `h3_probe_t2v_pdmd_kijai_*` resize arms are probes. The constants are
   `h3_config.PDMD_*`, and `bench/check_distill_settings.py` grades every PDMD
   graph against the contract below. As planned:
   - FlashGen's t2v graph with the LoRA name swapped;
   - `ManualSigmas` replaced by the `simple` scheduler at 4 and 2 steps;
   - `MiniMaxH3LoRABranch` at strength 1.0, the published `lora_scale`.

   Constants go in `workflows/h3_config.py` beside FlashGen's, with
   provenance. `bench/check_distill_settings.py` must recognise the files,
   as `classify_flashgen` does FlashGen's, so that a graph at the wrong shift
   or step count goes red.
2. **Full-rank ComfyUI conversions: built the same day.**
   `bench/convert_pdmd_lora.py` writes `minimax_h3_pdmd_{4,2}step_rank128_comfy`
   from the published files, with the mapping this note verified and no
   resize, as `FLASHGEN_R64_LORA` is for FlashGen.
   - Every module's change equals the published one, and the converter
     refuses anything less (records:
     `bench/results/2026-10-01_pdmd_{4,2}step_rank128_conversion.json`).
   - Both at-the-call loaders place them at scale 1.0.
   - The fused q/k/v is block-diagonal at three times the published rank, so
     its branch does more arithmetic than kijai's. Per
     `bench/results/2026-10-01_lora_branch_profile.md` the branch is mostly
     memory traffic, so the cost should be small, but it is unmeasured.
   - These are the default for a PDMD graph. Kijai's files are the arm that
     says whether the resize shows.
3. **A blind comparison against FlashGen**, the sibling, through the
   `h3-ab-session` process. It must be judged on content-independent axes:
   seed-matched clips from two distills are different scenes
   (`docs/h3_distills.md`, the box at the top). **Rendered and blinded
   2026-10-01, waiting on the owner's scores.**
   - Five owner-picked scenes: three t2va, ref2va and i2va. Each scene has
     three arms: full-rank PDMD, kijai's resize and FlashGen. Every arm ran at
     Sol `start_percent` 0.2.
   - Manifest: `bench/pdmd_vs_flashgen_arms.json`. Rows:
     `bench/results/2026-10-01_pdmd_vs_flashgen.jsonl`.
   - Blind session `2026-10-01_pdmd_vs_flashgen`, scored on the pairs-only
     page. Eight of ten pairs were scored and joined with `--partial`
     (`bench/results/2026-10-01_2026-10-01_pdmd_vs_flashgen_verdict.json`). The
     owner's reading of the other two is pending.
   - Sol `start_percent` 0.2 on every arm is the FlashGen and PDMD default
     again since 0.184.3 (0.184.1 had moved it to 0.0 and was reverted), so a
     later render at today's defaults repeats these settings.
4. **Length.** The README's 14 s example at our canvas is a ready first scene
   (`coderef/pdmd/jobs/giant_cat_harbor_768p_4nfe.json`).
