# The sister checkouts: what each one is good for

last updated: 2026-09-10 (section "What moved by 2026-09-10" added; the tables are the 2026-08-28 read)

`coderef/` holds the reference implementations. `ls -l coderef/` is the list of
what is currently on disk — some symlinks, some real clones — and this page is
what each one is *for*: what it implements, what has actually been compared
against it, and what it is not evidence of.

**Written by a person. Not generated** — the generator that builds
[`index.md`](index.md) never touches this file.

**Revisions are an observation point, not a pin.** Every one below was read on
the date in the header. A sister checkout moves under you; re-read before
quoting. Two of the H3 engines moved during the 2026-08-28 comparison pass
itself.

**Do not import Python from `coderef/`.** CLAUDE.md's rule, with the escaped
instance that earns it: requiring the clone and prepending it to `sys.path` is
how a bench script made itself unrunnable on a box that had the wheel and no
checkout. Use the clone for sources you cannot import; import the rest.

---

## The four H3 implementations worth comparing against

These are the ones that implement MiniMax H3 end to end. All four were compared
against our node chain on 2026-08-28; the findings live in
[`../custom_node_gaps.md`](../custom_node_gaps.md), which this page routes to
rather than restating.

| checkout | revision read | what it is | reach for it when |
|---|---|---|---|
| `sglang` | `803b4fb31c` | **the vendor's own serving path.** The closest thing to ground truth for what MiniMax intended | you need to know what the release actually does at a stage |
| `LightX2V` | `5169278f` | inference engine; **origin of the SLA work and the Turbo LoRAs we load** | anything about SLA, DMD step distillation, offload, or what a LoRA was distilled under |
| `DiffSynth-Studio` | `102fe99` | model library with a native H3 pipeline, its own converters and a LoRA path | you need a second opinion on a state-dict namespace or a converter |
| `diffusers` | `9f7aee482` | model library with a native H3 pipeline, a named H3 scheduler, and a conversion script | you need the canonical tensor namespace, or a clean statement of the sampler |

Two owner documents already exist for the first of these and are the authority
over anything here: [`../research/sglang_h3_pipeline.md`](../research/sglang_h3_pipeline.md)
for what sglang does stage by stage, and
[`../research/sglang_comparison.md`](../research/sglang_comparison.md) for what
its serving path does that we do not.

### What the 2026-08-28 pass established about them

Recorded here because it is a property of the *references*, not of our code:

- **Three-against-one is a real signal and it fired twice.** sglang, DiffSynth
  and diffusers agree on feeding one prepared reference tensor to both towers,
  and on running the video VAE more precisely than we do. Both are open.
- **Agreement is broad and worth banking.** The reference label rules, the
  encoder layer, the absence of a chat template, and the VAE normalisation
  statistics all agree across implementations. When four implementations agree,
  a fifth reading is not the cheapest next step.
- **Neither model library is independent evidence about the seven markers.**
  Both inherit the release tokenizer without touching the ids in code. Two more
  implementations is not two more votes.
- **No engine implements PDD.** diffusers, LightX2V, DiffSynth and sglang were
  each searched. See [`../research/pdd/pdd_implementations.md`](../research/pdd/pdd_implementations.md).

---

## The ComfyUI-side references

| checkout | revision read | what it is |
|---|---|---|
| `comfy-kitchen-sol` | `bd3fc78` | **the most-cited clone here.** Its `.cu` files ship in no wheel, so `morton.md` and `sol_upstream.md` quote it by path. The built branch is installed, so import the Python rather than requiring the clone |
| `comfy-kitchen` | `7490d87` | the upstream of the above |
| `ComfyUI-UtilsCollection` | `5bac35b` | a third-party pack with its own PDD path. Two of our guards were **adopted from it** |
| `Minimax-H3-Turbo` | `02e26d5` | the vendor README that publishes the distilled sigma grid `bench/check_distill_grid.py` grades against — a grid from the vendor, not one we computed |
| `sage-fork` | `56a5be4` | our SageAttention fork |
| `SLA` | `7db4039` | the sparse top-k attention reference |
| `TurboDiffusion` | `e3d6136` | step-distillation reference |

---

## The upstream and infrastructure clones

Not H3 implementations. Listed so nobody mistakes one for a comparison target.

| checkout | what it is | what it is not |
|---|---|---|
| `MiniMax-H3` | the release repository | not a runnable pipeline for our purposes |
| `MiniMax-Music3` | a different model | not H3 |
| `transformers`, `vllm`, `llm-compressor` | encoder-side and quantisation infrastructure | say nothing about the DiT |
| `triton`, `flashinfer`, `nanobind` | kernel infrastructure | |
| `Sana`, `h3-turbo-eval` | adjacent research | |

---

## What moved by 2026-09-10

Read on 2026-09-10 by fetch, at the revisions named here; the tables above keep
their 2026-08-28 revisions. Sol-side movement (Sana's Sol-H3, sglang's SubBlock
work) lives in [`../sol_upstream.md`](../sol_upstream.md).

- **`vllm-omni`** (`ffcaaa943`; an H3 serving engine not in the tables above).
  `af74a5a15` (#7062) derives a LightX2V Turbo file's sampler contract from its
  filename alone -- 768p files at video shift 6, 544p at 12, audio 3 -- and
  scales by the file's declared alpha over rank, falling back to alpha 8 only
  for a file that declares none (`vllm_omni/diffusion/models/minimax_h3/lora.py`).
  That matches ComfyUI's alpha/rank scaling, confirmed on the v1.1 and v1.2 768p
  files, whose `.alpha` tensors carry the declared value. By its rule the two
  8-step v1.0 768p files on disk (fl2v and ref2v) would run at 6/3;
  `bench/check_distill_settings.py` deliberately classifies neither, and no
  shipped graph loads them. `30d6a0b4e` (#7191) pins cuDNN flags around the
  keyframe encode: [`../open_experiments.md`](../open_experiments.md) #30.
  `715b8b874` (#6720) validates the text-conditioning handoff and changes no
  numerics.
- **`LightX2V`** (`fabad304`). `95e9b86b` (#1503) adds a persistent AdaLN cache,
  an offline-built table keyed on the exact float32 bits of each timestep and
  enabled across its H3 DMD configs. `35d4aaa8` (#1464) wraps kitchen's
  `int8_linear` in a `torch.library.custom_op` so torch.compile can trace INT8
  ConvRot; core calls it unwrapped (`comfy/ops.py`), and this repo never
  compiles. `e1088278` (#1506) adds optional FP8 Conv3D modes to the video VAE
  encoder, off by default. `fabad304` (#1511) redefined H3 `infer_steps` from
  sigma grid points to evaluations, with the same behaviour;
  `bench/check_distill_settings.py` reads those configs.
- **`flashinfer`**: `4fa42525` adds a MiniMax-H3 MXFP8 pre-attention kernel for
  SM100a/SM103a and `01587699` documents H3 run parameters. Neither runs on this
  card.
- **`DiffSynth-Studio`**: `ce9f454` (#1678) adds an optional H3 training
  adapter. **`diffusers`**: `d30c748f5` touches H3 LoRA tests only.
- **`Minimax-H3-Turbo`**: the clone is still at `02e26d5`. Its README has rows
  for the v1.0 768p 4-step and 8-step files only, and neither it nor the HF
  model README carries a card for v1.1 or v1.2. The HF repo added a v1.1 768p
  fp8 file on 2026-09-10.
- **`ComfyUI-UtilsCollection`** (`d6a9600`). H3 reference nodes landed
  2026-09-05 to 09-09: save, load and apply VAE-encoded reference latents; a
  reference-video component that resamples to H3's frame rate and the audio
  VAE's sample rate and pads its audio's end to the audio VAE's hop; a media config whose "even keyframes" mode
  places reference-video chunks as keyframes; a "VLM guide" that splices a
  separately encoded Qwen forward in front of the prompt text; and an encode
  cache keyed on the encoder object, its patches and the token bytes, which
  refuses a mismatched file and lives in ComfyUI's temp directory, so it does
  not outlast a restart. Its `d1921ae` reverted per-section encoding after
  reported generation distortion and caches the joint encode only.
- **Hugging Face, not cloned**: community FastH3 conversions (NVFP4 rotated,
  GGUF, a dense-datafree ComfyUI file); `junchaoh-cs/SolarWM-H3-33B` is gated
  and was not read.

---

## The trap this page exists to prevent

**Two models live in this repo, and the words for their parts do not
disambiguate the stage.** "Attention" and "capture" each name something at the
DiT *and* at the Qwen3-VL encoder. A fact about one is not a fact about the
other, and three instances of carrying a DiT-side fact to an encoder-side
conclusion happened in a single day. CLAUDE.md holds the full rule; the tell is
always a type or a module prefix, never the vocabulary of the claim.

This applies with extra force to the sister checkouts, because a clone gives you
a confident, well-written source for the wrong stage.
