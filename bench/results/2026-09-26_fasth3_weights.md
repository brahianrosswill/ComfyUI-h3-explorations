# FastH3 V2 from its bf16 weights (2026-09-26)

The owner's plan: use FastH3's real weights, not a LoRA extracted from int8
(`2026-09-26_fasth3_lora_rank.json` shows the int8 files cannot carry it).
Predictions were written first, in `2026-09-26_fasth3_weights_predictions.md`.
All of this ran on the CPU. Inputs:
- `minimax_h3_fl2va_pruned_bf16` (Comfy-Org);
- `fastvideo_fasth3_8step_v2_pruned_bf16` (FastVideo-FastH3-Comfy);
- their shipped int8 twins (`h3_config.MODELS`);
- `h3_config.PDD_FL2VA_LORA` and `FLASHGEN_R64_LORA`.

## Records

| record | tool | what |
|---|---|---|
| `2026-09-26_fasth3_weights_identity.json` | `bench/checkpoint_delta_map.py --identity --every 25` | bf16 to int8 requantisation against the shipped files, 8 modules per pair; non-backbone tensors compared across the two bf16 files |
| `2026-09-26_fasth3_weights_map.json` | `bench/checkpoint_delta_map.py --map` | all 200 backbone linears: `||dW||/||W||` for FastH3, PDD and FlashGen, and the pairwise cosines |
| `2026-09-26_fasth3_weights_adaln.json` | `bench/compare_adaln_modulation.py` | every block's modulation at all 1025 time rows: the change, at the rungs against between them, and a nearest-t' remap test |

## Findings

1. **The bf16 files are the true sources.** Requantising either one reproduces
   its shipped int8 file to rounding ties under
   `bake_pdd_checkpoint.TIES_CRITERION`. Codes differ by at most 1, on about
   1e-7 of the weights. A merge built from bf16 lands in the same regime as the
   files we run.
2. **The two files do not share a time table or an adaln basis.**
   `adaln_t_table` and every `adaln_proj` differ by several times their own
   norm. Each conversion picked its own basis, so adaln is compared on
   modulation outputs only. `rope.inv_freq` and the token-refiner norms are
   identical. Everything else trained moves slightly: the patch projections,
   the token refiner linears, the output heads, and some block norms.
3. **FastH3 barely changes the backbone.** Its median `||dW||/||W||` across
   the 200 linears is about 45 times smaller than PDD's and 4 times smaller than
   FlashGen's (the map record, `median` by kind and depth). The change is flat
   across depth and kind.
4. **FastH3 changes the timestep conditioning.** Every block's modulation moves
   about twenty times more, relative to itself, than its linears do. The
   change is smooth in t, slightly larger toward clean, and flat at the rungs.
   A nearest-t' search removes a small share of it (the adaln record's
   `remap`), so it is not a remap of time.
5. **FastH3 and FlashGen agree in the late blocks.** Their per-module cosine is
   near zero in blocks 0-29 and rises in 30-49, peaking at `blocks.43.mlp.fc1`.
   For matrices this size, chance is about 1e-4. Both are data-free
   distribution-matching distills (DMD2 and VSD) of the same teacher. PDD is
   near-orthogonal to both at every depth.

## The predictions, graded

- **P1, rung-concentrated adaln change: falsified.** The change is flat across
  the rungs in both streams. Caveat: FastVideo's converter for the pruned file
  is not public (the forward-parity audit, below). So part of finding 4
  could be conversion error rather than training. The unpruned V2 release would
  separate the two.
- **P2, identity:** the backbone half holds (finding 1). The time table half
  fails (finding 2), and the fallback it named, comparing outputs, is what ran.
- **P3, small and near-orthogonal deltas:** holds for FastH3 against PDD.
  FastH3 against FlashGen is orthogonal early but agrees late (finding 5), which
  names where a #5 sum is least likely to be awful.
- **P4, the novsa arm drops the gate branch:** a code reading, confirmed by the
  audit below.

## Forward parity, FastVideo against core (code reading, 2026-09-26)

A read-only audit compared FastVideo's H3 forward with
`comfy/ldm/minimax/model.py` for T2AV:
- rope ids and rotation;
- packing order;
- the time input (clean t in [0, 1]) and the audio time derivation;
- modulation order and rows;
- qk-norm, attention scale and velocity sign.

All match. FastH3 goes off its training contract in ComfyUI only through the
harness: the shift, the sigmas, the sampler and the VSA settings.
`h3_probe_t2v_fasth3_8step_contract` already matches all four (10/3, the rungs
via ManualSigmas, euler, keep 20% from the first step). The template arm does
not.

What core's VSA node cannot match:
- it forces the ±1 diagonal blocks exact;
- it rounds the top-k count where FastVideo takes the ceiling;
- its fused kernel is int8, and the dense rows run on kitchen int8 attention.

The trainer class named in the V2 metadata, `MiniMaxH3DMDModel`, is not in
the public checkout. So "trained through the shared forward" is inferred.

## What this means for the owner's #2-#5

**Superseded 2026-09-27 by the swap renders** (`2026-09-27_fasth3_swap.md`).
The conditioning swap is functionally near-inert, and the gates and backbone
carry FastH3. The planning below was built on the weight-norm inference and
is kept as written.


- **#3's lead arm is an adaln transplant:** FastH3's modulation on base
  weights, and the reverse. Finding 4 says that is where FastH3 lives, and a
  backbone-only transplant would move almost nothing.
- **#2's α dial** should scale the adaln and the gates. On the backbone it is
  near a no-op.
- **#5's sums** (PDD + α·FastH3) meet little interference in the backbone
  (finding 5).
- **Carrying the adaln across checkpoints** needs the modulation refitted into
  the target's basis, the way `bench/convert_pdd_lora.py` bakes PDD's adaln.

## The #3 swap checkpoints (built 2026-09-26, not yet rendered)

`bench/build_adaln_swap.py` (91db4fc2) moves the conditioning set
(`adaln_t_table` and all 51 `adaln_proj.linear` weight and bias pairs, 103
tensors) whole between the shipped int8 files, with nothing requantised.
Each output was reopened and compared with its donor byte for byte (the
103 tensors), and with its backbone on a spread of the rest. The output's
`__metadata__` names the backbone, the donor and the commit.

| file (`models/diffusion_models/`) | backbone | conditioning | asks |
|---|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot_fasth3adaln` | fl2va | FastH3 | does FastH3's conditioning alone make the base sample in 8 steps? |
| `fastvideo_fasth3_8step_v2_pruned_int8_convrot_baseadaln` | FastH3 (gates included) | fl2va | what is left of FastH3 without its conditioning? |

Both would be rendered on FastH3's contract (10/3, the rungs, euler), with
FastH3 and the base at 8 steps as the endpoints. Not queued: the follow-up
batch is at the owner's cap.

## #4 from the unpruned weights (2026-09-26)

Record: `2026-09-26_fasth3_weights_adaln_unpruned.json`, from
`bench/compare_adaln_unpruned.py`. It compares the diffusers-form fl2va
transformer (MiniMaxAI release, root `transformer/`) with FastH3 V2's
(`FastVideo/FastVideo-FastH3-8-Step-V2`, `transformer/`), both evaluated as
core evaluates a non-curve checkpoint, at the pruned tables' 1025 t.

- **Both conversions are near exact.** Each pruned file is within 2e-4 of its
  own unpruned modulation (`conv_base`, `conv_oth`). The caveat on P1 is
  gone: the 5% seen on the pruned files is training, not conversion.
- **FastH3's conditioning change is all in the time embedder.** The base's
  per-block projections on FastH3's time embedding reproduce the whole change
  (`temb_only` equals `train` in every block). FastH3's projections on the
  base's time embedding move modulation by about 1e-4 (`proj_only`), the same
  order as its backbone linears.

So FastH3 V2 is, to first order, the base with one retrained MLP (the shared
time embedder, 256 to 5376 to 2688), the 50 VSA gates, and near-zero drift
everywhere else. The `fasth3adaln` swap checkpoint above is therefore
"base plus FastH3's time embedder", and the #2 α dial reduces to blending
two time embeddings. In curve form that blend is exact without a refit: put
both tables side by side (16 columns) and the two projections side by side,
weighted 1-α and α. Core reads the table width as `time_embed_dim`
(`comfy/model_detection.py`).

## PDD on FastH3's conditioning: the #5 sidecar (built 2026-09-27, not rendered)

PDD's adaln update is baked per checkpoint into that checkpoint's curve basis
(`bench/convert_pdd_lora.py --pruned`), solved against `silu(time_embedder(t))`.
To stack PDD on FastH3's time embedder, it is re-baked against both.
`bench/build_pdd_base_shim.py` writes a minimal `--base`: the time embedder
from a diffusers-form transformer, and the fingerprint and int8 probe from the
pruned checkpoint the sidecar pairs with.

- **Control, exact.** Official MiniMaxAI fl2va time embedder, paired with the
  shipped pruned fl2va. It reproduces the shipped
  `minimax_h3_fl2va_pdd_8step_comfy` bit for bit: all 732 tensors identical,
  only provenance metadata differs. The shim method adds nothing of its own.
- **The build.** FastH3 V2's time embedder paired with
  `minimax_h3_fl2va_pruned_int8_convrot_fasth3adaln`, writing
  `models/loras/h3/minimax_h3_fl2va_fasth3temb_pdd_8step_comfy`. Adaln
  baked into 50 blocks, worst per-block reconstruction 1.67e-4 (the control's
  was 1.10e-4). The node's live-table check binds the sidecar to that
  checkpoint and refuses it on any other.

The arm this makes: fl2va backbone, FastH3's time embedder, and PDD's
backbone, adaln and heads at strength 1, on PDD8's schedule. Rendered only
after the swap arms show what FastH3's time embedder does alone.

## The #2 dial files (built 2026-09-27, not rendered)

`bench/build_adaln_blend.py` (135b449e), on FastH3's backbone and gates, with
the conditioning blended between fl2va (α = 0) and FastH3 (α = 1):
`fastvideo_fasth3_8step_v2_pruned_int8_convrot_temb_a05` and `_a075`. Each
is a 16-column table, and each is verified against the blend of the sources'
modulation at all 1025 rows (worst 1.9e-4 and 2.4e-4). Motivated by FastH3
carrying the most fine detail of the distills on 13 of 13 scenes
(`2026-09-26_distill_signatures.md`).
