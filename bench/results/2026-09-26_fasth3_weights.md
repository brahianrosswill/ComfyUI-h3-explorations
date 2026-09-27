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

- **#3's lead arm is an adaln transplant:** FastH3's modulation on base
  weights, and the reverse. Finding 4 says that is where FastH3 lives, and a
  backbone-only transplant would move almost nothing.
- **#2's α dial** should scale the adaln and the gates. On the backbone it is
  near a no-op.
- **#5's sums** (PDD + α·FastH3) meet little interference in the backbone
  (finding 5).
- **Carrying the adaln across checkpoints** needs the modulation refitted into
  the target's basis, the way `bench/convert_pdd_lora.py` bakes PDD's adaln.
