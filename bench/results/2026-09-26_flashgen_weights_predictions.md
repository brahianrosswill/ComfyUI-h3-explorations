# FlashGen's weights: predictions before looking

Written 2026-09-26, before any number below was computed. The owner asked,
through the VAE session, for FlashGen to get the same weight-level look as
FastH3's full weights (`2026-09-26_fasth3_weights_predictions.md`). CPU only.

**What is known before looking**, from the file's keys alone:
- The source (`minimax_h3_4step_lora_flashgen_v1.0_768p_bf16.safetensors`) is
  a PEFT LoRA at a stored rank of 64 on 259 modules: every block's qkv, out,
  fc1, fc2 and adaln, the refiner's 8, and the final layer's adaln.
- Its adaln LoRA reads the unpruned 2688-dim time embedding. Our conversion
  (`bench/convert_flashgen_lora.py`) re-expresses it on the pruned
  checkpoint's 8-curve basis.
- kijai's resize (sv_fro 0.95) came out at an average rank of 13
  (`h3_config.FLASHGEN_LORA`).

## Predictions

- **FP1, rank.** The effective rank is far below the stored 64.
  - 95% of each module's Frobenius energy sits in about 13 singular values on
    average, consistent with kijai's resize, and 99% in under 40.
  - It is uneven by kind: qkv needs more rank than the MLP, and late blocks
    more than early ones.
  - The rank-64 conversion keeps the source's non-adaln modules exactly.
    Its adaln is exact on the curve manifold to within a few percent.
- **FP2, change map.** FlashGen's delta is small everywhere (per-module
  ||dW||/||W|| around 1e-3, as the requant probe found at block 0). It grows
  toward the late blocks, as PDD's does.
  - Its per-module cosine with PDD's delta is near zero (a different
    objective, trajectory against distribution).
  - Its cosine with FastH3's is small but above PDD's, since both are
    distribution matching.
  - Both cosines stay under 0.1.
- **FP3, adaln in t.** FlashGen's modulation change is smooth in t, with no
  structure at its four training sigmas.
  - This follows from the basis: on the pruned checkpoint, modulation is a
    linear combination of 8 smooth curves, so it cannot hold a bump localised
    at 4 rungs. FastH3 showed the same, which falsified the VAE session's P1.
  - The change is largest over the high-noise band it was supervised in
    (t from 0.68 to 1), in relative terms against the base's modulation.
  - Below 0.6, where FlashGen never ran, the change is an extrapolation of
    the same curves, not zero.
  - That matters for the reverse step-switch: FlashGen finishing from 0.632
    sits just below its lowest training sigma, 0.679.
- **FP4, the dial.** Scaling FlashGen's delta by alpha below 1 moves the
  grade toward the base's. The 0.8 and 1.2 strength arms in the follow-up
  batch are this dial at two points.

Each gets a dated verdict line when its number is in.

## Verdicts, 2026-09-26

The numbers are in `2026-09-26_flashgen_weights.json`, from
`bench/analyze_flashgen_weights.py`.

- **FP1, rank: held on the headline, failed on both shapes.**
  - **Held:** the effective rank is far below 64. 95% of the energy sits in
    18 singular values on average (predicted about 13) and 99% in 31
    (predicted under 40). The conversion keeps every non-adaln module
    exactly: singular values agree to 5e-16.
  - **Failed, by kind:** fc2 needs the most rank, not qkv.
  - **Failed, by depth, and the wrong way round:** blocks 34 to 49 are nearly
    rank 2, against about 30 in blocks 0 to 16. FlashGen's substantive change
    is in the early blocks, where content and layout form, and the late
    blocks carry a very low-dimensional adjustment.
  - The source's adaln LoRA is rank 2 to 3 per block.
- **FP3, adaln in t: held.**
  - The change is smooth in t, with no structure at the four training
    sigmas.
  - It is slightly larger in the trained band than below 0.6.
  - It is tiny: about a tenth of a percent of the base's modulation, where
    the VAE session measured FastH3 at about 5%. FlashGen barely touches
    timestep conditioning.
  - So the follow-up batch's `flashgen_noadaln` arm will likely look almost
    like full FlashGen, and Claude's P3 in the run predictions ("without
    adaln: less contrast") is probably wrong. That is inference until the arm
    lands.
- **FP2, change map: held on size and order, failed on the ceiling.** From
  the VAE session's map (`2026-09-26_fasth3_weights_map.json`, all 200
  backbone linears, at strength 1).
  - FlashGen's median ||dW||/||W|| is 4.5e-4, against PDD's 4.8e-3 and
    FastH3's 1.1e-4.
  - **Held:** its median cosine with PDD's delta is near zero (1.4e-3).
  - **Held:** its cosine with FastH3's is above PDD's (median 9.0e-3).
  - **Failed:** "both under 0.1" does not hold in the late blocks.
    FlashGen·FastH3 climbs to 0.043 in blocks 30-39 and 0.099 in 40-49,
    peaking at 0.162 (blocks.43.mlp.fc1). Chance for matrices this size is
    about 1e-4.
  - **Inference, the finding worth testing:** two data-free
    distribution-matching distills of one teacher share a direction in the
    late blocks, exactly where FlashGen's change is nearly rank 2. That
    shared direction is a candidate for the look both carry: a bolder grade
    than the base's. The block transplant below is the test: if the grade
    lives there, FlashGen on blocks 0-33 alone keeps its motion and loses the
    grade shift.
- **FP4** waits for the strength arms.
- **Suggested by the rank map, not yet run:** a block transplant through
  `MiniMaxH3LoRABranch.blocks`, FlashGen on 0-33 only against 34-49 only. It
  asks whether FlashGen's look and its adherence loss live where its
  high-rank change is.
