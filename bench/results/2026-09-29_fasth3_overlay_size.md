# FastH3-based checkpoints as exact overlays on fl2va: how small (2026-09-29)

The owner asked how to ship FastH3-based research checkpoints without a
re-upload of FastH3's full file. Measured with
`bench/measure_checkpoint_overlay.py`, CPU only, against the base
`minimax_h3_fl2va_pruned_int8_convrot`. Records:

- `2026-09-29_overlay_fasth3v2.json`: FastH3 V2 as released, with the gates'
  rank from its bf16 source;
- `2026-09-29_overlay_fl2va_plus_gates.json`: #35's fl2va plus FastH3's gates;
- `2026-09-29_overlay_fasth3v2_no_gates.json`: #35's FastH3 without its gates.

Sizes are in the JSON; this page says what they mean.

## Findings

1. **FastH3's backbone is fl2va's, give or take one int8 step.** Across all
   200 backbone linears, a small fraction of codes differ (`changed_fraction`),
   every one of them by exactly 1 (`max_abs_code_diff`). Every row scale
   differs slightly (`scales_equal_count` is 0). So the backbone travels
   exactly as the changed positions, their sign, and the new scales
   (`overlay_bytes_*`), not as 19 GB of codes.
2. **The gates are the bulk, and not low-rank.** The 50 `to_gate_compress`
   linears exist only in FastH3 (`only_in_target_bytes`). Keeping 99% of a
   gate's energy needs most of its rank (`gate_energy_rank`), so a LoRA-style
   gate would be a lossy approximation, not a smaller copy.
3. **FastH3's token refiner is int8 where fl2va's is bf16**
   (`int8_in_target_other_dtype_in_base`), so an exact overlay carries it
   whole.
4. **The control holds.** fl2va plus gates changes no code and no scale: its
   overlay is the gates and nothing else.

## So what: the overlay each checkpoint would need

| target | what the overlay carries |
|---|---|
| FastH3 V2, exact | gates, code diff and scales, the refiner, and FastH3's adaln, time table and other small tensors |
| #35 fl2va plus gates | the gates only |
| #35 FastH3 without gates | everything in the first row but the gates |

Each is a fraction of the 22 GB file; the JSON has each part's bytes.

## What it would take (not built)

- **A builder**: target and base in, one safetensors overlay out (changed
  code positions and deltas, replaced scales, whole tensors that differ or
  are new, the base's sha256 in the metadata).
- **A CPU check that it is exact**: base plus overlay must equal the target
  tensor for tensor, as `reference/README.md` does for the FlashGen files.
  The three targets above are on disk, so no render is needed.
- **A loader node**: read the base state dict, apply the overlay, and hand it
  to core's `load_diffusion_model_state_dict`, so core's model detection sees
  the gate keys and builds the gate modules itself. It costs the CPU RAM a
  normal load already does.
- **Smaller still, each unvalidated**: entropy-code the code diff (it needs a
  decoder dependency); store FastH3's refiner as a diff against a requantized
  fl2va refiner; hold the gates at lower precision. Each changes what ships,
  so each needs its own exactness or quality check.
