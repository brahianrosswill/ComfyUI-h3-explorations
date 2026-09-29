# FastH3's int8 token refiner against fl2va's bf16 one (2026-09-29)

fl2va stores its token refiner in bf16 and FastH3 in int8 convrot, so the
FastH3 overlay (`2026-09-29_fasth3_overlay_exact.md`) carries the refiner
whole. Is that piece a learned change, or fl2va's refiner in another format?
`bench/analyze_refiner_requant.py`, CPU only, per tensor in
`2026-09-29_refiner_requant.json`. One checkpoint pair, no render.

## What it shows

- **It is not fl2va's refiner requantized exactly.** With the format FastH3's
  file declares, fl2va's bf16 weights give codes that match FastH3's for most
  positions but not all (`codes_equal_fraction`), up to two steps apart
  (`max_abs_code_diff`), with row scales that differ slightly
  (`scale_median_rel_diff`, `scale_max_rel_diff`).
- **It is close.** FastH3's refiner sits about as far from fl2va's bf16 weights
  as fl2va's own requantization does (`err_fasth3` against `err_requant`), so
  most of the piece is format, not change.
- **A small real difference remains, whose cause this cannot tell.** The
  distance between the two dequantized weights (`err_between`) is well above
  the control's (`err_control`, FastH3's own weights through the same bf16 round
  trip). That fits a small learned change in the refiner or a different source
  precision before FastH3 was quantized; nothing here separates them.

## What it does not show

- Whether the refiner's precision or its small change is visible in a render.
  Nothing here rendered, and on the int8 W8A8 model small effects sit near the
  measurement floor (board: `cross-session-patterns`).
- Anything about blocks outside the refiner.

## So what

Selecting the FastH3 overlay with `refiner` off gives FastH3's backbone and
gates on fl2va's own bf16 refiner. That is a near-null change on paper, a
cheap arm to look at, and a loader selection that already loads on CPU.
