# FastH3 V2 as an exact overlay on fl2va, by piece (2026-09-29)

The owner's question: ship FastH3-based research checkpoints without a 22 GB
re-upload. `2026-09-29_fasth3_overlay_size.md` measured what an overlay would
hold. This builds it and proves it. CPU only, no render.

Code: `checkpoint_overlay.py` (the format), `bench/build_checkpoint_overlay.py`,
`bench/check_checkpoint_overlay.py`, and the node `MiniMaxH3OverlayLoader`
(`overlay_loader.py`).

Records, one per target, each with the base and target sha256, the overlay's
size and the bytes of every piece:
- `2026-09-29_overlay_build_fasth3_v2.json`
- `2026-09-29_overlay_build_fl2va_plus_gates.json`
- `2026-09-29_overlay_build_fasth3_no_gates.json`

## What was shown

Run: `check_checkpoint_overlay.py --models <models dir>`.

1. **Base plus the full overlay is FastH3 V2, tensor for tensor** (same keys,
   dtype, shape and values). The two files' sizes are in the JSON.
2. **A selection is exactly the checkpoint it names.** On the FastH3 V2
   overlay, "gates only" equals #35's `hybrid__fl2va-weights__plus-fasth3v2-gates`
   and "all blocks, refiner, adaln and the outside tensors, no gates" equals
   `hybrid__fasth3v2-weights__no-gates`. Neither hybrid file was used to build
   the FastH3 overlay.
3. **`gate_scale` 0.5 moves the gate row scales and nothing else**, and halves
   them exactly. With no gates in the selection it is an error.
4. **The check can fail.** A base with one int8 code changed does not give
   FastH3 V2, and differs in exactly that tensor; hybrid A as the base is
   refused by sha256. The synthetic half breaks a delta, a piece, a base and a
   target, and each goes red.
5. **The loader node builds the gate modules.** With `MiniMaxH3OverlayLoader`
   on CPU (no server), "gates only" gives a model whose blocks carry
   `to_gate_compress`, built by core's own model detection from the keys. A
   scratch run, not a recorded check; a render through the node is not done.

## Findings

- **The int8 code differences are not all one step.** The size page said every
  changed code moved by exactly 1; the records say `max_abs_delta` 2 (in the
  build JSON, and `max_abs_code_diff` in the measurement JSON). The overlay
  stores the true int8 difference, so nothing depends on the guess.
- **The piece sizes** are in each build JSON's `bytes_by_piece`. The gates and
  the refiner are the bulk of FastH3's overlay, the backbone code diff is the
  small part, and `fl2va_plus_gates` is the gates alone: the control in the
  size page holds.
- **Small tensors outside the gates and refiner differ too** (some blocks'
  norms, the patch and condition projections, the final layer). They sit in the
  backbone piece of their block or in `io`, which the size page did not name.

## Not shown

- A render through the loader, or that it matches FastH3 V2's clip. The tensors
  are equal, so it should; nothing here rendered.
- The node on the running server: it needs the owner's restart.
- Entropy coding, a requantized refiner diff or lower-precision gates: each
  changes what ships and needs its own check.
