# Sage fp8++ rotated against kitchen int8 on the steps before Sol's window (2026-10-01)

The accuracy half of the sage-as-dense-fallback question the owner reopened on 2026-10-01. The
speed half is `2026-10-01_sage_floor_timing.md` (sage a little over one percent faster on the
sampler). Kitchen already led on every dense-tail cell inside Sol's window
(`2026-09-27_sol_redesign_test2.md`); no capture had covered the two steps before the window,
which every block runs dense.

**Result: kitchen int8 is closer to exact fp32 attention than sage fp8++ rotated on all 14
cells, the early-step blocks and the dense tail alike, over all rows and over the video and
audio rows.** Sage rotated beats unrotated sage everywhere, most on the dense-tail blocks 45 and
49, but not kitchen. bf16 SDPA, the floor, is below both on every cell.

## How

- **Capture:** `bench/sage_floor_capture_arms.json` on a server armed with
  `blocks=0:8:24:40:45:48:49,steps=0:1,cycle=8`: the shipped t2v finish
  (`h3_text_to_video_pdd8_flashgen_finish`), `t2va_slapstick_moving_piano`, 345 frames, seed
  730451892, kitchen `0.2.36+sol.aade8d5.up.3f7210f`. Blocks 0 to 40 were captured on the
  `outside_range` route (dense before Sol's window), 45 to 49 on `dense_block`. The capture lives
  under `h3_captures/2026-10-01_dense_early_steps` with its `manifest.json` (validated by
  `check_capture_manifest.py`) and `retention.json` (keep until 2026-10-31). The runner's JSONL
  row was lost when its process was interrupted; the render's `/history` entry
  (`454f83a3-6607-477b-a657-7191a935f06e`) is the record that it ran, and the manifest joins it.
- **Grade:** `grade_dense_kernels_on_captures.py <capture> --heads 56 --json
  2026-10-01_dense_early_steps_grade.json`: relative L2 against fp32 dense on sampled rows,
  every head.

## Two fixes on the way

- `h3_capture.py` stamped `comfy_args.fast` as a Python set whenever the launcher passed no
  `--fast`, which it has not since 2026-09-28; JSON cannot hold a set, so no capture since then
  could get a manifest. The stamp now writes a list.
- `bench/generate_capture_manifest.py` reads records stamped before that fix by writing a set as
  a sorted list.

## What follows

Neither half makes a case for sage: it loses accuracy on every dense cell graded so far and wins
about one percent of sampler time. Kitchen int8 stays Sol's dense fallback
(`h3_config.DENSE_BACKEND_NODE`); nothing changes. If the `start_percent` panel moves the default
to 0.0, the early steps stop being dense and only the tail is left, where kitchen also leads.
