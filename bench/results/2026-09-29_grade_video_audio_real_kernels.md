# Block 49's grade over the rows the model reads, on the real kernels (2026-09-29)

`bench/grade_dense_kernels_on_captures.py`, which now prints a second line per cell
over the video and audio query rows (the segments the final layer reads, from the
capture manifest's segment table), on the two step-2 captures that
`2026-09-27_sol_redesign_test2.md` graded: `2026-09-27_sol_test2_pdd8_t2v` and
`_ref2va`. Blocks 24 and 49, step 2, three kernels against fp32 dense attention on
the same bf16 inputs. CPU-free: a GPU job with ComfyUI stopped, no render.
Results: `2026-09-29_grade_video_audio_{t2v,ref2va}_h56.json` (all 56 heads).
`2026-09-29_grade_video_audio_{t2v,ref2va}.json` are the same run at the grader's
default of 8 heads and are not comparable; see "The 8-head run" below.

## Result, all 56 heads, block 49 step 2

| capture and rows | kitchen int8 | sage fp8++ | sage fp8++ rotated |
|---|---|---|---|
| t2v, all rows | 0.0140 | 0.0494 | 0.0210 |
| t2v, video and audio rows | 0.0136 | 0.0473 | 0.0207 |
| ref2va, all rows | 0.0367 | 0.1458 | 0.0405 |
| ref2va, video and audio rows | 0.0148 | 0.0559 | 0.0237 |
| **ref2va over t2v, all rows** | 2.61x | 2.95x | 1.93x |
| **ref2va over t2v, video and audio rows** | 1.09x | 1.18x | 1.15x |

Block 24 is unchanged by the row selection in both captures (all-row and
video-and-audio grades within 0.0002, ref2va over t2v about 0.95 to 0.96 on
either line).

## Reading

1. **The all-row numbers reproduce the 2026-09-27 record exactly** (ref2va block 49:
   0.0367, 0.1458, 0.0405), so the run measures the same cell.
2. **Over the rows the model reads, ref2va's block 49 is about as hard as t2v's**:
   1.1 to 1.2 times, against 1.9 to 3.0 times over every row. The selection removes
   about 14% of ref2va's query rows (text 7.0%, reference image 6.8% of 120,582;
   t2v's text is 0.6% of 104,565) and takes 60% of its kitchen error with it. On
   t2v it changes the grade by under 5%. So the "hardest cell" reading was mostly
   the grade counting rows whose last-block output is never read, as
   `2026-09-29_ref2va_block49_hardest_cell.md` and
   `2026-09-29_ref2va_block49_verify.md` argued from CPU emulations. This is the
   first check on the kernels' own outputs.
3. **It shrinks; it does not vanish.** ref2va's video and audio rows are 9 to 18%
   harder than t2v's on every kernel. One scene per capture cannot say whether that
   is the model or the scene.

## Not shown

- Any effect in a render. The rows with the extra error are not read by the final
  layer, and nothing here says the video rows' 9 to 18% shows.
- Other blocks and steps. Only blocks 24 and 49 at step 2 were graded (block 24 as
  a control); the earlier record's blocks 45 and 48 and step 6 were not rerun with the
  row selection.
- One capture per model, one scene each; the two captures have different lengths.
- Which heads carry the ref2va excess (below).

## The 8-head run

The grader's default is `--heads 8`, the first eight heads. A first run of this
grade used it and read very differently: ref2va over t2v of 1.2 to 1.4 times on
both lines, and no change from the row selection. The 2026-09-27 records used all 56
heads. The first eight heads of the ref2va capture read block 49 at 0.0209 (kitchen)
against 0.0367 over all 56, while t2v barely moves (0.0173 against 0.0140), so the
ref2va error that the row selection removes sits in heads outside the first eight.
That was not measured head by head. A block 49 grade at 8 heads under-reads ref2va
and cannot answer a question about rows; use `--heads 56`.
