# Why ref2va's block 49 is the hardest INT8 cell (2026-09-29)

`2026-09-27_sol_redesign_test2.md` finds ref2va's block 49 the hardest cell for
every int8 kernel and does not say why. This is a CPU look at the two captured
cells (`2026-09-27_sol_test2_pdd8_t2v` and `_ref2va`, step 2). It reproduces
the direction of the finding with a simulated unrotated INT8, not the kernels'
numbers, and names a cause.

Records, each written by its script under `bench/`:
- `2026-09-29_original_block49_scan.json` (`scan_original_block49.py`): the
  release's own weights;
- `2026-09-29_k_by_segment.json` (`analyze_k_by_segment.py`): K's INT8 stress
  by row segment;
- `2026-09-29_attention_mass.json` (`analyze_attention_mass.py`): peakiness and
  where video queries' attention goes;
- `2026-09-29_error_by_query_segment.json` (`analyze_error_by_query_segment.py`):
  the output error by query segment.

## What it shows

1. **Not the weights.** The original `FL2VA/` and `Ref2VA/` checkpoints, and the
   diffusers-format copies, carry the same loud K channels at block 49 and the
   same exactly-zero text-row modulation chunks (8 to 11). Block 49's zero
   chunks are the same in all four, and in none of the contrast blocks.
2. **Not K's stress, and not peakiness.** Per row segment, K's INT8 stress
   (`err_raw`, `err_smooth`, `crest`, `top4_share`) is alike in the two cells,
   reference-image rows included, and the share of heads with under 20
   effective keys (`heads_eff_under_20`) is the same.
3. **The queries decide it.** The grade of a cell
   (`bench/grade_dense_kernels_on_captures.py`, `rel_l2_against`) is over the
   whole output, so every row counts as a query, text and reference rows
   included. In this simulation the video rows' error is about the same in both
   cells (`rel_err`, `video`), but the text and reference-image rows' error at
   block 49 is many times the video rows', and ref2va has many times more of
   those rows. In the ref2va cell they hold most of the squared error
   (`share_of_sq_error`); in the t2v cell the video rows hold most of it.
   `cell_rel_err_all_rows` is several times larger for ref2va.
4. **Those rows are never read.** The final layer takes only the video and audio
   segments (`comfy/ldm/minimax/model.py`, `FinalLayer.forward`, `x[a:b]`), so the
   last block's output for text and reference rows is dropped. The text rows'
   modulation gates being exactly zero (all four original checkpoints) agrees
   with that: nothing trained those rows' post-attention update. This reading of
   the zeros is reasoned, not shown.

So "ref2va's block 49 is the hardest cell" is largely a property of the grade,
not of the render: it counts error in rows the model discards, and ref2va has
8424 text rows and two reference images of 4096 rows each where t2v has 599 text rows
(the capture's `segments`; the manifest's `token_accounting` lists 14568 `text_tokens`, which is a
remainder that also holds 6144 of the reference rows, and gives 1024 `latent_rows` per image against
4096 in the segments, an unexplained difference; the segments are what the analyses used). Video rows, which decide the
picture, are about as hard in both.

## Not shown

- The kernels' own numbers. This is K-only unrotated INT8 on sampled queries
  (`--queries` per segment), step 2, one scene per cell; the test 2 kernels also
  round Q, P and V, and kitchen's is rotated. The direction, and the split by
  segment, may not carry to a rotated kernel; a run of the same split on the
  kernels' outputs would say.
- Whether reference rows' last-block output is read anywhere else. The final layer
  is the only consumer found in `model.py`.
- Whether any of it is visible in a render: video rows' error is what would show.

## So what

A grade of a captured cell is better taken over video and audio query rows for
block 49 (and the text and reference rows reported apart), at least for the last
block. The test 2 verdicts on kernel ranking are unaffected in direction: the
kernels are ranked within a cell, on the same rows.
