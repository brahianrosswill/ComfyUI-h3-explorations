# Verification of the ref2va block 49 finding (h3l-02), 2026-09-29

Board direction `ref2va-verify`, second half (the partition-delta half is
`2026-09-29_ref2va_partition_delta_verify.md`). Under test:
`2026-09-29_ref2va_block49_hardest_cell.md`, by a session that named several
errors of its own on this task. That record is not edited. CPU only, no render.

Run by a verification pass I delegated (no git, no repo edits, GPU forbidden),
then spot checked by me. What I re-derived myself is marked.

## Verdict

The central claim survives every control that could be run on the CPU: ref2va's
block 49 grades hardest mainly because the cell grade counts text and reference
rows the model discards. Two details of the record are wrong or overstated, and
the version of the control on the real kernels' outputs was not run (it needs
the card).

## Claim by claim

1. **The original weights carry block 49's oddity: confirmed.**
   `bench/scan_original_block49.py` reproduces `2026-09-29_original_block49_scan.json`
   exactly (verifier). Block 49's K-norm energy is concentrated in the same four
   channels in all four release directories and in no other block scanned; the
   adaln chunks 8 to 11 are zero at block 49 in all four and in none of the
   contrast blocks. Those chunks are the text tag's `gate_msa`, `shift_mlp`,
   `scale_mlp` and `gate_mlp` (`bench/analyze_checkpoint_delta.py`).
2. **Segment tables and K stress: confirmed, with a qualifier on peakiness.**
   The capture manifests' segments are contiguous to the sequence length, the row
   counts agree with the geometry (video is 102 latent frames of 1344/32 by 768/32
   tokens; a 2048 by 2048 reference is 4096 rows), and all three analysis scripts
   use them; the `ref_img_2` naming is right. `analyze_k_by_segment.py` reproduces
   its committed record exactly, and K's stress at block 49 is alike in every
   segment of both cells.
   - **The committed attention-mass record was not made with the docstring's
     command.** The script's default is 384 queries; the committed
     `2026-09-29_attention_mass.json` equals a 256-query run (I confirmed the
     block 49 t2v `eff_keys_median_head` matches
     `2026-09-29_attention_mass_t2v_b49_q256.json`). The median of that statistic moves a lot with the
     sample size in one cell (verifier's runs at 96 to 512 queries), and ref2va's is
     higher, so "peakiness alike" holds only for `heads_eff_under_20`, which is
     stable across sample sizes and equal in the two cells. The medians say ref2va
     is flatter, not harder.
3. **The grade counts every query row: confirmed.** The grade is
   `rel_l2_against` over the whole tensor (`bench/analyze_sol_error.py`);
   `grade_dense_kernels_on_captures.py` never selects rows (its default is 8
   heads; test 2 used all 56). `analyze_error_by_query_segment.py` reproduces its
   record exactly. From that record (I recomputed it): the all-row cell error
   is about four times larger for ref2va at block 49, but over video and audio
   rows only the gap is far smaller, and block 24 shows no gap either way. In
   ref2va the text and reference rows hold most of the squared error; in t2v the
   video rows do. The record's "video rows about the same" is soft: video-only
   error is somewhat higher in ref2va.
4. **The last block's text and reference rows are never read: confirmed, one
   error found.** `FinalLayer.forward` reads only the video and audio segments
   (I read `comfy/ldm/minimax/model.py`); nothing consumes the hidden state after
   the block loop. The record says the text rows' post-attention gates are exactly
   zero so nothing trained them. That does not hold for the whole ref2va "text"
   span: it contains vision-embed rows tagged with the video modality
   (`text_token_tags`), whose block 49 modulation is not zero. The dropped-row claim
   is unaffected. The number of such rows was not counted (the tags are not in the
   capture).
5. **What the K-only simulation gets wrong: partly, but the conclusion holds.** It
   quantises K per row, unrotated, with Q, P and V left in float. Kitchen's
   kernel rotates with a Hadamard transform, shifts K, scales Q per 4 rows and
   K per 32 keys, V per channel and P as uint8. The verifier wrote a closer
   emulation (`bench/emulate_kitchen_int8_by_segment.py`, variants A to D; E to J
   add V and P and are uncalibrated, so they are not evidence). Its rotated K plus Q variant lands near kitchen's real grades in
   `2026-09-27_sol_redesign_test2.md` for both cells, and the conclusion holds across
   sample sizes and seeds (`2026-09-29_block49_emulation_*.json`).
6. **Row counts (8424 text, two 4096 reference images, 599 t2v text): confirmed** against
   the segment tables; the tokenised cross-check could not run (importing `comfy`
   starts CUDA even with `--cpu`). The capture's own `manifest.json` was rewritten
   today to the corrected 4096 per reference, while the repo copy
   `2026-09-27_capture_manifest_sol_test2_ref2va.json` still carries the old
   14568 text tokens and 2048 reference tokens; the record's account of the
   manifest bug is right.

## The control

Same emulation, cell error over video and audio rows only, t2v against ref2va.
For the variants A, C and D the three-to-four-times gap in all-row error collapses
to about the same ratio as the rows' own difference (`cell_video_audio` against
`cell_all_rows` in the `2026-09-29_block49_emulation_{t2v,ref2va}_*.json`; I
recomputed the ratios from them). It shrinks and does not vanish: ref2va's video
rows are somewhat harder, and a single video-only figure shifts with the query
sample. Variant B (unrotated K in groups) is too seed-dependent to lean on. So the
finding stands as "mostly the grade", not "entirely".

## Consequence

The record's own recommendation stands: grade block 49 over video and audio query
rows and report text and reference rows apart (`bench/grade_dense_kernels_on_captures.py`
counts every row and does not do this; it is not implemented). Kernel rankings
within a cell are unaffected in direction.

## Not verified

Real-kernel per-segment error (needs the card, and h3dude's say), the V and P
quantisation's contribution in the real kernels, the count of video-tagged rows in
the ref2va text span, and whether the guessed Hadamard sign pattern matters.
