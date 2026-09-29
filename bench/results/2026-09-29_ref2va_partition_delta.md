# How FL2VA and Ref2VA differ, in the release's own weights (2026-09-29)

The owner asked what, if anything, is wrong with ref2va (community reports say
something is; no one has said what). This maps the two original bf16 partitions
(`<release>/FL2VA/transformer` and `Ref2VA/transformer`) against each other.
CPU only, no render. Records: `2026-09-29_partition_delta_map.jsonl`
(`bench/map_partition_delta.py`), `2026-09-29_adaln_chunks_original.json`
(`bench/analyze_adaln_chunks_original.py`), `2026-09-29_time_warp.json`
(`bench/analyze_time_warp.py`).

## What it shows

1. **Nothing is missing, replaced or degenerate.** Both partitions hold the same
   keys, shapes and dtypes; every tensor but `rope.inv_freq` differs; no tensor
   is zero or wildly off; the diffusers-format copies in `transformer/` and
   `transformer_ref/` match the native checkpoints on the tensors compared
   (adaln, K norm, norm1 at blocks 0, 25 and 49). The exactly-zero text-row
   modulation of block 49 is in both (`2026-09-29_ref2va_block49_hardest_cell.md`).
2. **The two are one model fine-tuned two ways, by a few percent.** The backbone
   linears differ by a similar relative amount in every block (a narrow band in
   `rel_delta`), and the difference is close to a noise control, not
   concentrated: the top-16 singular directions hold little more of it than they
   do of a Gaussian matrix of the same norm (`top_energy` against
   `noise_top_energy`). The norms and the K/Q
   norms differ least.
3. **The one structured difference is the timestep path.** `time_embedder.proj_in`,
   `time_embedder.proj_out`, every block's `adaln_proj` and the final layer's
   `adaln_proj` differ far more in a few directions than noise would (their
   `top_energy` is many times `noise_top_energy`).
   - It is **not a time warp.** The best re-timing of fl2va's embedding to match
     ref2va's is the identity (`best_flow_shift.s` about 1, `best_affine` a=1,
     b=0; `warped_rms_rel` equals `unwarped_rms_rel`).
   - Most of the embedding difference is **a vector constant in t**
     (`delta_constant_share`); the rest is a rank-1 to rank-2 function of t
     (`delta_centered_top_shares`). So ref2va's conditioning is fl2va's plus a
     fixed offset and a small smooth part, which is what a mode or task offset
     would look like. This is a reading, not a test.
4. **The audio side differs more in two places: the first block's audio MLP
   modulation, and the tail.**
   - *Block 0.* The older record (`2026-08-20_dit_internals.json`, pruned int8
     files) found the time-varying part of the modulation differing most on the
     audio `shift_mlp`, `scale_mlp` and `gate_mlp` chunks (15 to 17) of block 0.
     This reproduces on the release's own weights (`mod_tv_rel_by_chunk` in
     `2026-09-29_time_warp.json`): those three chunks differ roughly two to four times more than
     any other chunk of block 0, and blocks 25 and 48 show no such standout. It is a
     first-block, audio-side difference, and a real one; nothing in the repo
     explains it.
   - *Tail.* The per-modality weight deltas are about equal across video, text and
     audio through most of the depth and rise for audio in blocks 45 to 49, and
     the audio output head differs more than the video head. The weight rows
     compare the weights, the time-varying part compares what they do over t, so
     the block-0 standout does not appear in the first view and the two do not
     contradict each other.

## Not shown

- That any of this is a defect. A fine-tune for a different task should differ;
  nothing here separates intended change from damage, and no base or reference
  model exists here to say what ref2va should be.
- Whether the block-0 audio modulation difference or the audio tail matters for
  any behaviour.
- Anything about behaviour. The community reports (Hugging Face discussions 50, 82
  and 91, MiniMax-H3 issue 17) are about identity, audio and noise; weights alone
  cannot say whether they are real.

## Next

The reference rows' role: how much the video queries read them by depth
(`bench/analyze_attention_mass.py` on the ref2va capture), then a paired capture
with a reference on both partitions (`workflows/h3_probe_capture_ref3_api.json`
and its fl2va twin), which needs the card.

## Correction notes, 2026-09-29, after verification

`2026-09-29_ref2va_partition_delta_verify.md` (mrblue) re-ran this on an independent
float64 path and built the control this record lacked. The structure holds: the
two partitions differ by a few percent, structured only in the timestep path, and
it is not a time warp (`bench/control_time_warp.py` recovers known warps of about
two percent or more; the real Ref2VA reads the identity). What was wrong or
overstated above:

- **Block 0's audio modulation "standout"** is partly a small-denominator ratio:
  chunk 15 carries a small share of FL2VA's time-varying norm. Read point 4 as "a
  relatively large change to a small quantity", not "a real one".
- **The audio tail's rise starts near block 30, not block 45.** The audio-to-video
  weight-delta ratio is about 1 until block 27; block 49 is high for all three
  modalities. Audio adaln bias also differs more than video's in blocks 0 to 2.
- **"Close to a noise control, not concentrated" holds for the bulk only.**
  `blocks.49.mlp.fc2` holds its delta in a few directions at many times the noise
  control, and blocks 0 and 1 attention somewhat.
- **The data file changed.** `2026-09-29_partition_delta_map.jsonl` was written by an
  older version of `bench/map_partition_delta.py`: 261 of 535 rows had `cos` above
  1, and `rel_delta` on the largest adaln tensors was about 1% low (float32 norms).
  The script now uses float64 norms and a seeded SVD, and the file was regenerated;
  `top_energy` is still a randomized, approximate measure.
