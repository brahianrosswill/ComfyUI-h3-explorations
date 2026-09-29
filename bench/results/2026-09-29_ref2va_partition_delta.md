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
4. **Audio differs more than video only at the tail.** Per-modality modulation
   weight deltas are about equal across video, text and audio through most of the
   depth and rise for audio in blocks 45 to 49; the audio output head differs
   more than the video head. The older record's biggest chunks
   (`2026-08-20_dit_internals.json`, audio `shift_mlp`, `scale_mlp`, `gate_mlp`)
   are the time-varying part of the modulation in the pruned int8 files, a
   different quantity, and this does not reproduce them on the weights.

## Not shown

- That any of this is a defect. A fine-tune for a different task should differ;
  nothing here separates intended change from damage, and no base or reference
  model exists here to say what ref2va should be.
- Anything about behaviour. The community reports (Hugging Face discussions 50, 82
  and 91, MiniMax-H3 issue 17) are about identity, audio and noise; weights alone
  cannot say whether they are real.

## Next

The reference rows' role: how much the video queries read them by depth
(`bench/analyze_attention_mass.py` on the ref2va capture), then a paired capture
with a reference on both partitions (`workflows/h3_probe_capture_ref3_api.json`
and its fl2va twin), which needs the card.
