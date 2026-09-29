# Verification of the ref2va partition-delta finding (h3l-03), 2026-09-29

Board direction `ref2va-verify`, first half (the block 49 finding, h3l-02, has its
own record). The record under test is `2026-09-29_ref2va_partition_delta.md`,
written by another session that named several errors of its own on this task.
That record is not edited. CPU only, no render.

Run by a verification pass I delegated (a subagent told not to use git or edit
anything, working from the scripts' docstrings and its own code), then spot
checked by me. What I re-ran myself is marked.

Records:
- `2026-09-29_partition_delta_float64.json` (`bench/verify_partition_delta_float64.py`):
  the release read directly, float64, exact top-16 energy from the Gram matrix's
  eigenvalues. Re-run by me; its numbers equal the verifier's log.
- `2026-09-29_time_warp_control.json` (`bench/control_time_warp.py`): the
  time-warp test given known warps. Re-run by me; it equals the verifier's.

## Where the two differ: confirmed

Same keys and shapes in both partitions; only `rope.inv_freq` is identical
(verifier). Backbone linears differ by a few percent in a narrow band; the norms
and K/Q norms least. The timestep path is structured, on the independent
float64 path: `time_embedder.proj_in`, `proj_out`, block 0 and block 25's
`adaln_proj` and the final layer's `adaln_proj` hold far more of their delta in
16 directions than the noise control does, while block 0 and 25's qkv and
block 25's fc1 sit near noise (`top16` against `noise_top16` in the JSON). So
findings 1 to 3's structure holds.

Defects in the record's underlying data, none cited by a claim in the md:

1. **`2026-09-29_partition_delta_map.jsonl` was written by an older version of
   its script.** 261 of 535 rows have `cos` above 1 (checked by me), up to
   1.115 (block 0's adaln); the current script gives none. Float32 norms on the
   260-million-element adaln weights come out several percent low, so
   `rel_delta` there is off by about 1%. The script's docstring claim that norm-based
   cosine avoids float32 drift is false for those tensors. Regenerate the jsonl
   before anyone reads `cos` from it.
2. **The concentration exceptions are not mentioned.** `blocks.49.mlp.fc2` holds
   its delta in a few directions at many times the noise control, and blocks 0
   and 1 attention somewhat (verifier's numbers, from the same map). The record's
   "close to a noise control, not concentrated" holds for the bulk, not for these.
3. `top_energy` in the jsonl comes from an unseeded randomised SVD and moves by
   a few hundredths between runs; it is a noisy measure, not a recorded constant.

## "Not a time warp": confirmed, and the test can fail

`bench/analyze_time_warp.py` reproduces its committed record exactly (verifier),
and the identity result holds in float64 (constant share and rms).

The control is the part the original lacked. Fed FL2VA's own embedder at a known
warp (an affine scale, an offset, a flow shift), the statistic recovers it: the
affine parameters come back to the fit grid's step and the flow shift exactly, and
a nonparametric warp collapses the mismatch. Fed the real Ref2VA embedder plus a
known warp (the case the real comparison is in), the recovered parameters still
track the injected ones: a scale of 0.98, 0.95, 0.9 returns about 0.98, 0.96,
0.91; offsets of +0.02 and -0.05 come back as injected; flow shifts of 1.1, 1.25,
2.0 come back as 1.14, 1.3, 2.07 (`B_real_plus_warp` in the control JSON). Real
Ref2VA reads a scale of 1.0, offset 0, flow shift 1.04. So a re-timing of about two
percent or more would show; there is none.

Caveat on the criterion: "the best warp removes most of the mismatch" is not a usable
test on the real embedder, because the constant offset (about 95% of the delta)
dominates: even a flow shift of 2 removes only part of it. The recovered
parameters are what discriminate, and the record reports them. The search covers
a scale of 0.8 to 1.2 with t clamped to [0, 1], so a stretch above 1 is blind in
the last few percent of t; the model feeds t in [0, 1]. It tests the timestep
embedder's input only, not audio-schedule or reference-row timesteps.

## "Partition" and the modality chunks: confirmed, no naming bug found

"Partition" is the FL2VA and Ref2VA checkpoint directories. Chunks of
`adaln_proj` are 3 modalities by 6 parameters, chunk c being modality c // 6,
parameter c % 6, with video 0, text 1, audio 2 matching `seg_tag` in the model
(verifier, read from the model file). Chunks 15 to 17 are the audio MLP
modulation, as the record says.

## Stronger than the evidence (verifier's reading; not re-run by me)

- **Block 0's audio "standout".** The relative time-varying delta of chunks 15
  to 17 is real, but chunk 15 carries a small share of FL2VA's time-varying norm,
  so its high ratio is partly a small denominator. Weighted by share, chunks 15
  and 17 are not larger than some other chunks; only chunk 16 is comparable. "A
  real one" overstates it; "a relatively large change to a small quantity" fits.
- **The audio tail.** The audio-to-video adaln weight-delta ratio is about 1
  until block 27 and rises from about block 30, not from block 45; block 49 is
  high for all three modalities. The record's "through most of the depth ... rise
  in blocks 45 to 49" places the rise too late.
- **Audio adaln bias in blocks 0 to 2** differs more than video's and is not
  mentioned.
- "A mode or task offset" is labelled a reading in the record; plain fine-tune
  drift is not ruled out.

## Not verified

The block 49 zero-modulation claim (the other record), any behavioural relevance,
and the functional effect of the audio tail.
