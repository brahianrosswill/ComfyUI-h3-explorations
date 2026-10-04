# Masked video to video: research notes

Research notes on replacing one subject in a source video and keeping the
rest: tracking and masking the subject, what the mask does on the token grid,
the composite, and the models that could help. The owner asked on 2026-10-04
that every session working on this keep its notes here, so they are tracked
and shared.

This folder holds research and working notes. It is not the authority for how
the node behaves: `video_mask.py` and `audio_freeze_song.py` are, and the arms
that were rendered are `bench/masked_v2v_arms.json` and
`bench/masked_v2v_band_arms.json`.

## How to add notes

- One file per session per day: `YYYY-MM-DD_<session>.md`. The date in the
  name makes it a dated record, so measured numbers may live in it
  (`docs/prose_measurements.md`).
- Write in your own file. Another session's file is theirs to change; answer
  it from yours and link to it.
- Say how you know each thing. The labels used so far: card (a model card or
  the Hub API), paper, code (read in this pack or in ComfyUI core), seen
  (frames looked at), measured (a script's output), reported (another
  session's result), inferred, unverified.
- When a claim is corrected, keep it and mark it, so a reader who heard the
  first version finds the correction beside it.
- No media in git. The source clips, the reference still, renders and contact
  sheets stay on the shares or under `internal/`, named here by filename only.
- Add your file to the list below.

## Notes

- [`2026-10-04_mrhf.md`](2026-10-04_mrhf.md): which Hugging Face models help
  beyond SAM 3.1 (effects outside the silhouette, matting, sync and identity
  instruments, sub-part masks); how core's SAM3 tracker behaves with several
  people; the shot table for the band segment; what the flicker beside the
  subject in the first clip is, and a fix to test.
- [`2026-10-04_mrblue.md`](2026-10-04_mrblue.md): what a code review of the
  mask path established and what it leaves open: a test that needs no
  sampling to tell whether the remnant is painted at decode or by the model,
  the one node that would let the two-sampler graphs carry a mask, what a
  soft band of mask values does, and two attributions nobody has made.
