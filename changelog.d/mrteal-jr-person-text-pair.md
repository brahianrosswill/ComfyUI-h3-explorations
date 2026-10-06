bump: patch

### Added

- **The shipped motion graph's new default text, rendered against the
  measured one**: `bench/results/2026-10-06_masked_v2v_person_text.md`,
  with its rows and figures beside it. Two renders of
  `workflows/daily/h3_mask_ref2va_motion_api.json` on the band window, one
  seed, differing only in the prompt node's `subject`. With "a man" the
  shipped graph reproduces the 2026-10-05 arm frame for frame, which is the
  render that day left owed. With "a person", the default, the turn still
  completes and ends with the source's, and starts later; the motion
  metric's zero-shift verdict reads that as not following. The readings
  were fixed before the control rendered. Neither clip has been watched, so
  the default is held on these renders, not verified.
