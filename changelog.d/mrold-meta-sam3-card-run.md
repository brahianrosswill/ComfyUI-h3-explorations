bump: patch

### Added

- **A record of the first run of `meta_sam3/` on the card**:
  `bench/results/2026-10-06_meta_sam3_card_run.md` and its `.json`. Meta's
  SAM 3.1 video predictor from the copy, outside ComfyUI, on two of the
  masked lane's test clips: what a session costs in seconds and memory, the
  per-shot sessions against the lane's stored subject mask, one-frame
  requests, the object cap, the model off the card and back, and a probe of
  the presence score. Each finding is marked measured, read from code,
  inferred or not established, and the record keeps what went wrong in the
  run and what was not run. It is not a comparison of Meta's pipeline with
  ComfyUI core's port. The clips are named by file and nothing in it says
  what they show.

### Changed

- `meta_sam3/README.md` points at the record where it says which setting was
  not Meta's default when the pack ran the copy.
