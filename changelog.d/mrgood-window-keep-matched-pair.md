bump: patch

### Added

- **The window keep on the card**,
  `bench/results/2026-10-06_window_keep_matched_pair.md`: one window of the
  ref2va motion graph rendered twice on one server with the same seed, the
  second run on the source latent and the conditioning the first one kept
  (`window_keep.py`, 0.204.0). The two stored window latents are equal on
  both streams and the videos are the same bytes; the source encode and the
  conditioning are gone from the second run's stage seconds; nothing wrote
  into a kept value. That is the acceptance the keep's card set. It is one
  short window and an unchanged seed; the record says what that leaves
  unshown.
- `bench/results/2026-10-06_masked_render_time_breakdown.md` points at it
  from the line that said no run on kept values had been timed.
