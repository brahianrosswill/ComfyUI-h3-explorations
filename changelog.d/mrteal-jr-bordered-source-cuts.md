bump: minor

### Fixed

- **A source with bars at its edges lost every cut.** The Subject Track's
  cut score compared whole frames, and a bar's edge is the same in every
  frame: on a 4:3 picture padded onto the wide canvas every cut scored
  lower, the clip's scores lost their gap, the automatic threshold fell
  back to its fixed value and found none, and one mask was carried across
  eleven cuts (found by mrsun on an offline copy of the score; confirmed on
  the node). The score now leaves out rows and columns at a frame's edge
  that stay one flat value through the loaded stretch
  (`subject_track.py::flat_borders`), and the report says what it left out.
  On that file it finds the unpadded original's eleven cuts at the same
  frames; a source without bars scores exactly as before. `MASK_VERSION`
  is 8, so a kept mask is tracked again once. No input, default or tooltip
  changed. `bench/check_subject_track.py` item 9 was red on the old code.
  `bench/measure_subject_yaw.py` finds its shots with the same function and
  gains the fix. Record: `bench/results/2026-10-06_bordered_source_cuts.md`.
