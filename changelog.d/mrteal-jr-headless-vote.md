bump: minor

### Fixed

- **The Subject Track's automatic pick could take a thing for the subject.**
  With nothing named, each shot's favourite votes with its shot's length,
  and a favourite needed no head. On the first window of the one-person clip
  the opening shot is the longest and shows only a hanging microphone on the
  frame where it is judged, so the microphone was picked and masked for the
  whole shot while the subject, who walks in partway through, stayed
  unmasked. Now a favourite on whom no head is found does not vote while
  another shot's has one (`subject_track.py::main_subject`), the rule a
  match already follows; with no head anywhere every favourite votes as
  before, and a frame the user names is taken as named. The report gains a
  line naming a favourite left out. `MASK_VERSION` is 7, so a kept mask from
  before is tracked again once. No input, default or tooltip changed.
  `bench/check_subject_track.py` item 8 holds the window's shape and was red
  on the old code. Rerun at defaults on the lane's three windows: the
  one-person window is right on all three shots, and the other two windows'
  masks are frame for frame what they were
  (`bench/results/2026-10-06_subject_track_defaults.md`).

### Added

- **The Subject Track and the part node at their defaults on the lane's
  three windows**, with nothing sampled:
  `bench/results/2026-10-06_subject_track_defaults.md` and its `.json`. What
  it found: the microphone pick fixed above; a first shot of the
  car window that no value of `match` can find, because another person's
  close-up scores the same, and that `corrections` does find, in the first
  run of a correction on a real clip; a cut the tracker misses on the band
  window at no cost there; and the Sapiens2 part node finding hair and face
  on every frame where the subject is a person and not mid-dissolve. It
  judges stills of the mask, not its edge and not a render.
- **The benchmark's clip file names a third window and what each window
  needs.** `bench/turn_metric_eye_verdicts.json` gains `friday_0s`, and
  every window gains its still and a `tracking` block: the real cuts, which
  shots the subject is in, who read them and on what, and the Subject
  Track inputs beyond `h3_config.SUBJECT_TRACK` that the window needs (none
  for two, one correction for the third). `bench/check_subject_yaw.py`
  holds the block to the cuts' shots, every frame once. The third window
  has no render, so the scoreboard lists it as not measured.
