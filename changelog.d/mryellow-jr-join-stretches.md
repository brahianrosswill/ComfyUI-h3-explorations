bump: minor

### Added

- **`bench/join_stretches.py` joins the windows of several song-node runs
  over consecutive stretches of one clip into one file.** A clip too long
  to load whole is rendered as several runs, one per stretch; each run
  joins its own windows under its own stretch of the track, and
  concatenating the finished files would join the audio at every boundary.
  This takes every stretch's window files in order, copies them without
  re-encoding through the song node's own join, and encodes the clip's own
  audio for the whole span once. `--review` joins the windows' `_with_mask`
  files the same way. It refuses, writing nothing, when what it is given
  does not add up to the span: a folder with no windows or a gap in their
  numbers, a folder given twice, a stretch that does not start where the
  one before it ends, frames that do not end at the span's end, or an
  output that exists without `--overwrite`; a joined file whose own frame
  count is not its windows' is removed. Its docstring says how to seed the
  stretches so they sample as one long run would.
  `bench/check_audio_freeze.py` holds the join on two stand-in runs under a
  clip whose audio carries a tone at a known place, and each refusal.
  Built by mrhow; applied and committed by mryellow_jr after mrhow's session
  closed.

Not run on real window files or on a clip: its first real use is the first
render made in stretches.
