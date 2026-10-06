bump: patch

### Fixed

- **A stored window is reused only when its file holds what this run would
  write.** 0.206.2 left one case open: a song run whose windows were all
  stored before it, queued again unchanged, reused its last window as it was
  stored, with the frames past the track still on it, and nothing ever
  rendered that window again because its key says how it was sampled, not
  how its file was cut. `loop_resume.save_window` now records how many
  frames a window's video holds, and the song node compares that with what
  the run writes of the window (`loop_plan.frames_written`) before reusing
  it. A window that differs stops the reused prefix and renders once more,
  cut at the frames, and its line in the report, preview included, says
  "renders again: its stored file holds N frames and this run writes M". A
  store from before this change has no count and is read as written whole,
  which is what it was (`loop_resume.stored_frames`). Found by an
  independent read of 0.206.2 (mrop). The masked graphs were never exposed.
  One consequence to expect and not chase: a last window that 0.206.2
  itself stored under a track ending inside it was cut correctly but
  carries no count, so it too renders once more.
- **A cut that reaches what the last window writes is refused before
  anything is encoded** (`loop_plan.frames_written`), where the slice would
  have wrapped and written the wrong frames without a word. The planner
  never plans one; the check now holds that.

### Changed

- The song node's first report line says how many frames are written when
  the track ends before its windows do; until now it gave the planned count
  alone, and in a preview nothing said otherwise.
- The line for a source that ends inside a window says how many of the held
  frames are past the track's end and not written. On a clip shorter than
  the extent the report used to say both that the last frame was held and
  that the same frames were dropped.
- `bench/check_audio_freeze.py`: the tail cut is held as a planner property
  over every window length and a spread of contexts, timelines and track
  lengths (the cut never reaches the last window, the files sum to the kept
  count, only the last window's tail moves), in place of a match on the
  node's source text; a store with no count reads back as whole and
  only a last window under a short track fails the comparison; the join
  case writes its windows with the node's own writer, counts packets with
  the ffmpeg the node uses so it needs no ffprobe, and adds a pair ending
  on a one-frame file, which a context of 90 or 141 can plan. Its docstring
  says which red against the old command is the bug and which states the
  contract. A deliberate break of each new piece turned it red in a scratch
  copy; the two breaks made in the node itself are caught by matches on its
  source, red by construction.

Not run through a window: the reuse comparison and the cut are held through
the pure functions and the node's source text. No stored run from before
0.206.2 has been queued again on this code, and no render has gone through
the changed node.
