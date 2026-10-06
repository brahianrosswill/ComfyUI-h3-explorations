bump: patch

### Changed

- **The masked graphs load the frames the song node's plan reads, and no
  more.** The generator set the source loader's `frame_load_cap` to the
  extent plus one whole window, and the Subject Track works on every frame
  the loader hands it: on the shipped thirty seconds the tracker was working
  on frames nothing renders
  (`bench/results/2026-10-06_masked_render_time_breakdown.md`, "The first
  run of a stretch"). The cap is now the planner's own count for the graph's
  extent, window and context (`loop_plan.frames_read`), and the eight masked
  graphs are rebuilt with it. `bench/check_video_mask.py` item 11 fails any
  shipped masked graph whose cap differs from what the plan reads; it was
  red on all eight before the rebuild. What the tracker saves is not
  measured yet: its time need not be linear in frames.
- **Every kept mask is tracked once more after this lands.** The loader's
  cap is part of the key a mask is kept under, so the first run of each
  masked graph on a clip tracks afresh.
- **The extent and the loader's cap are two widgets for one number, and a
  run that falls short now says so.** Until now a spare window hid up to one
  window of that coupling; with the cap exact, an extent raised without
  raising the cap gives a run that covers only what was loaded. The song
  node does not refuse it, since a clip that is simply shorter than the
  extent has to render untouched. It prints one line near the top of its
  report, and logs it as a warning, when the track is shorter than the
  extent asked for, with the frames the source holds, the frames the asked
  extent reads and the cap to raise; and when a source's picture ends before
  its track (`loop_plan.extent_shortfall`). A run that covers what it was
  asked for prints nothing new. The shipped graphs as generated will print
  the line on every run until their clip is replaced: the placeholder clip
  is shorter than the thirty seconds they ask for. That is the line being
  true, not a regression; the node cannot tell a short file from a low cap,
  so the advice about the cap is conditional.

### Fixed

- **The join returns every frame its windows hold.** `loop_output.join_and_mux`
  left the finished file's length to ffmpeg's `-shortest`, which with copied
  video stops where the audio's last packet ends. Under a track longer than
  the video nothing was lost, which is the case every masked render had been
  in, because the loader's spare window also loaded spare audio. Under a
  track exactly as long as the video, or shorter, a join of two or more
  windows came back a frame to four short of what it held, and on a still
  picture dozens of frames long. The exact cap above puts every masked
  render in the losing case, so the two land together. The join is now told
  how many frames its files hold and gives ffmpeg that as a duration, with
  the track padded with silence to reach it; it cuts nothing.
  **Which older files to distrust:** a render of two or more windows whose
  track was exactly as long as its video came back a frame to four short at
  its end, and one whose track ended first (every whole-track render, where
  the track ends inside the last window) was cut wherever the muxer
  stopped, not at the track. Both were seen on synthetic windows only; no
  older whole-track render has been counted. The two thirty-second masked
  renders of 2026-10-06 were not affected: their track was longer than
  their video, and each holds every frame its windows do (counted, 753 of
  753).
- **A track that ends before its windows do is cut at the frames.** The
  last window of a run ends at or past the end of the track it was planned
  from. The frames past the track used to be left to the join's cut; they
  now come off the last window before it is encoded
  (`loop_plan.frames_kept`), so the video ends with its track on a whole
  frame, and the report says how many were dropped. When the track on hand
  is longer than the windows, as with an extent of its first seconds,
  every frame is kept, as before.
- **Not fixed here: a run stored before this change and queued again
  unchanged.** Every window's key still matches, so the last window is
  reused as it was stored, with its tail, and it never renders again. When
  its track ends inside that window, which is the usual whole-track song,
  the join is given the kept count against files that hold more and cuts a
  copied stream, which on synthetic windows left stray frames past the cut.
  No report line says so yet. Turning `reuse_windows` off for one run
  rewrites the window. The masked graphs are not exposed: their track is
  the loader's audio, so their stored windows stopped matching when the cap
  moved. Found by an independent read of this change (mrop); the fix, a
  stored window that carries the frames it was written with, follows as its
  own entry.
- `bench/check_audio_freeze.py` holds the join through ffmpeg: two windows,
  noisy and flat, under a track exactly as long, shorter and longer, every
  frame back and evenly spaced. It fails on the old command at the sizes the
  loss was first seen.

### Added

- `loop_plan.frames_covered` and `loop_plan.frames_read`: the frames a run
  of windows covers, in one place. The song node's report line and the
  generator both call them. `bench/check_audio_freeze.py` holds
  `frames_read` to the furthest frame any placed window slices from the
  track, and holds the new line's cases.
