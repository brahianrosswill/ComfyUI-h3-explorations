bump: minor

### Added

- **The Subject Track looks again for a subject its track lets go inside a
  shot.** A tracker call is seeded once and never detects again, so a subject
  the tracker let go was lost for the rest of the shot however long they
  were back in view. A run of frames the track leaves empty is now
  probed every `PROBE_STRIDE` frames from the side that is tracked, and the
  track is seeded again on the first frame that shows the subject and run
  both ways over the empty run (`regain`). What stays empty stays empty, and
  the report and the shot table name those frames (`frames_without_subject`,
  `regained`, `probes_after_a_loss`, `gallery_frames`). A shot corrected by
  hand is not searched.
- **Who the subject is, is decided against a gallery of the shot's own
  track, never by being the only detection** (`gallery_frames`,
  `gallery_scores`, `clear_best`). Up to `GALLERY_MOST` signatures are taken
  under the tracked mask on frames spread over the shot's first track, the
  frames of the smallest and the largest mask among them. A person found
  later scores their best plain similarity to it, the lower of the two
  places a person is compared, and is taken at or above `REGAIN_SAME` and
  `REGAIN_MARGIN` clear of the next person on that frame. The provenance of
  both constants is beside them in `subject_track.py`.
- **`subject_from` on `MiniMaxH3SubjectTrack`, a new optional input, the
  last one.** The shot table's JSON now carries the picked shot's gallery
  (`gallery`). Wire an earlier run's `shot_table` output here, or give the
  path of the `..._shots.json` it saved, and the pick on this load is made
  against that gallery and not by `pick`'s rule; if nobody on any frame
  looked at is that person, nothing is picked and the report says so with
  the best score seen. For a long clip rendered in pieces. Left empty, the
  node picks as it did.

### Changed

- `MASK_VERSION` on the Subject Track goes from 8 to 9: the same inputs can
  now give a different mask.

The record is `bench/results/2026-10-06_subject_track_regain_and_handover.md`:
one clip, the tracker run in a process of its own with core's SAM 3, nothing
sampled. It states where the line between the subject and another person
sits and how thin it is, and that the score falls when a person is much smaller
than the gallery shows them. Not run: the hand-over through the node on a
server, and the first clip under the new code beyond the checks' stand-ins.
`bench/check_subject_track.py` (items 10 and 11) and
`bench/check_shot_table.py` (`gaps_are_listed`,
`gallery_travels_in_the_table`) hold the behaviour.
