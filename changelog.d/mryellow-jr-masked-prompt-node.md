bump: minor

### Added

- **`MiniMaxH3MaskedPrompt`** (`masked_prompt.py`): a node that writes the
  masked video-to-video prompt, so the long reference-format text is never
  typed. Three choices: `subject` (a person, a man, a woman), `voice` (the
  main voice on the track, or silent) and `picture_gives` (the whole
  person, or the head and hair). The rest is read off the Masked Source
  wired into its `source`: what is replaced, and whether a motion reference
  is on, in which case the text names `<Video 1>`. `extra` adds sentences to
  the shot as written. The text is shown on the node on every run.
- **`masked_prompt_text.py`** holds every sentence as a constant, with what
  each has rendered on. It reproduces, byte for byte, the generic swap texts
  written on 2026-10-04 and the text the motion arms measured. The
  head-and-hair and silent texts are new and have not been rendered.
- `bench/check_masked_prompt.py`, and its row in `docs/checks.md`.
- Four bank entries, each a copy of what the node writes:
  `ref2va_masked_person_swap`, `ref2va_masked_person_motion`,
  `ref2va_masked_person_head`, `ref2va_masked_person_silent`.

### Changed

- **Both shipped masked graphs take their prompt from the node**, and so do
  their `workflows/daily/` copies. `h3_video_to_video_masked_song_pdd8` held
  `ref2va_masked_subject_swap`, which says one performer, alone, standing
  through a fourteen-second static shot; it now renders
  `ref2va_masked_person_swap`, which claims none of that.
  `h3_video_to_video_masked_song_ref2va_motion` held the measured text,
  which says "the man"; it now renders the same text for "the person",
  since a shipped graph holds a placeholder still. That form has not been
  rendered. `docs/wiki/decisions.md`, 2026-10-06.
- The Masked Source's `source` output says what it replaces, which the
  prompt node reads.
- `workflows/prompts.py`: `carriers` and `describe` resolve a prompt that a
  Masked Prompt node writes, so the graders and the render records read the
  text the encoder reads. `describe` now follows a linked prompt, where it
  read only a typed one.
- `docs/wiki/masked_v2v.md` has the node as the third piece.

### Fixed

- `prompt_bank/ref2va_masked_subject_motion.txt` ended in a newline, and
  with its `<Video 1>` label and its camera sentence it failed
  `bench/build_prompt_bank.py --check` since it was replaced on 2026-10-05.
  Nothing in the sweep runs that gate. The file is stripped, and the two
  findings that are properties of the lane (the song node builds
  `<Video 1>` itself; the text names no camera on purpose) are recorded in
  the manifest as other entries record theirs.

### Not done

- No render: the graphs were built without a server on this code. A live
  validation and a first render on the node's text are owed.
- A clothing role for `the wired parts`, and several stills of one subject.
