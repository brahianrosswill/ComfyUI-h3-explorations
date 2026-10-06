bump: minor

### Added

- **A song run can continue another run's render** (`continue_from` on
  `MiniMaxH3AudioFreezeSong`, appended last, optional): the path of another
  run's last stored window (the `.safetensors` beside its video in its
  `_windows` folder). This run's first window then takes that window's last
  frames as its context, exactly as each window takes the one before it in
  one run, and writes its length less that context; the run's track and
  video must start `context_frames` before its first new frame. The first
  window's key is chained to the key stored in the continued latent, so a
  first stretch rendered again renders the stretch after it again, and
  moving the file changes nothing. The run's file and its mask review start
  on the first new frame. It exists for a clip too long to load whole,
  rendered as several runs over consecutive stretches and joined by
  `bench/join_stretches.py`; the tooltip says how to seed the later runs so
  they sample as one long run would. `loop_plan.frames_written` takes the
  head the first window does not write; `loop_resume.stored_key` reads a
  stored window's key and refuses anything that is not a stored window.
  Built by mrhow, who broke it nine ways in a scratch copy (nine of nine
  red); applied from their script and committed by mryellow_jr after their
  session closed. The mask review's check now holds `save_mask_review` to
  be appended after `source`, not last.

Not rendered: nothing has run through the node with the input set. The
seam between a continued stretch and the one before it is the first thing
a render with it must show, measured and looked at, before any long render
uses it.
