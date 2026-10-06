bump: minor

### Added

- **The motion reference can be zoomed in on the subject.** A fourth choice
  of the Masked Source's `motion_reference`, `subject only, zoomed in`:
  `subject only` shown in a box around the subject and not in the whole
  frame. `subject only` scales the whole frame down and greys the rest, so a
  subject that is small in the frame is a handful of the text encoder's
  tokens and most of the picture is grey; the owner's verdict on the lane's
  second test clip was that the movement was not followed, and their answer
  to why was to zoom in. One fixed box per shot, the union of the subject's
  boxes over the window's frames of that shot with room around it
  (`video_mask.MOTION_BOX_ROOM`, reasoned) on the canvas multiple: fixed, so
  the subject still travels against a frame that holds still, and the
  framing changes only where the source cuts.
- **What bounds the picture is a rule and not a setting**
  (`video_mask.zoom_plan`): it never holds more pixels than the whole-frame
  reference at `motion_short_edge` would, so the zoom never costs more than
  `subject only`; no shot is shown smaller than `subject only` shows it; a
  box that is the whole frame gives `subject only`'s picture value for
  value; and nothing is enlarged past the canvas's own pixels. A video has
  one size, so a window with shots of different sizes shows the smaller one
  centred on grey at its own scale. `bench/check_video_mask.py` holds each
  of the four with a case.
- **The source record carries the tracked subject's box per frame**
  (`subject_boxes`: one row a frame, `(x0, y0, x1, y1)` on the loaded frames
  with the far side exclusive, a row of -1 where the subject is absent, from
  `sapiens2_parts.mask_boxes`). The Masked Source replaces the tracker's
  mask with the region it regenerates, so on a parts graph nothing after it
  knew where the whole subject was; the zoom is framed on the subject
  whatever is replaced. On a run that reads a kept mask the tracker does not
  run and the boxes are the kept region's, which the log says.
- The preview strip draws each shot's box on the plate and shows the zoomed
  picture beside it; the Masked Source's log and the song node's report say
  the boxes and the picture's size per window. The conditioning kept across
  runs (`window_keep.cond_key`) holds the window's boxes and the shot
  table, so a changed box encodes again.

Not rendered: no arm has run on this choice, and no shipped or daily graph
uses it. The existing three choices are unchanged; `bench/check_video_mask.py`
holds the whole-frame box to `subject only`'s tensor.
