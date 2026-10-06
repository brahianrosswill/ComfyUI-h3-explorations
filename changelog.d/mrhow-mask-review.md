bump: minor

### Added

- **A masked render is saved with a view of what it regenerated.** With a
  Masked Source wired, `MiniMaxH3AudioFreezeSong` writes
  `<prefix>_NNNNN_with_mask.mp4` beside the render: the render on top and,
  below it, the source it started from with the mask the source carries
  tinted one colour, what else regenerates around it another, and, under
  `only what changed`, an outline of what the composite kept from the
  render. The same frames in step, the same track, and a small legend in the
  corner of the lower row that names each colour and gives the window's
  regenerated share, since the file is opened away from the graph. Asked for
  by the owner after a wide shot's region turned out several times the
  subject's own area, which nothing showed without a script. Built by
  mrgood; put on the join and stored-window changes of 0.206.2 and 0.206.3
  and finished by mrhow.
- **The legend names the mask for what it is.** The first colour is "the
  tracked subject" only when the Masked Source replaces the whole subject;
  with `head and hair` it is "the head and hair", and with `the wired parts`
  it is "the parts taken" (`video_mask.mask_layer_name`). On a parts graph
  the mask and the tracked subject can lie in different places, and a
  legend that called one by the other's name would say the opposite of what
  the render did. A `replace` the view has no name for is refused.
- The lower row is drawn from what the sampler was given: the token mask
  brought back to pixels, so the grow, the snap to tokens and the temporal
  runs are all in the picture (`video_mask.overlay_pieces`). It is built
  and piped a cycle of frames at a time, written per window beside the
  window's video and joined as the windows are joined, through the same
  join, so it holds exactly the frames the render does. A reused window
  stored without one is stacked from its stored video, with no outline, and
  the report says which.
- The view is a list of labelled layers (`video_mask.OverlayLayer`: a name,
  a colour, a mask per frame); each pixel takes the first filled layer that
  covers it, and the legend is built from the list. Today's layers are the
  three above; a part node's classes or a tracker's other people are more
  entries, with no other change.
- **The review never costs the render, and is never a file from another
  render.** A review is written under a temporary name and renamed when its
  encode has finished, so a file under the final name is always whole: a
  run that died inside that encode would otherwise leave every later run to
  fail at the review's join, after all its windows had sampled. A window
  that renders again removes its old review where it removes its old
  latent, so a later run cannot join the last render's picture to the new
  one. And a failure anywhere in a review is a line of the report and a
  warning in the log: the render's file, its metadata, its shot table and
  the node's outputs are as they would be with the switch off, and the next
  run stacks whichever reviews are missing from the stored videos. Stopping
  a run during a review still stops it.
- `save_mask_review` on the song node, appended last, on by default: it
  does nothing without a source, and turning it on or off never re-renders a
  stored window. The masked graphs write it.
- `bench/check_video_mask.py` item 12 holds the view: the frames from the
  trim on, the mask's, the margin's and an untouched pixel, the outline,
  the legend, the layer's name for each `replace`, an added layer, the
  stacked picture, a render whose tail was cut, the input's place and
  default, and that every graph with a source writes it; and the review's
  writer and its wrapper for real, with the song node's use of them read
  from its source, since the node's windows need the models. Each case was
  seen red against a deliberate break in a scratch copy, on a line that
  names the failure.

### Fixed

- `bench/check_video_mask.py` called a helper, `_fail`, that it never
  defined, in every case of the motion reference since 0.190.7: a failing
  case there ended the run with a NameError before the cases after it were
  reached. It is defined now. With the motion reference deliberately broken
  the check prints the failed case; without the helper the same break is
  the NameError. Found by mrop.

### Changed

- Under `only what changed` the composite's weight is held through the
  window's write, for the outline, and freed after the review; it used to
  be freed before the write.

What it costs a window is the "mask review" stage in the node's own
seconds. It is on by default on every masked graph and adds a drawing pass
and a second encode at twice the height to each window.
