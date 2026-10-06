bump: patch

### Added

- **Where a masked song render's time goes, by node and by stage.**
  `bench/masked_render_time_breakdown.py` joins three things on a render's
  prompt id: the runner's row (`bench/run_graph_arms.py`), the stage seconds
  `MiniMaxH3AudioFreezeSong` reports for itself, and the pipeline telemetry
  record. It prints a table per render and writes the numbers: seconds and
  share by node and by stage, each stage by window, what the card and the
  processors did during each stage, core's model loads placed in the stage
  they fell in, the seconds of each sampling step from the server's log,
  and what a repeat of the same stretch would pay again. The repeat figure
  is by rule (the tool's `REPEAT_KEEPS`, read from the node's code), not
  timed, and the output says so. A record made by it carries only the
  patch values a render's seconds depend on.
- **The first record from it**,
  `bench/results/2026-10-06_masked_render_time_breakdown.md`: two renders of
  one stretch on the ref2va motion graph, compared stage with stage. It
  re-ranks the masked lane's speed plan: sampling is most of a render at
  both canvases; the plate's VAE encode is the largest stage after it and,
  with the conditioning, is what a repeat run pays again without needing
  to; a canvas-size working copy does not make the loader faster; keeping
  the still's part of the conditioning across windows is worth too little
  to build; and the composite's time is spent on the processors. The record
  holds the figures and what it does not establish.
