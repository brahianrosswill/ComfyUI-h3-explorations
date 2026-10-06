bump: minor

### Added

- **`h3_video_to_video_masked_parts_song_pdd8`**: the first generated graph
  that wires Sapiens2. `MiniMaxH3Sapiens2Loader` and `MiniMaxH3SubjectParts`
  feed the Masked Source's `parts`, `replace` is `the wired parts`, and the
  prompt node is told the still gives the head and hair, which is what the
  part node's own ticks select. No matting model is loaded: the Masked
  Source reads the part mask, not the matte. `h3_config.SAPIENS2`,
  `SUBJECT_PARTS`, `MASKED_PARTS_SOURCE`, `MASKED_PARTS_PROMPT`.
- **`h3_video_to_video_masked_review`**: the look before a masked render.
  The default masked graph with the song node on `preview`, so nothing
  samples and no model loads, with the tracker's numbered tiles and shot
  table saved (`MiniMaxH3SaveShotTable`) and the Masked Source's preview
  shown. The mask it tracks is kept for the render graphs.
- Both have `workflows/daily/` copies, `h3_mask_parts_pdd8` and
  `h3_mask_review`.

### Changed

- `bench/check_video_mask.py` lets a graph wire the Subject Track's tiles
  when every song node it feeds is on `preview`: such a graph renders
  nothing, so there is no kept mask to defeat. A graph that renders is held
  to the rule as before.
- `bench/check_widget_deviations.py` declares the three widgets the new
  graphs set off their node defaults.
- The Masked Source's log line read "replacing the the wired parts"; it now
  names the choice as the widget shows it.
- `docs/wiki/masked_v2v.md` describes both graphs and no longer says the
  Sapiens2 nodes are in no shipped graph.

### Verified

- Every generated graph validates against a server running this code,
  yesterday's and today's masked graphs included; the 0.199.0 graphs were
  built without one.

### Not done

- Neither graph has rendered. The part node's ticks and margins are the
  node's own defaults, inherited, not judged for edge quality.
- A clothing text for the prompt node, which a garment swap through the
  parts graph needs.
