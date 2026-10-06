bump: patch

### Added

- **The song node reports where its time went.** `MiniMaxH3AudioFreezeSong`
  encodes, samples and decodes inside itself, so nothing outside it could
  time its stages: a render record held one figure for the whole node, and
  the pipeline telemetry sees it as one block. Its report now carries, per
  window, the seconds spent on the source encode, the window's setup,
  sampling, decode, composite and the write, and at the end the seconds by
  stage over the whole run with the track encode, the conditioning and the
  join. Wall clock read at stage ends; nothing that is rendered changes.
