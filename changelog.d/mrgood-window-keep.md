bump: minor

### Added

- **A window that renders again reuses its source encode and its prompt
  encode from an earlier run.** `MiniMaxH3AudioFreezeSong` encodes and
  samples inside itself, so ComfyUI's node cache could not hand a second
  run of the same stretch the parts that had not changed: with a new seed,
  sampler or schedule every window's plate went through the video VAE again
  and every prompt through the text encoder. `window_keep.py` keeps both in
  the server's memory for the session. A kept value is found by the objects
  it was made from (the loader's frames, the VAE, the encoder, each
  reference) together with the window and the settings that change what is
  encoded, so a hit is what a fresh encode would give and equal content in a
  new object is a miss on purpose. Nothing kept is on the card. A kept value
  that something has written into since is dropped, logged and made again.
  The report says which windows used a kept latent or conditioning. The
  conditioning half is reached by every song graph, not only the masked
  ones: with no Masked Source its key is the text, the length, the canvas,
  the encoder and the references. What it
  is worth against a whole run:
  `bench/results/2026-10-06_masked_render_time_breakdown.md`, "A repeat
  run", a sum of stages; no run on kept values has been timed yet.
- `bench/check_window_keep.py` holds the keep's key, its budget, where its
  tensors sit, the write guard and the song node's use of the switch.

### Changed

- **`reuse_windows` on the song node covers the keep too.** On, a window
  that renders again reads the keep; off, nothing is read from memory or
  from disk, and what the run encodes replaces what was kept. No input was
  added; the tooltip says it.
- Known limit, in the module: a Masked Source that executes again hands on
  a new mask, so a conditioning made with a motion reference is encoded
  again on that run. The source latents still hit.
