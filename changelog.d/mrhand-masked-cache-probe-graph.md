bump: minor

### Added

- The generator takes `masked_cache`: `MiniMaxH3FrozenVideoCache` on the song node's model (node 89), after Sol-Attn and everything else on the chain, at `h3_config.FROZEN_VIDEO_CACHE`; the scheduler keeps reading the model before it. It needs `freeze_song_source`. One graph carries it, `workflows/h3_probe_v2v_masked_song_ref2va_motion_cache_api.json`: the ref2va masked motion graph with the cache and nothing else changed. No shipped or daily graph changes. Not run: the first masked window through the cache is the next step on board card `use-frozen-row-cache`.

### Changed

- `h3_config.FROZEN_VIDEO_CACHE` carries the node's `halo` input at the node's own default, a width of no tokens. The audio-refine graphs that wire the cache gain that one input and nothing they compute changes: `frozen_video_cache._dilate` returns the grid untouched at that width, and a video frozen whole has nothing to widen. The constant's comment said the cache had not run on the card; it now points at the record of the refine pass that did, and `docs/wiki/decisions.md` has the line.
- `docs/prompt_bank.md`: `ref2va_masked_person_motion` ships in one more graph. `docs/wiki/next_steps.md` and `docs/wiki/masked_v2v.md` say where the cache is wired and that the graph is a probe.
