bump: patch

### Added

- `docs/research/masking/2026-10-06_mrsun.md`: a reading of Meta's SAM 3.1 video pipeline as built (what a session does on each frame, with the builder's values) against core's port, which has Meta's networks and its own shorter session logic. It attributes the Subject Track's three complaints about core's tracker: new objects across a cut is Meta's behaviour too, detection stopping for good at the object cap is core's, and a confident detection rewriting a tracked mask is Meta's design applied by core on every frame. Nothing was run.
