bump: patch

### Changed

- **A data file beside a record belongs when the record names it; a dump
  of every point does not** (owner, 2026-10-06). The data file holds what
  the record's numbers are built from, at the grain the record cites; a
  full per-frame or per-step dump stays on the share beside the run's
  outputs and the record names its path; a data file nothing names is
  removed. `docs/prose_measurements.md`, "Where numbers live". Removed in
  the same commit: the tracked data files under `bench/results/` that no
  tracked file names (the list is in the commit message).
