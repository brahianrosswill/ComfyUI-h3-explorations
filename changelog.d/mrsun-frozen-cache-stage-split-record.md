bump: patch

### Added

- `bench/results/2026-10-06_frozen_cache_stage_split.md`, with its rows and the log lines it reads: the first card run of the frozen video cache's stage timers, on the three audio-refine arms of `bench/frozen_cache_arms.json`. It measures what a cached step costs with almost nothing live, which the masked lane's cost model had only derived, splits it by stage, and restates that model's ceilings for today's masked windows with the measured figure. No masked window has run through the cache.
