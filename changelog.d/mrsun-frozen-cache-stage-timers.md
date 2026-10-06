bump: patch

### Changed

- `MiniMaxH3FrozenVideoCache`: with `verify` on, each cached step's seconds are split by stage (`frozen_video_cache.STAGES`: the kept states brought to the card, qkv over every row, the attention, the live rows' own work, and everything between blocks), on the call's record and in the log. Only a verified step waits for the card between stages, so a plain run's timing is untouched. `bench/check_frozen_video_cache.py` item 8 holds that the stages add up to the call and that nothing is timed with `verify` off. Checked on the CPU only; no card run yet.
