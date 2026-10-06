bump: patch

### Added

- **The sweep runs the prompt bank's gate.** `bench/build_prompt_bank.py
  --check` was red at a commit whose sweeps were all green, because
  `bench/run_checks.py` runs `check_*.py` and the gate is not named that
  way (found by mryellow_jr, 2026-10-06). `bench/check_prompt_bank.py` runs
  the builder's gate and returns its exit code. It grades nothing itself.
