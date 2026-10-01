# PDMD

Projected Distribution Matching Distillation (arXiv 2609.35768): a few-step
student of the base H3, published as 4-step and 2-step LoRAs and as a full
4-step transformer by `pdmd2026` on Hugging Face. Kijai's rank-reduced ComfyUI
conversions are the files that brought it here.

- [`2026-10-01_what_pdmd_is.md`](2026-10-01_what_pdmd_is.md): what the two
  LoRAs are, how they were trained, their sampling contract in ComfyUI terms,
  whether kijai's conversion is faithful, why they load at the call, and why
  FlashGen is the closest sibling. Ends with the open work, in order.

Sources:
- the paper: `internal/refs/pdmd/2609.35768v1.pdf`, gitignored;
- the trainer's repo: `coderef/pdmd`, read-only (`docs/wiki/references.md`);
- the measurements: `bench/measure_pdmd_lora_conversion.py` and
  `bench/probe_int8_lora_requant.py`, with their records in `bench/results/`,
  dated 2026-10-01.
