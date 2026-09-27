# FastH3 V2 weights: predictions before looking (2026-09-26)

Written before either bf16 file finished downloading, so before any number
from `bench/checkpoint_delta_map.py` or an adaln comparison existed. The
reads that grade these go in their own records; this file is not edited
after them.

Grounds: FastVideo trains FastH3 V2 data-free (DMD2, teacher score only) at
video/audio shifts 10/3 on the fixed rungs `[999, 874, 749, 624, 500, 375,
250, 125]` (`coderef/FastVideo/docs/inference/fasth3-distilled.md`), with
the VSA compression gate zero-initialised from the base and always on under
training (`coderef/FastVideo/fastvideo/models/dits/minimax_h3.py`, the
`to_gate_compress` comments). Core's dense forward ignores that gate
(`comfy/ldm/minimax/model.py`, the `to_gate_compress` comment).

- **P1 (#4, timestep conditioning).** Its only supervision is at the eight
  rung times, so the adaln modulation change from base to FastH3 is larger
  at the rung times than between them. Falsified if the change is smooth
  across t with no structure at the rungs; that would read as a learned
  time remap rather than per-rung fitting.
- **P2 (identity).** The two bf16 files share the time table and the curve
  basis bit for bit, and each rebuilds its shipped int8 file under
  `bake_pdd_checkpoint.TIES_CRITERION`. If the time tables differ, adaln is
  compared on modulation outputs only.
- **P3 (change map).** The FastH3 delta is small against the weights and
  close to orthogonal to both LoRA deltas on the same module (cosine near
  zero), because DMD2 and PDD/VSD optimise different objectives from the
  same teacher. A cosine well above zero on some kind or depth would name
  where the distills agree, which is where #5's sums are least likely to
  be awful.
- **P4 (novsa arm).** `fasth3_8step_contract_novsa` departs from the
  contract twice: dense for sparse attention, and the learned gate branch
  dropped. A worse look there cannot be charged to sparsity alone.
