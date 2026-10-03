# Does a different reference still break a per-head tau table? (2026-10-03)

**Result: the reference does not break it. A guard with no margin does not
carry exactly, in either direction. No table ships.** The saving and the
ranking carry almost whole to a different still; on a fifth to a third of
blocks one head ends up somewhat worse than the worst head the shipped tau
runs. The lever is the owner's to keep or park; this record's reading of the
rule is at the end.

**The question** (owner, 2026-10-03): "lets see if a different reference
breaks it", with the rule that if it does, "that kills that idea". Stated
before the run, by ditman: fit a table on reference A, apply it to the same
prompt and seed with reference B; the lever closes if most of the saving in
routed share is lost on B or a head on B goes past the per-head guard; the
rank correlation of per-head error between A and B is the supporting number.

**Conditions.** Two sweep renders (`sol_tau_sweep.py`, `H3_SOL_SWEEP`), one
server, `bench/run_tau_sweep.py --scene refA --scene refB=image:<still>` on
`workflows/distill_experiments/h3_probe_r2v_step_switch_pdd8_flashgen_h080_savelat_api.json`:
market prompt, seed 730451892, 1344x768, 345 frames, both samplers, all 50
blocks, eight tau values from 0.25 to 3.0, on the fallback's trajectory. A is
the graph's own still, B the backstage scene's. Kitchen
`0.2.37+sol.6272371.up.be003b7`, pack at `847c587a`. Fitted and scored by
`bench/calibrate_sparse_table.py --target equal_error --dense-blocks
38,39,40,41,42,49` into `2026-10-03_per_head_ref_kill.json`; the sweep's own
record is gitignored under `data/sparse/captures/`.

**Layout.** Shared 2048 encoder view, the still's VAE rows wired, one
reference. A: 119,485 tokens, 7,360 reference rows. B: 119,357 tokens, 7,296
reference rows. So B changes what the reference shows far more than how many
rows it has. The 512 encoder view became the default the same day
(`a36ca547`); on that layout the conditioning prefix is about half as long
and a table would have to be fitted again.

## Fitted on A, scored on B

Per stage, over the 44 blocks the shipped graph runs sparse. "Error" is a
block's pooled relative error on the video rows against the dense kernel, as
a ratio to the shipped tau's on the same render. "Routed" is routed key
blocks on the video rows, the same way.

| | first stage (PDD8, six evaluations) | finisher (FlashGen, two) |
|---|--:|--:|
| routed, where fitted (A) | 0.892 | 0.869 |
| routed, on B | 0.894 | 0.872 |
| share of the saving surviving on B | 0.98 | 0.97 |
| block error on B: median, worst block | 0.995, 1.031 | 0.993, 1.073 |
| blocks where a head passes the guard on B | 9 of 44 | 15 of 44 |
| that head's excess over the guard: median, worst | 1.02, 1.06 | 1.07, 1.19 |
| rank correlation of per-head error, A against B: median, lowest block | 0.993, 0.980 | 0.988, 0.960 |

- **The saving carries.** Almost all of it survives a different still.
- **The ranking carries.** Which heads tolerate sparsity is the same on both
  stills, block by block. For scale, the 2026-10-02 probe records gave 0.996
  across a seed and 0.963 across a prompt and seed
  (`2026-10-03_r2v_finish_time_budget.md`, "What a tau per head would buy").
- **The guard does not carry exactly.** The guard is "no head worse than the
  worst head the shipped tau runs in that block", enforced on A with no
  margin. On B a head passes it on a fifth of the first stage's blocks and a
  third of the finisher's, by a few percent in the first stage and by up to
  a fifth in the finisher.
- **That is not the reference's doing.** Fitted on B and scored on A, the
  same thing happens at the same size: the guard is passed on 11 and 12
  blocks, by up to 1.15 and 1.21, with the saving again intact. A bound met
  with no slack on one render is passed on another.

## What a table is worth, full size

Fitted on both stills with each block's held-out excess taken off its budget
(`summary` in the JSON), video rows only, conditioning rows exact as shipped:
routed key blocks fall to 0.90 of the shipped tau's in the first stage and
0.88 in the finisher. By the sweep's own kernel times (a call's milliseconds
are linear in its routed blocks to an r2 of 0.9997), that is about 9 s of a
predicted 135 s of Sol time in the first stage and about 3.4 s of 45 s in the
finisher: roughly 12 s of a render whose sampler is about 337 s. Fitted and
scored on the same two renders, so an upper bound.

## Reading the rule

By the owner's words, a different reference did not break it. By the rule as
written down before the run, its second clause is met: heads on B pass the
guard. Both are true, and they point different ways, so the lever is left
for the owner with this recommendation: **park it.** Not because the
reference broke it, but because of what it is worth and what it would take:
about 12 s a render at best; a lossy change that only the owner's glance can
accept (`2026-10-03_sol_output_check.md`); a guard that needs a margin, which
costs part of the 12 s; and a layout that changed the same day, so the table
would be fitted again before it could be used.

## What this does not show

- One prompt, one seed, two stills of nearly the same reference geometry. A
  still with a different aspect, two references, or an encoder-only or
  512-view layout is untested.
- Error is local, per call, against the dense INT8 kernel, on the dense
  trajectory. Nothing was rendered with a table and nothing was judged.
- The time saved is predicted from routed blocks, not measured on a render.
