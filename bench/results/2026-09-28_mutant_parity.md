# h3-mutant-distill's node against this pack's: render parity (2026-09-28)

`H3ExactLoRA` (`standalone/h3_mutant_distill/exact_lora.py`, blob
`aab3b1c4`, unchanged from its first commit `7bb0fd09`) is a trimmed rewrite
of `MiniMaxH3PDDLoRA` (`backbone_apply="exact branch"`, heads patched) and
`MiniMaxH3LoRABranch`. This asks whether it renders the same.

**Result: on all six recipes, the final video and audio latents are
`torch.equal` to the pack's.**

## How

- **Graphs:** `bench/check_mutant_parity.py graphs` takes each recipe's pack
  graph and writes it twice: as shipped (`pack`), and with the pack's LoRA
  nodes swapped for `H3ExactLoRA` (`ours`). Both save the final AV latent,
  split into video and audio. Everything else is the shipped graph: Sol-Attn,
  kitchen attention, the pack's conditioning, seed 730451892, the graph's own
  canvas and length.
- **Sigmas pinned on two pairs.** The i2v and ref2va PDD graphs feed the
  sampler from the PDD node's own SIGMAS output, which `H3ExactLoRA` does not
  have. Both arms of those pairs get the same `ManualSigmas`
  (`h3_config.PDD8_SIGMAS`) instead, so the pair still differs only in the
  LoRA node.
- **Renders:** `bench/run_graph_arms.py`, one run per arm, pack then ours,
  on the server started 14:53 with the node loaded. Rows:
  `2026-09-28_mutant_parity.jsonl` (the three t2v pairs) and
  `2026-09-28_mutant_parity_i2v_r2v.jsonl` (i2v and ref2va). No row carries
  an error.
- **Compare:** `check_mutant_parity.py compare` on the saved `.latent` files
  (`latents/mutant_parity_<recipe>_<arm>_<stream>_*` on the output share).

## Results

| recipe | pack graph | replaced | video | audio |
|---|---|---|---|---|
| PDD8 then FlashGen finish, t2v | `h3_text_to_video_pdd8_flashgen_finish` | PDD node + branch | equal | equal |
| PDD6, t2v | `h3_text_to_video_pdd_manual_sigmas` | PDD node | equal | equal |
| FlashGen on blocks 34-49, t2v | `distill_experiments/h3_text_to_video_flashgen_late_blocks` | branch | equal | equal |
| PDD8, i2v | `distill_experiments/h3_first_frame_to_video_pdd_savelat` | PDD node | equal | equal |
| PDD8, ref2va | `h3_image_ref_plus_text_to_video_pdd` | PDD node | equal | equal |
| FlashGen, ref2va | `distill_experiments/h3_probe_r2v_flashgen_4step` | branch | equal | equal |

## Controls and caveats

- **The comparison can fail:** PDD6's latent against the finish recipe's is
  red. `check_mutant_parity.py static` has its own controls (a block
  selection one grid point off, a strength scaled by 1.0001), both red.
- **`finish_pack` is a cache hit** (`suspect_cache_hit`, sampler 0 s). Its
  first submission sampled with the pack's nodes and then failed at the save
  (SaveLatent cannot take the nested AV latent; fixed in `5b453bc8`). The
  resubmission reused that sampler output. The latent compared is therefore
  a genuine pack render, from the same server process.
- The row `substrate.git_commit` moves between rows because the tree kept
  moving. None of those commits changed `pdd_lora.py`, `lora_branch.py` or
  `pdd_math.py`; `d0812265` only moved `exact_lora.py`, unchanged.
