# Block 49: why INT8 attention loses accuracy on H3's last blocks, and what has been done about it

Last updated: 2026-09-15 (moved from `docs/research/` and revised). Written from the sage fork's session at the owner's
request. Model throughout: **MiniMax H3, the pruned int8 convrot fl2va
checkpoint** (`h3_config.MODELS["unet_fl2va"]`), the one the capture set was
rendered with and the one every graph here ships; the weights finding holds
for every H3 DiT checkpoint on disk, see "So what". Capture: the 2026-09-03
base16 t2v set at 1344x768, S=104,361 (395 text, 1,150 audio, 102,816 video
rows), kept to 2026-09-20.

## So what, revised 2026-09-15

**It is not a bug, and it is not ours.** It is a design property of INT8
attention meeting this model: every INT8 attention kernel quantizes K with
one scale shared across a row's or a block's 128 channels, which is fine
where channels are alike, and H3's last blocks are not alike -- the released
weights put an order of magnitude of gain on four channels there. Nobody's
kernel is wrong; the model's weights and the quantizer's granularity are a
bad match at three blocks.

**The blast radius is every H3 user on quantized attention.** Stock
comfy-kitchen's Sol kernel (Comfy-Org's and kijai's; the per-row K
quantizer shares its scale across channels), stock SageAttention upstream
and every fork of it, in every mode including the "accurate" fp16 one (all
of them quantize QK to INT8), and by the same mechanism NVLabs' own INT8
Sol kernels, unmeasured here. Every H3 checkpoint variant: the loud channels
are identical across all fourteen full DiT files on this box, and the
turbo/SLA/PDD LoRAs carry no norm weights, so they inherit it. Untouched:
anyone on full-precision attention (flash or SDPA in bf16), which has no
scale to share.

**It is a quality effect, not a correctness one.** Renders complete and are
plausible. The error lands on the last block's sharp read of the text rows
(section 5), so if it is ever visible it will be as prompt adherence at
the output head, not as texture. Whether it is visible is unknown, for
everyone, not only here.

**It is fixable where the quantizers are, and this box owns both.** The
identity `q . k == (q * f) . (k / f)` lets K's loud channels be rebalanced
against Q before quantization at no cost to the attention math. Two forms
exist, both off by default:

| lever | where | reaches | block 49 INT8 error | cost |
|---|---|---|---|---|
| `MiniMaxH3ChannelBalance` (this pack) | per-channel factor from the checkpoint's norm weights, folded into `q_norm`/`k_norm` at load | sage steps and Sol steps | Sol -13%, sage -7% (8 heads) | none at render time |
| `qk_balance` (sage fork v0.7.19) | per-head factor from per-call channel norms, inside the per-thread quantizer, gated per head | sage steps only | sage -23% (all heads) | +0.7% call, no memory |

Neither reaches the rest: with the best lever on, block 49 still sits at
several times block 0, because the block's attention shape amplifies
whatever rounding remains, and only finer K scaling inside a kernel touches
that.

**Where this stands, and what it would take to call it solved:**

1. *Diagnosed.* Closed. Sections 1-5 are the evidence; the checkpoint scan
   and attention-target record is `bench/results/2026-09-14_block49_checkpoint_scan_and_targets.txt`.
2. *Two levers built and measured on captures.* Closed for what they are.
   Nothing is switched on, so a render today is exactly what it was before
   this page existed.
3. *Open: is any of it visible?* The blind multi-scene comparison in
   `docs/SOLATTN.md`'s decision standard, with one probe arm wired to the
   node (inputs in the generator's hands: `balance` "loud blocks (from
   weights)", alpha 0.5) and one with the fork's `qk_balance` on, against
   the unchanged graph. The only instrument for this question.
4. *Open: Sol's quantizer.* `quant_k_rows` / `quant_q_rows` in the kitchen
   fork do not take the per-head factor; the routed steps get only the
   weights fold. The same factor into those two functions, sharing the
   key-mean pass they already do, is the remaining half; `bench/grade_channel_balance.py`
   grades it. Kitchen build as of 2026-09-15 is `0.2.34+sol.2aff3c5`
   (`docs/sol_upstream.md`), which changed nothing on this axis.
5. *Open, and not ours alone:* the mechanism, the checkpoint scan and the
   fold numbers are a contribution the kitchen maintainers could act on
   for every user; the sage-side change is one commit anyone forking sage
   could take. Neither has been sent anywhere.
6. *Deeper, not started:* finer K scaling inside the kernels (per-channel
   groups, or splitting the loud channels into their own scale), the LLM
   world's per-channel key quantization done in an attention kernel. Real
   kernel work on either side, uncertain payoff beyond the fifth-to-third
   already recoverable.

## The answer in four sentences

Block 49's `k_norm.weight` puts an order of magnitude more gain on a few
channels than on the rest, so after RMSNorm four K channels carry ~93% of the
block's K energy. Both INT8 attention kernels on these graphs quantize K with one
scale across all 128 channels (sage per 64-token block, Sol per key row), so
those four set the scale and the other 124 keep about three bits. Block 49's attention is also
the peakiest in the model, a handful of effective keys per query on its worst
heads with logits spanning over a hundred, so a small relative error in K
becomes a large logit error and the softmax flips. Loud channels make K's
rounding coarse; peaky attention makes the coarse rounding expensive; the
product is a block whose INT8 error is five times block 0's, almost all of it
on the K side.

Everything below is the evidence for each clause, with where it lives.

## 1. It is the weights, and it is not only block 49

Ranking all fifty blocks of the shipped checkpoint by the energy share of
the four loudest `k_norm.weight` channels (`bench/check_channel_balance.py`
prints it; no card, no capture):

| block | top-4 K-norm energy share | max / median weight |
|---|---|---|
| 49 | 70% | 16x |
| 45 | 31% | 7x |
| 48 | 25% | 6x |
| every other block | 4-6% | 1.0-1.6x |

`q_norm.weight` peaks at the same channels at the same blocks (69% at 49).
The weight peaks at block 49 sit at channels 82 and 19; H3's RoPE is
split-half over channels 0-95 and rotates (i, i+48) together, so each peak
spreads into its mate and the activations show the pairs 34/82 and 19/67.
That is the four-channel set the 2026-08-20 head-magnitude analysis found
(`docs/roadmap.md`, "block 49 attributed, at the input level") and the sage
fork's capture spike reproduced exactly (share 93.2% at block 49, 5-11% at
blocks 0, 32, 40).

Blocks 45 and 48 were never captured. The weights say they carry the same
defect at a third to a half of block 49's strength.

## 2. It is K's rounding, not Q's, not the PV side

A CPU decomposition on the captured block-49 and block-0 cells, fp32 exact
attention on 1,024 sampled query rows (256 text, 256 audio, 512 video) over
all keys and all 56 heads, with each of sage's quantization steps simulated
alone (`tests/spikes/spike_h3_block49_error_anatomy.py` in the sage fork;
K per 64-token block with one scale, Q per token; the fp8 PV side is not
simulated, see the kernel records for that split):

| arm | block 49 | block 0 | 49 / 0 |
|---|---|---|---|
| K int8 alone | 0.0645 | 0.0031 | 21x |
| Q int8 alone | 0.0159 | 0.0052 | 3x |
| Q and K int8 | 0.0666 | 0.0062 | 11x |

K's rounding is essentially the whole QK error at block 49, and it is
twenty times block 0's. Q's is small because its scale is per token and no
single channel dominates a token the way the loud channels dominate a
64-token block of K.

The fp8 PV side, from the kernel records on the same cell
(`spike_h3_real_activations.py`, fp8++ against the fp16 kernel): 0.0472
against 0.0409 at block 49, so the fp8 P and V storage adds about a seventh
on top of the INT8 QK error. Real, and not the mechanism.

## 3. It is amplified by how block 49 attends

Same run, the exact reference's own shape, median over heads and sampled rows:

| | row max p | effective keys | logit range (max minus mean) |
|---|---|---|---|
| block 0, text queries | 0.002 | 25,602 | 7.0 |
| block 0, video queries | 0.003 | 10,653 | 8.9 |
| block 49, text queries | 0.076 | 340 | 15.6 |
| block 49, video queries | 0.095 | 165 | 17.5 |
| block 49, worst-4 heads | 0.453 | 5 | 145.5 |

Block 0 spreads attention over ten thousand keys; block 49 over a few
hundred, and its four worst heads over about five, with logits a hundred
times wider than block 0's. An INT8 error in K is a relative error in the
logit; the same relative error on a logit of 145 moves the softmax by orders
of magnitude more than on a logit of 7. This is why the same quantizer, at
the same granularity, costs five times more at the last block.

The four worst heads by K-rounding error at block 49 are 17, 9, 11 and 22,
each at five to seven times the block's median. Head 11 is the head that
loses under token routing at every step
(`2026-09-04_sol_token_aug_grade.md`), which had no mechanism recorded; a
head that attends to five keys with a hundred-logit range is a candidate.
Hypothesis, not measured.

Text query rows carry the most error at block 49 (0.11 against 0.04 for
video rows). They are sparse queries under the shipped
`sink_conditioning=exact_kv_and_rows`, which runs the audio rows dense and
the text rows not; that is a sparsity-term observation and outside this
page's instrument, noted because the two rankings coincide.

## 4. What can be done, measured

The dot product is invariant under a per-channel rescale of q against k, so
the channels can be rebalanced before quantization at no cost to the math.
Sol's routing threshold is invariant under the same rescale. Measured on the
block-49 cell, sage fp8++ as served, mean rtol against fp32 attention:

| factor form | block 49 | block 0 | where it lives |
|---|---|---|---|
| none | 0.0472 | 0.0085 | |
| per head, from the capture, RoPE-pair-equal (a=0.5) | 0.0381 (-19%) | 0.0088 (+3.5%) | superseded by the in-quantizer form below |
| per channel from the checkpoint's norm weights, pair-equal (a=0.5), CPU simulation of the QK side | 0.0587 vs 0.0666 plain (-12%) | 0.0062 vs 0.0062 (neutral) | `MiniMaxH3ChannelBalance`, folded into the norm weights, free |

**Built 2026-09-15 in the sage fork (v0.7.19, `qk_balance`):** the
per-head factor inside the per-thread INT8 quantizer itself, computed per
call from copy-free channel norms, gated per head on K's loud-channel
share, no RoPE-pair constraint (it acts after RoPE), no calibration, no
extra q/k copy. Kernel-level on the same cells: block 49 0.0472 -> 0.0364
(-22.9%), blocks 40 and 0 unchanged; cost +0.7% on the call at the frame
ceiling and no change in peak memory. Off by default there until a render
check; the record is the fork's CHANGELOG v0.7.19. That is the form to
reach for on the sage steps; this node's weights fold remains the form
that also reaches Sol's kernel, until the same factor is put into Sol's
quantizer.

The alpha sweep put 0.5 at the optimum on both captured block-49 steps;
fully equalizing K (a=1.0) is bad everywhere because Q then carries the whole
scale. The per-channel form gets roughly half the per-head gain because the
loud channels differ in strength across heads and a [128] weight cannot
express that; it is exactly neutral at a flat block, which the per-head form
is not.

**Both kernels share the mechanism, at different strengths.** Read from the
installed kitchen build's source (`comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh`,
`quant_k_rows`, the checkout the wheel was built from): Sol quantizes K per
key row, after subtracting the per-channel key mean, with one INT8 scale
across the row's 128 channels; sage quantizes K per 64-token block with one
scale across the block. Sol's is finer along tokens and has mean-centring
built in, so its loud-channel penalty is smaller than sage's on the same
block, but a row's scale is still set by its loudest channel, so the
rebalancing reaches it. Its routing threshold (eager reference,
`comfy_kitchen/backends/eager/sol_attn.py`: query-block centroid squared
times the per-channel variance of the key-block centroids) is invariant
under the paired rescale in exact arithmetic, so the fold changes which
keys Sol quantizes coarsely, not which blocks it routes.

`MiniMaxH3ChannelBalance` (`channel_balance.py`) is the per-channel form:
two weight patches per block through `ModelPatcher.add_patches`, combo
`balance` off by default, "loud blocks (from weights)" selecting by the
ranking above, "named blocks" taking `dense_blocks` syntax.
`bench/check_channel_balance.py` pins the fold's exactness, its RoPE safety,
the off default and the shipped ranking without a GPU.
`bench/grade_channel_balance.py` is step 3 of the experiment in
`docs/SOLATTN.md`: the Sol kernel's own INT8 term, plain against balanced,
through `analyze_sol_error.py`'s decomposition (kernel called as the node
calls it, since the oracle's own kernel wrapper predates comfy-kitchen#117).
Run 2026-09-14, tau 1.0, first 8 heads, factor from the weights,
`bench/results/2026-09-14_channel_balance_{b49_s15,b0_s15}.json`:

| cell | arm | Sol sparsity_l2 | Sol quant_l2 | Sol total_l2 | sage fp8++ l2 |
|---|---|---|---|---|---|
| block 49, step 15 | plain | 0.0324 | 0.0265 | 0.0415 | 0.0487 |
| block 49, step 15 | balanced | 0.0323 | **0.0231 (-12.9%)** | 0.0393 | 0.0453 |
| block 0, step 15 | plain | 0.1247 | 0.0024 | 0.1244 | 0.0036 |
| block 0, step 15 | balanced | 0.1248 | 0.0024 (+0.9%) | 0.1244 | 0.0036 |

Sol's kernel benefits, by about the same fraction as sage's, and the fold is
neutral at block 0 on both. Routing is unchanged: the eager Sol reference
moves by exactly as much as exact attention does under the bf16 re-rounding
of the balanced inputs (1.69e-2 against 1.67e-2 at block 49, 5.4e-4 against
5.4e-4 at block 0), which is the invariance argument holding in practice.
The sparsity term does not move. So on the shipped stack the fold lowers
the last block's INT8 term for both the dense-window steps and the routed
steps, at no cost, and does nothing anywhere the weights are flat.

## 5. What the peaky heads attend to

Measured on the same two cells, 768 sampled query rows per cell over all
keys, all 56 heads, exact fp32 softmax
(`bench/results/2026-09-14_block49_checkpoint_scan_and_targets.txt`):

| | block 0 | block 49 |
|---|---|---|
| attention mass on the 395 text keys, median head | 0.4% | 12.2% |
| attention mass on the 1,150 audio keys, median head | 17.7% | 16.2% |
| queries whose top-1 key is a text key, median head | 4% | 20% |
| queries sharing one top-1 key (sink signature), median / max head | 2% / 67% | 7% / 34% |
| loudest key row's norm over the median, worst heads | 1.0-1.1x | 1.0-1.4x |

The four worst heads by K-rounding error (17, 9, 11, 22) put 6-38% of
their mass on text keys and their top-1 keys are text tokens (rows 0 and
114) for a tenth to a fifth of queries, with no key-norm outlier. Head 11,
the token-routing loser, is the heaviest text reader of the four at 38%.
So the block-49 error is concentrated on the prompt read, not on a sink
token and not on video-to-video attention.

## 6. What this does not establish

- Whether a fifth less INT8 error at the last block is visible in a clip.
  Nothing here is perceptual; the blind comparison in `docs/SOLATTN.md`'s
  decision standard is the only instrument for that.
- Whether blocks 45 and 48 behave like 49 under balancing. The weights say
  they carry the defect; no capture exists to grade them.
- The remaining four-fifths. After balancing, block 49 still sits at several
  times block 0, and section 3 says why: the attention shape is the
  amplifier, and no input-side rescale changes it. Finer K granularity in
  the kernel (per-channel or per-32-token scales) is the lever for that, and
  it is a kernel change on either side, not a weights fold.

## 7. The kernel-side fix, built 2026-09-15: `qk_balance` in the sage fork

### What was done

The weights fold in section 4 is capped by the architecture: the only
per-channel weights after the projection are the q/k RMSNorm gains, which
are shared across heads, so a per-head factor, which the simulation put at
twice the gain, had nowhere to live. The sage fork's per-thread INT8
quantizer is the other place the factor can be applied, and it streams q
and k exactly once, so the multiply is free. Built there as `qk_balance`
(sage fork v0.7.19, its `CHANGELOG.md` and `tests/test_qk_balance.py`):

- Both Triton quant kernels take a factor pointer, `[B, H_kv, C]` fp32,
  and multiply it in right after the load, before the absmax: Q by `f`,
  K by `1/f`. `q . k == (q * f) . (k / f)`, so the attention math is
  unchanged; only the INT8 rounding moves.
- `f = rms_k^0.5 / rms_q^0.5` per (batch, kv head, channel), geometric
  mean one per head, computed per call from `torch.linalg.vector_norm`
  with fp32 accumulation, so no fp32 or bf16 copy of q or k is ever made.
  That is the difference from `smooth_k`, which this stack rejected for
  materializing a K copy at the frame ceiling.
- Gated per head on the energy share of K's four loudest channels; below
  the threshold the head's codes are bit-identical to the plain path.
  Under GQA the factor is per kv head and each query head reads its
  group's.
- No calibration, no capture, no RoPE-pair constraint (it acts after
  RoPE), no per-block list: blocks 45, 48 and 49 open on their own.

### What it measured

Same cells and reference as the rest of this page, sage fp8++ as served:

| cell | plain | `qk_balance` (gate 0.2, the default) | gate 0.5 |
|---|---|---|---|
| block 49, step 15 | 0.0472 | **0.0364 (-22.9%)** | 0.0412 (-12.8%) |
| block 40, step 15 | 0.0426 | 0.0426 (+0.1%) | (no head opens) |
| block 0, step 15 | 0.0085 | 0.0085 (-0.2%) | 0.0085 (+0.0%) |

Per head, in the CPU simulation of the QK side: block 49 loses a third of
its QK error at the default gate and the worst head (17) three-quarters
of its own; block 0 shows a 3% cost in that simulation that the real
kernel's Q rounding and fp8 PV error dilute to nothing, which is why the
default was chosen on the kernel rows. Cost: the two norm passes, +3.3 ms
on the quant step at 104k rows, +0.7% on the whole call, and no change in
peak memory.

For scale against the other numbers on this page: the balanced fp8++ call
at block 49 (0.0364) is below the fp16 kernel's unbalanced error on the
same cell (0.0409), so on this block the fast path with balancing is now
more accurate than the accurate path without it.

### Why it matters

- It removes the identified mechanism at the source, in the code that
  quantizes, rather than working around it in the weights. Every H3
  variant benefits identically because the loud channels are identical
  across them; any other model with late-block outlier channels benefits
  without anyone naming a block.
- It lands the correction on the block that reads the prompt at the
  output head, with nothing after it to absorb error, which is the most
  plausible place for INT8 attention to show up as prompt adherence.
- It costs nothing that this card is short of: no memory, and under a
  percent of the call.
- It is the same idea the LLM quantization world settled on
  (SmoothQuant's migration of difficulty from activations to the other
  operand; KVQuant's per-channel keys), applied to the attention inputs
  of a DiT, in the dynamic per-call form rather than the static
  calibrated one.

### What it does not yet cover

- **Sol's steps.** The routed steps run Sol's kernel, whose quantizer
  (`quant_k_rows` / `quant_q_rows` in the kitchen fork) does not have the
  factor yet. Until it does, `qk_balance` reaches only the sage steps:
  the dense window, every `dense_blocks` entry, and the token-refiner
  calls. This node's weights fold is the form that reaches Sol today, at
  the smaller, head-shared gain (section 4). Putting the same factor into
  Sol's quantizer, sharing the key-mean pass it already does, is the
  remaining half.
- **Whether it is visible.** Off by default in the fork. Turning it on
  changes numerics on every served render, so it wants the blind
  comparison this repo requires of a default, and the freeze session
  told first.

## 8. Records

- Sage fork `CHANGELOG.md`: decision log "sm89 q/k quantization" and
  "`smooth_k` on H3: graded across the trajectory"; workload intel "MiniMax
  H3, block 49: per-channel K balancing". Harnesses under `tests/spikes/`
  there: `spike_h3_real_activations.py`, `spike_h3_k_channel_balance.py`,
  `spike_h3_block49_error_anatomy.py`.
- This repo: `docs/roadmap.md` "block 49 attributed, at the input level"
  (2026-08-20) and `bench/results/2026-08-20_head_magnitudes*.json`;
  `docs/SOLATTN.md` "The defaults, re-read against the sage-side error
  records, 2026-09-14"; `bench/results/2026-09-14_channel_balance_*.json`.
