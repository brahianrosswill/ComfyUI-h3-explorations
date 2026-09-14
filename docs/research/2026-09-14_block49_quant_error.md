# Why block 49 has so much quantization error, and what can be done about it

Last updated: 2026-09-14. Written from the sage fork's session at the owner's
request. Model throughout: **MiniMax H3, the pruned int8 convrot fl2va
checkpoint** (`h3_config.MODELS["unet_fl2va"]`), the one the capture set was
rendered with and the one every graph here ships. Capture: the 2026-09-03
base16 t2v set at 1344x768, S=104,361 (395 text, 1,150 audio, 102,816 video
rows), kept to 2026-09-20.

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
| per head, from the capture, RoPE-pair-equal (a=0.5) | 0.0381 (-19%) | 0.0088 (+3.5%) | needs a per-head pass; not built |
| per channel from the checkpoint's norm weights, pair-equal (a=0.5), CPU simulation of the QK side | 0.0587 vs 0.0666 plain (-12%) | 0.0062 vs 0.0062 (neutral) | `MiniMaxH3ChannelBalance`, folded into the norm weights, free |

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

## 5. What this does not establish

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

## Records

- Sage fork `CHANGELOG.md`: decision log "sm89 q/k quantization" and
  "`smooth_k` on H3: graded across the trajectory"; workload intel "MiniMax
  H3, block 49: per-channel K balancing". Harnesses under `tests/spikes/`
  there: `spike_h3_real_activations.py`, `spike_h3_k_channel_balance.py`,
  `spike_h3_block49_error_anatomy.py`.
- This repo: `docs/roadmap.md` "block 49 attributed, at the input level"
  (2026-08-20) and `bench/results/2026-08-20_head_magnitudes*.json`;
  `docs/SOLATTN.md` "The defaults, re-read against the sage-side error
  records, 2026-09-14"; `bench/results/2026-09-14_channel_balance_*.json`.
