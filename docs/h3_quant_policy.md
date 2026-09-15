# H3 quantization policy: which blocks get which precision, and the plan to earn each row

Last updated: 2026-09-15. Owner-approved plan, executing. Model throughout:
MiniMax H3, every DiT checkpoint variant on this box (the finding behind
this page is a base-model property; `docs/h3_block49_quant_error.md`).

## The idea

A quantized H3 is not one setting. It is a per-block policy: for each of the
fifty DiT blocks, what precision the attention kernel quantizes q/k/v to,
and what precision the linears are stored at. Until 2026-09-15 that policy
was "INT8 everywhere" for attention and "int8 convrot everywhere" for the
linears, with the first-two-blocks-dense recipe inherited from upstream and
never measured here. The block-49 work replaced the guess with a mechanism
and a first visible result (`docs/h3_block49_quant_error.md`, sections 1-6):
INT8 attention costs something visible at the last blocks, both free levers
help, and exact attention on the three lopsided blocks helps most.

This page is where the policy lives once each row has evidence. Rows carry a
status; an empty status means nobody has measured it and the row is a
placeholder, not a recommendation.

## The policy table (attention)

| blocks | attention | status | evidence |
|---|---|---|---|
| 0-44, 46, 47 | INT8 (sage per-thread with `qk_balance`; Sol as shipped) | proposed | flat K-norm weights; balancing gated off per head where nothing is loud |
| 45, 48, 49 | bf16 (`MiniMaxH3ExactBlocks`) until Tier 1/2 close the gap; then INT8 balanced | proposed, one scene one seed | market scene 2026-09-15: shipped morphs, balanced does not, exact tail best |

## The policy table (linears)

| blocks | linears | status | evidence |
|---|---|---|---|
| all | int8 convrot (`qkv_proj`, `out_proj`, `fc1`, `fc2`) | shipped | inherited; no sensitivity measurement exists here |
| tail | bf16? | unmeasured | TaoMate protects 0, 1, 47, 48, 49 on its W8A8 path; near-miss against our loud set, and a different surface |

## The plan, in tiers

Each tier is usable on its own; each later tier removes a cost of the one
before.

**Tier 0 (config only, today): the policy as a graph.** `h3_probe_t2v_policy`
carries the channel-balance node on the loud blocks, sage in `fp8++
balanced`, and blocks 45/48/49 on exact attention. A few percent of render
time for the full fix at those blocks plus the free levers elsewhere. The
market scene is its regression witness: the porter morphing as he turns is
the visible failure any future change is checked against.

**Tier 1 (days): the free levers reach Sol's steps.** The per-head q/k
balance factor into Sol's own quantizer (`quant_k_rows` / `quant_q_rows`,
computed in the preprocess pass beside the key mean it already takes), so
the routed steps get what the sage steps get. Graded by
`bench/grade_channel_balance.py` on captures; then the witness re-rendered
with balanced everywhere and no bf16 blocks. If that matches exact tail,
the bf16 row disappears and the policy is "levers on."

**Tier 2 (a week): INT8 that survives the loud blocks.** Mixed-granularity K
in both quantizers: the loud channels permuted into a 16-channel group with
its own scale, two accumulators. Goal: INT8 at block 49 as good as bf16, so
the tail needs no exception and every user of these kernels gets it.

**Tier 3 (a day, independent): a sensitivity-ranked bake for the linears.**
Capture one step's linear inputs across all fifty blocks, score each
linear's tolerance for 4 bits with the kitchen's own quantizers, bake a
mixed checkpoint with the tail protected on evidence. The VRAM half of a
better quant.

## Today's order (2026-09-15), GPU-sequential, code in parallel

1. Plan page (this), policy graph in the generator. No card.
2. Kitchen branch off `h3-build`: Sol quantizer balance factor. CUDA, then
   build; install only if the grade says yes.
3. When the card frees: the six diner renders (three arms, two seeds),
   scored as they land, originals in listed order.
4. Captures of blocks 45 and 48, so the two blocks the weights flag can be
   graded like 49 was.
5. Grade the rebuilt wheel on captures; re-render the witness with
   balanced-everywhere and no bf16 blocks.
6. Fill this page's rows with what the day earned; flip the defaults it
   justifies, telling the freeze session first; tag the served sage build
   if the fork changed.

Not today: Tier 2 and Tier 3.

## Instrumentation this plan needs

- Captures of blocks 45 and 48 (every measurement so far is on 49).
- The audio question: the ceiling arm sounded louder and crisper on one
  seed. Outside the mechanism; if the diner seeds agree it is its own
  thread.
- A second scene at two seeds before any row moves from "proposed" to
  "measured" (the diner batch).

## Status log

- 2026-09-15: page created; Tier 0 graph added to the generator; Tier 1
  kernel work started in the kitchen fork.
