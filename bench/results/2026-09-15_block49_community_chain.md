# The community chain on the market prompt: kitchen int8 attention dense + Sol (2026-09-15)

The chain most H3 users run: ComfyUI core's Model Attention Backend on
"comfy kitchen attention" (kitchen's `int8_attention`, which rotates q/k
before INT8 and is immune to the loud-channel mechanism,
`2026-09-15_ck_int8_attention_block49.json`) as the dense kernel, Sol on
the routed steps, sage absent. Our Sol node stands in for core's
block-sparse node on the same kernel. Graphs `h3_probe_t2v_ck*_api.json`
(generator `dense_attn="ck"`), market prompt, seed 730451892, the seed of
every other market arm today. Served wheel `0.2.34+sol.5284cfb`.

Server log for these renders (`user/comfyui_8188.log`): "[h3-sol] chaining
onto an existing attention override" (the backend node's), "chain assert,
call-time: no sage: probes ... reached no sage kernel", and for the third
arm "[h3-sol] keeping blocks [45, 48, 49] dense of 50".

| clip | arm | what carries the block-49 mechanism | wall time |
|---|---|---|---|
| ck | as most people run it | Sol's routed steps only; the dense steps are on the rotated kernel | 500 s |
| ck_balanced | + Sol `qk_balance` | nothing unrotated and unbalanced | 500 s |
| ck_dense_tail | + blocks 45/48/49 handed to the dense kernel (`dense_blocks`) | nothing at those blocks; Sol elsewhere | 516 s |

For comparison, our own chain on the same prompt and seed: sage-dense
default 508 s, all levers 511 s, levers plus bf16 tail 569 s. The dense tail
costs 16 s here against 60 s there, because this chain's fallback is the
INT8 rotated kernel, not bf16.

## Outputs

`Video/h3_probe_t2v_ck_00001-audio.mp4`, `Video/h3_probe_t2v_ck_balanced_00001-audio.mp4`,
`Video/h3_probe_t2v_ck_dense_tail_00001-audio.mp4`; captioned stack
`Video/block49_eval/stack_market_community_chain.mp4`. Prompt ids:
- `h3_probe_t2v_ck`: `0e6a1247-6e2c-4d5f-b51b-e43169d4ab26`
- `h3_probe_t2v_ck_balanced`: `2ad0ed48-5f4d-4359-8276-3012c1bf42a5`
- `h3_probe_t2v_ck_dense_tail`: `b4eceff6-99eb-40e0-af67-3c4ae6db872b`

## What it decides

Against `h3_t2v_00019` (our sage-dense default, the porter morphs) and
`h3_probe_t2v_levers_00001` (our chain with every lever on): if the plain
community chain already looks like our levers arm, the dense half of our
problem was our Sage node and nobody else's; if it morphs like our default,
the routed steps carry it visibly for everyone on block-sparse attention
and the Sol balance or the dense tail is worth shipping to them.

## Owner's scoring

(unfilled)
