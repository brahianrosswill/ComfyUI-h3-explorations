# Tier 1 pair plus the default, restaurant-kitchen prompt (2026-09-15)

Third scene for the block-49 question. Model: MiniMax H3 fl2va pruned
int8 convrot. Prompt: `prompt_bank/t2va_restaurant_kitchen.txt`, unchanged.
Seed 730451892 on every arm. Kernel: the served `0.2.34+sol.5284cfb`
(the `default` arm rendered after the TaoMate session's restart onto the
same wheel plus its new node; nothing in the attention chain changed).

**On the word "default".** Earlier records in this series say "shipped"
for the arm with no rebalance. That means the owner's default
text-to-video graph on this box as of this date (`h3_text_to_video_api.json`:
sage `fp8++` plus Sol at the recipe in `workflows/h3_config.py`), not
anyone else's defaults and not a released setting. From this record on it
is called `default`.

| clip | arm | attention chain |
|---|---|---|
| default | the owner's default graph | sage fp8++ + Sol, no rebalance, INT8 everywhere |
| levers | every free lever | + balance node on 45/48/49, sage `fp8++ balanced`, Sol `qk_balance` |
| policy | levers + bf16 tail | the above + blocks 45/48/49 on bf16 attention |

## Outputs

Under the server's output directory, `Video/block49_kitchen/`:

- `levers_s730451892_00001-audio.mp4` (prompt id `55444991-dcb5-4cf3-8b32-e52894a6e2cb`)
- `policy_s730451892_00001-audio.mp4` (prompt id `73bdd13e-8c04-4ae7-93fa-0b3dd8e44c7f`)
- `default_s730451892_00001-audio.mp4` (prompt id `e7bff358-0b69-4145-8144-e4046bdd20e5`)

## Wall time (server history, seconds)

| arm | seconds |
|---|---|
| levers | 485 |
| policy | 560 |
| default | 486 |

## Owner's scoring

(unfilled)

## Stacks

Captioned vertical stacks (video only), built by `bench/stack_labeled_clips.py`:
`Video/block49_kitchen/stack_kitchen_default_rebalanced_bf16tail.mp4` and,
for the market scene, `Video/block49_eval/stack_market_default_rebalanced_bf16tail.mp4`.
Band order top to bottom: default, rebalanced, rebalanced + bf16 tail;
each caption carries the arm, seed, sampler and wall time.
