# The Sol override takes core's container entry (2026-10-03)

`sol_attn_h3.py::make_override` now gives its override a `container_function`
when the fallback under it has one, so core's `wrap_attn` hands q, k and v
over in their single-owner containers and a call the node declines passes
them on untaken. This record is what that changed, measured. Board lever
`container-protocol`; asked for by the owner ("is this not low hanging
fruit?").

## Before any change: the two kitchen entries agree

The container entry swaps `comfy_kitchen.int8_attention` for
`prequantize_int8_attention` then `int8_attention_from_prequantized` on every
dense call. On three cells of the 2026-09-27 ref2va capture (blocks 49, 45
and 0; 120,582 tokens, 56 heads, bf16) the two return the same bytes. The
dense cells below repeat that comparison (`kitchen_entries_equal`).

## One attention call, with and without the entry

`bench/measure_sol_container_protocol.py`, rows in
`2026-10-03_sol_container_protocol.json`. Each cell goes through core's
`optimized_attention` as the H3 forward calls it, over the override core's
`set_model_optimized_attention` builds for kitchen's int8 backend. "Tensor
entry" is the same override with the attribute removed, which is the node as
it was. Nothing else on the card; peak is `torch.cuda.max_memory_allocated`
over the call, q, k and v included.

| cell | route | peak, tensor entry | peak, container entry | outputs | ms, second call |
|---|---|--:|--:|---|---|
| block 49, step 2 | dense_block | 9.52 GB | 7.79 GB | same bytes | 895.9 / 894.9 |
| block 45, step 6 | dense_block | 9.52 GB | 7.79 GB | same bytes | 899.0 / 897.6 |
| block 0, step 2 | sol | 10.24 GB | 10.24 GB | same bytes | 578.3 / 578.5 |
| block 24, step 6 | sol | 10.24 GB | 10.24 GB | same bytes | 484.5 / 481.7 |

- **A dense call's peak falls by 1.73 GB, one of the three bf16 tensors, not
  by all 5.19 GB of them.** The peak moves to the prequantize step, where the
  bf16 tensors and their int8 copies are alive together. The kernel itself
  then runs with 4.33 GB allocated where it ran with 9.52 GB (first check
  above, direct calls).
- **A Sol call peaks above a dense call on either entry.** On a render where
  Sol takes any call, attention's high-water mark is therefore a Sol call,
  and this change does not lower it. What it lowers is the dense calls: the
  blocks in `dense_blocks`, and every block of an evaluation outside the
  sigma window.
- No time change on either route.
- With no attention node under Sol, or the sage override, the override has no
  container entry and nothing changes: core's entry passes no `func`, so
  there is nothing to hand the containers to.

## One render, end to end

`workflows/distill_experiments/h3_probe_r2v_step_switch_pdd8_flashgen_h080_savelat_api.json`
at seed 730451892 on a freshly started, unarmed server, with the change in
the working tree. Row: `2026-10-03_sol_container_protocol_e2e.jsonl`. The
same graph (the rows' `graph_sha256` agree) at the same seed was rendered
that morning before the change (`2026-10-03_r2v_finish_e2e.jsonl`, second
row).

- All four saved latents are the same bytes as the morning's: video and
  audio, after the PDD stage and after the finisher. Control: the morning's
  two seeds differ from each other.
- Sampler 338.8 s on the first render after a server start, against 338.8 s
  for the morning's first render after a start. The server log names the
  fallback (`attention_comfy_kitchen_int8`, core's `ModelAttentionBackend`)
  and shows Sol routing.

## Not measured

- Peak memory inside a render. The per-call figures are with an empty card;
  a render has the DiT partly resident and the allocator under core's memory
  management.
- A graph with the sage node under Sol, or with nothing under it. Both keep
  the old path by construction; `bench/check_sol_node_equivalence.py` holds
  that the entry exists exactly when the fallback has one.
