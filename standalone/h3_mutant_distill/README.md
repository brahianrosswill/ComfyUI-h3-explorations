# h3-mutant-distill

**Experimental. YMMV.** Combinations of MiniMax H3's few-step distills for
ComfyUI, and one node to run them. Each was judged by eye, by one person, on
one seed, on an RTX 4090 at 1344x768 and 345 frames.

## Recipes

Text to video. One workflow each in `example_workflows/`, with a note on why.

| workflow | what | model evaluations |
|---|---|---|
| `h3_t2v_pdd8_flashgen_finish` | PDD8 from sigma 1.0 to 0.8, then FlashGen to 0 | 6 + 2 |
| `h3_t2v_pdd6` | the PDD8 LoRA on a 6-step schedule, for close-ups and low motion | 6 |
| `h3_t2v_flashgen_late_blocks` | FlashGen applied to DiT blocks 34-49 only; a curiosity | 4 |
| `h3_t2v_fasth3_contract` | FastVideo's FastH3 on FastVideo's own sampling settings; core nodes only | 8 |

## Install

Install through ComfyUI-Manager, or clone this repo into `custom_nodes/`. Open
a workflow: the frontend offers to download each missing model from its
Hugging Face URL.

## The node

`H3 Exact LoRA (FlashGen / PDD)` adds a LoRA at the call, `y = W x + B (A x)`,
instead of merging it into the weights. ComfyUI merges a LoRA into the int8
checkpoints by requantizing, which loses a change smaller than one int8 step:
most of FlashGen, and much of PDD. For a PDD file it also applies the
modulation update and fuses the per-step output heads.

## How it was made

- **FlashGen:** [Beidouqixing/minimax-h3-4step-lora-flashgen](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen),
  converted to ComfyUI at its full rank 64: keys renamed, q/k/v rows
  reordered, and the modulation update re-expressed in the pruned
  checkpoint's 8-column time basis. Hosted at
  [fbjr/h3-mutant-distill](https://huggingface.co/fbjr/h3-mutant-distill).
- **PDD8:** [alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs),
  converted, from [fbjr/MiniMax-H3-Acc-LoRAs-sidecar](https://huggingface.co/fbjr/MiniMax-H3-Acc-LoRAs-sidecar).
- **FastH3:** [FastVideo/FastVideo-FastH3-Comfy](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy), unchanged.
- **The node** is a trimmed copy of `MiniMaxH3PDDLoRA` and
  `MiniMaxH3LoRABranch` from
  [ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations),
  where the recipes, the records behind them and the parity check live.
- The judged renders also ran Sol-Attn, that repo's sparse attention. These
  workflows use ComfyUI's kitchen attention instead.

## License

Code: MIT (`LICENSE`). The models are MiniMax H3 Model Derivatives under the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
