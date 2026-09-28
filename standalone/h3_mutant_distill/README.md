# h3-mutant-distill

**Experimental. YMMV.** Combinations of MiniMax H3's few-step distills for
ComfyUI, and one node to run them. Each was judged by eye (blinded as best as I could),
by me, often on 1-2 seeds max, on an RTX 4090 at 1344x768 and 345 frames.

## Models

One row per adapter. Nothing here was trained: each file is someone else's
distill, converted for ComfyUI or used as published. The adapters are built
for, and tested only on, the pruned int8 convrot H3 checkpoints.

| file (folder) | what it does | how it was made | used by |
|---|---|---|---|
| [`minimax_h3_fl2va_pdd_8step_comfy.safetensors`](https://huggingface.co/fbjr/h3-mutant-distill/resolve/main/minimax_h3_fl2va_pdd_8step_comfy.safetensors) (`loras/`) | PDD8: alibaba-pai's Parallel Decoding Distillation at 8 steps. A backbone LoRA, a modulation update, and 32 per-interval output heads; each step uses the heads for the slice of the trajectory it covers | [alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs), FL2VA 8-step, converted: diffusers names to ComfyUI's, q/k/v fused, SwiGLU halves reordered, alpha tensors added, the modulation update pre-solved into the pruned checkpoint's 8-column time basis | `pdd8_flashgen_finish`, `pdd6` |
| [`minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy.safetensors`](https://huggingface.co/fbjr/h3-mutant-distill/resolve/main/minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy.safetensors) (`loras/`) | FlashGen: a 4-step distill LoRA for text to video, trained by distribution matching (VSD, data-free) | [Beidouqixing/minimax-h3-4step-lora-flashgen](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen) at its full rank 64, converted: keys renamed with an alpha per module, q/k/v rows reordered to ComfyUI's layout, the modulation update re-expressed in the pruned fl2va checkpoint's 8-column time basis | `pdd8_flashgen_finish`, `flashgen_late_blocks` |
| [`fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors`](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/resolve/main/diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors) (`diffusion_models/`) | FastH3: FastVideo's 8-step distill of H3 for text to video. A full checkpoint, not an adapter | FastVideo's own (DMD2, data-free, trained with VSA sparse attention), used unchanged | `fasth3_contract` |

## Recipes

Text to video for now but entirely possible this works for i2va and ref2va, just haven't tested it enough. 
One workflow each in `example_workflows/`, with a note on why.

| workflow | what | model evaluations |
|---|---|---|
| `h3_t2v_pdd8_flashgen_finish` | PDD8 from sigma 1.0 to 0.8, then FlashGen to 0 | 6 + 2 |
| `h3_t2v_pdd6` | the PDD8 LoRA on a 6-step schedule. Only for close-ups and low-motion scenes, and iffy even there; included anyway | 6 |
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

- The models: see [Models](#models).
- **The node** is a trimmed copy of `MiniMaxH3PDDLoRA` and
  `MiniMaxH3LoRABranch` from
  [ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations),
  where the recipes, the records behind them and the parity check live.
- The judged renders also ran Sol-Attn, that repo's sparse attention. These
  workflows use ComfyUI's kitchen attention instead.

## License

Code: MIT (`LICENSE`). The models are MiniMax H3 Model Derivatives under the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
