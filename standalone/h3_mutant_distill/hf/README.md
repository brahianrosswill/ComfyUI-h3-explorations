---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model: Beidouqixing/minimax-h3-4step-lora-flashgen
tags:
  - comfyui
  - minimax-h3
  - text-to-video
  - lora
---

# h3-mutant-distill

**Experimental. YMMV.** Weights for the recipes in
[h3-mutant-distill](https://github.com/fblissjr/h3-mutant-distill),
a ComfyUI node pack.

## Files

`minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy.safetensors`:
[Beidouqixing/minimax-h3-4step-lora-flashgen](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen)
(published under Apache-2.0) converted for ComfyUI and the pruned int8 fl2va
checkpoint
([Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)). Nothing
was trained. The conversion:

- keys renamed from PEFT to ComfyUI's, with an alpha tensor per module so
  strength 1.0 is the publisher's merge scale;
- q/k/v `lora_B` rows reordered from per-head interleaved to ComfyUI's bands;
- the modulation `lora_A` re-expressed in the pruned checkpoint's 8-column
  time basis, its mean moved into `diff_b`;
- full rank 64 kept.

The file's metadata records the source and the conversion. Load it with the
pack's `H3 Exact LoRA` node, not `LoraLoaderModelOnly`: merging into the int8
weights rounds away most of it.

The PDD8 file the recipes also use is
[fbjr/MiniMax-H3-Acc-LoRAs-sidecar](https://huggingface.co/fbjr/MiniMax-H3-Acc-LoRAs-sidecar).

## License

A MiniMax H3 Model Derivative, under the MiniMax H3 Community License
Agreement (`LICENSE`); see `NOTICE`.
