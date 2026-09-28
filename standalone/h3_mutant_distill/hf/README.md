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

## h3 mutant distills

Experimental distill adapters for h3 in comfyui safetensor format. YMMV.
Requires using nodes from
[h3-mutant-distill](https://github.com/fblissjr/h3-mutant-distill) because
ComfyUI's LoRA loader merges a LoRA into the int8 checkpoint by
requantizing it, which rounds away most of these adapters; the node applies
them at the call instead.

## Files

Both are [Beidouqixing/minimax-h3-4step-lora-flashgen](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen)
(published under Apache-2.0), converted for ComfyUI. Nothing was trained.
Load them with the pack's `H3 Exact LoRA` node, not `LoraLoaderModelOnly`:
merging into int8 weights rounds away most of the LoRA.

### FlashGen for fl2va (text to video)

[`minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy.safetensors`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy.safetensors)

- **Loads on:** `minimax_h3_fl2va_pruned_int8_convrot`.
- **Taken from FlashGen:** every module of the LoRA (attention q/k/v and
  output, MLP fc1 and fc2, each block's and the final layer's modulation,
  the token refiner) at its full rank 64. Nothing is dropped or resized.
- **Taken from the checkpoint:** its 8-column time basis (`adaln_t_table`),
  which the modulation part is refit onto.
- **The gist:** FlashGen is a 4-step distill of H3 for text to video, trained
  by distribution matching (VSD, data-free). It renders in 4 steps alone. In
  the recipes it also finishes PDD8's last two steps, where it fixed sign
  text PDD8 garbled.

### FlashGen for ref2va (reference to video)

[`minimax_h3_flashgen_4step_v1.0_768p_ref2va_pruned_rank64_comfy.safetensors`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/minimax_h3_flashgen_4step_v1.0_768p_ref2va_pruned_rank64_comfy.safetensors)

- **Loads on:** `minimax_h3_ref2va_pruned_int8_convrot`.
- **Taken from FlashGen:** the same modules, at the same full rank.
- **Taken from the checkpoint:** ref2va's own time basis. fl2va's and
  ref2va's differ, so the two files are not interchangeable.
- **The gist:** an untested transfer. FlashGen was trained only for text to
  video on fl2va. One render held both references and the likeness by eye;
  it has not been compared against PDD8.

### How both were converted

- keys renamed from PEFT to ComfyUI's, with an alpha tensor per module so
  strength 1.0 is the publisher's merge scale;
- q/k/v `lora_B` rows reordered from per-head interleaved to ComfyUI's bands;
- the modulation `lora_A` re-expressed in the checkpoint's 8-column time
  basis, its mean moved into `diff_b`.

Each file's metadata records its source and conversion. The converter is
[`reference/convert_flashgen_lora.py`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/reference/convert_flashgen_lora.py),
with the commands to rebuild both files in
[`reference/README.md`](https://huggingface.co/fbjr/h3-mutant-distill/blob/main/reference/README.md).

### PDD8

The PDD8 sidecars the recipes also use are on
[fbjr/MiniMax-H3-Acc-LoRAs-sidecar](https://huggingface.co/fbjr/MiniMax-H3-Acc-LoRAs-sidecar):
[`minimax_h3_fl2va_pdd_8step_comfy.safetensors`](https://huggingface.co/fbjr/MiniMax-H3-Acc-LoRAs-sidecar/blob/main/minimax_h3_fl2va_pdd_8step_comfy.safetensors)
for fl2va and
[`minimax_h3_ref2va_pdd_8step_comfy.safetensors`](https://huggingface.co/fbjr/MiniMax-H3-Acc-LoRAs-sidecar/blob/main/minimax_h3_ref2va_pdd_8step_comfy.safetensors)
for ref2va. They are alibaba-pai's PDD8 LoRAs, converted by
[`bench/convert_pdd_lora.py`](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/bench/convert_pdd_lora.py).

## License

A MiniMax H3 Model Derivative, under the MiniMax H3 Community License
Agreement (`LICENSE`); see `NOTICE`.
