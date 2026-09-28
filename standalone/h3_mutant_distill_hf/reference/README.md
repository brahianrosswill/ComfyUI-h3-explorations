# Rebuilding the FlashGen files

`convert_flashgen_lora.py` turns the publisher's FlashGen LoRA into the two
files in this repo. It needs `torch`, `safetensors` and `comfy_kitchen`
(installed with ComfyUI), and runs on the CPU.

## Inputs

- `minimax_h3_4step_lora_flashgen_v1.0_768p_bf16.safetensors` from
  [Beidouqixing/minimax-h3-4step-lora-flashgen](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen);
- the release, [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3):
  its `FL2VA/` and `Ref2VA/` transformer shards, for the time embedder and
  the layout check;
- the pruned int8 checkpoints from
  [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):
  `minimax_h3_fl2va_pruned_int8_convrot.safetensors` and
  `minimax_h3_ref2va_pruned_int8_convrot.safetensors`.

## Commands

    python convert_flashgen_lora.py --partition FL2VA \
        --lora minimax_h3_4step_lora_flashgen_v1.0_768p_bf16.safetensors \
        --release MiniMax-H3 \
        --pruned minimax_h3_fl2va_pruned_int8_convrot.safetensors \
        --out minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy.safetensors

    python convert_flashgen_lora.py --partition Ref2VA \
        --lora minimax_h3_4step_lora_flashgen_v1.0_768p_bf16.safetensors \
        --release MiniMax-H3 \
        --pruned minimax_h3_ref2va_pruned_int8_convrot.safetensors \
        --out minimax_h3_flashgen_4step_v1.0_768p_ref2va_pruned_rank64_comfy.safetensors

The script checks the q/k/v and fc1 layout against the checkpoint, the
modulation fit, and that each emitted delta equals the source's, and refuses
to write a file that fails.

A rebuild matches the published file tensor for tensor, but not byte for
byte: the safetensors header can come out in another order. Compare tensors:

    python -c "import sys, torch; from safetensors.torch import load_file as l; \
    a, b = l(sys.argv[1]), l(sys.argv[2]); \
    print(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a))" rebuilt.safetensors published.safetensors

The modulation fit is a float64 least-squares solve, so another machine's
math library may differ in the last bits.
