"""A bench-only ComfyUI node: save the exact conditioning and latent a sampler would get.

NOT part of the pack. `nodes.py` never imports it; a server sees it only when
launched with an `--extra-model-paths-config` whose `custom_nodes` entry names
`bench/comfy_capture_nodes/`, which `bench/measure_encoder_quant_dit.py capture`
writes and prints. The directory is read from `H3_BENCH_CAPTURE_DIR` in the
server's environment so no path is ever typed into a graph.

The whole conditioning list is pickled with `torch.save`, extras included
(`minimax_refs`, `minimax_keyframes`, `minimax_token_tags`), because the DiT
reads all of them: a saved text tensor alone cannot drive a ref2va forward.
"""

import os

import torch


class H3BenchSaveConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",),
                             "samples": ("LATENT",),
                             "name": ("STRING", {"default": "capture"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "h3/bench"

    def save(self, conditioning, samples, name):
        out = os.environ.get("H3_BENCH_CAPTURE_DIR")
        if not out:
            raise RuntimeError("H3_BENCH_CAPTURE_DIR is not set in the server's environment")
        os.makedirs(out, exist_ok=True)

        def cpu(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            if isinstance(x, dict):
                return {k: cpu(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return type(x)(cpu(v) for v in x)
            return x

        torch.save({"conditioning": cpu(conditioning), "samples": cpu(samples)},
                   os.path.join(out, f"{name}.pt"))
        return {}


NODE_CLASS_MAPPINGS = {"H3BenchSaveConditioning": H3BenchSaveConditioning}
