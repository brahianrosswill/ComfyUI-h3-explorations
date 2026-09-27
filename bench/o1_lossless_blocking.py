#!/usr/bin/env python3
"""O1's own test: is dark-region blocking the model's or the video encoder's?

The owner's theory O1 (`bench/results/2026-09-26_distill_run_predictions.md`):
the distills render dark regions "blocky and splotchy". An mp4 blocks on 8 and
16 pixels, so blocking measured on the mp4 cannot tell the encoder from the
model. This decodes a slice of the render's saved video latent straight to
pixels with the video VAE (no encoder anywhere), takes the same frames from
the mp4, and compares the dark-over-bright block-edge excess
(`measure_clip_temporal.block_ratio`) on both.

Reading: if the excess is in the lossless frames, it is the model's (or the
VAE's). If it appears only in the mp4, it is the encoder's.

The slice is the first `--latent-frames` latent frames, which decode to
`(n - 2) // 5 * 17 + 5` video frames; the mp4's first as many frames are the
same frames. Runs on CPU (`--cpu`) or on the card; the VAE is
`h3_config.MODELS["video_vae"]`, what the mp4 was decoded with.

    python bench/o1_lossless_blocking.py LATENT.latent CLIP.mp4 [--latent-frames 7] [--cpu] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(REPO / "workflows"))

from measure_clip_temporal import DARK, BRIGHT, block_ratio  # noqa: E402


def ratios(gray_frames: np.ndarray) -> dict:
    out = {}
    for p in (8, 16):
        for name, sel in (("dark", lambda f: f < DARK), ("bright", lambda f: f > BRIGHT)):
            vals = [block_ratio(f.astype(np.float32), sel(f), p) for f in gray_frames]
            vals = [v for v in vals if v is not None]
            out[f"{name}_block{p}"] = float(np.median(vals)) if vals else None
    return out


def to_gray(rgb: np.ndarray) -> np.ndarray:
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("latent", type=Path)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--latent-frames", type=int, default=7)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(COMFY))
    sys.argv = [sys.argv[0]] + (["--cpu"] if args.cpu else [])
    import comfy.options
    comfy.options.enable_args_parsing()
    import safetensors.torch
    import torch
    import comfy.sd
    import comfy.utils
    import folder_paths
    from h3_config import MODELS

    lat = safetensors.torch.load_file(str(args.latent))["latent_tensor"][:, :, :args.latent_frames].float()
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", MODELS["video_vae"])))
    # no_grad, not inference_mode: on CPU the load builds Parameters inside
    # the call, which inference tensors refuse.
    with torch.no_grad():
        img = vae.decode(lat)
    img = img.reshape(-1, *img.shape[-3:])                       # [F, H, W, 3] in 0..1
    lossless = to_gray((img.clamp(0, 1) * 255).round().cpu().numpy())
    n = lossless.shape[0]
    h, w = lossless.shape[1:]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(args.clip), "-frames:v", str(n),
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    mp4 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w)[:n]

    rec = {"measured_by": "bench/o1_lossless_blocking.py", "latent": args.latent.name, "clip": args.clip.name,
           "vae": MODELS["video_vae"], "frames": n,
           "lossless": ratios(lossless), "mp4": ratios(mp4),
           "mean_abs_luma_diff_mp4_vs_lossless": float(np.abs(mp4.astype(np.int16) - lossless.astype(np.int16)).mean())}
    for k in ("lossless", "mp4"):
        r = rec[k]
        print(f"{k:<9} dark8 {r['dark_block8']:.3f} bright8 {r['bright_block8']:.3f}   "
              f"dark16 {r['dark_block16']:.3f} bright16 {r['bright_block16']:.3f}")
    print(f"mean |mp4 - lossless| luma: {rec['mean_abs_luma_diff_mp4_vs_lossless']:.3f} over {n} frames")
    if args.json:
        args.json.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
