#!/usr/bin/env python3
"""#38: which save format removes the dark blocking, at what size.

O1 (`bench/results/2026-09-27_o1_lossless.md`) put the dark-region blocking
on the save, not the model: the VAE's own frames show none, and the house
format (VHS `video/h264-mp4`, `yuv420p`, crf 19) adds it. This decodes the
first `--latent-frames` latent frames of each saved latent to lossless pixels
with the video VAE, then encodes those same frames with each candidate
format and reads them back, so every candidate is judged against one
reference with no render.

Each candidate's ffmpeg arguments are VHS's `main_pass` for that format
(`ComfyUI-VideoHelperSuite/video_formats/*.json`), fed as VHS feeds them:
8-bit `rgb24` on stdin at `h3_config.FPS`. The two `ffv1` rows are lossless
codecs after the YUV conversion, so they are the floor that bit depth and
4:2:0 alone set; a lossy row can only approach its floor.

Measures, per clip and candidate, against the VAE's frames:
- the block-edge ratios `o1_lossless_blocking.ratios` computes (dark and
  bright, 8 and 16 px);
- mean absolute luma error, over the frame and over the lossless frame's dark
  pixels (`measure_clip_temporal.DARK`);
- bitrate in kbit/s, video stream only (no audio is muxed).
Read back through ffmpeg with the stream's bt709 matrix and tv range stated,
so no row carries a colour-conversion error the others do not.

    python bench/encode_format_ab.py LATENT.latent [...] [--latent-frames 22] [--json OUT]

For the owner's eye (added 2026-09-27): `--latent-frames all` decodes the
whole clip, and `--keep DIR` keeps each encode as
`<latent stem>__<candidate>.<ext>`, so a pair can be watched side by side:

    python bench/encode_format_ab.py LATENT.latent --latent-frames all \
        --only h264_crf19_8bit h265_crf22_10bit --keep "$H3_COMFY_OUTPUT/Video/review_38"

On the card: the VAE decode is the only GPU work, with no server running.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(REPO / "workflows"))

from measure_clip_temporal import DARK  # noqa: E402
from o1_lossless_blocking import ratios, to_gray  # noqa: E402

TAGS = ["-vf", "scale=out_color_matrix=bt709", "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709"]
# name -> (extension, codec arguments). The lossy rows are VHS's main_pass
# for that format at the stated pix_fmt and crf (inherited); `h264_crf19_8bit`
# is what every generated graph saves today (the O1 record).
CANDIDATES = {
    "h264_crf19_8bit": ("mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19"]),
    "h264_crf14_8bit": ("mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14"]),
    "h264_crf10_8bit": ("mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10"]),
    "h264_crf19_10bit": ("mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p10le", "-crf", "19"]),
    "h265_crf22_10bit": ("mp4", ["-c:v", "libx265", "-vtag", "hvc1", "-pix_fmt", "yuv420p10le", "-crf", "22",
                                 "-preset", "medium", "-x265-params", "log-level=quiet"]),
    "h265_crf18_10bit": ("mp4", ["-c:v", "libx265", "-vtag", "hvc1", "-pix_fmt", "yuv420p10le", "-crf", "18",
                                 "-preset", "medium", "-x265-params", "log-level=quiet"]),
    "av1_crf23_10bit": ("webm", ["-c:v", "libsvtav1", "-pix_fmt", "yuv420p10le", "-crf", "23"]),
    "ffv1_floor_8bit": ("mkv", ["-c:v", "ffv1", "-pix_fmt", "yuv420p"]),
    "ffv1_floor_10bit": ("mkv", ["-c:v", "ffv1", "-pix_fmt", "yuv420p10le"]),
}


# swscale's default 10-bit to rgb24 path reads 255 back as 253 (measured on an
# ffv1 ramp, 2026-09-27; the encode itself was exact, Y 64..940). Accurate
# rounding and full chroma interpolation read it back exactly, so every row
# uses them and the 8-bit and 10-bit rows share one read-back.
READ_BACK = ("scale=in_color_matrix=bt709:in_range=tv:out_range=pc:"
             "flags=accurate_rnd+full_chroma_int,format=rgb24")


def encode(rgb: np.ndarray, fps: float, codec: list[str], out: Path) -> None:
    h, w = rgb.shape[1:3]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", *codec, *TAGS, str(out)]
    subprocess.run(cmd, input=rgb.tobytes(), check=True, env={**os.environ, "SVT_LOG": "1"})


def read_back(path: Path, n: int, h: int, w: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(n),
                          "-vf", READ_BACK,
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)[:n]


def score(ref_gray: np.ndarray, got_gray: np.ndarray) -> dict:
    err = np.abs(got_gray.astype(np.int16) - ref_gray.astype(np.int16))
    dark = ref_gray < DARK
    return {**ratios(got_gray), "mae_luma": float(err.mean()),
            "mae_luma_dark": float(err[dark].mean()) if dark.any() else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("latents", type=Path, nargs="+")
    ap.add_argument("--latent-frames", default="22",
                    help="how many latent frames to decode from the start, or `all`")
    ap.add_argument("--only", nargs="*", choices=sorted(CANDIDATES), help="a subset of the candidates")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--keep", type=Path, help="keep each encode here instead of discarding it")
    args = ap.parse_args()
    if args.latent_frames != "all" and not args.latent_frames.isdigit():
        ap.error("--latent-frames takes a count or `all`")
    frames = None if args.latent_frames == "all" else int(args.latent_frames)
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(COMFY))
    sys.argv = [sys.argv[0]]
    import comfy.options
    comfy.options.enable_args_parsing()
    import safetensors.torch
    import torch
    import comfy.sd
    import comfy.utils
    import folder_paths
    from h3_config import FPS, MODELS

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", MODELS["video_vae"])))
    names = args.only or list(CANDIDATES)
    rec = {"measured_by": "bench/encode_format_ab.py", "vae": MODELS["video_vae"], "fps": FPS,
           "latent_frames": args.latent_frames, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "candidates": {k: CANDIDATES[k][1] for k in names}, "clips": {}}
    for lat_path in args.latents:
        lat = safetensors.torch.load_file(str(lat_path))["latent_tensor"][:, :, :frames].float()
        with torch.no_grad():
            img = vae.decode(lat)
        rgb = (img.reshape(-1, *img.shape[-3:]).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        del img
        n, h, w, _ = rgb.shape
        ref = to_gray(rgb)
        clip = {"frames": n, "seconds": n / FPS, "lossless": ratios(ref), "candidates": {}}
        with tempfile.TemporaryDirectory() as td:
            for name in names:
                ext, codec = CANDIDATES[name]
                out = Path(td) / f"{name}.{ext}"
                encode(rgb, FPS, codec, out)
                row = score(ref, to_gray(read_back(out, n, h, w)))
                row["kbps"] = out.stat().st_size * 8 / (n / FPS) / 1000
                clip["candidates"][name] = row
                if args.keep:
                    shutil.copy2(out, args.keep / f"{lat_path.stem}__{name}.{ext}")
        rec["clips"][lat_path.name] = clip
        print(f"\n{lat_path.name}  ({n} frames)")
        lr = clip["lossless"]
        print(f"  {'lossless (VAE)':<18} dark8 {lr['dark_block8']:.3f}  dark16 {lr['dark_block16']:.3f}")
        for name, r in clip["candidates"].items():
            print(f"  {name:<18} dark8 {r['dark_block8']:.3f}  dark16 {r['dark_block16']:.3f}  "
                  f"mae {r['mae_luma']:.2f}  mae_dark {r['mae_luma_dark'] or 0:.2f}  {r['kbps']:.0f} kbps")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if args.json:
        args.json.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
