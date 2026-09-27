#!/usr/bin/env python3
"""The temporal side of "low bitrate": boiling texture, detail lost in motion, dark blocks.

Written 2026-09-26 for the owner's reading of FastH3 ("my eyes see it as
like... low bitrate streaming video") and their theory O1 (dark scenes going
"blocky and splotchy"), in `bench/results/2026-09-26_distill_run_predictions.md`.
`measure_clip_resolution.py` measures detail frame by frame. This measures
what only shows across frames, plus O1's dark-versus-bright blocking.

**Report only. It grades nothing and has no threshold.** On luma, over
consecutive frame pairs (`--stride` pairs apart). Farneback optical flow runs
on a half-resolution copy and warps the previous frame onto the current one.
Pixels are "static" where the flow is under `STATIC_PX` and "moving" where it
is over `MOVING_PX` (both in full-resolution pixels).

- `boil`: on static pixels, mean |Lap(cur) - Lap(warp(prev))| over mean
  |Lap(cur)|. This is fine texture that changes frame to frame on surfaces that
  do not move; 0 is perfectly stable texture.
- `motion_detail`: mean |Lap| on moving pixels over mean |Lap| on static
  pixels, i.e. how much detail survives motion. Real motion blur lowers it
  too, so read it against the base on the same scene.
- `dark_block8` / `bright_block8` (and 16): gradient strength across 8- (or
  16-) pixel boundaries over off-boundary gradient, within dark pixels (luma
  under `DARK`) and within bright pixels. An encoder blocks on 8 and 16, and
  H3's latent pixel is 16. `dark_block` well above `bright_block` is O1's
  signature. Comparing the mp4 with a lossless decode of the same latent
  separates the codec from the model; that is O1's own test.
- `band`: in smooth 16x16 tiles (std under `SMOOTH_STD`, range at least 4
  levels), the distinct luma levels over (range + 1). Low values are banding.

Medians over pairs are printed; `--json` also keeps per-pair series.

    python bench/measure_clip_temporal.py CLIP.mp4 [...] [--stride 2] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

STATIC_PX = 0.5
MOVING_PX = 3.0
DARK = 0.20 * 255
BRIGHT = 0.45 * 255
SMOOTH_STD = 3.0


def read_gray(path: Path):
    w, h = map(int, subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w)


def block_ratio(img: np.ndarray, mask: np.ndarray, period: int) -> float | None:
    gx = np.abs(np.diff(img, axis=1))
    mx = mask[:, :-1] & mask[:, 1:]
    cols = (np.arange(gx.shape[1]) + 1) % period == 0
    on = gx[:, cols][mx[:, cols]]
    off = gx[:, ~cols][mx[:, ~cols]]
    gy = np.abs(np.diff(img, axis=0))
    my = mask[:-1, :] & mask[1:, :]
    rows = (np.arange(gy.shape[0]) + 1) % period == 0
    on = np.concatenate([on, gy[rows, :][my[rows, :]]])
    off = np.concatenate([off, gy[~rows, :][my[~rows, :]]])
    if on.size < 100 or off.size < 100 or off.mean() == 0:
        return None
    return float(on.mean() / off.mean())


def band_fill(img: np.ndarray) -> float | None:
    h, w = (img.shape[0] // 16) * 16, (img.shape[1] // 16) * 16
    tiles = img[:h, :w].reshape(h // 16, 16, w // 16, 16).transpose(0, 2, 1, 3).reshape(-1, 256)
    vals = []
    for t in tiles:
        rng = int(t.max()) - int(t.min())
        if t.std() < SMOOTH_STD and rng >= 4:
            vals.append(len(np.unique(t)) / (rng + 1))
    return float(np.median(vals)) if vals else None


def measure(path: Path, stride: int) -> dict:
    frames = read_gray(path)
    series = {k: [] for k in ("boil", "motion_detail", "dark_block8", "bright_block8",
                              "dark_block16", "bright_block16", "band")}
    for i in range(stride, frames.shape[0], stride):
        prev, cur = frames[i - 1], frames[i]
        small_p = cv2.resize(prev, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        small_c = cv2.resize(cur, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(small_c, small_p, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.resize(flow, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_LINEAR) * 2.0
        mag = np.linalg.norm(flow, axis=2)
        gx, gy = np.meshgrid(np.arange(cur.shape[1], dtype=np.float32), np.arange(cur.shape[0], dtype=np.float32))
        warped = cv2.remap(prev, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        lap_c = np.abs(cv2.Laplacian(cur.astype(np.float32), cv2.CV_32F))
        lap_w = np.abs(cv2.Laplacian(warped.astype(np.float32), cv2.CV_32F))
        static, moving = mag < STATIC_PX, mag > MOVING_PX
        if static.sum() > 1000 and lap_c[static].mean() > 0:
            series["boil"].append(float(np.abs(lap_c - lap_w)[static].mean() / lap_c[static].mean()))
            if moving.sum() > 1000:
                series["motion_detail"].append(float(lap_c[moving].mean() / lap_c[static].mean()))
        f = cur.astype(np.float32)
        dark, bright = cur < DARK, cur > BRIGHT
        for p in (8, 16):
            for name, m in (("dark", dark), ("bright", bright)):
                r = block_ratio(f, m, p)
                if r is not None:
                    series[f"{name}_block{p}"].append(r)
        b = band_fill(cur)
        if b is not None:
            series["band"].append(b)
    med = {k: (float(np.median(v)) if v else None) for k, v in series.items()}
    return {"frames": int(frames.shape[0]), "pairs": len(range(stride, frames.shape[0], stride)),
            "median": med, "series": series}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = {"measured_by": "bench/measure_clip_temporal.py", "stride": args.stride,
           "thresholds": {"STATIC_PX": STATIC_PX, "MOVING_PX": MOVING_PX, "DARK": DARK,
                          "BRIGHT": BRIGHT, "SMOOTH_STD": SMOOTH_STD}, "clips": {}}
    cols = ("boil", "motion_detail", "dark_block8", "bright_block8", "dark_block16", "bright_block16", "band")
    print(f"{'clip':<60} " + " ".join(f"{c:>14}" for c in cols))
    for clip in args.clips:
        row = measure(clip, args.stride)
        out["clips"][clip.name] = row
        print(f"{clip.name[:60]:<60} " + " ".join(
            f"{'-' if row['median'][c] is None else format(row['median'][c], '.3f'):>14}" for c in cols))
    if args.json:
        args.json.write_text(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
