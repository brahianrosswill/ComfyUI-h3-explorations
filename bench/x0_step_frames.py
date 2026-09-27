#!/usr/bin/env python3
"""Where in time each sampling step's x0 prediction still moves. CPU only.

Written 2026-09-26 for the owner's 1 s clone on `subway_chase__pdd8`
(fastdude's x0 capture, `step_x0_observer.py`). Given the per-step x0 video
latents and the final latent of one render, it reports per LATENT frame:

- `to_final`: ||x0_step - final|| / ||final|| for every step;
- `change`: ||x0_step - x0_prev|| / ||final||, what that step moved.

A structure that enters late (a doubled figure appearing in the last steps)
shows as late-step `change` concentrated on the frames it occupies. Latent
frames map to video frames by H3's temporal packing (1, 4, 4, 4, 4 per five
latent frames; `comfy/ldm/minimax/model.py::_video_t_spans`), so each row
carries its video frame range and time at `h3_config.FPS`.

`--preview a:b` also writes one PNG per step of latent frames a..b-1 projected
to RGB with core's `MiniMaxH3Video.latent_rgb_factors`: coarse, 1/16 scale,
enough to see a second figure appear, not to judge quality.

    python bench/x0_step_frames.py FINAL.latent STEP_00.latent ... [--preview 5:10 --preview-dir DIR] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import safetensors.torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

PATTERN = (1, 4, 4, 4, 4)


def frame_ranges(n: int) -> list[tuple[int, int]]:
    out, start = [], 0
    for k in range(n):
        w = PATTERN[k % 5]
        out.append((start, start + w - 1))
        start += w
    return out


def load(p: Path) -> np.ndarray:
    return safetensors.torch.load_file(str(p))["latent_tensor"][0].double().numpy()   # [C, T, H, W]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("final", type=Path)
    ap.add_argument("steps", type=Path, nargs="+")
    ap.add_argument("--preview", default=None, help="latent frame range a:b")
    ap.add_argument("--preview-dir", type=Path, default=None)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    final = load(args.final)
    steps = [load(p) for p in sorted(args.steps)]
    T = final.shape[1]
    fnorm = np.sqrt((final ** 2).sum(axis=(0, 2, 3)))                         # per latent frame
    to_final = [np.sqrt(((x - final) ** 2).sum(axis=(0, 2, 3))) / fnorm for x in steps]
    change = [np.sqrt(((b - a) ** 2).sum(axis=(0, 2, 3))) / fnorm for a, b in zip(steps, steps[1:])]
    ranges = frame_ranges(T)
    fps = h3_config.FPS

    if change:
        print(f"{T} latent frames, {len(steps)} steps; last-step change by latent frame (top 8):")
        last = change[-1]
        for k in np.argsort(last)[::-1][:8]:
            a, b = ranges[k]
            print(f"  latent {k:3d}  video {a}-{b}  t={a / fps:.2f}s  change {last[k]:.4f}  "
                  f"median over frames {np.median(last):.4f}")
    rec = {"measured_by": "bench/x0_step_frames.py", "final": args.final.name,
           "steps": [p.name for p in sorted(args.steps)], "fps": fps,
           "latent_frames": [{"k": k, "video_frames": list(ranges[k]),
                              "t_s": round(ranges[k][0] / fps, 3),
                              "to_final": [round(float(x[k]), 5) for x in to_final],
                              "change": [round(float(x[k]), 5) for x in change]} for k in range(T)],
           "median_change_per_step": [round(float(np.median(c)), 5) for c in change]}

    if args.preview:
        from PIL import Image
        sys.path.insert(0, str(REPO.parents[1]))
        from comfy.latent_formats import MiniMaxH3Video
        fmt = MiniMaxH3Video()
        W = np.array(fmt.latent_rgb_factors)                                   # [C, 3]
        bias = np.array(fmt.latent_rgb_factors_bias)
        a, b = (int(x) for x in args.preview.split(":"))
        out = args.preview_dir or args.final.parent
        out.mkdir(parents=True, exist_ok=True)
        for i, x in enumerate([*steps, final]):
            tiles = []
            for k in range(a, b):
                rgb = np.einsum("chw,cr->hwr", x[:, k], W) + bias
                tiles.append(np.clip((rgb + 1) * 127.5, 0, 255).astype(np.uint8))
            img = Image.fromarray(np.concatenate(tiles, axis=1)).resize(
                (tiles[0].shape[1] * 4 * len(tiles), tiles[0].shape[0] * 4), Image.NEAREST)
            name = f"x0_step{i:02d}" if i < len(steps) else "x0_final"
            img.save(out / f"{args.final.stem}_{name}_lat{a}-{b - 1}.png")
        rec["preview"] = {"latent_frames": [a, b - 1], "dir_basename": out.name}
        print(f"previews: {out}")

    if args.json:
        args.json.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
