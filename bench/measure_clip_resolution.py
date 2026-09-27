#!/usr/bin/env python3
"""How much real detail a clip carries, per clip and per region: its effective resolution.

Written 2026-09-26 for the owner's words: FastH3 "perceptively looked like it
was lower resolution and then some parts of pdd did. Maybe that's because the
pixels near each other didn't have enough difference? Like what could have
been 128x128 of a gray suit has way more pixels where the ones next to it are
way more similar to it." A patch looks low-resolution when it has little
detail at fine scales. Throw its fine pixels away, scale it back up, and
nothing is lost. That is measurable.

**Report only. It grades nothing and has no threshold.** Per clip, over every
`EVERY`-th frame at the clip's own resolution, on luma Y (BT.709 weights on
the sRGB-coded values):

- `loss2` / `loss4`: RMS error between each frame and itself downscaled by 2
  (or 4) with area averaging and scaled back up bilinearly. Low `loss4` means
  the frame holds little detail finer than four pixels: in effect a
  quarter-resolution picture. Compare clips, not absolute values.
- `hf`: the share of the frame's spectral energy, DC excluded, above a quarter
  of the Nyquist frequency.
- `flat`: the share of 32x32 tiles whose luma standard deviation is below
  `FLAT_STD`, the owner's "gray suit" patches. Genuinely flat content (a wall,
  a sky) scores here too, so read it against the scene.
- `eff_res`: the median over 64x64 tiles of the largest factor, from 1, 2, 4
  and 8, whose down-and-up loss stays under `EFF_TOL`. A tile at 4 carries
  about a quarter of its pixels' worth of detail.
- `block16`, `block32`, `block64`: how much stronger the horizontal and
  vertical gradients are on a 16-, 32- or 64-pixel grid than off it (1.0
  means no grid). 16 is H3's latent pixel, 32 a DiT token (2x2 latent patch),
  64 a VSA cube's spatial side. Seams at exactly those spacings point at the
  model, not at the codec, whose blocks are 8 and 16.

- `poster`: the share of 32x32 tiles with strong contrast (luma std above
  `POSTER_MACRO`) but almost no pixel-scale texture (mean absolute difference
  from a 3x3 box blur below `POSTER_MICRO`). Flat patches with hard edges
  between them: posterized, flat-shaded "polygons". High-frequency energy
  (`hf`) cannot see this, because the hard edges add energy while the patches
  hold none. Added after native crops at 10 s showed FastH3's stairs as flat
  orange blotches, the owner's "low res ps2 polygons".
- `motion_sharp`: Laplacian energy in pixels that moved (frame-to-frame
  difference above `MOVE`) over Laplacian energy in pixels that did not. Below
  1 means moving regions are blurrier than still ones: resolution lost to
  motion blur, which the crops showed on PDD8's running figures.

**What it has not yet shown, 2026-09-26.** On the subway and diner clips, no
whole-clip column puts FastH3 lowest in detail: it carries the most
fine-scale energy of the four models. At the owner's "stairs at 10 s", a
region check (in session, not this tool) found FastH3's brightness grainier
than the base's, about 1.6 times the pixel-scale texture, and neither its
brightness nor its colour posterized. A frame-to-frame texture correlation to
tell detail from shimmer was confounded: each model framed that moment
differently, and one camera was moving. So region measures must follow each
model's own shots, with camera motion removed (optical flow), before they can
compare models. Until then, "looks low-resolution" is not measured by this
tool.

`--map OUT.png` also writes, per clip, a heatmap of `eff_res` per 64x64 tile
over the mean frame, so the low-resolution regions can be seen.

    <comfy venv python> bench/measure_clip_resolution.py [--json OUT] [--map DIR] <clip.mp4> ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

#: Frames sampled: every Nth. **Reasoned**: 15 frames of a 345-frame clip at
#: full resolution is enough to average over motion and cheap to process.
EVERY = 24
#: A 32x32 tile is "flat" below this luma std. **Reasoned**: about 2.5 code
#: values of 255, under what reads as texture on a monitor.
FLAT_STD = 0.01
#: The down-and-up RMS loss under which a tile is said to hold no detail finer
#: than that factor. **Reasoned**: about 2 code values of 255.
EFF_TOL = 0.008
TILE_FLAT, TILE_EFF = 32, 64
#: Posterized tile thresholds. **Reasoned**: a tile spanning about 13 code
#: values of 255 in luma std, whose pixels differ from their 3x3 neighbourhood
#: mean by under about 1.5 code values.
POSTER_MACRO, POSTER_MICRO = 0.05, 0.006
#: A pixel "moved" when it changes by more than this between consecutive
#: frames. **Reasoned**: about 10 code values of 255, over compression noise.
MOVE = 0.04


def frames_pairs(path: Path, w: int, h: int) -> np.ndarray:
    """Consecutive frame pairs, each (n, n+1) with n a multiple of EVERY."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                          f"select=lt(mod(n\\,{EVERY})\\,2)", "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255.0
    y = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    return y[: len(y) // 2 * 2].reshape(-1, 2, h, w)


def _box3(y: np.ndarray) -> np.ndarray:
    p = np.pad(y, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return sum(p[:, i:i + y.shape[1], j:j + y.shape[2]] for i in range(3) for j in range(3)) / 9.0


def _lap(y: np.ndarray) -> np.ndarray:
    return np.abs(4 * y[:, 1:-1, 1:-1] - y[:, :-2, 1:-1] - y[:, 2:, 1:-1]
                  - y[:, 1:-1, :-2] - y[:, 1:-1, 2:])


def frames(path: Path) -> np.ndarray:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                          f"select=not(mod(n\\,{EVERY}))", "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def down_up(y: np.ndarray, k: int) -> np.ndarray:
    """Area-downscale by k, bilinear back up to the crop that tiles by k."""
    import torch
    t = torch.from_numpy(y).unsqueeze(1)
    h, w = y.shape[-2] // k * k, y.shape[-1] // k * k
    t = t[..., :h, :w]
    d = torch.nn.functional.avg_pool2d(t, k)
    u = torch.nn.functional.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
    return (u.squeeze(1) - t.squeeze(1)).numpy()


def tile_loss(err: np.ndarray, tile: int) -> np.ndarray:
    n, h, w = err.shape
    h, w = h // tile * tile, w // tile * tile
    e = err[:, :h, :w].reshape(n, h // tile, tile, w // tile, tile)
    return np.sqrt((e ** 2).mean(axis=(0, 2, 4)))


def blockiness(y: np.ndarray, period: int) -> float:
    gx = np.abs(np.diff(y, axis=2)).mean(axis=(0, 1))    # per column boundary
    gy = np.abs(np.diff(y, axis=1)).mean(axis=(0, 2))    # per row boundary
    def ratio(g):
        idx = np.arange(len(g))
        on = g[(idx + 1) % period == 0]
        off = g[(idx + 1) % period != 0]
        return on.mean() / max(off.mean(), 1e-9)
    return float((ratio(gx) + ratio(gy)) / 2)


def measure(path: Path, map_dir: Path | None) -> dict:
    y = frames(path)
    loss = {k: down_up(y, k) for k in (2, 4, 8)}
    spec = np.abs(np.fft.rfft2(y - y.mean(axis=(1, 2), keepdims=True))) ** 2
    fy = np.abs(np.fft.fftfreq(y.shape[1]))[:, None]
    fx = np.fft.rfftfreq(y.shape[2])[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2) / 0.5
    hf = float(spec[:, r > 0.25].sum() / max(spec.sum(), 1e-12))
    n, h, w = y.shape
    th, tw = h // TILE_FLAT * TILE_FLAT, w // TILE_FLAT * TILE_FLAT
    tiles = y[:, :th, :tw].reshape(n, th // TILE_FLAT, TILE_FLAT, tw // TILE_FLAT, TILE_FLAT)
    flat = float((tiles.std(axis=(2, 4)) < FLAT_STD).mean())
    eff = np.ones(tile_loss(loss[2], TILE_EFF).shape)
    for k in (2, 4, 8):
        eff = np.where(tile_loss(loss[k], TILE_EFF) < EFF_TOL, k, eff)
    micro = np.abs(y - _box3(y))
    mt = micro[:, :th, :tw].reshape(n, th // TILE_FLAT, TILE_FLAT, tw // TILE_FLAT, TILE_FLAT).mean(axis=(2, 4))
    macro = tiles.std(axis=(2, 4))
    poster = float(((macro > POSTER_MACRO) & (mt < POSTER_MICRO)).mean())
    pairs = frames_pairs(path, w, h)
    moved = np.abs(pairs[:, 1] - pairs[:, 0])[:, 1:-1, 1:-1] > MOVE
    lap = _lap(pairs[:, 1])
    ms = float(lap[moved].mean() / max(lap[~moved].mean(), 1e-9)) if moved.any() else float("nan")
    out = {
        "clip": path.name, "size": [w, h], "frames_sampled": n,
        "poster": round(poster, 4), "motion_sharp": round(ms, 3),
        "moved_share": round(float(moved.mean()), 3),
        "loss2": round(float(np.sqrt((loss[2] ** 2).mean())), 4),
        "loss4": round(float(np.sqrt((loss[4] ** 2).mean())), 4),
        "hf": round(hf, 4), "flat": round(flat, 4),
        "eff_res": float(np.median(eff)),
        "eff_res_share_ge4": round(float((eff >= 4).mean()), 3),
        "block16": round(blockiness(y, 16), 3), "block32": round(blockiness(y, 32), 3),
        "block64": round(blockiness(y, 64), 3),
        "eff_map": eff.astype(int).tolist(),
    }
    if map_dir is not None:
        _heatmap(y.mean(axis=0), eff, map_dir / f"{path.stem}_effres.png")
    return out


def _heatmap(mean_frame: np.ndarray, eff: np.ndarray, out: Path) -> None:
    """The mean frame in grey, each 64x64 tile tinted by its eff_res: none at 1,
    yellow at 2, orange at 4, red at 8."""
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = mean_frame.shape
    rgb = np.repeat(mean_frame[..., None], 3, axis=-1)
    tint = {2: (1.0, 0.9, 0.2), 4: (1.0, 0.55, 0.1), 8: (0.9, 0.1, 0.1)}
    for (i, j), k in np.ndenumerate(eff):
        if k in tint:
            sl = rgb[i * TILE_EFF:(i + 1) * TILE_EFF, j * TILE_EFF:(j + 1) * TILE_EFF]
            sl[:] = 0.55 * sl + 0.45 * np.array(tint[int(k)])
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-i", "-", str(out)], input=img.tobytes(), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--map", type=Path, default=None)
    args = ap.parse_args()
    keys = ["loss4", "hf", "flat", "poster", "motion_sharp", "moved_share",
            "block16", "block32", "block64"]
    print(f"{'clip':<52}" + "".join(f"{k:>18}" for k in keys))
    rows = []
    for clip in args.clips:
        r = measure(clip, args.map)
        rows.append(r)
        print(f"{clip.stem[:52]:<52}" + "".join(f"{r[k]:>18}" for k in keys))
    if args.json:
        args.json.write_text(json.dumps({"tool": Path(__file__).name, "every": EVERY,
                                         "flat_std": FLAT_STD, "eff_tol": EFF_TOL,
                                         "rows": rows}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
