#!/usr/bin/env python3
"""Audio of render arms side by side: level, dynamics and spectrum per clip.

Written 2026-09-26 for the owner's "compare audio measurements from pdd's
before vs merged ones": PDD8 on the exact branch (0.154.0 on) against the
merged path it replaced, per scene, same prompt, length and seed. Generic:
any labels whose clips and saved audio latents sit on the output share under
`run_graph_arms.py`'s `_<label>` naming.

Per clip (the muxed `-audio.mp4`):
- EBU R128 integrated loudness (LUFS), loudness range (LU) and true peak
  (dBTP), from ffmpeg's `ebur128` filter;
- spectral centroid (Hz) and energy in four bands (dB relative to the clip's
  total), from the decoded 48 kHz mono mix.

Per scene, for each arm after the first: the relative L2 distance of its saved
AUDIO latent from the first arm's, which needs no decode.

One clip per arm: a per-scene difference is a paired draw from two samples,
not a measurement of the arm. Report how many scenes share a sign.

    python bench/compare_audio_pairs.py --scene subway_chase --scene radio_drama \\
        --arm pdd8 --arm pdd8_merge [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _paths import comfy_output  # noqa: E402

BANDS = ((20, 200), (200, 2000), (2000, 8000), (8000, 20000))
SR = 48000


def newest(root: Path, pattern: str):
    hits = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def r128(path: Path) -> dict:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn",
                          "-af", "ebur128=peak=true", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    summary = err[err.rfind("Summary:"):]
    def get(pat):
        m = re.search(pat, summary)
        if m is None:
            raise SystemExit(f"{path.name}: ffmpeg's ebur128 summary did not parse")
        return float(m.group(1))
    return {"lufs": get(r"I:\s+(-?[\d.]+) LUFS"), "lra": get(r"LRA:\s+(-?[\d.]+) LU"),
            "true_peak": get(r"Peak:\s+(-?[\d.]+) dBFS")}


def spectrum(path: Path) -> dict:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    n = 1 << 14
    frames = [x[i:i + n] * np.hanning(n) for i in range(0, len(x) - n, n // 2)]
    p = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0)
    f = np.fft.rfftfreq(n, 1 / SR)
    tot = p.sum()
    out = {"centroid_hz": float((f * p).sum() / tot)}
    for lo, hi in BANDS:
        e = p[(f >= lo) & (f < hi)].sum()
        out[f"band_{lo}_{hi}_db"] = float(10 * np.log10(max(e, 1e-30) / tot))
    return out


def audio_latent(root: Path, label: str):
    p = newest(root / "latents", f"*_audio_{label}_0*_.latent")
    if not p:
        return None
    import safetensors.torch
    return safetensors.torch.load_file(str(p))["latent_tensor"].double().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", action="append", required=True)
    ap.add_argument("--arm", action="append", required=True, help="first arm is the reference")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    root = comfy_output()
    rec = {"measured_by": "bench/compare_audio_pairs.py", "arms": args.arm, "scenes": {}}
    for scene in args.scene:
        row = {}
        ref_lat = None
        for i, arm in enumerate(args.arm):
            label = f"{scene}__{arm}"
            clip = newest(root / "Video", f"*_{label}_0*-audio.mp4")
            if not clip:
                print(f"  {label}: no clip")
                continue
            m = {"clip": clip.name, **r128(clip), **spectrum(clip)}
            lat = audio_latent(root, label)
            if i == 0:
                ref_lat = lat
            elif lat is not None and ref_lat is not None and lat.shape == ref_lat.shape:
                m["audio_latent_rel_to_first"] = float(np.linalg.norm(lat - ref_lat) / np.linalg.norm(ref_lat))
            row[arm] = m
            print(f"  {label:<34} {m['lufs']:6.1f} LUFS  LRA {m['lra']:4.1f}  TP {m['true_peak']:5.1f}  "
                  f"centroid {m['centroid_hz']:6.0f} Hz  " +
                  " ".join(f"{k.split('_')[1]}-{k.split('_')[2]} {v:5.1f}" for k, v in m.items() if k.startswith("band_")) +
                  (f"  lat {m['audio_latent_rel_to_first']:.3f}" if "audio_latent_rel_to_first" in m else ""))
        rec["scenes"][scene] = row
    if args.json:
        args.json.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
