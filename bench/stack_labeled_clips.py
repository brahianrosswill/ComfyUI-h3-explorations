#!/usr/bin/env python3
"""Stack N clips vertically with a caption on each, for side-by-side viewing.

`stack_eval_clips.py` takes two clips and can blind them; this takes any
number, never blinds, and burns a caption into each band, for the case
where the viewer already knows the arms and wants them lined up. Video
only: the stack carries no audio stream, so a judgement about sound goes
back to the singles.

    python bench/stack_labeled_clips.py -o out.mp4 \
        clip_a.mp4 "default: no rebalance | seed 730451892 | 508 s" \
        clip_b.mp4 "rebalanced | ..." clip_c.mp4 "bf16 on 45/48/49 | ..."

Each clip is scaled to the first clip's width; the caption sits in a
translucent band at the top of its clip. ffmpeg with drawtext is required.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", metavar="CLIP LABEL", help="clip path and its caption, alternating")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--font", default=FONT)
    ap.add_argument("--font-size", type=int, default=30)
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()
    if len(args.pairs) % 2 or len(args.pairs) < 4:
        raise SystemExit("give at least two CLIP LABEL pairs")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not on PATH")
    clips = args.pairs[0::2]
    labels = args.pairs[1::2]
    inputs = []
    for c in clips:
        inputs += ["-i", c]
    band = args.font_size + 24
    chains = []
    for i, label in enumerate(labels):
        text = label.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        chains.append(
            f"[{i}:v]scale=iw:ih,"
            f"drawbox=x=0:y=0:w=iw:h={band}:color=black@0.55:t=fill,"
            f"drawtext=fontfile='{args.font}':text='{text}':fontsize={args.font_size}:"
            f"fontcolor=white:x=16:y=12[v{i}]"
        )
    stack = "".join(f"[v{i}]" for i in range(len(clips))) + f"vstack=inputs={len(clips)}[out]"
    graph = ";".join(chains + [stack])
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", graph, "-map", "[out]", "-an",
           "-c:v", "libx264", "-crf", str(args.crf), "-preset", "medium", "-pix_fmt", "yuv420p", args.output]
    print(" ".join(cmd[:3]), "...", args.output)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
