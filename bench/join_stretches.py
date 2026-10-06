#!/usr/bin/env python3
"""Join the windows of several song-node runs over consecutive stretches of one clip into one file.

    <comfy venv python> bench/join_stretches.py --clip <the source clip> --start 36.0 --end 157.0 \\
        --stretch <run 1's _windows folder>@36.0 --stretch <run 2's _windows folder>@75.875 ... \\
        --out <one file>.mp4 [--review] [--overwrite]

A clip too long to load whole is rendered as several runs, each over its own stretch
(`audio_freeze_song.py` works a window at a time, but the frames a graph loads are held whole).
Each run joins its own windows under its own stretch of the track. This joins all of them: every
stretch's window files, in order, copied without re-encoding, under the clip's own audio for the
whole span, extracted once. It calls the song node's own join (`loop_output.join_and_mux`), so the
file is made the way a single run's is. Do not concatenate the runs' finished files instead: their
audio would be joined at every stretch boundary.

`--stretch FOLDER@SECONDS` is a run's windows folder and the clip time of the first frame that run
WROTE: its loader's `start_time` for a run that starts cold, and that plus the context for a run
continued from the stretch before it. `--review` joins the windows' `_with_mask` files instead.

When the stretches are rendered, give each run the seed of the run before it plus the number of
windows that run rendered: inside one run window k samples on seed + k - 1, so a second stretch
queued on the first one's seed would sample its first window on the first stretch's first seed
again. Nothing here reads the seed; it is said here because this is where the stretches meet.

It refuses, and writes nothing, when what it is given does not add up to the span: a folder with no
windows or a gap in their numbers, a folder given twice, a stretch that does not start where the one
before it ends (a stretch missing, doubled or out of order), or frames that do not end at `--end`.
What it found is in the message. `bench/check_audio_freeze.py` holds the join and each refusal.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent

WINDOW = re.compile(r"^(?P<stem>.+)_window_(?P<n>\d+)(?P<review>_with_mask)?\.mp4$")


def _pack():
    """The pack's `loop_output` as a module of a stand-in package, as the checks load the pack's modules."""
    for extra in (str(REPO), str(COMFY)):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    import comfy.cli_args
    comfy.cli_args.args.cpu = True          # nothing here touches a card
    pkg = sys.modules.get("_h3pack")
    if pkg is None:
        pkg = types.ModuleType("_h3pack")
        pkg.__path__ = [str(REPO)]
        sys.modules["_h3pack"] = pkg
    return importlib.import_module("_h3pack.loop_output")


def window_files(folder: str, review: bool = False) -> list[str]:
    """A run's window videos (or their mask reviews) in order, numbered from 1 with no gap; refused otherwise."""
    found: dict[str, dict[int, str]] = {}
    for name in sorted(os.listdir(folder)):
        m = WINDOW.match(name)
        if m and bool(m.group("review")) == bool(review):
            found.setdefault(m.group("stem"), {})[int(m.group("n"))] = os.path.join(folder, name)
    what = "mask review" if review else "window"
    if not found:
        raise ValueError(f"{folder} holds no {what} files (`<name>_window_N{'_with_mask' if review else ''}.mp4`)")
    if len(found) > 1:
        raise ValueError(f"{folder} holds the windows of more than one run ({sorted(found)}); give a folder of one run")
    numbers = next(iter(found.values()))
    if sorted(numbers) != list(range(1, len(numbers) + 1)):
        raise ValueError(f"{folder}: its {what} files are numbered {sorted(numbers)}, not 1 to {len(numbers)} without a gap")
    return [numbers[n] for n in sorted(numbers)]


def count_frames(ffmpeg: str, path: str) -> int:
    """How many video frames a file holds, counted as packets by ffmpeg itself (no ffprobe needed)."""
    out = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "framecrc", "-"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise ValueError(f"{path} could not be read as a video: {out.stderr.strip()[-200:]}")
    return sum(1 for line in out.stdout.splitlines() if line and not line.startswith("#"))


def lay_out(stretches: list[tuple[str, float]], start: float, end: float, fps: int, count, review: bool = False):
    """(every file in order, frames in all) for stretches that tile the span from `start` to `end` exactly.

    `count(path)` gives a file's frames. Refused, with what was found, unless each stretch starts within
    half a frame of where the one before it ends, the first at `start` and the last ending at `end`.
    """
    seen, files, at, table = set(), [], float(start), []
    for folder, begins in stretches:
        real = os.path.realpath(folder)
        if real in seen:
            raise ValueError(f"{folder} is given twice")
        seen.add(real)
        mine = window_files(folder, review)
        frames = sum(count(p) for p in mine)
        table.append(f"{folder}: {len(mine)} file(s), {frames} frames, from {float(begins):.4f}s")
        if abs(float(begins) - at) > 0.5 / fps:
            raise ValueError(
                f"{folder} starts at {float(begins):.4f}s and the frames before it end at {at:.4f}s: a stretch is "
                "missing, given twice or out of order, or its start is not the first frame it wrote.\n  "
                + "\n  ".join(table))
        files += mine
        at = float(begins) + frames / fps
    if abs(at - float(end)) > 0.5 / fps:
        raise ValueError(f"the stretches end at {at:.4f}s and the span was given as ending at {float(end):.4f}s.\n  "
                         + "\n  ".join(table))
    return files, int(round((at - float(start)) * fps))


def read_audio(ffmpeg: str, clip: str, start: float, seconds: float):
    """(waveform [1, 2, samples] float32, rate): the clip's own audio over the span, decoded once, at its own rate."""
    import torch
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", clip], capture_output=True, text=True).stderr
    m = re.search(r"Audio:.*?(\d+) Hz", probe)
    if m is None:
        raise ValueError(f"{clip} has no audio stream that ffmpeg can name")
    rate = int(m.group(1))
    raw = subprocess.run([ffmpeg, "-v", "error", "-ss", f"{float(start):.6f}", "-t", f"{float(seconds):.6f}", "-i", clip,
                          "-vn", "-map", "0:a:0", "-ac", "2", "-ar", str(rate), "-f", "f32le", "-"], capture_output=True)
    if raw.returncode != 0:
        raise ValueError(f"ffmpeg could not read the audio of {clip}: {raw.stderr.decode(errors='replace')[-200:]}")
    wave = torch.frombuffer(bytearray(raw.stdout), dtype=torch.float32).reshape(-1, 2).t().unsqueeze(0).contiguous()
    return wave, rate


def join(clip: str, start: float, end: float, stretches: list[tuple[str, float]], out: str,
         review: bool = False, overwrite: bool = False) -> int:
    """Write `out`; return its frame count. Nothing is written when anything is refused."""
    lo = _pack()
    ffmpeg, fps = lo._ffmpeg(), int(lo.FPS)
    if os.path.exists(out) and not overwrite:
        raise ValueError(f"{out} exists; pass --overwrite to replace it")
    files, frames = lay_out(stretches, start, end, fps, lambda p: count_frames(ffmpeg, p), review)
    wave, rate = read_audio(ffmpeg, clip, start, frames / fps)
    # written beside its final name and renamed when it is whole and counted
    part = os.path.splitext(out)[0] + ".part.mp4"
    try:
        with tempfile.TemporaryDirectory() as scratch:
            lo.join_and_mux(files, wave, rate, part, scratch, "stretches", frames)
        got = count_frames(ffmpeg, part)
        if got != frames:
            raise ValueError(f"the joined file holds {got} frames, not the {frames} its windows do")
        os.replace(part, out)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", required=True, help="the source clip, for its audio")
    ap.add_argument("--start", type=float, required=True, help="clip time of the span's first frame, seconds")
    ap.add_argument("--end", type=float, required=True, help="clip time just past the span's last frame, seconds")
    ap.add_argument("--stretch", action="append", required=True, metavar="FOLDER@SECONDS",
                    help="a run's _windows folder and the clip time of the first frame it wrote; once per run, in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--review", action="store_true", help="join the windows' _with_mask files, not their videos")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    stretches = []
    for item in args.stretch:
        folder, sep, begins = item.rpartition("@")
        if not sep:
            ap.error(f"--stretch {item!r} is not FOLDER@SECONDS")
        stretches.append((folder, float(begins)))
    try:
        frames = join(args.clip, args.start, args.end, stretches, args.out, args.review, args.overwrite)
    except ValueError as exc:
        print(f"REFUSED  {exc}")
        return 1
    print(f"wrote {args.out}: {frames} frames, {len(stretches)} stretch(es), audio from {args.clip} at {args.start:g}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
