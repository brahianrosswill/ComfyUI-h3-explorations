"""Where a frozen-audio loop's files go, and what they carry.

`docs/h3_audio_freeze.md` owns the lane. The two nodes in this pack that write
video themselves, `MiniMaxH3AudioFreezeSong` and `MiniMaxH3JoinWindows`, write
through here so the two cannot drift apart.

**The finished file** is `<prefix>_NNNNN.mp4`, the counter from core's
`folder_paths.get_save_image_path`. Not VHS's `-audio.mp4` spelling: core's
counter parses the digits after the prefix up to the next `_` or `.`, so a lone
`<prefix>_00001-audio.mp4` does not advance it, and with the PNG switched off
the next run would overwrite the last (tested against
`folder_paths.py::get_save_image_path`, 2026-09-14).

**Metadata** rides in the mp4 as a `comment` tag holding the prompt graph and
the workflow, in VHS's own shape and escaping
(`comfyui-videohelpersuite/videohelpersuite/nodes.py::ffmpeg_process`), so what
reads a VHS file reads these. `write_metadata_png` adds `<prefix>_NNNNN.png`,
the first frame carrying the same chunks, which is what a drag into the
frontend reads.

**Window files** live in `<prefix>_windows/`, one per window slot and
overwritten in place by the next run of the same graph. The folder name has no
digits after the prefix, so core's counter ignores it.
"""

from __future__ import annotations

import contextlib
import datetime
import io as _bytes_io
import json
import os
import subprocess

from comfy_api.latest import io, ui
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .audio_freeze import _ffmpeg, _write_wav


def window_dir(full_out: str, filename: str) -> str:
    """The working folder for a graph's window files, beside its finished files."""
    return os.path.join(full_out, f"{filename}_windows")


def window_path(work_dir: str, filename: str, number: int) -> str:
    """Window `number`, counted from 1."""
    return os.path.join(work_dir, f"{filename}_window_{int(number)}.mp4")


def _metadata_payload(prompt, extra_pnginfo) -> dict:
    # VHS's `video_metadata`: the prompt graph as a JSON string, every
    # extra_pnginfo entry (the workflow) as its own value
    payload = {}
    if prompt is not None:
        payload["prompt"] = json.dumps(prompt)
    for key, value in (extra_pnginfo or {}).items():
        payload[key] = value
    return payload


def _write_ffmetadata(path: str, payload: dict) -> None:
    # ffmpeg's metadata file format escapes = ; # \ and newline (VHS's order)
    text = json.dumps(payload)
    for old, new in (("\\", "\\\\"), (";", "\\;"), ("#", "\\#"), ("=", "\\="), ("\n", "\\\n")):
        text = text.replace(old, new)
    with open(path, "w") as f:
        f.write(";FFMETADATA1\ncomment=" + text)


def join_and_mux(files: list[str], waveform, rate: int, out_path: str, scratch_dir: str, stem: str,
                 prompt=None, extra_pnginfo=None) -> None:
    """Concatenate `files` without re-encoding, mux the track cut to the video, embed the metadata.

    The windows must share every muxer setting or the stream copy fails. The
    concat list, the track and the metadata file are written to `scratch_dir`
    and removed whether or not ffmpeg succeeds.
    """
    list_path = os.path.join(scratch_dir, stem + "_concat.txt")
    wav_path = os.path.join(scratch_dir, stem + "_track.wav")
    meta_path = os.path.join(scratch_dir, stem + "_metadata.txt")
    try:
        with open(list_path, "w") as f:
            for p in files:
                f.write("file '" + p.replace("'", "'\\''") + "'\n")
        _write_wav(wav_path, waveform, rate)
        _write_ffmetadata(meta_path, _metadata_payload(prompt, extra_pnginfo))
        cmd = [_ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path,
               "-i", wav_path, "-i", meta_path, "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "2",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
               "-metadata", "creation_time=now", out_path]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed joining into {out_path}: "
                               f"{proc.stderr.decode(errors='replace')[-400:]}")
    finally:
        for p in (list_path, wav_path, meta_path):
            with contextlib.suppress(FileNotFoundError):
                os.remove(p)


def write_metadata_png(png_path: str, video_path: str, prompt=None, extra_pnginfo=None) -> str:
    """The video's first frame as a PNG carrying the prompt and workflow chunks, as VHS writes it."""
    proc = subprocess.run([_ffmpeg(), "-v", "error", "-i", video_path, "-frames:v", "1",
                           "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not read the first frame of {video_path}: "
                           f"{proc.stderr.decode(errors='replace')[-400:]}")
    info = PngInfo()
    if prompt is not None:
        info.add_text("prompt", json.dumps(prompt))
    for key, value in (extra_pnginfo or {}).items():
        info.add_text(key, json.dumps(value))
    info.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])
    with Image.open(_bytes_io.BytesIO(proc.stdout)) as frame:
        # compress_level 4: inherited from VHS's metadata image
        frame.save(png_path, pnginfo=info, compress_level=4)
    return png_path


def saved_outputs(out_path: str, subfolder: str, png_path: str | None):
    """The VHS_FILENAMES value (the PNG first, the video last, as VHS lists them) and the preview."""
    filenames = (True, ([png_path] if png_path else []) + [out_path])
    preview = ui.PreviewVideo([ui.SavedResult(os.path.basename(out_path), subfolder, io.FolderType.output)])
    return filenames, preview
