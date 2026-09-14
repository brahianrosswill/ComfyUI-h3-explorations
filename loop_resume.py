"""Resume a frozen-audio loop from the first window that changed.

`docs/h3_audio_freeze.md` owns the lane; `audio_freeze_song.py` is the node
that uses this. Imports nothing from the pack, so
`bench/check_audio_freeze.py` exercises it without the node.

**What a window's key says.** A window is reused when the key stored beside
it equals the key this run computes for it, and a key is built from what
produced the window:

- the root: a hash of everything upstream of the song node in the queued API
  prompt (models, LoRA, attention settings, loaders, sampler, sigmas,
  references) plus the song node's own literal inputs that reach every
  window (canvas, context, mask, level, crf), minus `SONG_PER_WINDOW` and the
  file-only switches; and a hash of the track's samples;
- the window's own text, frame count, start and seed;
- the previous window's key, because a window's head is the previous
  window's tail.

The song node's prompt text, `extent`, window length, seed and modes are
deliberately not in the root. They reach each window through its own text,
frames, start and seed, so editing a later prompt block or covering more of
the track leaves the earlier windows' keys alone.

**Reuse is a prefix.** Windows are reused in order until the first one whose
key does not match, and everything from there renders again. A window that
renders again need not reproduce its old latent bit for bit, so a later
stored window, keyed on the same inputs, could sit on a slightly different
head; the prefix rule never reuses past a render.

**What the key cannot see.** A file replaced on disk under the same name (a
checkpoint, a LoRA, a reference still) leaves every key unchanged. The node's
`reuse_windows` switch is the way out.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch

import comfy.nested_tensor
import comfy.utils

#: Song-node inputs that reach a window only through that window's own text,
#: frames, start and seed, or that change files beside the windows and not the
#: windows. `lists` fills the text, so a changed list moves only the windows
#: whose filled-in text changed. Reasoned, from `MiniMaxH3AudioFreezeSong.execute`.
SONG_PER_WINDOW = ("prompt", "extent", "extent.seconds", "window_frames", "seed",
                   "prompt_mode", "window_mode", "filename_prefix", "save_metadata_png",
                   "keep_windows", "reuse_windows", "lists")


def _is_link(value) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], (str, int)) and isinstance(value[1], int))


def graph_signature(prompt, node_id, skip=()) -> str | None:
    """A hash of `node_id` and everything upstream of it in an API prompt.

    Node ids are replaced by visit order, as core's own cache key does
    (`comfy_execution/caching.py::CacheKeySetInputSignature`), so renumbering
    a graph changes nothing. `skip` names inputs of `node_id` itself to leave
    out. None when the prompt or the node is missing: no key, no reuse.
    """
    if not isinstance(prompt, dict) or node_id is None or str(node_id) not in prompt:
        return None
    order: dict[str, int] = {}
    records = []

    def visit(nid) -> int:
        nid = str(nid)
        if nid in order:
            return order[nid]
        node = prompt.get(nid)
        if not isinstance(node, dict):
            raise KeyError(nid)
        order[nid] = len(order)
        inputs = {}
        for name, value in sorted((node.get("inputs") or {}).items()):
            if nid == str(node_id) and name in skip:
                continue
            inputs[name] = ["link", visit(value[0]), value[1]] if _is_link(value) else value
        records.append((order[nid], node.get("class_type"), inputs))
        return order[nid]

    try:
        visit(node_id)
    except KeyError:
        return None
    blob = json.dumps(sorted(records, key=lambda r: r[0]), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def track_hash(waveform: torch.Tensor, rate: int) -> str:
    h = hashlib.sha256(str(int(rate)).encode())
    h.update(waveform.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def root_key(signature: str | None, track: str) -> str | None:
    return None if signature is None else hashlib.sha256(f"{signature}:{track}".encode()).hexdigest()


def window_key(root: str, number: int, text: str, frames: int, start: float, seed: int,
               previous: str | None) -> str:
    blob = json.dumps({"root": root, "window": int(number), "text": text, "frames": int(frames),
                       "start": round(float(start), 6), "seed": int(seed), "previous": previous},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def window_paths(work_dir: str, filename: str, number: int) -> tuple[str, str]:
    """(video, latent) for window `number`, counted from 1."""
    base = os.path.join(work_dir, f"{filename}_window_{int(number)}")
    return base + ".mp4", base + ".safetensors"


def save_window(work_dir: str, filename: str, number: int, key: str, samples, trim: int,
                next_start: float) -> str:
    """Store a rendered window's sampled latent and what the next window needs from it.

    Written after the window's video, so a latent on disk means its video
    finished.
    """
    _video, latent_path = window_paths(work_dir, filename, number)
    streams = samples.unbind() if getattr(samples, "is_nested", False) else [samples]
    tensors = {f"stream_{i}": s.detach().cpu().contiguous() for i, s in enumerate(streams)}
    meta = {"key": key, "trim": str(int(trim)), "next_start": repr(float(next_start)),
            "nested": str(bool(getattr(samples, "is_nested", False)))}
    comfy.utils.save_torch_file(tensors, latent_path, metadata=meta)
    return latent_path


def read_window(work_dir: str, filename: str, number: int) -> dict | None:
    """The stored key, trim and next start of window `number`, without loading its tensors."""
    video_path, latent_path = window_paths(work_dir, filename, number)
    if not (os.path.isfile(video_path) and os.path.isfile(latent_path)):
        return None
    try:
        from safetensors import safe_open
        with safe_open(latent_path, framework="pt") as f:
            meta = f.metadata() or {}
        return {"key": meta["key"], "trim": int(meta["trim"]), "next_start": float(meta["next_start"]),
                "video": video_path, "latent": latent_path}
    except Exception:  # noqa: BLE001 -- an unreadable store is a store to render over, not an error
        return None


def load_window_latent(latent_path: str) -> dict:
    """A stored window's sampled latent as the `previous` the window node takes."""
    sd, meta = comfy.utils.load_torch_file(latent_path, return_metadata=True)
    streams = [sd[k] for k in sorted(sd, key=lambda k: int(k.split("_")[1]))]
    nested = (meta or {}).get("nested") == "True"
    samples = comfy.nested_tensor.NestedTensor(streams) if nested else streams[0]
    return {"samples": samples}
