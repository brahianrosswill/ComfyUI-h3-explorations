"""Keep a subject mask on disk across runs, so a restart does not pay for the tracker again.

Masked video to video (`video_mask.py`) needs one mask per source frame.
Making it is the slow part of a test render that is not sampling: the SAM 3
checkpoint load, the tracker over every frame, and for `head and hair` a
detector pass per phrase on every frame the subject is in
(`bench/results/2026-10-04_masked_v2v_band_arms.jsonl`, `per_node_s`). ComfyUI
already keeps a node's output while the server stays up and its inputs do not
change; a restart for new code throws that away, and the mask of a clip does
not change between the arms of a test.

So `MiniMaxH3MaskedSource` keeps its finished mask here and asks core for its
`mask`, `segmenter` and `segmenter_clip` inputs only when nothing kept matches
(`check_lazy_status`). On a hit core never runs the nodes behind them.

**No bookkeeping** (the owner's rule for these nodes, 2026-10-03): nothing to
save, load or name. A mask is found by a key made from what decides it:

- every node upstream of the Masked Source and their settings, and the Masked
  Source's own settings that change the mask (`loop_resume.graph_signature`,
  with the settings that do not change it left out by the caller);
- the frames themselves, by a fingerprint of a strided sample, so a different
  video under the same file name is a different key;
- the size and modification time of any input file an upstream node names, so
  a mask painted in another program and saved again is a different key.

A changed setting is a different key, so a kept mask cannot be stale for the
settings it is asked for; it is only ever unused, and unused ones go first.

**Exact.** The mask is stored as it was computed, float32, losslessly
compressed. A hit returns the same bytes the tracker gave, so a render with a
kept mask is the render with a tracked one.

**Where.** ComfyUI's user directory, `h3_masked_source/`: local disk, not the
media share, kept across restarts (the temp directory is emptied at start).
`STORE_BYTES` caps it, least recently used out first.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch

from . import loop_resume

logger = logging.getLogger(__name__)

#: Disk the kept masks may take, in bytes. Reasoned: a tracked mask of a
#: half-minute clip compresses to tens of megabytes at most (soft edges are
#: the only part that is not runs of zeros and ones), so this holds every clip
#: of a working week and is small beside one checkpoint.
STORE_BYTES = 4 * 1024 ** 3
DIRNAME = "h3_masked_source"
#: Frames sampled for the fingerprint, and the spatial stride. Reasoned: enough
#: that two different clips cannot share a key by accident, small enough to
#: hash in milliseconds; it is not a proof of equality, the settings and the
#: file's size and time are in the key beside it.
FINGERPRINT_FRAMES = 64
FINGERPRINT_STRIDE = 8
#: What a kept file that is damaged or cut short raises on reading.
UNREADABLE = (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile, zlib.error)


def root() -> Path:
    import folder_paths
    return Path(folder_paths.get_user_directory()) / DIRNAME


def frames_fingerprint(frames: torch.Tensor) -> str:
    """A hash of the frames' shape and of a strided sample of them."""
    step = max(1, int(frames.shape[0]) // FINGERPRINT_FRAMES)
    sample = frames[::step, ::FINGERPRINT_STRIDE, ::FINGERPRINT_STRIDE].detach().to(torch.float32).cpu().contiguous()
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{tuple(frames.shape)}|{step}|".encode())
    h.update(memoryview(sample.numpy()).cast("B"))
    return h.hexdigest()


def _upstream(prompt, node_id) -> list[dict]:
    """The node and every node upstream of it in an API prompt, in visit order."""
    seen, out = set(), []

    def visit(nid):
        nid = str(nid)
        node = prompt.get(nid)
        if nid in seen or not isinstance(node, dict):
            return
        seen.add(nid)
        out.append(node)
        for value in (node.get("inputs") or {}).values():
            if loop_resume._is_link(value):
                visit(value[0])

    visit(node_id)
    return out


def input_file_stats(prompt, node_id) -> list:
    """(name, size, modified) for every input-folder file an upstream node names."""
    import folder_paths
    stats = []
    for node in _upstream(prompt, node_id):
        for value in (node.get("inputs") or {}).values():
            if not isinstance(value, str) or not value or len(value) > 512:
                continue
            try:
                if not folder_paths.exists_annotated_filepath(value):
                    continue
                st = os.stat(folder_paths.get_annotated_filepath(value))
            except (OSError, ValueError):
                continue
            stats.append((value, int(st.st_size), int(st.st_mtime_ns)))
    return sorted(set(stats))


def mask_key(prompt, node_id, frames: torch.Tensor, skip=()) -> str | None:
    """The key a mask is kept under, or None when there is no queued prompt to read (a direct call).

    `skip` names the node's own inputs that do not change its mask.
    """
    signature = loop_resume.graph_signature(prompt, node_id, skip=skip)
    if signature is None:
        return None
    blob = json.dumps({"graph": signature, "frames": frames_fingerprint(frames),
                       "files": input_file_stats(prompt, node_id)}, sort_keys=True)
    return hashlib.blake2b(blob.encode(), digest_size=20).hexdigest()


def _path(key: str) -> Path:
    return root() / f"{key}.npz"


def has(key: str | None, shape) -> bool:
    """True when a mask of `shape` ([N, H, W]) is kept under `key`. Reads the header only."""
    if key is None:
        return False
    path = _path(key)
    try:
        with np.load(path) as z:
            return tuple(int(v) for v in z["shape"]) == tuple(int(v) for v in shape)
    except UNREADABLE:
        return False


def load(key: str | None, shape) -> torch.Tensor | None:
    """The kept mask, float32 [N, H, W], or None. A file that does not read is removed."""
    if key is None:
        return None
    path = _path(key)
    try:
        with np.load(path) as z:
            if tuple(int(v) for v in z["shape"]) != tuple(int(v) for v in shape):
                return None
            mask = torch.from_numpy(z["mask"])
    except FileNotFoundError:
        return None
    except UNREADABLE as exc:
        logger.warning("[h3] kept mask %s does not read (%s); removed", path.name, exc)
        path.unlink(missing_ok=True)
        return None
    os.utime(path)                      # read is use: least recently used goes first
    return mask


def save(key: str | None, mask: torch.Tensor) -> float:
    """Keep `mask` ([N, H, W]) under `key`, then drop the least recently used past the budget. Seconds taken."""
    if key is None:
        return 0.0
    t0 = time.perf_counter()
    folder = root()
    folder.mkdir(parents=True, exist_ok=True)
    data = mask.detach().to(torch.float32).cpu().contiguous().numpy()
    tmp = folder / f".{key}.{os.getpid()}.tmp.npz"
    try:
        np.savez_compressed(tmp, mask=data, shape=np.asarray(data.shape, dtype=np.int64))
        os.replace(tmp, _path(key))     # a reader sees the whole file or none of it
    finally:
        tmp.unlink(missing_ok=True)
    evict()
    return time.perf_counter() - t0


def evict(budget: int | None = None) -> list[str]:
    """Remove the least recently used kept masks until the folder is inside the budget. Names removed."""
    budget = STORE_BYTES if budget is None else int(budget)
    try:
        files = sorted(root().glob("*.npz"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return []
    total = sum(p.stat().st_size for p in files)
    gone = []
    for p in files[:-1]:                # never the newest: the one just kept, or just read
        if total <= budget:
            break
        total -= p.stat().st_size
        p.unlink(missing_ok=True)
        gone.append(p.name)
    if gone:
        logger.info("[h3] kept masks: removed %d least recently used to stay inside %d bytes", len(gone), budget)
    return gone
