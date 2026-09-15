"""TaoLiveAIGC's TaoMate-H3 streaming runtime: the parts that need no ComfyUI.

`docs/h3_taomate.md` section 7 is the design. This module is the one home of
the runtime's inherited constants (the distilled grid, the chunk plan, the
cache policy, the audio teacher's states), and it holds the chunk geometry,
the split attention, the K/V cache, the colour matching and the frozen-track
stand-in for the teacher. The sampler node (`taomate_stream_sampler.py`), the
workflow generator, the converter and the checks all read it.

It imports nothing from ComfyUI, for the reason `h3_rules.py` gives: the node
cannot import `workflows/h3_config.py`, and both sides must agree. torch is
imported inside the functions that need it, so the generator and the checks
can read the constants without loading torch.

Every inherited value names its source, a path inside `UPSTREAM`, the revision
it was read at. Nothing here reads that tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

UPSTREAM = ("https://github.com/TaoLiveAIGC/TaoMate-H3/tree/"
            "ccc1a70adbf7f552a84a0cd7eeac0a6f3d461cad")

# ---- the distilled grid ------------------------------------------------------
#: Inherited, `src/taomate_h3/denoise_schedule.py` called from
#: `src/taomate_h3/model/pipeline.py`: a 50-point `linspace(1, 0)` base
#: schedule shifted pointwise as `s*q / (1 + (s-1)*q)`, keeping these indices,
#: at shift 12 for video and 3 for audio. Three intervals, three evaluations.
GRID_POINTS = 50
STATE_INDICES = (0, 16, 33, 49)
SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0
STEPS = len(STATE_INDICES) - 1
#: Inherited: `src/taomate_h3/model/denoise.py::minimax_h3_denoise_loop` is
#: "Euler-eta0", `x <- r*x + (1-r)*(x - sigma*v)` with `r` the sigma ratio,
#: which is ComfyUI's `euler` on a flow model.
SAMPLER = "euler"
#: Inherited: the runtime adds `update * alpha / rank` with no user multiplier
#: (`src/taomate_h3/inference/lora_checkpoint.py`).
STRENGTH = 1.0

# ---- the audio teacher -------------------------------------------------------
#: Inherited, `src/taomate_h3/teacher.py` and
#: `src/taomate_h3/inference/base10_teacher.py`: the base model with no adapter
#: runs a 10-point schedule at the audio shift, and its states 3, 6 and 9
#: replace the chunk's audio after the student's steps 1, 2 and 3.
TEACHER_POINTS = 10
TEACHER_STATES = (3, 6, 9)

# ---- chunk geometry ----------------------------------------------------------
#: Inherited, `src/taomate_h3/streaming/geometry.py` and `config.py`. The
#: frame cadence per latent is ComfyUI's too (`FRAME_PER_TOKEN` in
#: `comfy/ldm/minimax/model.py`).
FRAME_PER_LATENT = (1, 4, 4, 4, 4)
AUDIO_LATENTS_PER_SECOND = 40
VIDEO_FPS = 24
FIRST_REQUEST_LATENTS = 37
REQUEST_LATENTS = 35
PREFIX_LATENTS = 2
GROUP_LATENTS = len(FRAME_PER_LATENT)
CHUNK_GROUPS = (2, 2, 2, 1)

# ---- cache policy ------------------------------------------------------------
#: Inherited, `src/taomate_h3/streaming/cache.py::retain_sink_and_recent_commits`
#: and `config.py`: after each commit keep the first commit's video as a sink
#: plus this many most recent commits, and drop audio history at the start of
#: every request whose index is a positive multiple of the reset period.
RECENT_COMMITS = 2
AUDIO_RESET_REQUESTS = 12


def shifted_sigmas(points: int, shift: float, indices) -> list[float]:
    """The shifted `linspace(1, 0, points)` schedule at `indices`, as upstream builds it.

    Plain floats where upstream uses a float32 tensor; they differ below 1e-6.
    """
    last = points - 1
    out = []
    for index in indices:
        q = 1.0 - index / last
        out.append(shift * q / (1.0 + (shift - 1.0) * q))
    return out


def student_sigmas(shift: float) -> list[float]:
    """The adapter's retained sigmas at one shift."""
    return shifted_sigmas(GRID_POINTS, shift, STATE_INDICES)


def teacher_sigmas(shift: float) -> list[float]:
    """The audio teacher's sigma at each injected state."""
    return shifted_sigmas(TEACHER_POINTS, shift, TEACHER_STATES)


#: The video sigmas as a `ManualSigmas` string, six decimals. Only the video
#: vector is wired: core derives each audio sigma from it through
#: `time_shift_sigma` (12 to 3); the shift is pointwise over the same base
#: point, so that lands on `student_sigmas(SHIFT_AUDIO)`. Reasoned, and graded
#: by `bench/check_distill_grid.py::taomate_graphs_on_their_grid`.
MANUAL_SIGMAS = ", ".join(repr(round(s, 6)) for s in student_sigmas(SHIFT_VIDEO))


def frames_before(latent: int) -> int:
    """Pixel frames before global latent index `latent` (ComfyUI's cadence)."""
    full, rest = divmod(latent, GROUP_LATENTS)
    return full * sum(FRAME_PER_LATENT) + sum(FRAME_PER_LATENT[:rest])


def audio_boundary(frame: int) -> int:
    """The audio latent at pixel frame `frame`, rounded half to even as upstream does."""
    return round(Fraction(frame * AUDIO_LATENTS_PER_SECOND, VIDEO_FPS))


def total_latents(requests: int) -> int:
    return FIRST_REQUEST_LATENTS + (requests - 1) * REQUEST_LATENTS


def frame_count(requests: int) -> int:
    """Pixel frames of a run of `requests` requests: 124, then 119 per further one."""
    return frames_before(total_latents(requests))


def requests_for(latent_t: int) -> int | None:
    """How many requests a video latent length holds, or None if it is not a run length."""
    if latent_t < FIRST_REQUEST_LATENTS or (latent_t - FIRST_REQUEST_LATENTS) % REQUEST_LATENTS:
        return None
    return 1 + (latent_t - FIRST_REQUEST_LATENTS) // REQUEST_LATENTS


@dataclass(frozen=True)
class Chunk:
    request: int
    index: int   # within its request
    v0: int      # global video latent range [v0, v1)
    v1: int
    a0: int      # global audio latent range [a0, a1), per channel
    a1: int

    @property
    def video_latents(self) -> int:
        return self.v1 - self.v0

    @property
    def audio_latents(self) -> int:
        return self.a1 - self.a0


def run_plan(requests: int) -> list[Chunk]:
    """Every chunk of a run, in order, on the global video and audio timelines."""
    if requests < 1:
        raise ValueError("a streaming run needs at least one request")
    chunks, v = [], 0
    for r in range(requests):
        for c, groups in enumerate(CHUNK_GROUPS):
            n = groups * GROUP_LATENTS + (PREFIX_LATENTS if r == 0 and c == 0 else 0)
            chunks.append(Chunk(r, c, v, v + n,
                                audio_boundary(frames_before(v)),
                                audio_boundary(frames_before(v + n))))
            v += n
    return chunks


def request_video_start(request: int) -> int:
    """The global video latent where request `request` begins."""
    return 0 if request == 0 else total_latents(request)


def chunk_positions(full_positions, text_len: int, audio_t: int, frame_rows: int, chunk: Chunk):
    """`[text | audio ch0 | audio ch1 | video]` positions for one chunk.

    Sliced from the run's full `PackedLayout.position_ids`, whose segments for
    a text-to-audio-video graph are text, then `2 * audio_t` audio rows
    channel-major, then the video rows. Audio and video keep their global
    positions. Text is right-aligned to the start of the chunk's request, as
    upstream places the current prompt: `vpos(request start) + arange(L)`,
    which for the first request is the full layout's own text positions.
    """
    import torch

    audio_start = text_len
    video_start = text_len + 2 * audio_t
    text = full_positions[:text_len].clone()
    text[:, 0] += float(Fraction(5, 3) * frames_before(request_video_start(chunk.request)))
    ch0 = full_positions[audio_start + chunk.a0:audio_start + chunk.a1]
    ch1 = full_positions[audio_start + audio_t + chunk.a0:audio_start + audio_t + chunk.a1]
    video = full_positions[video_start + chunk.v0 * frame_rows:video_start + chunk.v1 * frame_rows]
    return torch.cat([text, ch0, ch1, video])


def _sdpa(q, k, v):
    import torch.nn.functional as F

    # [S, H, D] rows -> SDPA's [1, H, S, D] and back
    out = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0),
                                         k.transpose(0, 1).unsqueeze(0),
                                         v.transpose(0, 1).unsqueeze(0))
    return out[0].transpose(0, 1)


def stream_attention(q, k, v, text_rows: int, history_k=None, history_v=None,
                     text_sees_all: bool = False):
    """Upstream's attention routing for one chunk forward, `[S, H, D]` in and out.

    Rows are `[text | media]`. Text queries attend only to text; media queries
    attend to `[text | history | media]`, with no mask
    (`src/taomate_h3/streaming/attention_hook.py`). Keys carry their positions
    already, so the concatenation order changes nothing.

    `text_sees_all` is ComfyUI's stock routing, every row attending to every
    row. It exists only for the whole-clip equality check and refuses a
    history, which stock routing has no place for.
    """
    import torch

    if text_sees_all:
        if history_k is not None:
            raise ValueError("text_sees_all is the stock whole-clip routing; it takes no history")
        return _sdpa(q, k, v)
    text_out = _sdpa(q[:text_rows], k[:text_rows], v[:text_rows])
    if history_k is None:
        keys, values = k, v
    else:
        keys = torch.cat([k[:text_rows], history_k, k[text_rows:]])
        values = torch.cat([v[:text_rows], history_v, v[text_rows:]])
    media_out = _sdpa(q[text_rows:], keys, values)
    return torch.cat([text_out, media_out])


class StreamCache:
    """Per-block K/V of committed chunks, held off the card and streamed per block.

    One commit is, per block, the post-norm, post-RoPE K and the V of a chunk's
    media rows, in packed order: audio channel 0, audio channel 1, video.
    """

    def __init__(self, blocks: int, store_device: str = "cpu", pin: bool = True):
        import torch

        self.blocks = blocks
        self.store_device = torch.device(store_device)
        self.pin = bool(pin and self.store_device.type == "cpu" and torch.cuda.is_available())
        self.commits: list[dict] = []
        self._staged: dict | None = None

    def _store(self, t):
        import torch

        if self.store_device.type == "cpu":
            out = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=self.pin)
            out.copy_(t)
            return out
        return t.detach().to(self.store_device, copy=True)

    def begin_commit(self) -> None:
        self._staged = {"k": [None] * self.blocks, "v": [None] * self.blocks}

    @property
    def committing(self) -> bool:
        return self._staged is not None

    def stage(self, block: int, k, v) -> None:
        if self._staged is None:
            raise RuntimeError("stage() outside a commit")
        self._staged["k"][block] = self._store(k)
        self._staged["v"][block] = self._store(v)

    def finish_commit(self, audio_rows: int, video_rows: int) -> None:
        staged, self._staged = self._staged, None
        if staged is None:
            raise RuntimeError("finish_commit() without begin_commit()")
        missing = [i for i, t in enumerate(staged["k"]) if t is None]
        if missing:
            raise RuntimeError(f"commit is missing blocks {missing[:5]}")
        rows = audio_rows + video_rows
        bad = [i for i, t in enumerate(staged["k"]) if t.shape[0] != rows]
        if bad:
            raise RuntimeError(f"commit rows disagree with the chunk at blocks {bad[:5]}")
        self.commits.append({"k": staged["k"], "v": staged["v"],
                             "audio_rows": audio_rows, "video_rows": video_rows})

    def abort_commit(self) -> None:
        self._staged = None

    @staticmethod
    def _video_only(commit: dict) -> dict:
        a = commit["audio_rows"]
        if a == 0:
            return commit
        return {"k": [t[a:] for t in commit["k"]], "v": [t[a:] for t in commit["v"]],
                "audio_rows": 0, "video_rows": commit["video_rows"]}

    def retain(self) -> None:
        """Keep the first commit's video as the sink plus the most recent commits."""
        if len(self.commits) <= RECENT_COMMITS:
            return
        self.commits = [self._video_only(self.commits[0])] + self.commits[-RECENT_COMMITS:]

    def drop_audio(self) -> None:
        self.commits = [self._video_only(c) for c in self.commits]

    @property
    def tokens(self) -> int:
        return sum(c["audio_rows"] + c["video_rows"] for c in self.commits)

    def history(self, block: int, device, dtype):
        """This block's cached K and V on `device`, or `(None, None)` with no commits."""
        import torch

        if not self.commits:
            return None, None
        ks = [c["k"][block].to(device=device, dtype=dtype, non_blocking=True) for c in self.commits]
        vs = [c["v"][block].to(device=device, dtype=dtype, non_blocking=True) for c in self.commits]
        return torch.cat(ks), torch.cat(vs)


def match_to_anchor(rows, anchor):
    """Upstream's per-chunk colour matching on patchified video rows `[N, 96]`.

    Returns `(rows, anchor)`. The first call sets the anchor to this chunk's
    per-feature mean and population standard deviation and returns the rows
    unchanged; later calls map each feature onto it
    (`src/taomate_h3/streaming/runtime.py`, `affine_to_first_stream_chunk`).
    """
    cur = rows.float()
    mean = cur.mean(0, keepdim=True)
    std = (cur - mean).pow(2).mean(0, keepdim=True).sqrt().clamp_min(1e-6)
    if anchor is None:
        return rows, (mean, std)
    out = (cur - mean) / std * anchor[1] + anchor[0]
    return out.to(rows.dtype), anchor


def frozen_track_states(track, noise, sigmas) -> list:
    """The frozen track at each teacher state's audio sigma, standing in for the teacher.

    Flow-matching interpolation `sigma * noise + (1 - sigma) * track`. At the
    last state the sigma is zero and this is the clean track.
    """
    return [float(s) * noise + (1.0 - float(s)) * track for s in sigmas]
