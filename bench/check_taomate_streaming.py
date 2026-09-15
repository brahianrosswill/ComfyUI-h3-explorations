#!/usr/bin/env python3
"""CPU checks for `taomate_streaming.py`, the TaoMate-H3 streaming runtime's torch-only parts.

`docs/h3_taomate.md` section 7.3 step 1. No model, no card, no server. Each
case compares against something the module did not produce:

  sigmas         the student and teacher sigmas against upstream's float32
                 values, recorded from upstream's own `denoise_schedule.py`
                 run on 2026-09-15 (inherited constants below, not recomputed).
  chunk plan     the chunk and audio boundaries against upstream's
                 `streaming/geometry.py` tables, recorded the same way.
  core shapes    every run length against ComfyUI core's own
                 `comfy_extras/nodes_minimax_h3.py::temporal_shape`, so a
                 streaming latent is one core can build and decode.
  positions      chunk positions against ComfyUI core's `PackedLayout` for the
                 whole run: video and audio keep their global rows, text is
                 right-aligned to its request.
  attention      the split routing against one SDPA call with an explicit
                 boolean mask over the same keys.
  cache          history token counts per chunk against upstream's retention
                 arithmetic at 864x480, and the audio reset.
  colour match   the matched chunk's per-feature mean and std against the anchor.

Exit codes: 0 all pass, 1 a case failed (an unimportable ComfyUI is a failure,
not a skip: two cases grade against core).

    <comfy venv python> bench/check_taomate_streaming.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent
# ComfyUI's root FIRST and this repo's root after it: core's
# `comfy_extras/nodes_minimax_h3.py` does `import nodes`, which must find
# ComfyUI's and not this pack's (`docs/comfy_notes.md`, the `import nodes` trap).
sys.path.insert(0, str(COMFY))
sys.path.append(str(REPO))

import taomate_streaming as tm  # noqa: E402

# Inherited: upstream's float32 schedule values, printed by running
# `src/taomate_h3/denoise_schedule.py::select_time_shift_sigmas` and the
# teacher's `(10, 3.0)` call at `tm.UPSTREAM` on 2026-09-15.
UPSTREAM_STUDENT_VIDEO = [1.0, 0.96116507, 0.85333329, 0.0]
UPSTREAM_STUDENT_AUDIO = [1.0, 0.86086953, 0.59259260, 0.0]
UPSTREAM_TEACHER_AUDIO_STATES = [0.85714281, 0.59999996, 0.0]
# Inherited: `streaming/geometry.py` run at `tm.UPSTREAM` on 2026-09-15.
# Request 0 as (video latent range, audio latent range); later requests as
# their per-channel audio stops local to the request.
UPSTREAM_REQUEST0 = [((0, 12), (0, 65)), ((12, 22), (65, 122)),
                     ((22, 32), (122, 178)), ((32, 37), (178, 207))]
UPSTREAM_LATER_AUDIO_STOPS = {1: [56, 113, 170, 198], 2: [57, 113, 170, 198],
                              3: [57, 114, 170, 199]}
UPSTREAM_AUDIO_OFFSETS = [207, 405, 603, 802, 1000]
# Reasoned: float32 rounding of the schedule is below this.
SIGMA_TOL = 1e-6
# Reasoned: two float32 SDPA evaluations of the same attention agree far below this.
ATTN_TOL = 1e-5

failures: list[str] = []


def case(name, fn):
    try:
        note = fn()
        print(f"  ok    {name}" + (f"   {note}" if note else ""))
    except Exception as exc:  # a crash in a case is a failure of that case, not of the run
        failures.append(name)
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")


def _core_on_cpu():
    """Put ComfyUI core on its CPU path before anything imports `model_management`.

    Same two lines `bench/check_distill_grid.py::comfy_grid` runs; with no card
    visible, core otherwise raises at import.
    """
    try:
        import comfy.cli_args
    except ImportError as exc:
        raise AssertionError(f"ComfyUI core is not importable from {COMFY}: {exc}") from exc
    comfy.cli_args.args.cpu = True


def sigmas():
    for got, want, what in ((tm.student_sigmas(tm.SHIFT_VIDEO), UPSTREAM_STUDENT_VIDEO, "student video"),
                            (tm.student_sigmas(tm.SHIFT_AUDIO), UPSTREAM_STUDENT_AUDIO, "student audio"),
                            (tm.teacher_sigmas(tm.SHIFT_AUDIO), UPSTREAM_TEACHER_AUDIO_STATES, "teacher audio")):
        dev = max(abs(a - b) for a, b in zip(got, want))
        assert len(got) == len(want) and dev <= SIGMA_TOL, f"{what} {got} vs upstream {want}"
    manual = [float(s) for s in tm.MANUAL_SIGMAS.split(",")]
    assert max(abs(a - b) for a, b in zip(manual, UPSTREAM_STUDENT_VIDEO)) <= 1e-6, tm.MANUAL_SIGMAS


def chunk_plan():
    plan = tm.run_plan(4)
    got0 = [((c.v0, c.v1), (c.a0, c.a1)) for c in plan if c.request == 0]
    assert got0 == UPSTREAM_REQUEST0, f"request 0 {got0}"
    for r, stops in UPSTREAM_LATER_AUDIO_STOPS.items():
        chunks = [c for c in plan if c.request == r]
        start = chunks[0].a0
        got = [c.a1 - start for c in chunks]
        assert got == stops, f"request {r} audio stops {got}, upstream {stops}"
        assert [c.video_latents for c in chunks] == [10, 10, 10, 5]
    offsets = [tm.audio_boundary(tm.frame_count(r)) for r in range(1, 6)]
    assert offsets == UPSTREAM_AUDIO_OFFSETS, f"audio offsets {offsets}"
    assert tm.requests_for(37) == 1 and tm.requests_for(72) == 2 and tm.requests_for(102) is None


def core_shapes():
    _core_on_cpu()
    try:
        from comfy_extras.nodes_minimax_h3 import temporal_shape
    except ImportError as exc:
        raise AssertionError(f"ComfyUI core is not importable from {COMFY}: {exc}") from exc
    for r in range(1, 7):
        frames = tm.frame_count(r)
        f, latent_t, audio_t = temporal_shape(frames)
        assert f == frames, f"core snaps {frames} frames to {f}"
        assert latent_t == tm.total_latents(r), f"{frames} frames: core {latent_t} latents"
        assert audio_t == tm.run_plan(r)[-1].a1, f"{frames} frames: core {audio_t} audio latents"
    return f"run lengths {[tm.frame_count(r) for r in range(1, 4)]}... land on core's grid"


def positions():
    _core_on_cpu()
    try:
        from comfy.ldm.minimax.model import PackedLayout, FRAME_RESCALE
    except ImportError as exc:
        raise AssertionError(f"ComfyUI core is not importable from {COMFY}: {exc}") from exc
    text_len, lat_h, lat_w, requests = 7, 6, 10, 2
    latent_t = tm.total_latents(requests)
    audio_t = tm.run_plan(requests)[-1].a1
    full = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t)
    frame_rows = (lat_h // 2) * (lat_w // 2)
    for chunk in tm.run_plan(requests):
        pos = tm.chunk_positions(full.position_ids, text_len, audio_t, frame_rows, chunk)
        n_audio = 2 * chunk.audio_latents
        text, audio, video = pos[:text_len], pos[text_len:text_len + n_audio], pos[text_len + n_audio:]
        assert video.shape[0] == chunk.video_latents * frame_rows
        want_v = torch.tensor([text_len + FRAME_RESCALE * tm.frames_before(g)
                               for g in range(chunk.v0, chunk.v1)], dtype=torch.float64)
        got_v = video[:, 0].view(chunk.video_latents, frame_rows)
        assert torch.allclose(got_v, want_v[:, None].expand_as(got_v)), f"video t, chunk {chunk}"
        want_a = torch.arange(chunk.a0, chunk.a1, dtype=torch.float64) + text_len
        assert torch.equal(audio[:, 0], want_a.repeat(2)), f"audio t, chunk {chunk}"
        shift = float(Fraction(5, 3) * tm.frames_before(tm.request_video_start(chunk.request)))
        assert torch.allclose(text[:, 0], torch.arange(text_len, dtype=torch.float64) + shift)
        if chunk.request == 0:
            assert torch.equal(text, full.position_ids[:text_len])


def attention():
    torch.manual_seed(0)
    heads, dim, text_rows, hist_rows, media_rows = 3, 8, 5, 11, 9
    s = text_rows + media_rows
    q, k, v = (torch.randn(s, heads, dim) for _ in range(3))
    hk, hv = torch.randn(hist_rows, heads, dim), torch.randn(hist_rows, heads, dim)
    got = tm.stream_attention(q, k, v, text_rows, hk, hv)
    keys = torch.cat([k[:text_rows], hk, k[text_rows:]])
    values = torch.cat([v[:text_rows], hv, v[text_rows:]])
    mask = torch.zeros(s, keys.shape[0], dtype=torch.bool)
    mask[:text_rows, :text_rows] = True
    mask[text_rows:, :] = True
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None], keys.transpose(0, 1)[None], values.transpose(0, 1)[None],
        attn_mask=mask)[0].transpose(0, 1)
    dev = float((got - ref).abs().max())
    assert dev <= ATTN_TOL, f"split routing vs masked SDPA deviates {dev:.2e}"
    plain = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])[0].transpose(0, 1)
    assert float((tm.stream_attention(q, k, v, text_rows, text_sees_all=True) - plain).abs().max()) <= ATTN_TOL
    try:
        tm.stream_attention(q, k, v, text_rows, hk, hv, text_sees_all=True)
    except ValueError:
        pass
    else:
        raise AssertionError("text_sees_all accepted a history")
    return f"max deviation {dev:.1e}"


def cache():
    frame_rows = (480 // 32) * (864 // 32)
    blocks = 2
    c = tm.StreamCache(blocks, pin=False)
    seen = []
    for chunk in tm.run_plan(2):
        if chunk.index == 0 and chunk.request > 0 and chunk.request % tm.AUDIO_RESET_REQUESTS == 0:
            c.drop_audio()
        seen.append(c.tokens)
        a, vrows = 2 * chunk.audio_latents, chunk.video_latents * frame_rows
        c.begin_commit()
        for b in range(blocks):
            c.stage(b, torch.zeros(a + vrows, 1, 1), torch.zeros(a + vrows, 1, 1))
        c.finish_commit(a, vrows)
        c.retain()
    # Reasoned arithmetic from upstream's retention at 864x480, spec 2026-09-15:
    # request 0 chunks see 0, 4990, 9154, 13186; request 1 chunk 0 sees 11105.
    assert seen[:5] == [0, 4990, 9154, 13186, 11105], f"history tokens {seen[:5]}"
    hk, hv = c.history(0, "cpu", torch.float32)
    assert hk.shape[0] == c.tokens
    c.drop_audio()
    assert all(x["audio_rows"] == 0 for x in c.commits)
    return f"history per chunk {seen}"


def colour_match():
    torch.manual_seed(1)
    first = torch.randn(40, 96) * 2.0 + 0.5
    out, anchor = tm.match_to_anchor(first, None)
    assert torch.equal(out, first)
    later = torch.randn(30, 96) * 0.3 - 1.0
    matched, _ = tm.match_to_anchor(later, anchor)
    mean = matched.mean(0, keepdim=True)
    std = (matched - mean).pow(2).mean(0, keepdim=True).sqrt()
    assert torch.allclose(mean, anchor[0], atol=1e-4) and torch.allclose(std, anchor[1], atol=1e-4)


def main() -> int:
    print("taomate streaming, CPU")
    for name, fn in (("sigmas", sigmas), ("chunk plan", chunk_plan), ("core shapes", core_shapes),
                     ("positions", positions), ("attention", attention), ("cache", cache),
                     ("colour match", colour_match)):
        case(name, fn)
    if failures:
        print(f"\n{len(failures)} case(s) FAILED: {', '.join(failures)}")
        return 1
    print("\nall ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
