#!/usr/bin/env python3
"""The masked-source reduction and composite, and the ways each could keep the old subject.

`video_mask.py` is the module; `audio_freeze_song.py` uses it. Masked
video-to-video fails quietly: a subject pixel the mask loses is a piece of the
original left in the render, and the clip still plays. Each item is one way
that could happen.

1. **No subject frame is dropped in time.** A subject present in one frame of
   a run must regenerate that run's latent step. The control is core's own
   resize (`comfy.utils.reshape_mask`, trilinear) on the same mask, which
   must lose it, or this module is not buying what it says.
2. **The mask is already on core's token grid.** `token_mask`'s output run
   through `comfy/ldm/minimax/model.py::mask_row_values` comes back unchanged,
   so the region this module reports and composites is the region the model
   regenerates.
3. **The feather stays off the subject.** With `grow_pixels >= feather_pixels`
   the blend weight is exactly 1 on every original subject pixel, and exactly
   0 further than the feather from a regenerated token. The node refuses the
   other ordering.
4. **The composite returns the source where nothing regenerates**, bit for
   bit, and the render where the weight is 1.
5. **A window that outruns the source holds its last frame unmasked**, and a
   window that starts past the source is refused.
6. **The mask is cropped as the frames are, and loses nothing on the way
   down.** `fit_mask` restates core's centre crop so it can max-pool; a mask
   and its frames fitted from the same odd shape must still coincide, a
   one-pixel line must survive a large downscale (the control: core's
   bilinear resize of the same mask drops it below one half), and the node
   refuses a mask whose shape is not its frames'.
7. **The paint-out hides the subject from the encode and nothing else.**
   With `paint_out` the frames to encode differ from the fitted frames only
   under the hole, the hole covers every subject pixel, no filled pixel lies
   outside the regenerated tokens (so none can be shown), and a bright
   subject on a flat ground is gone from the encode frames. Off, the encode
   frames are the fitted frames.
8. **A part is taken only from the subject, and a missed frame is never
   left showing the original.** `select_part` keeps a part where it lies on
   the subject's mask and drops the same part on a neighbour. The region is
   the subject down to the part's lowest row (`above`), so it holds the
   subject's own pixels above that row and none below. A frame with no part
   takes the nearest found frame's row and is counted (`carry_missing`); with
   nothing found anywhere nothing is invented. The node refuses a part with
   no segmenter wired, an empty phrase list and an unknown choice, and its
   second output is the mask it used.
9. **`only what changed` keeps a subject and restores the margin.** With a
   render that equals the source except where the new subject is, the weight
   is 1 on the new subject and on every old-subject pixel inside the
   regenerated tokens, 0 in margin further than the feather and the
   difference's own blur from either, never above the whole-region weight,
   and 0 everywhere when the render repainted the source faithfully (a
   tracker's false positive). A difference under the ramp's foot is not
   kept. The node refuses an unknown choice.
10. **Every graph that wires a Masked Source wires it whole**: into a song
   node's `source`, its mask tracked over the same frames it carries, and the
   song node's track taken from the same loader as those frames.

No model, no CUDA, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_video_mask.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
WORKFLOWS = REPO / "workflows"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WORKFLOWS))
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True  # no CUDA context for a shape check; the sibling checks do the same
import comfy.utils  # noqa: E402
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, mask_row_values  # noqa: E402
import h3_config  # noqa: E402


def _load():
    """`video_mask` as a module of a stand-in package (`check_audio_freeze.py` says why)."""
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    spec = importlib.util.spec_from_file_location("_h3pack.video_mask", REPO / "video_mask.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_h3pack.video_mask"] = module
    spec.loader.exec_module(module)
    return module


vm = _load()

SOURCE = "MiniMaxH3MaskedSource"
SONG = "MiniMaxH3AudioFreezeSong"
# a small canvas on the model's grid: 16 px per latent cell, 2 cells per token
W, H, LAT_W, LAT_H = 192, 128, 12, 8
LATENT_T = 7
FRAMES = sum(vm.run_lengths(LATENT_T))


def check_temporal(problems):
    mask = torch.zeros(FRAMES, H, W)
    # one frame only, in the middle of the third run, a block the size of one cell
    run_start = sum(vm.run_lengths(2))
    frame = run_start + 1
    mask[frame, 32:48, 64:80] = 1.0
    tokens = vm.token_mask(mask, LATENT_T, LAT_H, LAT_W)
    if float(tokens[2].max()) != 1.0:
        problems.append("a subject present in one frame of a run did not regenerate that run's latent step")
    if float(tokens.sum()) != float(tokens[2].sum()):
        problems.append("a one-frame subject marked a latent step outside its own run")
    if sorted(tokens.unique().tolist()) not in ([0.0, 1.0], [1.0]):
        problems.append(f"token_mask is not binary: {tokens.unique().tolist()}")
    # the control: core's resize of the same mask
    core = comfy.utils.reshape_mask(mask, (1, 24, LATENT_T, LAT_H, LAT_W))[0, 0]
    if float(core.max()) >= 0.5:
        problems.append(
            "control failed: core's trilinear resize kept a one-frame subject at half strength or more "
            f"({float(core.max()):.3f}), so the per-run maximum is not what protects it")
    try:
        vm.token_mask(mask[:-1], LATENT_T, LAT_H, LAT_W)
        problems.append("a mask one frame short of the window was accepted")
    except ValueError:
        pass
    if tuple(FRAME_PER_TOKEN) != tuple(vm.run_lengths(len(FRAME_PER_TOKEN))):
        problems.append("run_lengths does not follow core's FRAME_PER_TOKEN")


def check_token_grid(problems):
    torch.manual_seed(0)
    for lat_h, lat_w in ((LAT_H, LAT_W), (7, 11)):  # the odd grid takes core's replicate pad
        # a few single pixels, so most tokens stay preserved and the pooling has edges to move
        mask = torch.zeros(FRAMES, lat_h * 16, lat_w * 16)
        for _ in range(6):
            f, y, x = (int(torch.randint(0, n, (1,))) for n in mask.shape)
            mask[f, y, x] = 1.0
        tokens = vm.token_mask(mask, LATENT_T, lat_h, lat_w)
        pad_h, pad_w = lat_h + lat_h % 2, lat_w + lat_w % 2
        rows = mask_row_values(tokens, LATENT_T, pad_h, pad_w)
        if rows is None:
            problems.append("core read a partly masked window as fully generating")
            continue
        again = rows.reshape(LATENT_T, pad_h // 2, pad_w // 2)
        back = again.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)[:, :lat_h, :lat_w]
        if not torch.equal(back, tokens):
            problems.append(f"core's patch pooling changes token_mask's output on a {lat_h}x{lat_w} grid")


def check_feather(problems):
    grow, feather = 32, 8
    subject = torch.zeros(FRAMES, H, W)
    subject[:, 48:80, 80:112] = 1.0
    tokens = vm.token_mask(vm.grow(subject, grow), LATENT_T, LAT_H, LAT_W)
    alpha = vm.pixel_alpha(tokens, H, W, feather)
    if tuple(alpha.shape) != (FRAMES, H, W):
        problems.append(f"pixel_alpha returned {tuple(alpha.shape)} for {FRAMES} frames of {H}x{W}")
        return
    if float(alpha[subject > 0.5].min()) != 1.0:
        problems.append("the feather reaches the subject's own pixels: the original would show at its edge")
    region = torch.nn.functional.interpolate(tokens.unsqueeze(1), size=(H, W), mode="nearest")[:, 0]
    region = region.repeat_interleave(torch.tensor(vm.run_lengths(LATENT_T)), dim=0)
    far = vm.grow(region, feather) < 0.5
    if far.any() and float(alpha[far].max()) != 0.0:
        problems.append("the blend weight is above 0 further than the feather from a regenerated token")
    # the node refuses a feather wider than the grow
    try:
        vm.MiniMaxH3MaskedSource.execute(torch.zeros(2, 8, 8, 3), torch.zeros(2, 8, 8), grow_pixels=4, feather_pixels=8)
        problems.append("the node accepted a feather wider than the grow")
    except ValueError:
        pass
    try:
        vm.MiniMaxH3MaskedSource.execute(torch.zeros(3, 8, 8, 3), torch.zeros(2, 8, 8))
        problems.append("the node accepted a mask with a different frame count from the frames")
    except ValueError:
        pass


def check_composite(problems):
    torch.manual_seed(1)
    render, source = torch.rand(FRAMES, H, W, 3), torch.rand(FRAMES, H, W, 3)
    alpha = torch.zeros(FRAMES, H, W)
    alpha[:, :64] = 1.0
    out = vm.composite(render, source, alpha)
    if not torch.equal(out[:, 64:], source[:, 64:]):
        problems.append("the composite does not return the source bit for bit where nothing regenerates")
    if not torch.equal(out[:, :64], render[:, :64]):
        problems.append("the composite does not return the render where the weight is 1")
    try:
        vm.composite(render[:-1], source, alpha)
        problems.append("a render one frame short of the source was composited")
    except ValueError:
        pass


def check_window(problems):
    have = FRAMES + 10
    frames = torch.rand(have, 72, 128, 3)
    mask = torch.zeros(have, 72, 128)
    mask[:, 20:40, 40:60] = 1.0
    source = {"frames": frames, "mask": mask, "grow_pixels": 0, "feather_pixels": 0}
    pixels, encode, tokens, _mask, held = vm.window(source, 0, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)
    if tuple(pixels.shape) != (FRAMES, H, W, 3) or tuple(tokens.shape) != (LATENT_T, LAT_H, LAT_W) or held:
        problems.append(f"a whole window came back as {tuple(pixels.shape)}, {tuple(tokens.shape)}, held {held}")
    if encode is not pixels:
        problems.append("with paint_out off the frames to encode are not the fitted frames themselves")
    # the last window of a loop: the source runs out inside it
    start = have - 12
    pixels, _encode, tokens, _mask, held = vm.window(source, start, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)
    if held != FRAMES - 12:
        problems.append(f"a window 12 frames from the source's end reported {held} held frames, expected {FRAMES - 12}")
    elif not torch.equal(pixels[12:], pixels[11:12].expand(FRAMES - 12, -1, -1, -1)):
        problems.append("the frames past the source's end are not its last frame held")
    elif float(tokens[-1].max()) != 0.0:
        problems.append("the held frames past the source's end are masked: they would regenerate")
    try:
        vm.window(source, have, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)
        problems.append("a window starting past the source's end was accepted")
    except ValueError:
        pass


def check_fit(problems):
    # a wide source onto a narrower canvas and a tall one onto a wider: both crops
    for src_h, src_w in ((270, 480), (400, 300)):
        frames = torch.zeros(1, src_h, src_w, 3)
        mask = torch.zeros(1, src_h, src_w)
        y0, y1, x0, x1 = src_h // 3, src_h // 2, src_w // 3, src_w // 2
        frames[:, y0:y1, x0:x1] = 1.0
        mask[:, y0:y1, x0:x1] = 1.0
        f = vm.fit_frames(frames, W, H)[..., 0]
        m = vm.fit_mask(mask, W, H)
        if tuple(m.shape) != (1, H, W):
            problems.append(f"fit_mask returned {tuple(m.shape)} for a {W}x{H} canvas")
            continue
        # every pixel the fitted frame shows as subject is inside the fitted mask
        if bool(((f > 0.5) & (m < 0.5)).any()):
            problems.append(f"a {src_w}x{src_h} mask and its frames do not coincide after the fit")
    line = torch.zeros(1, 1080, 1920)
    line[:, :, 1001] = 1.0
    if float(vm.fit_mask(line, W, H).max()) != 1.0:
        problems.append("a one-pixel line was lost when the mask was fitted down")
    core = comfy.utils.common_upscale(line.unsqueeze(1), W, H, "bilinear", "center")
    if float(core.max()) >= 0.5:
        problems.append("control failed: core's bilinear resize kept a one-pixel line at half strength or more")
    try:
        vm.MiniMaxH3MaskedSource.execute(torch.zeros(2, 8, 8, 3), torch.zeros(2, 8, 16))
        problems.append("the node accepted a mask whose shape is not its frames'")
    except ValueError:
        pass


def check_paint_out(problems):
    grow_px = 32
    frames = torch.full((FRAMES, H, W, 3), 0.2)
    mask = torch.zeros(FRAMES, H, W)
    mask[:, 48:80, 80:112] = 1.0
    frames[:, 48:80, 80:112] = 1.0  # a bright subject on a flat ground
    source = {"frames": frames, "mask": mask, "grow_pixels": grow_px, "feather_pixels": 8, "paint_out": True}
    pixels, encode, tokens, _mask, _held = vm.window(source, 0, FRAMES, W, H, LATENT_T, LAT_H, LAT_W)
    if not torch.equal(pixels, frames):
        problems.append("paint_out changed the frames the composite restores")
    changed = (encode != pixels).any(dim=-1)
    hole = vm.grow(mask, grow_px // 2) > 0.5
    if bool((changed & ~hole).any()):
        problems.append("paint_out changed pixels outside its hole")
    if bool(((mask > 0.5) & ~changed).any()):
        problems.append("a subject pixel survived the paint-out")
    region = torch.nn.functional.interpolate(tokens.unsqueeze(1), size=(H, W), mode="nearest")[:, 0]
    region = region.repeat_interleave(torch.tensor(vm.run_lengths(LATENT_T)), dim=0) > 0.5
    if bool((changed & ~region).any()):
        problems.append("a filled pixel lies outside the regenerated tokens: the composite would not hide it")
    if float(encode.max()) > 0.25:
        problems.append(f"the bright subject is still in the frames to encode (max {float(encode.max()):.3f})")
    # a fill of nothing is nothing
    if not torch.equal(vm.fill_subject(frames, torch.zeros(FRAMES, H, W)), frames):
        problems.append("fill_subject changed frames with an empty hole")


def check_part(problems):
    subject = torch.zeros(4, 64, 96)
    subject[:, 10:50, 20:40] = 1.0
    part = torch.zeros(4, 64, 96)
    part[0, 10:20, 22:38] = 1.0   # the subject's own, down to row 19
    part[0, 10:20, 70:86] = 1.0   # a neighbour's, far from the subject
    part[3, 10:30, 22:38] = 1.0   # frames 1 and 2 have none
    got = vm.select_part(subject, part)
    if float(got[0, 10:20, 22:38].min()) != 1.0:
        problems.append("select_part dropped the part that lies on the subject")
    if float(got[:, :, 60:].max()) != 0.0:
        problems.append("select_part kept a neighbour's part")
    bottoms = vm.part_bottom(got)
    if bottoms.tolist() != [19, -1, -1, 29]:
        problems.append(f"part_bottom read {bottoms.tolist()}, expected [19, -1, -1, 29]")
    filled, carried = vm.carry_missing(bottoms)
    if carried != 2 or filled.tolist() != [19, 19, 29, 29]:
        problems.append(f"carry_missing gave {filled.tolist()} carrying {carried}; each miss takes its nearest found frame")
    none, n = vm.carry_missing(torch.tensor([-1, -1]))
    if n != 0 or none.tolist() != [-1, -1]:
        problems.append("carry_missing invented a part where no frame had one")
    region = vm.above(subject, filled)
    if float(region[0, 10:20, 20:40].min()) != 1.0 or float(region[0, 20:].max()) != 0.0:
        problems.append("above does not hold exactly the subject's rows down to the part's lowest row")
    if float(region[:, :, 60:].max()) != 0.0 or float(region[1, 10:20, 20:40].min()) != 1.0:
        problems.append("above took pixels off the subject, or left a carried frame empty")
    for kwargs, what in (({"replace": "head and hair"}, "a part with no segmenter wired"),
                         ({"replace": "left arm"}, "an unknown replace"),
                         ({"replace": "head and hair", "segmenter": object(), "segmenter_clip": object(),
                           "part_phrases": " , "}, "an empty phrase list")):
        try:
            vm.MiniMaxH3MaskedSource.execute(torch.zeros(2, 8, 8, 3), torch.ones(2, 8, 8), **kwargs)
            problems.append(f"the node accepted {what}")
        except ValueError:
            pass
    out = vm.MiniMaxH3MaskedSource.execute(torch.zeros(2, 8, 8, 3), torch.ones(2, 8, 8))
    out = getattr(out, "args", out)
    if len(out) != 2 or not torch.equal(out[1], out[0]["mask"].to(torch.float32)):
        problems.append("the node's second output is not the mask it used")


def check_changed_alpha(problems):
    grow_px, feather = 32, 8
    old = torch.zeros(FRAMES, H, W)
    old[:, 48:80, 80:112] = 1.0
    tokens = vm.token_mask(vm.grow(old, grow_px), LATENT_T, LAT_H, LAT_W)
    source = torch.full((FRAMES, H, W, 3), 0.5)
    new = torch.zeros(FRAMES, H, W, dtype=torch.bool)
    new[:, 40:72, 96:128] = True        # the new subject, shifted: part inside the old outline, part in margin
    render = source.clone()
    render[new] = 0.9
    blank = torch.zeros(FRAMES, H, W)
    t = vm.CHANGE_THRESHOLD
    alpha = vm.changed_alpha(render, source, tokens, old, feather, grow_px // 2, t)
    whole = vm.pixel_alpha(tokens, H, W, feather)
    region = vm.pixel_alpha(tokens, H, W, 0) > 0.5
    deep = whole >= 1.0                 # further than the feather inside the region
    if tuple(alpha.shape) != (FRAMES, H, W):
        problems.append(f"changed_alpha returned {tuple(alpha.shape)}")
        return
    if bool((alpha > whole + 1e-6).any()):
        problems.append("the changed-only weight exceeds the whole-region weight somewhere")
    if float(alpha[new & deep].min()) != 1.0:
        problems.append("a new-subject pixel well inside the region is not fully kept")
    if float(alpha[(old > 0.5) & deep].min()) != 1.0:
        problems.append("an old-subject pixel is not fully kept: the original would show through")
    kept = new | (vm.grow(old, grow_px // 2) > 0.5)
    margin = region & ~(vm.grow(kept.float(), 2 * feather + vm.CHANGE_BLUR) > 0.5)
    if not bool(margin.any()):
        problems.append("the changed-alpha case has no margin to test")
    elif float(alpha[margin].max()) != 0.0:
        problems.append("margin the render left as the source keeps the render")
    # a faithful repaint with no old subject under it: everything is restored
    if float(vm.changed_alpha(source.clone(), source, tokens, blank, feather, grow_px // 2, t).max()) != 0.0:
        problems.append("a faithful repaint of the source keeps some of the render")
    # a difference under the ramp's foot is noise, not change
    faint = source + 0.5 * t
    if float(vm.changed_alpha(faint, source, tokens, blank, feather, grow_px // 2, t).max()) != 0.0:
        problems.append("a difference of half the threshold was kept")
    try:
        vm.MiniMaxH3MaskedSource.execute(torch.zeros(2, 8, 8, 3), torch.ones(2, 8, 8), composite="the nice bits")
        problems.append("the node accepted an unknown composite")
    except ValueError:
        pass


def check_motion_reference(problems):
    """`motion_reference`: `none` is None; `subject only` keeps the subject (widened by the margin) and greys the
    rest; `whole frame` keeps everything; the short edge is honoured and rounded to the canvas multiple."""
    vm = _load()
    torch.manual_seed(0)
    pixels = torch.rand(4, 64, 96, 3)
    mask = torch.zeros(4, 64, 96)
    mask[:, 16:48, 32:64] = 1.0
    if vm.motion_reference(pixels, mask, vm.MOTION_NONE, 64, 0) is not None:
        _fail(problems, "motion_reference: `none` must return None")
    whole = vm.motion_reference(pixels, mask, vm.MOTION_FRAME, 64, 0)
    if whole is None or tuple(whole.shape) != (4, 64, 96, 3) or not torch.allclose(whole, pixels, atol=1e-6):
        _fail(problems, "motion_reference: `whole frame` at the source's own short edge must be the frames unchanged")
    subject = vm.motion_reference(pixels, mask, vm.MOTION_SUBJECT, 64, 0)
    inside = subject[:, 16:48, 32:64]
    outside = torch.cat([subject[:, :16].flatten(), subject[:, 48:].flatten(), subject[:, :, :32].flatten(), subject[:, :, 64:].flatten()])
    if not torch.allclose(inside, pixels[:, 16:48, 32:64], atol=1e-6):
        _fail(problems, "motion_reference: `subject only` changed the subject's own pixels")
    if not torch.allclose(outside, torch.full_like(outside, 0.5), atol=1e-6):
        _fail(problems, "motion_reference: `subject only` must set everything outside the subject to mid grey")
    widened = vm.motion_reference(pixels, mask, vm.MOTION_SUBJECT, 64, 8)
    if torch.allclose(widened[:, 8:16, 32:64], torch.full((4, 8, 32, 3), 0.5), atol=1e-6):
        _fail(problems, "motion_reference: the margin must widen what is kept of the subject")
    small = vm.motion_reference(pixels, mask, vm.MOTION_FRAME, 32, 0)
    if tuple(small.shape) != (4, 32, 64, 3):
        _fail(problems, f"motion_reference: a 32 short edge on 64x96 frames should give 32x64, got {tuple(small.shape)}")
    try:
        vm.motion_reference(pixels, mask, "sideways", 64, 0)
        _fail(problems, "motion_reference: an unknown mode was accepted")
    except ValueError:
        pass
    if not all(k in vm.MASK_KEY_SKIP for k in ("motion_reference", "motion_short_edge", "motion_vae")):
        _fail(problems, "motion_reference's three inputs must not enter the kept mask's key: they do not change the mask")
    import reference_order as ro
    if ro.MOTION_NONE != vm.MOTION_NONE or ro.MASKED_SOURCE_CLASS != "MiniMaxH3MaskedSource":
        _fail(problems, "reference_order's literals for the Masked Source must equal video_mask's: the static label plan reads them")
    song_inputs = {"source": ["9", 0]}
    g = {"9": {"class_type": "MiniMaxH3MaskedSource", "inputs": {"motion_reference": vm.MOTION_SUBJECT}}}
    if ro.assign_labels(ro.plan_for(song_inputs, g)) != ["<Video 1>"]:
        _fail(problems, "a Masked Source with a motion reference must add one <Video N> to the song node's static label plan")
    g["9"]["inputs"]["motion_reference"] = vm.MOTION_NONE
    if ro.assign_labels(ro.plan_for(song_inputs, g)) != []:
        _fail(problems, "a Masked Source with no motion reference must add nothing to the static label plan")


def check_graphs(problems):
    seen = 0
    for path in h3_config.graph_paths(WORKFLOWS, include_bench=True):
        graph = json.loads(path.read_text())
        for nid, node in graph.items():
            if not isinstance(node, dict) or node.get("class_type") != SOURCE:
                continue
            seen += 1
            ins = node["inputs"]
            users = [n for n in graph.values() if isinstance(n, dict) and n.get("class_type") == SONG
                     and n["inputs"].get("source") == [nid, 0]]
            if not users:
                problems.append(f"{path.name}: Masked Source {nid} feeds no song node's `source`")
                continue
            frames_from = ins["frames"][0]
            # walk the mask back to the frames its tracker saw: the pack's Subject Track
            # takes them itself; core's chain reaches them through its track data
            mask_node = graph[ins["mask"][0]]
            if "track_data" in mask_node["inputs"]:
                track = graph.get(mask_node["inputs"]["track_data"][0], {})
                tracked_from = track.get("inputs", {}).get("images", [None])[0]
            else:
                tracked_from = mask_node["inputs"].get("frames", [None])[0]
            if tracked_from != frames_from:
                problems.append(f"{path.name}: the mask of Masked Source {nid} was not tracked over its own frames")
            for out in (1, 2):  # a consumer of the tracker's preview or report runs it on every queue
                if mask_node.get("class_type") == "MiniMaxH3SubjectTrack" and any(
                        v == [ins["mask"][0], out] for n in graph.values() if isinstance(n, dict)
                        for v in n.get("inputs", {}).values()):
                    problems.append(f"{path.name}: the Subject Track's preview or report is wired, which defeats the kept mask")
            for song in users:
                if song["inputs"].get("audio", [None])[0] != frames_from:
                    problems.append(f"{path.name}: the song node's track is not the audio of the source video")
                if "segmenter" not in ins or "segmenter_clip" not in ins:
                    problems.append(f"{path.name}: Masked Source {nid} has no segmenter wired, so `replace` cannot be changed without rewiring")
                if "references" not in song["inputs"]:
                    problems.append(f"{path.name}: a masked source with no reference still to replace the subject from")
    if not seen:
        problems.append("no shipped graph wires a Masked Source; item 10 checked nothing")


def main() -> int:
    problems: list[str] = []
    for check in (check_temporal, check_token_grid, check_feather, check_composite, check_window, check_fit, check_paint_out, check_part, check_changed_alpha, check_motion_reference, check_graphs):
        check(problems)
    for p in problems:
        print(f"FAIL  {p}")
    if not problems:
        print("ok    the masked source keeps every subject frame, sits on core's token grid, feathers off the "
              "subject, composites exactly, holds a short source, crops the mask as the frames, paints out only "
              "inside the regenerated tokens, takes a part only from the subject, restores the margin under "
              "`only what changed`, builds a motion reference on grey or whole at the short edge asked, and is wired whole in every graph")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
